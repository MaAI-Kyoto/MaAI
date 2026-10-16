"""Real-time 2ch (two lapel mics) speaker-bleed canceller.

BleedCanceller  : core processor working on ``block`` samples at 16 kHz
                  (bidirectional x 2-pass linear AEC + residual suppression v2)
StreamProcessor : wrapper accepting any sample rate / chunk length (resampling, HPF, blocking)

Signal flow (side A; side B is symmetric):
    A mic --delay m--> AEC1(ref = B mic)                -> e_A1 --+ (reference of B's 2nd pass)
    A mic --delay 2m-> AEC2(ref = e_B1) -> AES -> A out
    m = margin (default 32 samples = 2 ms). The mic is delayed slightly relative to the reference
    so that a causal filter can model the bleed even if the two input paths differ a little in delay.
Algorithmic latency: passes * margin + AES latency
    = 64 + (nfft - hop) + look-ahead = 64 + 384 + 0 samples = 28 ms at 16 kHz (no look-ahead by default).
"""
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from .aec import PBFDAFStream
from .aes2 import ResidualSuppressorV2Pair
from .delay import OnlineDelayEstimator
from .resample import StreamResampler

DEFAULTS = dict(
    sr=16000,
    block=256,
    margin=32,
    passes=2,
    n_part=12,                # filter length = block * n_part (192 ms at 16 kHz)
    mu_max=0.5,
    dtd_ratio_db=6.0,
    near_hold_ms=300.0,
    hpf_hz=60.0,
    aes=True,                 # False: linear AEC only (least distortion, more residual bleed)
    aes_t60=0.3,
    aes_pow_path=True,
    aes_beta=0.7,             # residual suppression strength
    aes_floor_db=-25.0,       # band gain floor during near-end speech
    aes_absent_db=-30.0,      # extra suppression while the near-end talker is silent
    aes_dd_alpha=0.6,         # decision-directed smoothing
    aes_near_thr_db=3.0,      # near-end speech decision threshold
    aes_lookahead_ms=0.0,     # look-ahead of the near-end speech decision (adds latency)
    aes_post_ms=250.0,        # hangover of the near-end speech decision
    aes_n_bands=48,           # number of ERB bands for the gain
    aes_cng=1.0,              # comfort noise level (0 disables)
    aes_refine=True,          # second stage using the partner's first-stage output
    aes_res_cap=True,         # cap the residual estimate at the observed power (prevents over-suppression
                              # of the near-end talker during double talk)
)


class DelayLine:
    def __init__(self, n):
        self.buf = np.zeros(n)

    def process(self, x):
        if len(self.buf) == 0:
            return x
        y = np.concatenate([self.buf, x])
        self.buf = y[len(x):]
        return y[: len(x)]


class VariableDelay:
    """Delay line whose delay can change on the fly (up to ``max_delay``)."""

    def __init__(self, max_delay):
        self.buf = np.zeros(max_delay)
        self.delay = 0

    def process(self, x):
        y = np.concatenate([self.buf, x])
        self.buf = y[len(x):]
        end = len(y) - self.delay
        return y[end - len(x):end]


class BleedCanceller:
    """Core real-time bleed canceller for two synchronized 16 kHz channels.

    Keyword arguments override :data:`DEFAULTS`. Call :meth:`process` with ``block`` samples
    per channel; the output is delayed by :attr:`latency` samples.
    """

    def __init__(self, **kw):
        unknown = set(kw) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"Unknown bleed canceller parameter(s): {sorted(unknown)}")
        p = dict(DEFAULTS, **kw)
        self.p = p
        B, sr = p["block"], p["sr"]
        self.block = B
        aec_kw = dict(block=B, n_part=p["n_part"], mu_max=p["mu_max"], dtd_ratio_db=p["dtd_ratio_db"],
                      near_hold_ms=p["near_hold_ms"], sr=sr)
        self.aec = [[PBFDAFStream(**aec_kw) for _ in range(2)] for _ in range(p["passes"])]  # [pass][ch]
        self.mic_delay = [[DelayLine(p["margin"] * (i + 1)) for _ in range(2)] for i in range(p["passes"])]
        self.aes2 = None
        if p["aes"]:
            self.aes2 = ResidualSuppressorV2Pair(
                block=B, sr=sr, beta=p["aes_beta"], floor_db=p["aes_floor_db"], t60=p["aes_t60"],
                dtd_ratio_db=p["dtd_ratio_db"], near_hold_ms=p["near_hold_ms"], use_pow_path=p["aes_pow_path"],
                n_bands=p["aes_n_bands"], dd_alpha=p["aes_dd_alpha"], absent_db=p["aes_absent_db"],
                near_thr_db=p["aes_near_thr_db"], lookahead_ms=p["aes_lookahead_ms"],
                post_ms=p["aes_post_ms"], cng=p["aes_cng"], refine=p["aes_refine"], res_cap=p["aes_res_cap"])
        self.delay_est = [OnlineDelayEstimator(hop=B, sr=sr, dtd_ratio_db=p["dtd_ratio_db"]) for _ in range(2)]
        # The AES reference is aligned with the estimated delay (bleed lags the reference by margin, as
        # in the offline version). The AEC reference is not re-aligned: its long filter absorbs the delay
        # and re-aligning on the fly would shift the filter taps.
        self.max_lag = self.delay_est[0].max_lag
        self.aes_ref_delay = [VariableDelay(self.max_lag) for _ in range(2)]
        aes_lat = self.aes2.latency if self.aes2 is not None else 0
        self.latency = p["passes"] * p["margin"] + aes_lat
        self.n_blocks = 0

    def process(self, a, b):
        """a, b: 16 kHz signals of length ``block``. Returns (a_out, b_out), delayed by ``latency`` samples."""
        mics = (np.asarray(a, np.float64), np.asarray(b, np.float64))
        # Delay estimation for monitoring (ch0: B->A, ch1: A->B)
        for c in range(2):
            d = self.delay_est[c].process(mics[c], mics[1 - c])
            if d is not None:
                target = int(np.clip(round(d), 0, self.max_lag))
                if abs(target - self.aes_ref_delay[c].delay) >= 2:
                    self.aes_ref_delay[c].delay = target

        refs = (mics[1], mics[0])  # 1st-pass reference = partner's raw mic
        for it, (aecs, dls) in enumerate(zip(self.aec, self.mic_delay)):
            lin, est, mic_d = [], [], []
            for c in range(2):
                m = dls[c].process(mics[c])
                e, y = aecs[c].process(m, refs[c])
                lin.append(e)
                est.append(y)
                mic_d.append(m)
            last_refs = refs
            refs = (lin[1], lin[0])  # next-pass reference = partner's linear AEC output
        self.n_blocks += 1
        if self.aes2 is None:
            return lin[0], lin[1]
        aes_refs = [self.aes_ref_delay[c].process(last_refs[c]) for c in range(2)]
        return self.aes2.process(lin, est, mic_d, aes_refs)

    def status(self):
        """Return a dict describing the current adaptation state (for monitoring)."""
        st = dict(blocks=self.n_blocks, latency_ms=1000 * self.latency / self.p["sr"],
                  delay_b_to_a=self.delay_est[0].delay, delay_a_to_b=self.delay_est[1].delay,
                  aes_ref_delay=[d.delay for d in self.aes_ref_delay],
                  aec=[[a.stats() for a in aecs] for aecs in self.aec])
        if self.aes2 is not None:
            st["aes_gain"] = list(self.aes2.last_gain)
            st["near_presence"] = list(self.aes2.last_presence)
        warn = []
        for name, d in (("B->A", st["delay_b_to_a"]), ("A->B", st["delay_a_to_b"])):
            if d is not None and d < -self.p["margin"]:
                warn.append(f"{name} delay {d:.1f} < -margin: the delay difference between the two mic "
                            f"input paths is too large (increase margin)")
        st["warnings"] = warn
        return st


class StreamProcessor:
    """Process 2ch chunks of any sample rate and length.

    ``push(x)``: x is a float array of shape (n, 2) at ``sr_in``. Returns the processed (m, 2) at
    ``out_sr`` (m varies because samples are buffered up to block boundaries).
    """

    def __init__(self, sr_in, out_sr=None, **kw):
        self.engine = BleedCanceller(**kw)
        p = self.engine.p
        self.sr = p["sr"]
        self.sr_in = sr_in
        self.out_sr = out_sr or self.sr
        self.down = StreamResampler(sr_in, self.sr, channels=2)
        self.up = StreamResampler(self.sr, self.out_sr, channels=2)
        if p["hpf_hz"]:
            self.sos = butter(2, p["hpf_hz"], "highpass", fs=self.sr, output="sos")
            self.zi = np.stack([sosfilt_zi(self.sos) * 0.0] * 2, axis=-1)  # (sections, 2, ch)
        else:
            self.sos = None
        self.pending = np.zeros((0, 2))

    def push(self, x):
        x = np.asarray(x, np.float64).reshape(-1, 2)
        y = self.down.process(x)
        if self.sos is not None and len(y):
            y, self.zi = sosfilt(self.sos, y, axis=0, zi=self.zi)
        self.pending = np.concatenate([self.pending, y], 0)
        B = self.engine.block
        nb = len(self.pending) // B
        outs = []
        for i in range(nb):
            blk = self.pending[i * B:(i + 1) * B]
            oa, ob = self.engine.process(blk[:, 0], blk[:, 1])
            outs.append(np.stack([oa, ob], 1))
        self.pending = self.pending[nb * B:]
        out = np.concatenate(outs, 0) if outs else np.zeros((0, 2))
        return self.up.process(out) if len(out) else out

    def latency_ms(self):
        """Algorithmic latency incl. block buffering and resampler group delay (excl. device buffers)."""
        lat = self.engine.latency + self.engine.block  # processing latency + max wait for a full block
        ms = 1000 * lat / self.sr
        if not self.down.passthrough:
            ms += 1000 * self.down.delay_in_samples / self.sr_in
        if not self.up.passthrough:
            ms += 1000 * self.up.delay_in_samples / self.sr
        return ms
