"""Streaming residual bleed suppressor v2 (real-time version of ``offline/aes2.py`` + refine).

Same model as the offline v2:
  ERB band gains (48 bands) / decision-directed Wiener gain / near-end speech detection with
  deep suppression of non-speech segments / comfort noise / second stage that uses the
  partner's first-stage output as its reference (refine).

Differences from the offline version:
  - The near-end speech mask look-ahead is ``lookahead_ms`` (default 0). The signal path is
    delayed by ``lookahead`` frames and the gain for the delayed frame uses speech decisions
    up to the newest frame (the look-ahead adds latency).
  - refine (stage 2) runs within the same frame: the stage-1 gains of both channels are computed
    first and the partner's stage-1 output spectrum |G1 E|^2 is used as the power-path
    reference of stage 2 (no extra latency). Stage 2 reuses the stage-1 learning and
    near-end speech decisions.

Both channels are processed together (refine needs the other channel's output).
``process()`` takes ``block`` samples; internally frames are processed every hop (= nfft/4).
Latency: (nfft - hop) + lookahead samples. Default 384 samples (24 ms at 16 kHz).
"""
from collections import deque

import numpy as np

from .._bands import _MinStat, erb_bands


class _Stage:
    """Residual bleed model of one stage (leak + power path + reverb tail) and its DD gain."""

    def __init__(self, K, W, n_pow_taps, beta, floor, decay, dd_alpha, leak_alpha, leak_min, pow_mu,
                 use_pow_path):
        self.W = W
        self.beta, self.floor, self.decay, self.dd_alpha = beta, floor, decay, dd_alpha
        self.leak_alpha, self.leak_min, self.pow_mu, self.use_pow_path = leak_alpha, leak_min, pow_mu, use_pow_path
        self.num = np.full(K, 1e-10)
        self.den = np.full(K, 1e-9)
        self.H = np.zeros((n_pow_taps, K))
        self.Xhist = np.zeros((n_pow_taps, K))
        self.R = np.zeros(K)
        self.prev = None  # (Gb, Eb, Rb)
        self.xpow = 1e-10  # long-term reference power (NLMS regularization)

    def residual(self, Ep, Yp, Xp, learn):
        self.Xhist[1:] = self.Xhist[:-1]
        self.Xhist[0] = Xp
        Hx = (self.H * self.Xhist).sum(0)
        if learn:
            self.xpow = 0.99 * self.xpow + 0.01 * Xp.sum()
            a = self.leak_alpha
            self.num = (1 - a) * self.num + a * Ep
            self.den = (1 - a) * self.den + a * Yp
            if self.use_pow_path:
                err = Ep - Hx
                reg = 1e-4 * (self.xpow / len(Xp)) ** 2
                self.H += self.pow_mu * err * self.Xhist / ((self.Xhist ** 2).sum(0) + reg + 1e-20)
                np.maximum(self.H, 0.0, out=self.H)
        leak = np.clip(self.num / self.den, self.leak_min, 1.0)
        R = np.maximum(leak * Yp, self.decay * self.R)
        if self.use_pow_path:
            R = np.maximum(R, Hx)
        self.R = R
        return R

    def gain(self, Eb, Rb):
        """Band gain (before the near-end absence suppression)."""
        b = self.beta
        gamma = Eb / (b * Rb)
        if self.prev is None:
            xi = np.maximum(gamma - 1, 0)
        else:
            Gp, Ep_, Rp = self.prev
            xi = self.dd_alpha * Gp ** 2 * Ep_ / (b * Rp) + (1 - self.dd_alpha) * np.maximum(gamma - 1, 0)
        Gb = np.maximum(xi / (1 + xi), self.floor)
        self.prev = (Gb, Eb, Rb)
        return Gb


class ResidualSuppressorV2Pair:
    def __init__(self, block=256, sr=16000, nfft=512, hop=None, beta=0.7, floor_db=-25.0, t60=0.3,
                 dtd_ratio_db=6.0, near_hold_ms=300.0, leak_alpha=0.005, leak_min=1e-3, n_pow_taps=None,
                 pow_mu=0.1, use_pow_path=True, n_bands=48, dd_alpha=0.6, absent_db=-30.0, near_thr_db=3.0,
                 lookahead_ms=0.0, post_ms=250.0, cng=1.0, refine=True, seed=0):
        hop = hop or nfft // 4
        assert block % hop == 0, "block must be a multiple of hop"
        self.block, self.hop, self.nfft = block, hop, nfft
        self.K = K = nfft // 2 + 1
        self.win = np.sqrt(np.hanning(nfft + 1)[:nfft])
        self.ola_scale = hop / (nfft / 2)
        if n_pow_taps is None:
            n_pow_taps = max(1, int(round(0.1 * sr / hop)))
        self.W, fc = erb_bands(K, sr, n_bands)
        self.speech = (fc >= 200) & (fc <= 4000)
        self.dtd = 10 ** (dtd_ratio_db / 10)
        self.hold_frames = int(near_hold_ms * 1e-3 * sr / hop)
        self.near_thr = 10 ** (near_thr_db / 10)
        self.absent = 10 ** (absent_db / 10)
        self.beta, self.cng, self.refine = beta, cng, refine
        self.L = int(round(lookahead_ms * 1e-3 * sr / hop))
        self.post = int(round(post_ms * 1e-3 * sr / hop))
        self.rel = np.exp(-hop / (0.02 * sr))  # smoothing when returning to non-speech (~20 ms)
        self.latency = (nfft - hop) + self.L * hop

        st = lambda: _Stage(K, self.W, n_pow_taps, beta, 10 ** (floor_db / 10),
                            10 ** (-6.0 * hop / sr / t60) if t60 > 0 else 0.0, dd_alpha,
                            leak_alpha, leak_min, pow_mu, use_pow_path)
        self.st1 = [st() for _ in range(2)]
        self.st2 = [st() for _ in range(2)] if refine else None
        self.ms = [_MinStat(K, sub_len=max(1, int(0.25 * sr / hop)), n_sub=6) for _ in range(2)]
        self.buf = np.zeros((2, 4, nfft))  # [ch][e, y, mic, ref]
        self.ola = np.zeros((2, nfft))
        self.hold = [0, 0]
        # Look-ahead FIFO (pre-filled with zero frames for a constant delay)
        zero = dict(E=np.zeros(K, complex), Ep=np.zeros(K), Yp=np.zeros(K), Eb=np.zeros(len(self.W)),
                    Rb1=np.full(len(self.W), 1e-20), N=np.zeros(K), learn=False, Mp=0.0)
        self.fifo = [deque([dict(zero) for _ in range(self.L)]) for _ in range(2)]
        self.raw = [deque([False] * (self.L + self.post + 1), maxlen=self.L + self.post + 1) for _ in range(2)]
        self.pres = [0.0, 0.0]
        self.rng = [np.random.default_rng(seed + c) for c in range(2)]
        self.last_gain = [1.0, 1.0]
        self.last_presence = [0.0, 0.0]

    def _analyze(self, c):
        """Analyze the newest frame (learning, residual estimate, speech decision) and push it to the FIFO."""
        fe, fy, fm, fr = self.buf[c]
        w = self.win
        E = np.fft.rfft(fe * w)
        Y = np.fft.rfft(fy * w)
        X = np.fft.rfft(fr * w)
        Ep = E.real ** 2 + E.imag ** 2
        Yp = Y.real ** 2 + Y.imag ** 2
        Xp = X.real ** 2 + X.imag ** 2
        em, er = fm @ fm, fr @ fr
        M = np.fft.rfft(fm * w)
        Mp_sum = float((M.real ** 2 + M.imag ** 2).sum())  # windowed mic power (stage-2 learning decision)
        if em > self.dtd * er:
            self.hold[c] = self.hold_frames
        elif self.hold[c] > 0:
            self.hold[c] -= 1
        learn = er > self.dtd * em and er > 1e-8 and self.hold[c] == 0
        R1 = self.st1[c].residual(Ep, Yp, Xp, learn)
        N = self.ms[c].update(Ep)
        W = self.W
        Eb, Rb1, Nb = W @ Ep, W @ R1 + 1e-20, W @ N
        b = self.beta
        sp = self.speech
        near_pow = np.maximum(Eb - 2.0 * b * Rb1 - Nb, 0.0)[sp].sum()
        self.raw[c].append(bool(near_pow > self.near_thr * (b * Rb1 + Nb)[sp].sum()))
        self.fifo[c].append(dict(E=E, Ep=Ep, Yp=Yp, Eb=Eb, Rb1=Rb1, N=N, learn=learn, Mp=Mp_sum))

    def _presence(self, c):
        """Near-end speech presence (0..1) of the FIFO head frame (look-ahead L, hangover post frames)."""
        mask = 1.0 if any(self.raw[c]) else 0.0
        p = max(mask, self.rel * self.pres[c] + (1 - self.rel) * mask)
        self.pres[c] = p
        return p

    def _frame(self):
        h, W = self.hop, self.W
        for c in range(2):
            self._analyze(c)
        fr = [self.fifo[c].popleft() for c in range(2)]
        G1b = [self.st1[c].gain(fr[c]["Eb"], fr[c]["Rb1"]) for c in range(2)]
        prs = [self._presence(c) for c in range(2)]
        gate = lambda g, pr: g if pr >= 1 else g * (pr + (1 - pr) * self.absent)
        outs = []
        for c in range(2):
            f = fr[c]
            if self.refine:
                o = 1 - c
                # Partner's stage-1 output (with absence suppression, without comfort noise)
                C1 = (gate(G1b[o], prs[o]) @ W) * fr[o]["E"]
                Xp2 = C1.real ** 2 + C1.imag ** 2
                # Stage 2 learns only when the partner's clean output is well above our own mic
                # (learning on frames where the reference was deeply suppressed makes H diverge)
                learn2 = f["learn"] and Xp2.sum() > self.dtd * f["Mp"]
                R2 = self.st2[c].residual(f["Ep"], f["Yp"], Xp2, learn2)
                Gb = self.st2[c].gain(f["Eb"], W @ R2 + 1e-20)
            else:
                Gb = G1b[c]
            pr = prs[c]
            Gb = gate(Gb, pr)
            G = Gb @ W
            O = f["E"] * G
            if self.cng > 0:
                cn = np.sqrt(np.maximum(f["N"] * (1 - G ** 2), 0.0) * self.cng)
                O = O + cn * np.exp(2j * np.pi * self.rng[c].random(self.K))
            self.last_gain[c] = float(G.mean())
            self.last_presence[c] = pr
            self.ola[c] += np.fft.irfft(O, self.nfft) * self.win * self.ola_scale
            outs.append(self.ola[c, :h].copy())
            self.ola[c, :-h] = self.ola[c, h:]
            self.ola[c, -h:] = 0.0
        return outs

    def process(self, e, y, mic, ref):
        """Each argument is a list of two channels (length ``block`` each). Returns (out_a, out_b), delayed by ``latency``."""
        h = self.hop
        out = [[], []]
        for s in range(0, self.block, h):
            for c in range(2):
                self.buf[c, :, :-h] = self.buf[c, :, h:]
                self.buf[c, 0, -h:] = e[c][s:s + h]
                self.buf[c, 1, -h:] = y[c][s:s + h]
                self.buf[c, 2, -h:] = mic[c][s:s + h]
                self.buf[c, 3, -h:] = ref[c][s:s + h]
            oa, ob = self._frame()
            out[0].append(oa)
            out[1].append(ob)
        return np.concatenate(out[0]), np.concatenate(out[1])
