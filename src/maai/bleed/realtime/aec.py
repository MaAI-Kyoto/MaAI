"""Streaming linear AEC: dual-filter PBFDAF (same algorithm as the offline ``offline/pbfdaf.py``).

Each ``process()`` call handles ``block`` samples (default 256 = 16 ms at 16 kHz) without adding
latency (the overlap-save output is time-aligned with the input block).

- background: NLMS that always adapts (per-bin normalization, gradient constraint)
- foreground: stable filter used for the output. Copied from / restored to the background
  following the Speex MDF criterion.
- Adaptation happens only in "far-end only" blocks (reference >> mic) and is frozen for
  ``near_hold_ms`` after the near-end talker speaks. The reference (the partner's lapel mic)
  also contains bleed of the near-end voice, so adapting during double talk would learn to
  cancel the near-end voice.
"""
import numpy as np

rfft, irfft = np.fft.rfft, np.fft.irfft


class PBFDAFStream:
    def __init__(self, block=256, n_part=12, mu_max=0.5, mu_init=0.3, dtd_ratio_db=6.0,
                 adapt_sec_needed=3.0, near_hold_ms=300.0, sr=16000):
        self.N = N = block
        self.K = K = N + 1
        self.P = n_part
        self.mu_max, self.mu_init = mu_max, mu_init
        self.dtd_ratio = 10 ** (dtd_ratio_db / 10)
        self.need = int(adapt_sec_needed * sr / N)
        self.hold_blocks = int(near_hold_ms * 1e-3 * sr / N)
        self.noise_floor = 1e-10 * N
        self.ss = 0.35 / n_part

        self.X = np.zeros((n_part, K), np.complex128)   # reference spectrum history (0 = newest)
        self.Wf = np.zeros((n_part, K), np.complex128)
        self.Wb = np.zeros((n_part, K), np.complex128)
        self.x_prev = np.zeros(N)
        self.zN = np.zeros(N)
        self.Psum = np.zeros(K)
        self.Eh = np.zeros(K)
        self.Yh = np.zeros(K)
        self.Pey = self.Pyy = 1.0
        self.leak = 0.25
        self.D = [0.0, 0.0, 0.0, 0.0]  # Davg1, Davg2, Dvar1, Dvar2
        self.adapted_blocks = 0
        self.adapted = False
        self.hold = 0
        self.n_fg_update = self.n_bg_reset = 0
        self.far_single = False

    def reset_filters(self):
        self.Wf[:] = 0
        self.Wb[:] = 0
        self.adapted = False
        self.adapted_blocks = 0

    def process(self, d, x):
        """d: mic block, x: reference block (length ``block``). Returns (e=cleaned, y=estimated bleed)."""
        N, zN = self.N, self.zN
        d = np.asarray(d, np.float64)
        x = np.asarray(x, np.float64)
        self.X = np.roll(self.X, 1, axis=0)
        self.X[0] = rfft(np.concatenate([self.x_prev, x]))
        self.x_prev = x
        Xo = self.X

        yf = irfft((self.Wf * Xo).sum(0), 2 * N)[N:]
        yb = irfft((self.Wb * Xo).sum(0), 2 * N)[N:]
        ef = d - yf
        eb = d - yb

        # foreground / background decision (Speex MDF)
        Sff, See = ef @ ef, eb @ eb
        Dbf = (ef - eb) @ (ef - eb) + 1e-12
        diff = Sff - See
        Davg1, Davg2, Dvar1, Dvar2 = self.D
        Davg1 = 0.6 * Davg1 + 0.4 * diff
        Dvar1 = 0.36 * Dvar1 + 0.16 * Sff * Dbf
        Davg2 = 0.85 * Davg2 + 0.15 * diff
        Dvar2 = 0.7225 * Dvar2 + 0.0225 * Sff * Dbf
        if (diff * abs(diff) > Sff * Dbf) or (Davg1 * abs(Davg1) > 0.5 * Dvar1) \
                or (Davg2 * abs(Davg2) > 0.25 * Dvar2):
            self.Wf[:] = self.Wb
            ef, yf = eb, yb
            Davg1 = Davg2 = Dvar1 = Dvar2 = 0.0
            self.n_fg_update += 1
        elif -diff * abs(diff) > 4.0 * Sff * Dbf:
            self.Wb[:] = self.Wf
            eb, yb = ef, yf
            Davg1 = Davg2 = Dvar1 = Dvar2 = 0.0
            self.n_bg_reset += 1
        self.D = [Davg1, Davg2, Dvar1, Dvar2]

        Ef = np.abs(rfft(np.concatenate([zN, ef]))) ** 2
        Yf = np.abs(rfft(np.concatenate([zN, yf]))) ** 2
        Eb = rfft(np.concatenate([zN, eb]))
        Ebp = np.abs(Eb) ** 2
        Ybp = np.abs(rfft(np.concatenate([zN, yb]))) ** 2

        self.Psum = 0.7 * self.Psum + 0.3 * (np.abs(Xo) ** 2).sum(0)
        ed, ex = d @ d, x @ x
        if ed > self.dtd_ratio * ex:
            self.hold = self.hold_blocks
        elif self.hold > 0:
            self.hold -= 1
        far_single = ex > self.dtd_ratio * ed and ex > self.noise_floor and self.hold == 0
        self.far_single = far_single

        # leak (residual / estimated bleed) estimate, updated only in far-end-only blocks
        Eh_cur, Yh_cur = Ef - self.Eh, Yf - self.Yh
        self.Eh = 0.95 * self.Eh + 0.05 * Ef
        self.Yh = 0.95 * self.Yh + 0.05 * Yf
        pyy = float(Yh_cur @ Yh_cur)
        if pyy > 0 and far_single:
            self.Pey = (1 - self.ss) * self.Pey + self.ss * float(Eh_cur @ Yh_cur)
            self.Pyy = max((1 - self.ss) * self.Pyy + self.ss * pyy, 1e-20)
        if self.adapted:
            self.leak = float(np.clip(self.Pey / self.Pyy, 0.005, 1.0))

        mu = None
        if not self.adapted:
            if far_single:
                mu = np.full(self.K, self.mu_init)
                self.adapted_blocks += 1
                self.adapted = self.adapted_blocks >= self.need
        elif far_single:
            mu = np.minimum(self.mu_max, self.leak * Ybp / (Ebp + 1e-12))
        if mu is not None:
            delta = 1e-6 * (self.Psum.mean() + 1e-10) + self.noise_floor
            G = np.conj(Xo) * ((mu / (self.Psum + delta)) * Eb)[None, :]
            g = irfft(G, 2 * N, axis=1)
            g[:, N:] = 0.0  # gradient constraint
            self.Wb += rfft(g, axis=1)
        return ef, yf

    def stats(self):
        return dict(adapted=self.adapted, leak=self.leak, fg_updates=self.n_fg_update, bg_resets=self.n_bg_reset)
