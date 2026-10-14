"""Online delay estimation (GCC-PHAT with exponential forgetting).

In the real-time version the AEC reference alignment is fixed (the mic is only delayed by
``margin`` samples) and the bleed delay itself is absorbed by the 192 ms adaptive filter.
This estimator is used for monitoring, for aligning the residual suppressor reference, and to
warn when the estimated delay leaves the range the filter can model (more negative than -margin).
"""
import numpy as np


class OnlineDelayEstimator:
    def __init__(self, hop=256, sr=16000, max_lag_ms=30.0, dtd_ratio_db=6.0, forget_sec=30.0,
                 update_sec=1.0, min_frames=200):
        self.hop = hop
        self.nfft = 4 * hop
        self.frame = 2 * hop
        self.win = np.hanning(self.frame)
        self.max_lag = int(max_lag_ms * 1e-3 * sr)
        self.dtd = 10 ** (dtd_ratio_db / 10)
        self.lam = 1.0 - hop / (forget_sec * sr)
        self.update_every = max(1, int(update_sec * sr / hop))
        self.min_frames = min_frames
        self.bm = np.zeros(self.frame)
        self.br = np.zeros(self.frame)
        self.S = np.zeros(self.nfft // 2 + 1, np.complex128)
        self.n_used = 0
        self.t = 0
        self.delay = None      # [samples] (positive: reference leads = usual bleed)
        self.strength = 0.0

    def process(self, mic, ref):
        h = self.hop
        self.bm[:h], self.bm[h:] = self.bm[h:], mic
        self.br[:h], self.br[h:] = self.br[h:], ref
        self.t += 1
        em, er = self.bm @ self.bm, self.br @ self.br
        if er > self.dtd * em and er > 1e-8:
            M = np.fft.rfft(self.bm * self.win, self.nfft)
            R = np.fft.rfft(self.br * self.win, self.nfft)
            self.S = self.lam * self.S + M * np.conj(R)
            self.n_used += 1
        if self.t % self.update_every == 0 and self.n_used >= self.min_frames:
            P = self.S / (np.abs(self.S) + 1e-20)
            cc = np.fft.irfft(P, self.nfft)
            cc = np.concatenate([cc[-self.max_lag:], cc[: self.max_lag + 1]])
            k = int(np.argmax(cc))
            frac = 0.0
            if 0 < k < len(cc) - 1:
                a, b, c = cc[k - 1], cc[k], cc[k + 1]
                den = a - 2 * b + c
                frac = 0.5 * (a - c) / den if den != 0 else 0.0
            self.delay = k - self.max_lag + frac
            self.strength = float(cc[k])
        return self.delay
