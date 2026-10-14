"""Stateful polyphase rational resampler (causal, accepts chunks of any length).

The filter is the same Kaiser-window FIR as ``scipy.signal.resample_poly``
(half_len = 10 * max(up, down)). The output lags by the group delay (len(h)-1)/2 / up
input samples (about 5 ms for 44.1k -> 16k).
"""
from math import gcd

import numpy as np
from scipy.signal import firwin


class StreamResampler:
    def __init__(self, sr_in, sr_out, channels=1):
        g = gcd(int(sr_in), int(sr_out))
        self.L, self.M = int(sr_out) // g, int(sr_in) // g
        self.channels = channels
        self.passthrough = self.L == 1 and self.M == 1
        if self.passthrough:
            return
        L, M = self.L, self.M
        maxr = max(L, M)
        half = 10 * maxr
        h = firwin(2 * half + 1, 1.0 / maxr, window=("kaiser", 5.0)) * L
        Q = int(np.ceil(len(h) / L))
        h = np.concatenate([h, np.zeros(Q * L - len(h))])
        self.Q = Q
        self.hp = h.reshape(Q, L).T.copy()  # hp[p, j] = h[p + j*L]
        self.hist = np.zeros((Q - 1, channels))  # previous inputs (global index n_in-Q+1 .. n_in-1)
        self.n_in = 0
        self.n_out = 0
        self.delay_in_samples = (2 * half) / 2 / L

    def process(self, x):
        """x: (n, channels) or (n,). Returns the output in the same layout."""
        squeeze = x.ndim == 1
        x = x.reshape(len(x), -1).astype(np.float64)
        if self.passthrough:
            return x[:, 0] if squeeze else x
        L, M, Q = self.L, self.M, self.Q
        buf = np.concatenate([self.hist, x], 0)  # global index of buf[0] = n_in - (Q-1)
        start = self.n_in - (Q - 1)
        self.n_in += len(x)
        # output n uses inputs up to floor(n*M/L)
        n_end = (self.n_in * L - 1) // M + 1 if self.n_in > 0 else 0
        n = np.arange(self.n_out, n_end, dtype=np.int64)
        self.n_out = n_end
        self.hist = buf[len(buf) - (Q - 1):] if Q > 1 else buf[:0]
        if len(n) == 0:
            out = np.zeros((0, x.shape[1]))
            return out[:, 0] if squeeze else out
        base = (n * M) // L - start
        p = (n * M) % L
        idx = base[:, None] - np.arange(Q)[None, :]          # (n, Q)
        valid = idx >= 0
        g = buf[np.clip(idx, 0, None)] * valid[..., None]      # (n, Q, ch)
        out = np.einsum("nq,nqc->nc", self.hp[p], g)
        return out[:, 0] if squeeze else out
