"""Helpers shared by the real-time and offline residual suppressors (v2)."""
from collections import deque

import numpy as np


def erb_bands(K, sr, n_bands=48, min_bins=2):
    """ERB-spaced triangular filter bank whose weights sum to 1 at every bin.

    Returns:
        tuple: (W of shape (n_bands', K), band center frequencies [Hz]).
    """
    nfft = 2 * (K - 1)
    erb = lambda x: 21.4 * np.log10(1 + 0.00437 * x)
    inv = lambda e: (10 ** (e / 21.4) - 1) / 0.00437
    centers = inv(np.linspace(erb(0), erb(sr / 2), n_bands))
    cbin = np.unique(np.round(centers / (sr / nfft)).astype(int))
    # Drop centers closer than min_bins
    keep = [cbin[0]]
    for c in cbin[1:]:
        if c - keep[-1] >= min_bins:
            keep.append(c)
    if keep[-1] != K - 1:
        keep[-1] = K - 1
    c = np.array(keep)
    k = np.arange(K)
    W = np.zeros((len(c), K))
    for i in range(len(c)):
        if i > 0:
            lo = c[i - 1]
            up = (k >= lo) & (k <= c[i])
            W[i, up] = (k[up] - lo) / max(c[i] - lo, 1)
        if i < len(c) - 1:
            hi = c[i + 1]
            dn = (k >= c[i]) & (k <= hi)
            W[i, dn] = (hi - k[dn]) / max(hi - c[i], 1)
        W[i, c[i]] = 1.0
    W /= W.sum(0, keepdims=True)
    return W, c * sr / nfft


class _MinStat:
    """Stationary noise PSD estimate by minimum statistics (sub-window method)."""

    def __init__(self, K, sub_len, n_sub, bias=1.5):
        self.sub_len, self.n_sub, self.bias = sub_len, n_sub, bias
        self.P = None
        self.cur = np.full(K, np.inf)
        self.mins = deque(maxlen=n_sub)
        self.cnt = 0

    def update(self, p):
        self.P = p.copy() if self.P is None else 0.85 * self.P + 0.15 * p
        np.minimum(self.cur, self.P, out=self.cur)
        self.cnt += 1
        if self.cnt >= self.sub_len:
            self.mins.append(self.cur.copy())
            self.cur = np.full_like(self.cur, np.inf)
            self.cnt = 0
        m = self.cur
        for x in self.mins:
            m = np.minimum(m, x)
        return self.bias * m
