"""Residual bleed suppression v2 (offline, musical-noise-reduced).

  1. Band gains: gains are computed on ERB-spaced bands and interpolated back to bins with
     triangular windows (as in WebRTC AEC3), which removes independent per-bin gain fluctuation
     (= musical noise).
  2. Decision-directed a-priori SER (Ephraim-Malah) Wiener gain: smoother over time than the
     instantaneous 1 - beta*R/|E|^2 and less distortion of the near-end voice.
  3. Near-end speech detection + deep suppression of non-speech segments. Being offline, the mask
     is widened with look-ahead (pre_ms) and hangover (post_ms) so onsets/offsets are not cut.
  4. Comfort noise: the stationary background of the near-end mic (minimum statistics) is added
     back with random phase in proportion to what the suppression removed, keeping the output
     background at the input level.
  5. 75% overlap (hop = nfft/4) for smoother gain changes over time.

Residual bleed model:
    R[t,k] = max(leak_k * |Y|^2, sum_l H[l,k] |X[t-l]|^2, decay * R[t-1,k])
leak_k and H are learned only in far-end-only frames.
"""
import numpy as np

from .._bands import _MinStat, erb_bands


def residual_echo_suppress_v2(e, y, mic, ref, sr=16000, nfft=512, hop=None, beta=0.7, floor_db=-25.0,
                              t60=0.3, dtd_ratio_db=6.0, near_hold_ms=300.0, leak_alpha=0.005,
                              leak_min=1e-3, n_pow_taps=None, pow_mu=0.1, use_pow_path=True,
                              n_bands=48, dd_alpha=0.6, absent_db=-30.0, near_thr_db=3.0,
                              pre_ms=60.0, post_ms=250.0, cng=1.0, res_cap=True, seed=0, return_clean=False):
    hop = hop or nfft // 4
    K = nfft // 2 + 1
    win = np.sqrt(np.hanning(nfft + 1)[:nfft])
    ola_scale = hop / (nfft / 2)  # OLA normalization of sqrt-Hann analysis/synthesis (1 at hop=nfft/2)
    if n_pow_taps is None:
        n_pow_taps = max(1, int(round(0.1 * sr / hop)))  # power path covers ~100 ms
    L = len(e)
    nfr = int(np.ceil(L / hop)) + nfft // hop
    tot = nfr * hop + nfft
    pad = lambda s: np.concatenate([np.zeros(nfft - hop), s.astype(np.float64),
                                    np.zeros(tot - (nfft - hop) - len(s))])
    e_p, y_p, m_p, r_p = pad(e), pad(y), pad(mic[:L]), pad(ref[:L])
    floor = 10 ** (floor_db / 10)
    absent = 10 ** (absent_db / 10)
    decay = 10 ** (-6.0 * hop / sr / t60) if t60 > 0 else 0.0
    dtd = 10 ** (dtd_ratio_db / 10)
    near_thr = 10 ** (near_thr_db / 10)
    hold_frames = int(near_hold_ms * 1e-3 * sr / hop)
    W, fc = erb_bands(K, sr, n_bands)
    speech = (fc >= 200) & (fc <= 4000)

    def analysis(learn_only):
        """learn_only=True: return raw near-end speech decisions. False: return the output (uses presence)."""
        num = np.full(K, 1e-10)
        den = np.full(K, 1e-9)
        H = np.zeros((n_pow_taps, K))
        Xhist = np.zeros((n_pow_taps, K))
        R = np.zeros(K)
        ms = _MinStat(K, sub_len=max(1, int(0.25 * sr / hop)), n_sub=6)
        hold = 0
        Rb_prev = None
        Eb_prev = None
        Gb_prev = None
        raw = np.zeros(nfr, bool)
        out = None if learn_only else np.zeros(tot)
        out_nc = None if learn_only else np.zeros(tot)  # without comfort noise (reference for the partner)
        rng = np.random.default_rng(seed)
        gsum = 0.0
        for t in range(nfr):
            s = t * hop
            E = np.fft.rfft(e_p[s:s + nfft] * win)
            Y = np.fft.rfft(y_p[s:s + nfft] * win)
            X = np.fft.rfft(r_p[s:s + nfft] * win)
            Ep = E.real ** 2 + E.imag ** 2
            Yp = Y.real ** 2 + Y.imag ** 2
            Xhist[1:] = Xhist[:-1]
            Xhist[0] = X.real ** 2 + X.imag ** 2
            em = m_p[s:s + nfft] @ m_p[s:s + nfft]
            er = r_p[s:s + nfft] @ r_p[s:s + nfft]
            if em > dtd * er:
                hold = hold_frames
            elif hold > 0:
                hold -= 1
            learn = er > dtd * em and er > 1e-8 and hold == 0
            Hx = (H * Xhist).sum(0)
            if learn:
                num = (1 - leak_alpha) * num + leak_alpha * Ep
                den = (1 - leak_alpha) * den + leak_alpha * Yp
                if use_pow_path:
                    err = Ep - Hx
                    H += pow_mu * err * Xhist / ((Xhist ** 2).sum(0) + 1e-20)
                    np.maximum(H, 0.0, out=H)
            leak = np.clip(num / den, leak_min, 1.0)
            R = np.maximum(leak * Yp, decay * R)
            if use_pow_path:
                R = np.maximum(R, Hx)
            if res_cap:
                # the residual bleed cannot exceed the observed power (the power path can overestimate
                # it during double talk and hide the near-end talker from the speech decision)
                R = np.minimum(R, Ep)
            N = ms.update(Ep)

            # band powers
            Eb = W @ Ep
            Rb = W @ R + 1e-20
            Nb = W @ N

            # raw near-end speech decision: is the near-end component (minus residual bleed and
            # background) large enough?
            near_pow = np.maximum(Eb - 2.0 * beta * Rb - Nb, 0.0)[speech].sum()
            raw[t] = near_pow > near_thr * (beta * Rb + Nb)[speech].sum()
            if learn_only:
                continue

            # decision-directed a-priori SER (near-end / residual bleed)
            gamma = Eb / (beta * Rb)
            if Gb_prev is None:
                xi = np.maximum(gamma - 1, 0)
            else:
                xi = dd_alpha * (Gb_prev ** 2) * Eb_prev / (beta * Rb_prev) + (1 - dd_alpha) * np.maximum(gamma - 1, 0)
            Gb = np.maximum(xi / (1 + xi), floor)
            # the DD recursion uses the gain before absence suppression (no lag at speech onsets)
            Gb_prev, Eb_prev, Rb_prev = Gb, Eb, Rb
            # deeper suppression while the near-end is silent (presence has look-ahead + hangover, 0..1)
            pr = presence[t]
            if pr < 1:
                Gb = Gb * (pr + (1 - pr) * absent)
            G = Gb @ W  # bands -> bins (weights sum to 1 at each bin)
            gsum += G.mean()
            O = E * G
            out_nc[s:s + nfft] += np.fft.irfft(O, nfft) * win * ola_scale
            if cng > 0:
                # add back the removed background (keeps the input background level)
                cn = np.sqrt(np.maximum(N * (1 - G ** 2), 0.0) * cng)
                ph = np.exp(2j * np.pi * rng.random(K))
                O = O + cn * ph
            out[s:s + nfft] += np.fft.irfft(O, nfft) * win * ola_scale
        if learn_only:
            return raw
        return out, out_nc, gsum / nfr

    raw = analysis(True)
    # near-end speech mask with look-ahead / hangover -> smooth 0..1 presence
    pre = int(pre_ms * 1e-3 * sr / hop)
    post = int(post_ms * 1e-3 * sr / hop)
    idx = np.flatnonzero(raw)
    mask = np.zeros(nfr, float)
    if len(idx):
        d = np.zeros(nfr + 1)
        np.add.at(d, np.maximum(idx - pre, 0), 1)
        np.add.at(d, np.minimum(idx + post + 1, nfr), -1)
        mask = (np.cumsum(d)[:nfr] > 0).astype(float)
    # smooth onsets/offsets over ~20 ms (forward + backward first-order IIR)
    a = np.exp(-hop / (0.02 * sr))
    sm = mask.copy()
    for i in range(1, nfr):
        sm[i] = max(mask[i], a * sm[i - 1] + (1 - a) * mask[i])
    for i in range(nfr - 2, -1, -1):
        sm[i] = max(sm[i], a * sm[i + 1] + (1 - a) * sm[i])
    presence = np.clip(sm, 0, 1)

    out, out_nc, mean_gain = analysis(False)
    sl = slice(nfft - hop, nfft - hop + L)
    stats = dict(mean_gain=float(mean_gain), near_presence_ratio=float(presence.mean()), n_bands=int(W.shape[0]))
    if return_clean:
        return out[sl].astype(np.float32), stats, out_nc[sl].astype(np.float32)
    return out[sl].astype(np.float32), stats
