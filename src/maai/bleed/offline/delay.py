"""Delay estimation: block-wise GCC-PHAT with clock drift compensation (offline)."""
import numpy as np


def gcc_phat_block(mic, ref, max_lag, sr, frame=512, ratio_db=6.0):
    """Estimate the delay [samples] of ``ref`` bleeding into ``mic`` (positive: ref leads).

    The cross spectrum is averaged only over frames where ``ref`` dominates ``mic`` (far-end only)
    to suppress the influence of the opposite-direction bleed.

    Returns:
        tuple: (delay, peak_strength), or (None, 0) if there are no usable frames.
    """
    n = min(len(mic), len(ref)) // frame
    if n < 4:
        return None, 0.0
    hop = frame // 2
    nfft = 2 * frame
    win = np.hanning(frame)
    starts = np.arange(0, n * frame - frame, hop)
    em = np.array([np.mean(mic[s:s + frame] ** 2) for s in starts])
    er = np.array([np.mean(ref[s:s + frame] ** 2) for s in starts])
    floor = np.percentile(er, 30) + 1e-12
    sel = (er > 10 ** (ratio_db / 10) * em) & (er > 4 * floor)
    if sel.sum() < 4:
        return None, 0.0
    S = np.zeros(nfft // 2 + 1, dtype=np.complex128)
    for s in starts[sel]:
        M = np.fft.rfft(mic[s:s + frame] * win, nfft)
        R = np.fft.rfft(ref[s:s + frame] * win, nfft)
        S += M * np.conj(R)
    S /= np.abs(S) + 1e-12
    cc = np.fft.irfft(S, nfft)
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
    k = int(np.argmax(cc))
    # parabolic interpolation for sub-sample accuracy
    if 0 < k < len(cc) - 1:
        a, b, c = cc[k - 1], cc[k], cc[k + 1]
        den = a - 2 * b + c
        frac = 0.5 * (a - c) / den if den != 0 else 0.0
    else:
        frac = 0.0
    return k - max_lag + frac, float(cc[k])


def estimate_delay(mic, ref, sr, block_sec=30.0, max_lag_ms=30.0, ratio_db=6.0):
    """Estimate per-block delays and return the bulk delay and drift (samples/sample)."""
    blk = int(block_sec * sr)
    max_lag = int(max_lag_ms * 1e-3 * sr)
    centers, delays, strengths = [], [], []
    for s in range(0, min(len(mic), len(ref)) - blk + 1, blk):
        d, st = gcc_phat_block(mic[s:s + blk], ref[s:s + blk], max_lag, sr, ratio_db=ratio_db)
        if d is not None:
            centers.append(s + blk / 2)
            delays.append(d)
            strengths.append(st)
    centers, delays, strengths = map(np.asarray, (centers, delays, strengths))
    if len(delays) == 0:
        return {"bulk_delay": 0.0, "drift": 0.0, "n_blocks": 0, "block_delays": []}
    good = strengths >= np.median(strengths) * 0.5
    med = float(np.median(delays[good]))
    # linear regression without outliers (drift estimate)
    inl = good & (np.abs(delays - med) < 0.5e-3 * sr)
    drift, intercept = 0.0, med
    if inl.sum() >= 3:
        drift, intercept = np.polyfit(centers[inl], delays[inl], 1)
    return {
        "bulk_delay": med,
        "drift": float(drift),
        "intercept": float(intercept),
        "n_blocks": int(len(delays)),
        "n_inliers": int(inl.sum()),
        "block_delays": [(float(c), float(d), float(s)) for c, d, s in zip(centers, delays, strengths)],
    }


def align_reference(ref, length, delay, margin, drift=0.0, drift_threshold=2.0):
    """Shift the reference so that it leads the mic by ``margin`` samples.

    Bleed = ref(t - delay), so delaying ref by (delay - margin) leaves a delay of ``margin`` samples
    for the adaptive filter (causal, with a short filter). ``drift`` [samples/sample] is the clock
    mismatch; time-varying compensation is applied only if the accumulated drift over the whole
    signal exceeds ``drift_threshold``.
    """
    t0 = delay - margin
    if abs(drift) * length <= drift_threshold:
        s = int(np.round(t0))
        out = np.zeros(length, dtype=ref.dtype)
        src = np.arange(length) - s
        ok = (src >= 0) & (src < len(ref))
        out[ok] = ref[src[ok]]
        return out
    # time-varying fractional delay: local cubic spline interpolation per chunk
    from scipy.interpolate import CubicSpline
    out = np.zeros(length, dtype=ref.dtype)
    step, pad = 1 << 18, 64
    for i in range(0, length, step):
        t = np.arange(i, min(i + step, length), dtype=np.float64)
        pos = t - (t0 + drift * t)
        lo = max(int(np.floor(pos.min())) - pad, 0)
        hi = min(int(np.ceil(pos.max())) + pad, len(ref))
        if hi - lo < 4:
            continue
        cs = CubicSpline(np.arange(lo, hi), ref[lo:hi], extrapolate=False)
        out[i:i + len(t)] = np.nan_to_num(cs(pos))
    return out
