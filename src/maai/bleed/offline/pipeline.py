"""Offline bleed removal for one recorded session (two lapel mics).

    mic A --+
            +- delay estimation -> reference alignment -> linear AEC (PBFDAF) -> residual suppression -> A cleaned
    mic B --+ (reference)
Side B is processed symmetrically with A as the reference.
"""
import json
import os
import time
import warnings

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

from .aes2 import residual_echo_suppress_v2
from .delay import align_reference, estimate_delay
from .pbfdaf import pbfdaf

DEFAULTS = dict(
    sr=16000,
    hpf_hz=60.0,
    margin=32,           # how many samples the reference leads the bleed
    block=256,
    n_part=12,           # filter length = block * n_part (192 ms at 16 kHz)
    mu_max=0.5,
    adapt_mode="far_single",
    near_hold_ms=300.0,  # freeze adaptation after near-end speech
    dtd_ratio_db=6.0,
    passes=2,
    drift_threshold=2.0,
    aes=True,            # False: linear AEC only (least distortion)
    aes_t60=0.3,
    aes_pow_path=True,   # use the power-domain bleed path model for the residual estimate
    aes_beta=0.7,        # residual suppression strength
    aes_floor_db=-25.0,  # band gain floor during near-end speech
    aes_absent_db=-30.0,  # extra suppression while the near-end talker is silent
    aes_dd_alpha=0.6,    # decision-directed smoothing
    aes_near_thr_db=3.0,  # near-end speech decision threshold
    aes_pre_ms=60.0,     # look-ahead of the near-end speech mask
    aes_post_ms=250.0,   # hangover of the near-end speech mask
    aes_n_bands=48,      # number of ERB bands for the gain
    aes_cng=1.0,         # comfort noise level (0 disables, 1 = input background level)
    aes_refine=1,        # times to re-run AES with the partner's AES output as the reference
    aes_res_cap=True,    # cap the residual estimate at the observed power (prevents over-suppression
                         # of the near-end talker during double talk)
)


def preprocess(x, sr_in, sr, hpf_hz):
    if sr_in != sr:
        g = np.gcd(int(sr_in), int(sr))
        x = resample_poly(x, sr // g, sr_in // g)
    if hpf_hz:
        x = sosfiltfilt(butter(2, hpf_hz, "highpass", fs=sr, output="sos"), x)
    return x


def linear_stage(mic, ref, delay, clock_drift, p, log):
    """Linear AEC. Returns (pbfdaf result dict, aligned reference)."""
    ref_al = align_reference(ref, len(mic), delay, p["margin"], clock_drift, p["drift_threshold"])
    t0 = time.time()
    res = pbfdaf(mic, ref_al, block=p["block"], n_part=p["n_part"], mu_max=p["mu_max"],
                 dtd_ratio_db=p["dtd_ratio_db"], adapt_mode=p["adapt_mode"],
                 near_hold_ms=p["near_hold_ms"], sr=p["sr"])
    log["aec"] = res["stats"]
    log["aec_sec"] = round(time.time() - t0, 1)
    return res, ref_al


def aes_stage(mic, res, ref_al, p, log):
    """Residual suppression. Returns (output, output without comfort noise)."""
    if not p["aes"]:
        return res["e"], res["e"]
    t0 = time.time()
    out, st, clean = residual_echo_suppress_v2(
        res["e"], res["y"], mic, ref_al, sr=p["sr"], beta=p["aes_beta"], floor_db=p["aes_floor_db"],
        t60=p["aes_t60"], dtd_ratio_db=p["dtd_ratio_db"], near_hold_ms=p["near_hold_ms"],
        use_pow_path=p["aes_pow_path"], dd_alpha=p["aes_dd_alpha"], absent_db=p["aes_absent_db"],
        near_thr_db=p["aes_near_thr_db"], pre_ms=p["aes_pre_ms"], post_ms=p["aes_post_ms"],
        cng=p["aes_cng"], n_bands=p["aes_n_bands"], res_cap=p["aes_res_cap"], return_clean=True)
    log["aes"] = st
    log["aes_sec"] = round(time.time() - t0, 1)
    return out, clean


def process_pair(a, b, sr_in, **kw):
    """Remove the mutual bleed between two synchronized lapel-mic recordings.

    Args:
        a (np.ndarray): Mono signal of speaker 1's mic.
        b (np.ndarray): Mono signal of speaker 2's mic (same clock / start as ``a``).
        sr_in (int): Sample rate of ``a`` and ``b``.
        **kw: Overrides of :data:`DEFAULTS`.

    Returns:
        tuple: (a_clean, b_clean, log). The outputs are float32 at ``sr`` (16 kHz by default).
    """
    unknown = set(kw) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"Unknown bleed removal parameter(s): {sorted(unknown)}")
    p = dict(DEFAULTS, **kw)
    sr = p["sr"]
    a = preprocess(np.asarray(a, np.float64), sr_in, sr, p["hpf_hz"])
    b = preprocess(np.asarray(b, np.float64), sr_in, sr, p["hpf_hz"])
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    # peak normalization (as in Seamless Interaction); the original scale is restored at the end
    ga = np.max(np.abs(a)) + 1e-12
    gb = np.max(np.abs(b)) + 1e-12
    a, b = a / ga, b / gb

    # Delay estimation: half the difference of the two directions' slopes = clock drift
    # (distance changes due to speaker movement have the same sign in both directions)
    dBA = estimate_delay(a, b, sr, ratio_db=p["dtd_ratio_db"])
    dAB = estimate_delay(b, a, sr, ratio_db=p["dtd_ratio_db"])
    clock_drift = (dBA["drift"] - dAB["drift"]) / 2.0
    log = dict(params=p, n_samples=n, peak_a=float(ga), peak_b=float(gb), clock_drift=clock_drift,
               delay_b_to_a={k: v for k, v in dBA.items() if k != "block_delays"},
               delay_a_to_b={k: v for k, v in dAB.items() if k != "block_delays"},
               a={}, b={})
    dl_a = (dBA["bulk_delay"], clock_drift)   # delay of the B reference relative to A's mic
    dl_b = (dAB["bulk_delay"], -clock_drift)

    # Linear AEC. From the 2nd pass on, the partner's linear AEC output is the reference (less of our
    # own voice in the reference). The non-linear AES output is not used for the linear filter.
    ref_b, ref_a = b, a
    for it in range(p["passes"]):
        la = log["a"].setdefault(f"pass{it + 1}", {})
        lb = log["b"].setdefault(f"pass{it + 1}", {})
        res_a, ral_a = linear_stage(a, ref_b, *dl_a, p, la)
        res_b, ral_b = linear_stage(b, ref_a, *dl_b, p, lb)
        ref_b, ref_a = res_b["e"].astype(np.float64), res_a["e"].astype(np.float64)

    # Residual suppression
    a_out, a_cl = aes_stage(a, res_a, ral_a, p, la)
    b_out, b_cl = aes_stage(b, res_b, ral_b, p, lb)
    # aes_refine: re-run AES using the partner's AES output (without comfort noise) as the reference.
    # Less of our own voice remains in the reference, so it no longer leaks into the power-domain bleed
    # estimate (the power model ignores phase, so a non-linearly processed reference is fine).
    for it in range(p["aes_refine"]):
        lr_a = log["a"].setdefault(f"refine{it + 1}", {})
        lr_b = log["b"].setdefault(f"refine{it + 1}", {})
        rb = align_reference(b_cl.astype(np.float64), n, dl_a[0], p["margin"], dl_a[1], p["drift_threshold"])
        ra = align_reference(a_cl.astype(np.float64), n, dl_b[0], p["margin"], dl_b[1], p["drift_threshold"])
        a_out, a_cl2 = aes_stage(a, res_a, rb, p, lr_a)
        b_out, b_cl = aes_stage(b, res_b, ra, p, lr_b)
        a_cl = a_cl2

    return (a_out * ga).astype(np.float32), (b_out * gb).astype(np.float32), log


def _to_int16(x):
    return np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)


def _read_mono(path):
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x[:, 0]
    return x, sr


def process_files(s1=None, s2=None, stereo=None, out_dir=".", prefix=None, **params):
    """Apply offline bleed removal to recorded files and write the results.

    Give either two mono files (``s1``, ``s2``) or one stereo file (``stereo``, L=s1, R=s2).
    Writes ``{prefix}_s1.wav``, ``{prefix}_s2.wav``, ``{prefix}_stereo.wav`` (16 kHz, PCM16)
    and ``{prefix}_bleed_log.json`` into ``out_dir``.

    Args:
        s1 (str, optional): Mono WAV of speaker 1's mic.
        s2 (str, optional): Mono WAV of speaker 2's mic.
        stereo (str, optional): Stereo WAV (L = speaker 1, R = speaker 2).
        out_dir (str): Output directory.
        prefix (str, optional): Output file prefix (defaults to the input file name).
        **params: Overrides of :data:`DEFAULTS`.

    Returns:
        dict: Paths of the written files (keys: s1, s2, stereo, log).
    """
    t0 = time.time()
    if stereo is not None:
        x, sr_in = sf.read(stereo, dtype="float64")
        if x.ndim != 2 or x.shape[1] < 2:
            raise ValueError(f"{stereo}: expected a file with at least 2 channels")
        a, b = x[:, 0], x[:, 1]
        src = stereo
    elif s1 is not None and s2 is not None:
        a, sr_in = _read_mono(s1)
        b, sr_b = _read_mono(s2)
        if sr_in != sr_b:
            raise ValueError(f"Sample rate mismatch: {sr_in} vs {sr_b}")
        src = s1
    else:
        raise ValueError("Specify either stereo=... or both s1=... and s2=...")
    if len(a) != len(b):
        warnings.warn(f"Length mismatch {len(a)} vs {len(b)}; truncating to the shorter one")

    a_out, b_out, log = process_pair(a, b, sr_in, **params)
    sr = log["params"]["sr"]

    prefix = prefix or os.path.splitext(os.path.basename(src))[0]
    os.makedirs(out_dir, exist_ok=True)
    paths = {k: os.path.join(out_dir, f"{prefix}_{k}.wav") for k in ("s1", "s2", "stereo")}
    paths["log"] = os.path.join(out_dir, f"{prefix}_bleed_log.json")
    ia, ib = _to_int16(a_out), _to_int16(b_out)
    sf.write(paths["s1"], ia, sr, subtype="PCM_16")
    sf.write(paths["s2"], ib, sr, subtype="PCM_16")
    sf.write(paths["stereo"], np.stack([ia, ib], 1), sr, subtype="PCM_16")
    log.update(input_s1=s1, input_s2=s2, input_stereo=stereo, input_sr=sr_in,
               clipped_samples=int(np.sum(np.abs(a_out) >= 1.0) + np.sum(np.abs(b_out) >= 1.0)),
               elapsed_sec=round(time.time() - t0, 1))
    with open(paths["log"], "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1, ensure_ascii=False, default=float)
    return paths
