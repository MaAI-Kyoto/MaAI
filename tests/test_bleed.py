"""Tests of the speaker-bleed removal (maai.bleed / MaaiInput.BleedCancel).

Two speakers' voices (example/wav_sample) are mixed with artificial bleed through known impulse
responses, following the schedule A only / B only / both. Checked:
  - ERLE (bleed reduction) of channel A in B-only segments >= 20 dB
  - correlation of channel A with the clean near-end voice in double-talk segments >= 0.95
for the real-time engine, the offline pipeline and the MaaiInput wrapper.

Run: python tests/test_bleed.py   (or pytest tests/test_bleed.py)
"""
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from maai import MaaiInput  # noqa: E402
from maai.bleed import StreamProcessor, process_pair  # noqa: E402
from maai.bleed.realtime.resample import StreamResampler  # noqa: E402

SR = 16000
WAV_A = os.path.join(ROOT, "example", "wav_sample", "jpn_inoue_16k.wav")
WAV_B = os.path.join(ROOT, "example", "wav_sample", "jpn_sumida_16k.wav")
_CACHE = {}


def rir(rng, delay, gain_db, t60=0.25, length=2400):
    h = np.zeros(length)
    t = np.arange(length - delay)
    tail = rng.standard_normal(length - delay) * 10 ** (-3 * t / (t60 * SR))
    tail[0] = 4.0  # direct sound
    h[delay:] = tail
    return h / np.sqrt(np.sum(h ** 2)) * 10 ** (gain_db / 20)


def build(dur=240, seed=0):
    """Synthetic two-speaker signals. Returns dict(mA, mB, sA, sB, lab, labels, seg)."""
    key = (dur, seed)
    if key in _CACHE:
        return _CACHE[key]
    rng = np.random.default_rng(seed)
    sA, _ = sf.read(WAV_A, frames=int(dur * SR))
    sB, _ = sf.read(WAV_B, frames=int(dur * SR))
    n = min(len(sA), len(sB))
    sA, sB = sA[:n] / np.std(sA[:n]), sB[:n] / np.std(sB[:n])

    # schedule (10 s units): A only, B only, both, ...
    seg = 10 * SR
    labels = np.array(["A", "B", "AB", "B", "A", "AB"])
    lab = labels[(np.arange(n) // seg) % len(labels)]
    sA = sA * np.isin(lab, ["A", "AB"])
    sB = sB * np.isin(lab, ["B", "AB"])

    # weak room reverb on the own voice (lapel mics also pick up diffuse reverb)
    own = lambda s_: s_ + fftconvolve(s_, rir(rng, 40, -15.0), mode="full")[:n]
    sA, sB = own(sA), own(sB)
    bleedA = fftconvolve(sB, rir(rng, 72, -8.0), mode="full")[:n]
    bleedB = fftconvolve(sA, rir(rng, 74, -8.0), mode="full")[:n]
    mA = sA + bleedA + 1e-3 * rng.standard_normal(n)
    mB = sB + bleedB + 1e-3 * rng.standard_normal(n)
    scale = 0.5 / max(np.abs(mA).max(), np.abs(mB).max())
    mA, mB, sA, sB = (v * scale for v in (mA, mB, sA, sB))
    _CACHE[key] = dict(mA=mA, mB=mB, sA=sA, sB=sB, lab=lab, labels=labels, seg=seg)
    return _CACHE[key]


def evaluate(sig, oA, burn_sec=60):
    mA, sA, lab = sig["mA"], sig["sA"], sig["lab"]
    n = len(mA)
    idx = np.arange(n) >= burn_sec * SR  # exclude the convergence period
    bonly, both = idx & (lab == "B"), idx & (lab == "AB")
    db = lambda num, den: 10 * np.log10(np.sum(num ** 2) / np.sum(den ** 2))
    return dict(
        erle=db(mA[bonly], oA[bonly]),
        sdr_in=db(sA[both], mA[both] - sA[both]),
        sdr_out=db(sA[both], oA[both] - sA[both]),
        corr=np.corrcoef(oA[both], sA[both])[0, 1],
    )


def stream(sp, x, chunk, rng=None):
    out = []
    i = 0
    while i < len(x):
        n = int(rng.integers(1, 2 * chunk)) if rng is not None else chunk
        out.append(sp.push(x[i:i + n]))
        i += n
    return np.concatenate(out, 0)


def _report(name, r):
    print(f"[{name}] ERLE {r['erle']:.1f} dB, double-talk SDR {r['sdr_in']:.1f} -> {r['sdr_out']:.1f} dB, "
          f"corr {r['corr']:.3f}")


def test_resampler():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((44100 * 2, 2))
    r = StreamResampler(44100, 16000, 2)
    out, i = [], 0
    while i < len(x):
        n = int(rng.integers(1, 882))
        out.append(r.process(x[i:i + n]))
        i += n
    y = np.concatenate(out, 0)
    ref = resample_poly(x, 160, 441, axis=0)
    d = int(round(r.delay_in_samples * 160 / 441))
    err = np.abs(y[d:len(ref)] - ref[: len(ref) - d]).max()
    print(f"[resampler] max err vs resample_poly: {err:.2e}")
    assert err < 1e-9


def test_realtime():
    sig = build()
    x = np.stack([sig["mA"], sig["mB"]], 1)
    sp = StreamProcessor(SR, hpf_hz=0)
    y = stream(sp, x, 160)
    k = sp.engine.latency
    oA = np.concatenate([y[k:, 0], np.zeros(k)])
    r = evaluate(sig, oA)
    _report(f"realtime, latency {1000 * k / SR:.0f} ms", r)
    assert r["erle"] >= 20 and r["corr"] >= 0.95

    # the output must not depend on the chunk size
    sp2 = StreamProcessor(SR, hpf_hz=0)
    y2 = stream(sp2, x[: SR * 30], 160, np.random.default_rng(1))
    diff = np.abs(y[:len(y2)] - y2).max()
    print(f"[realtime] chunk-size invariance max diff: {diff:.2e}")
    assert diff < 1e-9


def test_offline():
    sig = build()
    oA, oB, log = process_pair(sig["mA"], sig["mB"], SR, hpf_hz=0)
    r = evaluate(sig, oA.astype(np.float64))
    _report("offline", r)
    assert r["erle"] >= 20 and r["corr"] >= 0.95


def test_bleed_cancel_input():
    sig = build()
    x = np.stack([sig["mA"], sig["mB"]], 1)[: SR * 20]
    F = MaaiInput.Base.FRAME_SIZE

    # Reference: BleedCanceller directly (with the same causal HPF as StreamProcessor)
    sp = StreamProcessor(SR)
    ref = stream(sp, x, 256)

    # Two mono Chunk sources with different chunk sizes -> BleedCancel -> get_channel(0/1)
    c1, c2 = MaaiInput.Chunk(), MaaiInput.Chunk()
    bc = MaaiInput.BleedCancel(c1, c2)
    ch1, ch2 = bc.get_channel(0), bc.get_channel(1)
    q1, q2 = ch1.subscribe(), ch2.subscribe()
    bc.start()
    for i in range(0, len(x), F):
        c1.put_chunk(x[i:i + F, 0])
    for i in range(0, len(x), 400):
        c2.put_chunk(x[i:i + 400, 1])
    n_frames = (len(ref) // F) - 1
    out1 = np.array([ch1.get_audio_data(q1) for _ in range(n_frames)]).reshape(-1)
    out2 = np.array([ch2.get_audio_data(q2) for _ in range(n_frames)]).reshape(-1)
    n = len(out1)
    diff = max(np.abs(out1 - ref[:n, 0]).max(), np.abs(out2 - ref[:n, 1]).max())
    print(f"[BleedCancel] max diff vs StreamProcessor: {diff:.2e}, latency {bc.latency_ms:.1f} ms")
    assert diff < 1e-9

    # Disabled output = raw input delayed by the engine latency, enabled again = processed
    bc2 = MaaiInput.BleedCancel(MaaiInput.Chunk(), MaaiInput.Chunk())
    lat = bc2._engine.latency
    half = (len(x) // 2 // F) * F
    frames = []
    for i in range(0, half, F):
        frames += bc2._feed(x[i:i + F])
    bc2.set_enabled(False)
    for i in range(half, len(x), F):
        frames += bc2._feed(x[i:i + F])
    on = np.concatenate([f["output"] for f in frames if f["enabled"]])
    off = np.concatenate([f["output"] for f in frames if not f["enabled"]])
    clean = np.concatenate([f["clean"] for f in frames])
    assert np.abs(on - ref[:len(on)]).max() < 1e-9
    assert np.abs(clean - ref[:len(clean)]).max() < 1e-9
    raw_delayed = np.concatenate([np.zeros((lat, 2)), x])[len(on):len(on) + len(off)]
    assert np.abs(off - raw_delayed).max() < 1e-12
    print("[BleedCancel] on/off switching OK")


if __name__ == "__main__":
    test_resampler()
    test_realtime()
    test_offline()
    test_bleed_cancel_input()
    print("PASS")
