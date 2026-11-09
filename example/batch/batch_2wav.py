#!/usr/bin/env python3
"""
Offline batch inference over recorded audio with MaaiBatch.

The turn-taking (vap) and backchannel (bc) models score two 16 kHz WAV files
(one speaker each) in one pass; the Mimi encoder runs once and is shared.
"""

import os
import sys
import time

import numpy as np
import soundfile as sf

# For debugging purposes: import from the src directory without pip installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import MaaiBatch

WAV_DIR = os.path.join(os.path.dirname(__file__), "../wav_sample")


def main():
    ch1, sr1 = sf.read(os.path.join(WAV_DIR, "jpn_inoue_16k.wav"), dtype="float32")
    ch2, sr2 = sf.read(os.path.join(WAV_DIR, "jpn_sumida_16k.wav"), dtype="float32")
    assert sr1 == sr2 == 16000, "MaaiBatch expects 16 kHz audio"

    batch = MaaiBatch(
        configs=[
            {"mode": "vap", "lang": "jp"},
            {"mode": "bc", "lang": "jp"},
        ],
        frame_rate=12.5,
        context_len_sec=20,
        device="cpu",   # "cuda" for GPU
    )

    t0 = time.perf_counter()
    result = batch.process(ch1, ch2)
    elapsed = time.perf_counter() - t0

    n = len(result["time_sec"])
    print(f"{len(ch1) / 16000:.1f} s of audio -> {n} frames in {elapsed:.1f} s")

    # Moments where speaker 2 is predicted to take the turn, and where speaker 1
    # (the listener for the bc model) should backchannel.
    p_now = result["vap"]["p_now"]          # (n, 2)
    p_bc = result["bc"]["p_bc"]             # (n,)
    print("p_now(speaker 2) > 0.8 at [s]:", np.round(result["time_sec"][p_now[:, 1] > 0.8][:10], 2))
    print("p_bc > 0.5 at [s]:", np.round(result["time_sec"][p_bc > 0.5][:10], 2))

    # Per-frame dicts in the same format as Maai.get_result()
    frames = MaaiBatch.to_frames(result)
    print(frames[0])


if __name__ == "__main__":
    main()
