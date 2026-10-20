#!/usr/bin/env python3
"""
Example of the experimental three-party VAP model (MaaiTri) with a 3-channel
audio interface (one microphone per speaker).

    python vap_tri_3mic.py --model vap_3party_mimi_state_dict_jp_12.5hz_20000msec.pt \
        --mimi-onnx C:/Users/inoue/Desktop/mimi-ft/continuous_mimi_fp32_ckpt_fixed.onnx
"""

import argparse
import os
import sys

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# This allows you to import modules from the src directory without pip installing the package.
# Uncomment the line below if you need to run this script directly without installing the package.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import MaaiTri, MaaiInput, MaaiOutput


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic-device-index", type=int, default=-1)
    parser.add_argument("--model", default=None, help="local model file (default: hub)")
    parser.add_argument("--mimi-onnx", default=None, help="local fine-tuned Mimi ONNX (sidecar: <path>.json)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    mic = MaaiInput.Mic(mic_device_index=args.mic_device_index, channels=3)

    output = MaaiOutput.ConsoleBar()

    maai = MaaiTri(
        audio_ch1=mic.get_channel(0),
        audio_ch2=mic.get_channel(1),
        audio_ch3=mic.get_channel(2),
        lang="jp",
        device=args.device,
        local_model=args.model,
        mimi_local_onnx_fp32_path=args.mimi_onnx,
    )

    maai.start()

    while True:
        result = maai.get_result()
        output.update(result)


if __name__ == "__main__":
    test()
