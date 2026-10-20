#!/usr/bin/env python3
"""
Example of the experimental three-party VAP model (MaaiTri) with one 3-channel
16 kHz WAV file (channel k = speaker k+1).

    python vap_tri_3ch_wav.py

By default it reads J01_city.wav and the exported model from the desktop. The
fine-tuned Mimi ONNX is downloaded from the hub unless --mimi-onnx is given.
"""

import argparse
import os
import sys

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# This allows you to import modules from the src directory without pip installing the package.
# Uncomment the line below if you need to run this script directly without installing the package.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import MaaiTri, MaaiInput, MaaiOutput

DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", default=os.path.join(DESKTOP, "J01_city_MaAI.wav"), help="3-channel 16 kHz wav")
    parser.add_argument("--model", default=os.path.join(DESKTOP, "vap_3party_mimi_state_dict_jp_12.5hz_20000msec.pt"),
                        help="local model file (vap_3party_mimi_state_dict_*.pt)")
    parser.add_argument("--mimi-onnx", default=None, help="local fine-tuned Mimi ONNX (sidecar: <path>.json)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    wav = MaaiInput.Wav(wav_file_path=args.wav)
    if wav.channels != 3:
        raise ValueError(f"{args.wav} has {wav.channels} channels; expected 3")

    output = MaaiOutput.ConsoleBar()

    maai = MaaiTri(
        audio_ch1=wav.get_channel(0),
        audio_ch2=wav.get_channel(1),
        audio_ch3=wav.get_channel(2),
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
