#!/usr/bin/env python3
"""
Example of the experimental three-party VAP model (MaaiTri) with three WAV files
(one mono 16 kHz file per speaker).

    python vap_tri_3wav.py --model vap_3party_mimi_state_dict_jp_12.5hz_20000msec.pt \
        --mimi-onnx C:/Users/inoue/Desktop/mimi-ft/continuous_mimi_fp32_ckpt_fixed.onnx

Without --model / --mimi-onnx the files are downloaded from the hub.
The default WAVs are two-party samples used only as a demo input.
"""

import argparse
import os
import sys

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# This allows you to import modules from the src directory without pip installing the package.
# Uncomment the line below if you need to run this script directly without installing the package.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import MaaiTri, MaaiInput, MaaiOutput

WAV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wav_sample")


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav1", default=os.path.join(WAV_DIR, "jpn_inoue_16k.wav"))
    parser.add_argument("--wav2", default=os.path.join(WAV_DIR, "jpn_sumida_16k.wav"))
    parser.add_argument("--wav3", default=os.path.join(WAV_DIR, "eng_divesh_16k.wav"))
    parser.add_argument("--model", default=None, help="local model file (default: hub)")
    parser.add_argument("--mimi-onnx", default=None, help="local fine-tuned Mimi ONNX (sidecar: <path>.json)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    wav1 = MaaiInput.Wav(wav_file_path=args.wav1)
    wav2 = MaaiInput.Wav(wav_file_path=args.wav2)
    wav3 = MaaiInput.Wav(wav_file_path=args.wav3)

    output = MaaiOutput.ConsoleBar()

    maai = MaaiTri(
        audio_ch1=wav1,
        audio_ch2=wav2,
        audio_ch3=wav3,
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
