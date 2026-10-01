#!/usr/bin/env python3
"""
This script is an example of how to use the single-channel VAP model with a single WAV file.

When audio_ch2 is omitted, mode="vap" automatically uses the single-channel model.
mc=True (the default) selects the noise-robust model; set mc=False for the standard one.
The outputs p_now / p_future / vad are single float values for the input audio.
"""

import sys
import os

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# This allows you to import modules from the src directory without pip installing the package.
# Uncomment the line below if you need to run this script directly without installing the package.

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import Maai, MaaiInput, MaaiOutput

def test():

    wav = MaaiInput.Wav(wav_file_path="../wav_sample/jpn_inoue_16k.wav")

    output = MaaiOutput.ConsoleBar()

    maai = Maai(
        mode="vap",
        lang="jp",
        frame_rate=12.5,
        audio_ch1=wav,
        device="cpu",
        model_type="normal-ver2",
        use_mimi_onnx=True,
        mimi_onnx_precision="fp32",
    )

    maai.start()

    while True:
        result = maai.get_result()
        output.update(result)

if __name__ == "__main__":
    test()
