#!/usr/bin/env python3
"""
This script is an example of how to use the mono VAP model (vap_mono) with a single WAV file.

The vap_mono mode takes only one audio channel (audio_ch1).
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
        mode="vap_mono",
        lang="jp",
        frame_rate=10,
        audio_ch1=wav,
        device="cpu",
    )

    maai.start()

    while True:
        result = maai.get_result()
        output.update(result)

if __name__ == "__main__":
    test()
