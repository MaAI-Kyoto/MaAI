"""
This script is an example of how to use the mono VAD model (vad_mono) with a single WAV file.

The vad_mono mode takes only one audio channel (audio_ch1).
The output vad is a single float value for the input audio.

This example uses the 50 Hz CPC model (model_type="normal").
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
        mode="vad_mono",
        lang="jp",
        frame_rate=50,
        audio_ch1=wav,
        device="cpu",
        model_type="normal",
    )

    maai.start()

    while True:
        result = maai.get_result()
        output.update(result)

if __name__ == "__main__":
    try:
        test()
    except KeyboardInterrupt:
        print("Ending the test script.")
