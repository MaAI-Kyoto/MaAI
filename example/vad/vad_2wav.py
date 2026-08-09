"""
This script is an example of how to use the VAD (Voice Activity Detection) model
with two WAV files, one per speaker.

The output vad is a list of two float values, one per input channel.

This example uses the 12.5 Hz Mimi model (model_type="normal-ver2").
"""

import sys
import os

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# This allows you to import modules from the src directory without pip installing the package.
# Uncomment the line below if you need to run this script directly without installing the package.

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import Maai, MaaiInput, MaaiOutput

def test():

    wav1 = MaaiInput.Wav(wav_file_path="../wav_sample/jpn_inoue_16k.wav")
    wav2 = MaaiInput.Wav(wav_file_path="../wav_sample/jpn_sumida_16k.wav")

    output = MaaiOutput.ConsoleBar()

    maai = Maai(
        mode="vad",
        lang="jp",
        frame_rate=12.5,
        audio_ch1=wav1,
        audio_ch2=wav2,
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
    try:
        test()
    except KeyboardInterrupt:
        print("Ending the test script.")
