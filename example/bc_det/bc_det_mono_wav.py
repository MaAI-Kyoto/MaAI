"""
This script is an example of how to use the single-channel BC detection model
with a single WAV file.

When audio_ch2 is omitted, mode="bc_det" automatically uses the single-channel model.
The output p_bc_det is a single float value for the input audio.

Note that the model relies on the interlocutor's speech to tell a backchannel apart
from the beginning of a turn, so the mono variant is less accurate than the
two-channel one. Pass audio_ch2 when both channels are available.

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

    wav = MaaiInput.Wav(wav_file_path="../wav_sample/jpn_sumida_16k.wav")

    output = MaaiOutput.ConsoleBar()

    maai = Maai(
        mode="bc_det",
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
    try:
        test()
    except KeyboardInterrupt:
        print("Ending the test script.")
