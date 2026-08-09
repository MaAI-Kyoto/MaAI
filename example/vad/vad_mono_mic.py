"""
This script is an example of using a single microphone with the mono VAD model (vad_mono).

The vad_mono mode takes only one audio channel (audio_ch1).
The output vad is a single float value for the input audio.

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

    # Use the default mic
    mic = MaaiInput.Mic()

    output = MaaiOutput.ConsoleBar()

    maai = Maai(
        mode="vad_mono",
        lang="jp",
        frame_rate=12.5,
        audio_ch1=mic,
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
