"""
This script is an example of using two microphones with the VAD (Voice Activity Detection) model.

The vad mode detects the *current* voice activity of each speaker.
The output vad is a list of two float values, one per input channel.

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

    # Use the first mic for the first channel
    mic1 = MaaiInput.Mic(mic_device_index=0)

    # Use the second mic for the second channel
    mic2 = MaaiInput.Mic(mic_device_index=1)

    output = MaaiOutput.ConsoleBar()

    maai = Maai(
        mode="vad",
        lang="jp",
        frame_rate=12.5,
        audio_ch1=mic1,
        audio_ch2=mic2,
        device="cpu",
        model_type="normal-ver2",
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
