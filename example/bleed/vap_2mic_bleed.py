"""
This script is an example of using the VAP model with two face-to-face lapel mics,
removing the bleed (cross-talk) between them with MaaiInput.BleedCancel.
"""

import sys
import os

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# This allows you to import modules from the src directory without pip installing the package.
# Uncomment the line below if you need to run this script directly without installing the package.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import Maai, MaaiInput, MaaiOutput

def test():

    # Two lapel mics connected to one 2ch audio interface (L = speaker 1, R = speaker 2)
    mic = MaaiInput.Mic(mic_device_index=0, channels=2)
    bc = MaaiInput.BleedCancel(mic)

    # Two separate mono mics also work (the same clock is recommended)
    # bc = MaaiInput.BleedCancel(MaaiInput.Mic(mic_device_index=0), MaaiInput.Mic(mic_device_index=1))

    output = MaaiOutput.ConsoleBar()

    maai = Maai(
        mode="vap",
        lang="jp",
        frame_rate=10,
        audio_ch1=bc.get_channel(0),
        audio_ch2=bc.get_channel(1),
        device="cpu"
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
