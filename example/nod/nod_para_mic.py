"""
Example: single microphone + zero channel with the nod_para (nodding parameter) model.

Requires a converted weights file (.pt). Set MAAI_NOD_PARA_PT to its path, e.g.:

  set MAAI_NOD_PARA_PT=C:\\path\\to\\weights.pt
  python example/nod/nod_para_mic.py

Uses 12.5 Hz frame rate to match the fixed Maai nod_para training grid preset.
"""

import os
import sys

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# This allows you to import modules from the src directory without pip installing the package.
# Uncomment the line below if you need to run this script directly without installing the package.

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import Maai, MaaiInput, MaaiOutput


def test():
    local_model = "epoch2-val_loss_nod_all_4.54749.pt"
    if not local_model:
        print(
            "Set MAAI_NOD_PARA_PT to your nod_para .pt checkpoint path "
            "(flat state_dict or {state_dict, nod_param_stats}).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.isfile(local_model):
        print(f"File not found: {local_model}", file=sys.stderr)
        sys.exit(1)

    mic = MaaiInput.Mic()
    zero = MaaiInput.Zero()

    output = MaaiOutput.ConsoleBar()

    maai = Maai(
        mode="nod_para",
        lang="jp",
        frame_rate=12.5,
        audio_ch1=mic,
        audio_ch2=zero,
        device="cpu",
        model_type="normal-ver2",
        local_model=local_model,
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
