"""
Apply the speaker-bleed removal to lapel-mic recordings afterwards (offline).

The offline version estimates the bleed delay / clock drift over the whole file and uses a
look-ahead for the near-end speech detection, so it is slightly better than the real-time one.
Outputs (16 kHz, PCM16): {prefix}_s1.wav, {prefix}_s2.wav, {prefix}_stereo.wav (L = s1, R = s2)
and {prefix}_bleed_log.json. Any input sample rate is accepted.

Usage:
  python bleed_offline.py --s1 speaker1.wav --s2 speaker2.wav --out_dir out/
  python bleed_offline.py --stereo session.wav --out_dir out/       # L = speaker 1, R = speaker 2
  python bleed_offline.py --stereo rec/20261014_120000_raw.wav       # a recording of bleed_demo_gui.py
"""

import argparse
import json
import os
import sys

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# This allows you to import modules from the src directory without pip installing the package.
# Uncomment the line below if you need to run this script directly without installing the package.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai.bleed import OFFLINE_DEFAULTS, process_files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s1", help="WAV of speaker 1's mic (mono)")
    ap.add_argument("--s2", help="WAV of speaker 2's mic (mono)")
    ap.add_argument("--stereo", help="stereo WAV (L = speaker 1, R = speaker 2)")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--prefix", default=None, help="output file prefix (default: input file name)")
    ap.add_argument("--no_aes", action="store_true", help="linear AEC only (least distortion, more residual bleed)")
    ap.add_argument("--aes_beta", type=float, default=OFFLINE_DEFAULTS["aes_beta"],
                    help="residual suppression strength (larger removes more bleed but also more of "
                         "the own voice during overlaps)")
    ap.add_argument("--passes", type=int, default=OFFLINE_DEFAULTS["passes"])
    args = ap.parse_args()

    if not args.stereo and not (args.s1 and args.s2):
        ap.error("specify --stereo, or both --s1 and --s2")

    print("Processing... (this takes a while for long recordings)")
    paths = process_files(s1=args.s1, s2=args.s2, stereo=args.stereo, out_dir=args.out_dir, prefix=args.prefix,
                          aes=not args.no_aes, aes_beta=args.aes_beta, passes=args.passes)
    with open(paths["log"], encoding="utf-8") as f:
        log = json.load(f)
    sr = log["params"]["sr"]
    print(f"Done in {log['elapsed_sec']} s "
          f"(bleed delay 2->1 {1000 * log['delay_b_to_a']['bulk_delay'] / sr:.1f} ms, "
          f"1->2 {1000 * log['delay_a_to_b']['bulk_delay'] / sr:.1f} ms)")
    if log["clipped_samples"]:
        print(f"[Warning] {log['clipped_samples']} samples were clipped")
    for k in ("s1", "s2", "stereo", "log"):
        print(f"  {paths[k]}")


if __name__ == "__main__":
    main()
