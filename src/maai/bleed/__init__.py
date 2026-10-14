"""Speaker-bleed removal for two face-to-face lapel microphones.

When two people talk face to face with lapel mics, each mic also picks up the partner's voice.
This package removes that bleed with an AEC-style approach that uses the partner's mic as the
reference (2-pass linear AEC + residual suppression v2), so the near-end voice is kept even
during overlapping speech.

Real-time:
    - :class:`BleedCanceller` / :class:`StreamProcessor` (low level)
    - ``maai.input.BleedCancel`` wraps any MaaiInput source(s) (recommended)

Offline (recorded files):
    - :func:`process_pair` for numpy arrays, :func:`process_files` for WAV files
"""
from .realtime import DEFAULTS as REALTIME_DEFAULTS  # noqa: F401
from .realtime import BleedCanceller, StreamProcessor  # noqa: F401
from .offline import DEFAULTS as OFFLINE_DEFAULTS  # noqa: F401
from .offline import process_files, process_pair  # noqa: F401
