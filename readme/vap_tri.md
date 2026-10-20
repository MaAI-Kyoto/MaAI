<h1>
<p align="center">
Three-Party Turn-Taking (VAP) Model
</p>
</h1>

> [!WARNING]
> **🚧 This model is under development and is provided as a preview.**
> - Its prediction quality has not been fully evaluated.
> - The API (`MaaiTri`), outputs, model and file names may change without notice, and later models may not be compatible with this one.
> - Do not rely on it for production use or for reported results.

This model predicts **the near-future voice activity of three speakers**, i.e. which of the three participants is likely to speak next.

- Use the `MaaiTri` class (not `Maai`).
- The input is three 16kHz audio streams, one per speaker.
- Only Japanese (`lang="jp"`) and `frame_rate=12.5` are available.

<br>

## Quick Start

### 3-channel microphone input

```python
from maai import MaaiTri, MaaiInput, MaaiOutput

mic = MaaiInput.Mic(channels=3)

maai = MaaiTri(
    audio_ch1=mic.get_channel(0),
    audio_ch2=mic.get_channel(1),
    audio_ch3=mic.get_channel(2),
    lang="jp",
    device="cpu",
)
maai.start()

output = MaaiOutput.ConsoleBar()
while True:
    result = maai.get_result()
    output.update(result)
    # result["p_now"]     -> [float, float, float]
    # result["p_future"]  -> [float, float, float]
```

### 3-channel wav file input

```python
wav = MaaiInput.Wav(wav_file_path="path_to_your_3ch_wav_file")

maai = MaaiTri(
    audio_ch1=wav.get_channel(0),
    audio_ch2=wav.get_channel(1),
    audio_ch3=wav.get_channel(2),
    lang="jp",
    device="cpu",
)
```

Three separate 1-channel wav files can be used in the same way: pass `MaaiInput.Wav(...)` of each file to `audio_ch1`, `audio_ch2` and `audio_ch3`.

<br>

## Output

| Key | Description |
| --- | --- |
| `p_now` | `[float, float, float]`: probability that each speaker holds the floor in the next 0–600 ms |
| `p_future` | `[float, float, float]`: the same for 600–2000 ms ahead |
| `vad` | `[float, float, float]`: voice activity probability of each speaker at the current frame |
| `x1`, `x2`, `x3` | Input audio of each speaker for the current frame |
| `t` | Timestamp |

- All probabilities are in the range [0.0, 1.0].
- `p_now` and `p_future` are normalized among the three speakers, so the three values sum to 1.0.
- For typical turn-taking implementations, `p_now` is recommended.

```python
result["p_now"]     # e.g. [0.80, 0.15, 0.05]  -> speaker 1 is likely to be the next speaker
result["p_future"]  # e.g. [0.55, 0.30, 0.15]
```

### Per-bin probabilities (`return_p_bins=True`)

With `return_p_bins=True`, `p_bins` is also returned: for each speaker, the activity probabilities for the four bins (0–200, 200–600, 600–1200, 1200–2000 ms). Unlike `p_now` and `p_future`, these are not normalized among the speakers.

<br>

## Absent Speakers

When only two of the three participants are present, mark the absent one with `set_present` (or the `present` argument of `MaaiTri`):

```python
maai.set_present([1, 1, 0])  # speaker 3 is absent
```

An absent speaker gets 0 in `p_now`, `p_future` and `vad`, and the probabilities are normalized between the remaining two speakers.

<br>

## Notes

- The model uses the last 20 seconds of audio (`context_len_sec=20`, the length used in training).
- It runs three audio encoders, one per speaker, so it needs more computation than the two-party model. On a typical CPU it takes about 60 ms per 80 ms frame.
- `MaaiOutput.ConsoleBar` supports three-party results. `GuiBar`, `GuiPlot` and the TCP output do not yet; use the GUI sample script below instead.

<br>

## Sample Scripts

- [With 3-channel mic input](../example/vap_tri/vap_tri_3mic.py) 🎤
- [With a 3-channel wav file input](../example/vap_tri/vap_tri_3ch_wav.py) 🎵
- [With 3 wav file inputs](../example/vap_tri/vap_tri_3wav.py) 🎵
- [GUI viewer](../example/vap_tri/vap_tri_gui.py) 📈 — scrolling waveforms, voice activity and next-speaker probabilities of the three speakers, with play/pause and per-speaker presence buttons (requires `PyQt5` and `pyqtgraph`)
