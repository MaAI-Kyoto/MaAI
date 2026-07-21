<h1>
<p align="center">
Single-Channel Turn-Taking (VAP) Model (Mono-VAP)
</p>
</h1>
<p align="center">
README: <a href="vap_mono.md">English </a> | <a href="vap_mono_JP.md">Japanese (日本語) </a>
</p>

Please set the `mode` parameter of the `Maai` class to `vap_mono`.

This mode runs the same model and pretrained weights as the standard VAP model (`vap`), but exposes a single-channel interface: only `audio_ch1` is required, and silence (zero signal) is fed internally to the second channel.
It is intended for use cases where only one speaker's audio is available (e.g., a single microphone input for a spoken dialogue system).

The input requires 1-channel, 16kHz audio data.

## Output

The outputs `p_now` and `p_future` are single float values (not two-element lists) representing the input audio:

- Internally, the model computes the standard speaker-normalized `p_now` / `p_future` for the input channel versus the silent channel.
- The channel-1 value in the range [0.5, 1.0] is then linearly stretched to [0.0, 1.0]; values at or below 0.5 become 0.0.

```
p_mono = max(0.0, (p[0] - 0.5) * 2.0)
```

`p_now` represents the voice activity of the input speaker occurring in the next 0 to 600 milliseconds, and `p_future` represents 600 to 2000 milliseconds ahead.
`vad` is also a single float value for the input channel.

## Supported Languages, Frame Rates, and Context Lengths

Since `vap_mono` shares its checkpoints with the standard VAP model, the supported `lang`, `frame_rate`, and `context_len_sec` values are identical to those of [the VAP model](vap.md).

## Usage Example

```python
from maai import Maai, MaaiInput, MaaiOutput

mic = MaaiInput.Mic()

maai = Maai(
    mode="vap_mono",
    lang="jp",
    frame_rate=10,
    audio_ch1=mic,   # audio_ch2 is not needed
    device="cpu",
)
maai.start()

while True:
    result = maai.get_result()
    print(result["p_now"], result["p_future"], result["vad"])  # all single floats
```

Sample scripts:
- [With 1 mic input](../example/vap_mono/vap_mono_mic.py) 🎤
- [With 1 wav file input](../example/vap_mono/vap_mono_wav.py) 🎵
