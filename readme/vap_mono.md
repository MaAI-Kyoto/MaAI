<h1>
<p align="center">
Single-Channel Turn-Taking (VAP) Model (Mono-VAP)
</p>
</h1>
<p align="center">
README: <a href="vap_mono.md">English </a> | <a href="vap_mono_JP.md">Japanese (日本語) </a>
</p>

Please set the `mode` parameter of the `Maai` class to `vap_mono`.

This is a **dedicated single-channel model with its own pretrained weights** — not the standard two-speaker VAP model fed with a silent second channel. It encodes one audio stream, replaces the cross-channel transformer of the standard model with a plain causal transformer, and predicts the future activity of that one speaker directly.
It is intended for use cases where only one speaker's audio is available (e.g., a single microphone input for a spoken dialogue system).

The input requires 1-channel, 16kHz audio data.

## Output

`p_now` and `p_future` are single float values in the range [0.0, 1.0] (not two-element lists):

- `p_now` is the probability that the input speaker is active in the next 0 to 600 milliseconds.
- `p_future` is the same for 600 to 2000 milliseconds ahead.

Because there is no second speaker to compare against, these values are **not** normalized between speakers as in the standard `vap` model: each is the expected voice-activity ratio of the input speaker over the corresponding time range, already a probability.

`vad` is also a single float value for the input channel.

With `return_p_bins=True`, `p_bins` is a list of four per-bin activity probabilities (0–200, 200–600, 600–1200, 1200–2000 ms), and `p_bins_now` / `p_bins_future` are their averages over the `p_now` / `p_future` ranges.

## Supported Languages, Frame Rates, and Context Lengths

| lang | model_type | frame_rate | context_len_sec | `vap_mono` |
| ---- | ---------- | ---------- | --------------- | ---------- |
| jp | `normal` (CPC encoder) | 50 | 20 | Coming soon |
| jp | `normal-ver2` (Mimi encoder) | 12.5 | 20 | ✅ |
| en | `normal` (CPC encoder) | 50 | 20 | Coming soon |
| en | `normal-ver2` (Mimi encoder) | 12.5 | 20 | ✅ |
| ch | `normal` (CPC encoder) | 50 | 20 | Coming soon |
| ch | `normal-ver2` (Mimi encoder) | 12.5 | 20 | ✅ |

Japanese and Chinese with the Mimi encoder are released; the other combinations are in preparation and will follow.

Note that the 50 Hz model is the CPC-based one (`model_type="normal"`) and the 12.5 Hz model is the Mimi-based one (`model_type="normal-ver2"`). Unlike the standard [VAP model](vap.md), `vap_mono` offers a single frame rate per encoder and a fixed 20-second context.

## Training Data

The same data as the [VAD model](vad.md).

| lang | Training data |
| ---- | ------------- |
| jp | [Travel Agency Task Dialogue](https://aclanthology.org/2022.lrec-1.619/), [Human-Robot Dialogue](https://aclanthology.org/2025.naacl-long.367/), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) |
| en | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62), [Seamless Interaction](https://ai.meta.com/research/seamless-interaction/), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) |
| ch | [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) |

## Usage Example

```python
from maai import Maai, MaaiInput, MaaiOutput

mic = MaaiInput.Mic()

maai = Maai(
    mode="vap_mono",
    lang="jp",
    frame_rate=12.5,
    context_len_sec=20,
    audio_ch1=mic,   # audio_ch2 is not needed
    device="cpu",
    model_type="normal-ver2",
    use_mimi_onnx=True,
    mimi_onnx_precision="fp32",
)
maai.start()

while True:
    result = maai.get_result()
    print(result["p_now"], result["p_future"], result["vad"])  # all single floats
```

Sample scripts:
- [With 1 mic input](../example/vap_mono/vap_mono_mic.py) 🎤
- [With 1 wav file input](../example/vap_mono/vap_mono_wav.py) 🎵
