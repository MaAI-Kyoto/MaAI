<h1>
<p align="center">
Voice Activity Detection (VAD) Model
</p>
</h1>
<p align="center">
README: <a href="vad.md">English </a> | <a href="vad_JP.md">Japanese (日本語) </a>
</p>

Please set the `mode` parameter of the `Maai` class to `vad` (two channels) or `vad_mono` (single channel).

Unlike the turn-taking models, this model does not predict the future: it detects **whether each speaker is talking right now**.

The model takes both speaker channels and processes them jointly with cross-channel attention. This is the key difference from a conventional per-channel VAD: even when one speaker's voice leaks into the other speaker's microphone (crosstalk), the model can still decide who is actually talking.

The models were trained with noise and reverberation (RIR) augmentation, so the noise-robust recipe is the default. For this reason there is no separate `vad_mc` mode — `vad` *is* the multi-condition model.

The input requires 2-channel (or 1-channel for `vad_mono`), 16kHz audio data.

## Output

`vad` is a list of two float values in the range [0.0, 1.0], the voice activity probability of each input channel:

```python
result["vad"]  # e.g. [0.93, 0.02]  -> speaker 1 is talking, speaker 2 is not
```

For `vad_mono`, `vad` is a single float value for the input channel.

To obtain a binary decision, apply a threshold. `0.5` is the default; `0.54` gave the best F1 score on the development set used during training.

```python
is_speaking = [v >= 0.5 for v in result["vad"]]
```

## Supported Languages and Frame Rates

| lang | model_type | frame_rate | `vad` (2ch) | `vad_mono` (1ch) |
| ---- | ---------- | ---------- | ----------- | ---------------- |
| jp | `normal` (CPC encoder) | 10, 20, 50 | ✅ | ✅ |
| jp | `normal-ver2` (Mimi encoder) | 12.5 | ✅ | ✅ |
| en | `normal` (CPC encoder) | 10, 20, 50 | ✅ | ✅ |
| en | `normal-ver2` (Mimi encoder) | 12.5 | ✅ | ✅ |
| ch | `normal` (CPC encoder) | 10, 20, 50 | ✅ | ✅ |
| ch | `normal-ver2` (Mimi encoder) | 12.5 | ✅ | ✅ |

Note that the 10/20/50 Hz models are the CPC-based ones (`model_type="normal"`) and the 12.5 Hz model is the Mimi-based one (`model_type="normal-ver2"`).

## Training Data

| lang | Training data |
| ---- | ------------- |
| jp | [Travel Agency Task Dialogue](https://aclanthology.org/2022.lrec-1.619/), [Human-Robot Dialogue](https://aclanthology.org/2025.naacl-long.367/), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) |
| en | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62), [Seamless Interaction](https://ai.meta.com/research/seamless-interaction/), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) |
| ch | [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) |

## Usage Example

```python
from maai import Maai, MaaiInput, MaaiOutput

mic1 = MaaiInput.Mic(mic_device_index=0)
mic2 = MaaiInput.Mic(mic_device_index=1)

maai = Maai(
    mode="vad",
    lang="jp",
    frame_rate=12.5,
    audio_ch1=mic1,
    audio_ch2=mic2,
    device="cpu",
    model_type="normal-ver2",
    use_mimi_onnx=True,
    mimi_onnx_precision="fp32",
)
maai.start()

while True:
    result = maai.get_result()
    print(result["vad"])  # [float, float]
```

For the single-channel version, set `mode="vad_mono"` and pass only `audio_ch1`; `result["vad"]` is then a single float.

Sample scripts:
- [With 2 mic inputs](../example/vad/vad_2mic.py) 🎤
- [With 2 wav file inputs](../example/vad/vad_2wav.py) 🎵
- [With 1 mic input (mono)](../example/vad/vad_mono_mic.py) 🎤
- [With 1 wav file input (mono)](../example/vad/vad_mono_wav.py) 🎵
