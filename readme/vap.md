<h1>
<p align="center">
Turn-Taking (VAP) Model
</p>
</h1>
<p align="center">
README: <a href="vap.md">English </a> | <a href="vap_JP.md">Japanese (日本語) </a>
</p>

Please set the `mode` parameter of the `Maai` class to `vap`.

This model predicts **the near-future voice activity of the speakers**, which is the basis for turn-taking decisions in a spoken dialogue system.

## Model Selection

`mode="vap"` covers four model variants. Which one is used is decided automatically from the `mc` parameter and the audio inputs:

| | `mc=True` (default) | `mc=False` |
| --- | --- | --- |
| `audio_ch1` + `audio_ch2` (2-channel) | Noise-robust 2-channel model (`vap_mc`) | Standard 2-channel model (`vap`) |
| `audio_ch1` only, or `audio_ch2=MaaiInput.Zero()` (1-channel) | Noise-robust 1-channel model (`vap_mc_mono`) | Standard 1-channel model (`vap_mono`) |

- **Noise-robust (multi-condition) models** (`mc=True`) have been trained on data with various environmental noises added, and the gain of the speech audio was also randomly changed. They are therefore expected to operate more robustly in real-world environments than the standard models. If no noise-robust model exists for the given `lang` / `frame_rate` / `model_type`, the standard model is used instead.
- **1-channel models** are dedicated single-channel models with their own pretrained weights — not the 2-channel model fed with a silent second channel. They encode one audio stream, replace the cross-channel transformer with a plain causal transformer, and predict the future activity of that one speaker directly. They are intended for use cases where only one speaker's audio is available (e.g., a single microphone input for a spoken dialogue system). If no 1-channel model exists for the given combination, the 2-channel model is used with a silent second channel, and a warning is printed.

The variant actually loaded is available as `maai.variant` (e.g. `"vap_mc"`, `"vap_mc_mono"`).

> **Note:** Before this change, `mode="vap"` always used the standard model. Since `mc=True` is now the default, pass `mc=False` to keep the previous behavior. The old mode names `vap_mc`, `vap_mono` and `vap_mc_mono` still work but are deprecated.

The input is 16kHz audio data (2-channel or 1-channel).

## Output

### 2-channel models

`p_now` and `p_future` are lists of two float values in the range [0.0, 1.0], the probability that each speaker holds the floor over the corresponding time range. The two values are normalized between the speakers, so they sum to 1.0.

- `p_now` covers the next 0 to 600 milliseconds.
- `p_future` covers 600 to 2000 milliseconds ahead.

For typical turn-taking implementations, it is recommended to use `p_now`.

```python
result["p_now"]     # e.g. [0.87, 0.13]  -> speaker 1 is likely to be the next speaker
result["p_future"]  # e.g. [0.62, 0.38]
```

`vad` is a list of two float values, the voice activity probability of each input channel at the current frame (the same quantity as the [VAD model](vad.md), computed inside the VAP model).

With `return_p_bins=True`, `p_bins` is a list of per-speaker, per-bin activity probabilities over the four bins (0–200, 200–600, 600–1200, 1200–2000 ms), and `p_bins_now` / `p_bins_future` are their averages over the `p_now` / `p_future` ranges. Unlike `p_now` and `p_future`, these are not normalized between the speakers.

### 1-channel models

`p_now` and `p_future` are single float values in the range [0.0, 1.0] (not two-element lists):

- `p_now` is the probability that the input speaker is active in the next 0 to 600 milliseconds.
- `p_future` is the same for 600 to 2000 milliseconds ahead.

Because there is no second speaker to compare against, these values are **not** normalized between speakers: each is the expected voice-activity ratio of the input speaker over the corresponding time range, already a probability.

`vad` is also a single float value for the input channel.

With `return_p_bins=True`, `p_bins` is a list of four per-bin activity probabilities (0–200, 200–600, 600–1200, 1200–2000 ms), and `p_bins_now` / `p_bins_future` are their averages over the `p_now` / `p_future` ranges.

## Supported Languages and Frame Rates

Specify the language with the `lang` parameter of the `Maai` class.

`model_type` selects the model variant: `"normal-ver2"` is the newer variant that uses Mimi as the encoder, and `"normal"` is the existing variant used in previous releases, which uses the CPC encoder.

`frame_rate` specifies the number of samples processed per second by the VAP model. Please adjust this value according to your computing environment.

Each cell lists the available `frame_rate` values. "–" means the variant is not available, in which case the fallback described in [Model Selection](#model-selection) applies.

### `model_type="normal-ver2"` (Mimi encoder)

The 1-channel models are available only with a 20-second context (`context_len_sec=20`, the default).

| lang | 2ch (`vap`) | 2ch, mc (`vap_mc`) | 1ch (`vap_mono`) | 1ch, mc (`vap_mc_mono`) |
| ---- | ---- | ---- | ---- | ---- |
| jp | 12.5 | 12.5 | 12.5 | 12.5 |
| jp_kyoto | 12.5 | 12.5 | – | – |
| en | 12.5 | 12.5 | 12.5 | 12.5 |
| en_kyoto | 12.5 | 12.5 | – | – |
| ch | 12.5 | 12.5 | 12.5 | 12.5 |
| ch_kyoto | Coming soon | 12.5 | – | – |
| tri | 12.5 | 12.5 | – | – |
| tri_kyoto | 12.5 | 12.5 | – | – |

### `model_type="normal"` (CPC encoder)

| lang | 2ch (`vap`) | 2ch, mc (`vap_mc`) | 1ch (`vap_mono`) | 1ch, mc (`vap_mc_mono`) |
| ---- | ---- | ---- | ---- | ---- |
| jp | 5, 10, 20 | 5, 10, 20 | 10, 20, 50 | – |
| jp_kyoto | 5, 10, 20 | 5, 10, 20 | – | – |
| en | 5, 10, 20 | 5, 10, 20 | 10, 20, 50 | – |
| en_kyoto | 5, 10 | 5, 10 | – | – |
| ch | 5, 10, 20 | 5, 10, 20 | 10, 20, 50 | – |
| ch_kyoto | 5, 10 | 5, 10 | – | – |
| tri | 5, 10 | 5, 10 | – | – |
| tri_kyoto | 5, 10 | 5, 10 | – | – |

## Training Data

`tri` is the tri-lingual (JPN + ENG + CHN) model. The `*_kyoto` models are trained only on the Online Conversation Dataset and are released under the MIT license.

### 2-channel models

| lang | Training data | License |
| ---- | ------------- | ------- |
| jp | [Travel Agency Task Dialogue](https://aclanthology.org/2022.lrec-1.619/), [Human-Robot Dialogue](https://aclanthology.org/2025.naacl-long.367/), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | |
| jp_kyoto | [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | MIT |
| en | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62), [Seamless Interaction](https://ai.meta.com/research/seamless-interaction/), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | |
| en_kyoto | [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | MIT |
| ch | [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | |
| ch_kyoto | [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | MIT |
| tri | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62), [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15), [Travel Agency Task Dialogue](https://aclanthology.org/2022.lrec-1.619/), [Human-Robot Dialogue](https://aclanthology.org/2025.naacl-long.367/), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | |
| tri_kyoto | [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | MIT |

The noise-robust 2-channel models (`vap_mc`) use the same data with environmental noise and random gain augmentation applied (for `en`, the Switchboard corpus and the Online Conversation Dataset only).

### 1-channel models

The same data as the [VAD model](vad.md). The noise-robust 1-channel models (`vap_mc_mono`) use the same data with environmental noise and random gain augmentation applied.

| lang | Training data |
| ---- | ------------- |
| jp | [Travel Agency Task Dialogue](https://aclanthology.org/2022.lrec-1.619/), [Human-Robot Dialogue](https://aclanthology.org/2025.naacl-long.367/), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) |
| en | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62), [Seamless Interaction](https://ai.meta.com/research/seamless-interaction/), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) |
| ch | [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) |

## Usage Example

### 2-channel input

```python
from maai import Maai, MaaiInput

wav1 = MaaiInput.Wav(wav_file_path="path_to_your_user_wav_file")
wav2 = MaaiInput.Wav(wav_file_path="path_to_your_system_wav_file")

maai = Maai(
    mode="vap",
    lang="jp",
    frame_rate=10,
    audio_ch1=wav1,
    audio_ch2=wav2,
    device="cpu",
    # mc=False,  # use the standard (non noise-robust) model
)
maai.start()

while True:
    result = maai.get_result()
    print(result["p_now"])     # [float, float]
    print(result["p_future"])  # [float, float]
```

### 1-channel input

```python
from maai import Maai, MaaiInput

mic = MaaiInput.Mic()

maai = Maai(
    mode="vap",
    lang="jp",
    frame_rate=12.5,
    audio_ch1=mic,   # audio_ch2 is omitted -> 1-channel model
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
- [With 2 mic inputs](../example/vap/vap_2mic.py) 🎤
- [With 2 wav file inputs](../example/vap/vap_2wav.py) 🎵
- [With 1 mic input (the second channel is a zero signal)](../example/vap/vap_mic.py) 🎤
- [With 1 mic input, Mimi encoder (`model_type="normal-ver2"`)](../example/vap/vap_mic_ver2.py) 🎤
- [With 1 mic input, noise-robust model](../example/vap_mc/vap_mc_mic.py) 🎤
- [With 1 mic input, 1-channel model](../example/vap_mono/vap_mono_mic.py) 🎤
- [With 1 wav file input, 1-channel model](../example/vap_mono/vap_mono_wav.py) 🎵

## 📚 Publication

Please cite the following paper, if you made any publications made with this model. 🙏

Koji Inoue, Bing'er Jiang, Erik Ekstedt, Tatsuya Kawahara, Gabriel Skantze<br>
__Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection__<br>
International Workshop on Spoken Dialogue Systems Technology (IWSDS), 2024<br>
https://arxiv.org/abs/2401.04868<br>

```
@inproceedings{inoue2024iwsds,
    author = {Koji Inoue and Bing'er Jiang and Erik Ekstedt and Tatsuya Kawahara and Gabriel Skantze},
    title = {Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection},
    booktitle = {International Workshop on Spoken Dialogue Systems Technology (IWSDS)},
    year = {2024},
    url = {https://arxiv.org/abs/2401.04868},
}
```

If you use the multi-lingual VAP model, please also cite the following paper.

Koji Inoue, Bing'er Jiang, Erik Ekstedt, Tatsuya Kawahara, Gabriel Skantze<br>
__Multilingual Turn-taking Prediction Using Voice Activity Projection__<br>
Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING), pages 11873-11883, 2024<br>
https://aclanthology.org/2024.lrec-main.1036/<br>

```
@inproceedings{inoue2024lreccoling,
    author = {Koji Inoue and Bing'er Jiang and Erik Ekstedt and Tatsuya Kawahara and Gabriel Skantze},
    title = {Multilingual Turn-taking Prediction Using Voice Activity Projection},
    booktitle = {Proceedings of the Joint International Conference on Computational Linguistics and Language Resources and Evaluation (LREC-COLING)},
    pages = {11873--11883},
    year = {2024},
    url = {https://aclanthology.org/2024.lrec-main.1036/},
}
```

If you use the noise-robust (`mc`) models, please also cite the following paper.

Koji Inoue, Yuki Okafuji, Jun Baba, Yoshiki Ohira, Katsuya Hyodo, Tatsuya Kawahara<br>
__A Noise-Robust Turn-Taking System for Real-World Dialogue Robots: A Field Experiment__<br>
https://www.arxiv.org/abs/2503.06241<br>

```
@misc{inoue2025noisevap,
    author = {Koji Inoue and Yuki Okafuji and Jun Baba and Yoshiki Ohira and Katsuya Hyodo and Tatsuya Kawahara},
    title = {A Noise-Robust Turn-Taking System for Real-World Dialogue Robots: A Field Experiment},
    year = {2025},
    note = {arXiv:2503.06241},
    url = {https://www.arxiv.org/abs/2503.06241},
}
```
