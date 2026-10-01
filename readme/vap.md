<h1>
<p align="center">
Turn-Taking (VAP) Model
</p>
</h1>
<p align="center">
README: <a href="vap.md">English </a> | <a href="vap_JP.md">Japanese (日本語) </a>
</p>

This model predicts **the near-future voice activity of the speakers**, which is the basis for turn-taking decisions in a spoken dialogue system.

- Set `mode="vap"` in the `Maai` class.
- The input is 16kHz audio, either 2-channel (two speakers) or 1-channel (one speaker).
- The best model for your input is selected automatically (see [Model Selection](#model-selection)).

> **Note:** `mode="vap"` now uses the noise-robust model by default (`mc=True`). To get the previous behavior (standard model), pass `mc=False`.
> The old mode names `vap_mc`, `vap_mono` and `vap_mc_mono` still work but are deprecated.

<br>

## Quick Start

### 2-channel input (two speakers)

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
)
maai.start()

while True:
    result = maai.get_result()
    print(result["p_now"])     # [float, float]
    print(result["p_future"])  # [float, float]
```

### 1-channel input (one speaker)

Just omit `audio_ch2`.

```python
from maai import Maai, MaaiInput

mic = MaaiInput.Mic()

maai = Maai(
    mode="vap",
    lang="jp",
    frame_rate=12.5,
    audio_ch1=mic,
    device="cpu",
    model_type="normal-ver2",
)
maai.start()

while True:
    result = maai.get_result()
    print(result["p_now"], result["p_future"], result["vad"])  # single floats
```

<br>

## Model Selection

`mode="vap"` has four model variants. The variant is chosen from the `mc` parameter and the audio inputs:

| Input | `mc=True` (default) | `mc=False` |
| --- | --- | --- |
| **2-channel**: `audio_ch1` + `audio_ch2` | `vap_mc` (noise-robust) | `vap` (standard) |
| **1-channel**: `audio_ch1` only, or `audio_ch2=MaaiInput.Zero()` | `vap_mc_mono` (noise-robust) | `vap_mono` (standard) |

You can check the variant that was actually loaded with `maai.variant`.

### Noise-robust models (`mc=True`)

- Trained with various environmental noises added and with random changes to the speech gain (multi-condition training).
- Expected to work more robustly in real-world environments than the standard models.

### 1-channel models

- Dedicated single-channel models with their own pretrained weights, **not** the 2-channel model fed with a silent second channel.
- They encode one audio stream and predict the future activity of that one speaker directly.
- Intended for cases where only one speaker's audio is available (e.g., a single microphone in a spoken dialogue system).

### Fallback

Not every variant exists for every `lang` / `frame_rate` / `model_type` (see [Supported Languages and Frame Rates](#supported-languages-and-frame-rates)). When the preferred variant is missing:

- No noise-robust model → the standard model is used.
- No 1-channel model → the 2-channel model is used with a silent second channel, and a warning is printed.

<br>

## Output

The output format depends on whether a 2-channel or a 1-channel model is used.

| Key | 2-channel model | 1-channel model |
| --- | --- | --- |
| `p_now` | `[float, float]`: probability that each speaker holds the floor in the next 0–600 ms | `float`: probability that the input speaker is active in the next 0–600 ms |
| `p_future` | `[float, float]`: the same for 600–2000 ms ahead | `float`: the same for 600–2000 ms ahead |
| `vad` | `[float, float]`: voice activity probability of each channel at the current frame | `float`: voice activity probability of the input channel |

- All values are in the range [0.0, 1.0].
- For typical turn-taking implementations, `p_now` is recommended.
- **2-channel**: `p_now` and `p_future` are normalized between the speakers, so the two values sum to 1.0.
- **1-channel**: there is no second speaker to compare against, so the values are **not** normalized. Each is the expected voice-activity ratio of the input speaker.
- `vad` is the same quantity as the [VAD model](vad.md), computed inside the VAP model.

```python
# 2-channel
result["p_now"]     # e.g. [0.87, 0.13]  -> speaker 1 is likely to be the next speaker
result["p_future"]  # e.g. [0.62, 0.38]

# 1-channel
result["p_now"]     # e.g. 0.87  -> the input speaker is likely to be speaking in the next 600 ms
```

### Per-bin probabilities (`return_p_bins=True`)

With `return_p_bins=True`, the following keys are also returned:

- `p_bins`: activity probabilities for the four bins (0–200, 200–600, 600–1200, 1200–2000 ms). Per speaker for 2-channel models, a single list for 1-channel models.
- `p_bins_now` / `p_bins_future`: their averages over the `p_now` / `p_future` ranges.

Unlike `p_now` and `p_future`, these are not normalized between the speakers.

<br>

## Supported Languages and Frame Rates

- `lang`: language of the model.
- `model_type`: `"normal-ver2"` is the newer model that uses the Mimi encoder. `"normal"` is the existing model from previous releases, which uses the CPC encoder.
- `frame_rate`: number of frames processed per second. Adjust it to your computing environment.

### `model_type="normal-ver2"` (Mimi encoder, `frame_rate=12.5`)

<table>
<thead>
<tr>
<th rowspan="2" align="left">Language</th>
<th rowspan="2" align="left"><code>lang</code></th>
<th colspan="2">🎧🎧 2-channel</th>
<th colspan="2">🎧 1-channel</th>
</tr>
<tr>
<th>Standard<br><sub><code>vap</code></sub></th>
<th>Noise-robust<br><sub><code>vap_mc</code></sub></th>
<th>Standard<br><sub><code>vap_mono</code></sub></th>
<th>Noise-robust<br><sub><code>vap_mc_mono</code></sub></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2"><b>Japanese</b></td>
<td><code>jp</code></td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">✅</td>
</tr>
<tr>
<td><code>jp_kyoto</code> <sub>MIT</sub></td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td rowspan="2"><b>English</b></td>
<td><code>en</code></td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">✅</td>
</tr>
<tr>
<td><code>en_kyoto</code> <sub>MIT</sub></td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td rowspan="2"><b>Chinese</b></td>
<td><code>ch</code></td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">✅</td>
</tr>
<tr>
<td><code>ch_kyoto</code> <sub>MIT</sub></td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td rowspan="2"><b>Trilingual</b></td>
<td><code>tri</code></td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td><code>tri_kyoto</code> <sub>MIT</sub></td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
</tbody>
</table>

### `model_type="normal"` (CPC encoder)

Each cell lists the available `frame_rate` values.

<table>
<thead>
<tr>
<th rowspan="2" align="left">Language</th>
<th rowspan="2" align="left"><code>lang</code></th>
<th colspan="2">🎧🎧 2-channel</th>
<th colspan="2">🎧 1-channel</th>
</tr>
<tr>
<th>Standard<br><sub><code>vap</code></sub></th>
<th>Noise-robust<br><sub><code>vap_mc</code></sub></th>
<th>Standard<br><sub><code>vap_mono</code></sub></th>
<th>Noise-robust<br><sub><code>vap_mc_mono</code></sub></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2"><b>Japanese</b></td>
<td><code>jp</code></td>
<td align="center">5, 10, 20</td>
<td align="center">5, 10, 20</td>
<td align="center">10, 20, 50</td>
<td align="center">—</td>
</tr>
<tr>
<td><code>jp_kyoto</code> <sub>MIT</sub></td>
<td align="center">5, 10, 20</td>
<td align="center">5, 10, 20</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td rowspan="2"><b>English</b></td>
<td><code>en</code></td>
<td align="center">5, 10, 20</td>
<td align="center">5, 10, 20</td>
<td align="center">10, 20, 50</td>
<td align="center">—</td>
</tr>
<tr>
<td><code>en_kyoto</code> <sub>MIT</sub></td>
<td align="center">5, 10</td>
<td align="center">5, 10</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td rowspan="2"><b>Chinese</b></td>
<td><code>ch</code></td>
<td align="center">5, 10, 20</td>
<td align="center">5, 10, 20</td>
<td align="center">10, 20, 50</td>
<td align="center">—</td>
</tr>
<tr>
<td><code>ch_kyoto</code> <sub>MIT</sub></td>
<td align="center">5, 10</td>
<td align="center">5, 10</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td rowspan="2"><b>Trilingual</b></td>
<td><code>tri</code></td>
<td align="center">5, 10</td>
<td align="center">5, 10</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td><code>tri_kyoto</code> <sub>MIT</sub></td>
<td align="center">5, 10</td>
<td align="center">5, 10</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
</tbody>
</table>

✅ available ／ — not available (the [fallback](#fallback) applies) ／ <sub>MIT</sub> released under the MIT license

<br>

## Training Data

- `tri` is the tri-lingual (JPN + ENG + CHN) model.
- The `*_kyoto` models are trained only on the Online Conversation Dataset and are released under the MIT license.
- The noise-robust models (`mc`) use the same data as the corresponding standard models, with environmental noise and random gain augmentation applied.

### 2-channel models

| lang | Training data | License |
| ---- | ------------- | ------- |
| jp | [Travel Agency Task Dialogue](https://aclanthology.org/2022.lrec-1.619/), [Human-Robot Dialogue](https://aclanthology.org/2025.naacl-long.367/), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | |
| jp_kyoto | [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | MIT |
| en | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62), [Seamless Interaction](https://ai.meta.com/research/seamless-interaction/)\*, [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | |
| en_kyoto | [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | MIT |
| ch | [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | |
| ch_kyoto | [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | MIT |
| tri | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62), [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15), [Travel Agency Task Dialogue](https://aclanthology.org/2022.lrec-1.619/), [Human-Robot Dialogue](https://aclanthology.org/2025.naacl-long.367/), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | |
| tri_kyoto | [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | MIT |

\* Not used for the noise-robust model (`vap_mc`).

### 1-channel models

The same data as the [VAD model](vad.md).

| lang | Training data |
| ---- | ------------- |
| jp | [Travel Agency Task Dialogue](https://aclanthology.org/2022.lrec-1.619/), [Human-Robot Dialogue](https://aclanthology.org/2025.naacl-long.367/), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) |
| en | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62), [Seamless Interaction](https://ai.meta.com/research/seamless-interaction/), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) |
| ch | [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) |

<br>

## Sample Scripts

### 2-channel input

- [With 2 mic inputs](../example/vap/vap_2mic.py) 🎤
- [With 2 wav file inputs](../example/vap/vap_2wav.py) 🎵

### 1-channel input

- [With 1 mic input](../example/vap/vap_mic.py) 🎤
- [With 1 mic input, Mimi encoder (`model_type="normal-ver2"`)](../example/vap/vap_mic_ver2.py) 🎤
- [With 1 mic input, noise-robust model](../example/vap_mc/vap_mc_mic.py) 🎤
- [With 1 mic input, 1-channel model](../example/vap_mono/vap_mono_mic.py) 🎤
- [With 1 wav file input, 1-channel model](../example/vap_mono/vap_mono_wav.py) 🎵

<br>

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

### Multi-lingual model (`tri`)

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

### Noise-robust models (`mc`)

If you use the noise-robust models, please also cite the following paper.

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
