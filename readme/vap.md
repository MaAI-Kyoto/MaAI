<h1>
<p align="center">
Turn-Taking (VAP) Model
</p>
</h1>
<p align="center">
README: <a href="vap.md">English </a> | <a href="vap_JP.md">Japanese (日本語) </a>
</p>

Please set the `mode` parameter of the `Maai` class to `vap`.

This model predicts **the near-future voice activity of two speakers**, which is the basis for turn-taking decisions in a spoken dialogue system.

The model takes both speaker channels and processes them jointly with cross-channel attention, so the prediction for one speaker is always made in the context of what the other speaker is doing.

The input requires 2-channel, 16kHz audio data.

## Output

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

## Supported Languages and Frame Rates

Specify the language with the `lang` parameter of the `Maai` class.

`model_type` selects the model variant: `"normal-ver2"` is the newer variant that uses Mimi as the encoder, and `"normal"` is the existing variant used in previous releases, which uses the CPC encoder.

`frame_rate` specifies the number of samples processed per second by the VAP model. Please adjust this value according to your computing environment.

### `model_type="normal-ver2"` (Mimi encoder)

| lang | frame_rate |
| ---- | ---------- |
| jp | 12.5 |
| jp_kyoto | 12.5 |
| en | 12.5 |
| en_kyoto | 12.5 |
| ch | 12.5 |
| ch_kyoto | Coming soon |
| tri | 12.5 |
| tri_kyoto | 12.5 |

### `model_type="normal"` (CPC encoder)

| lang | frame_rate |
| ---- | ---------- |
| jp | 5, 10, 20 |
| jp_kyoto | 5, 10, 20 |
| en | 5, 10, 20 |
| en_kyoto | 5, 10 |
| ch | 5, 10, 20 |
| ch_kyoto | 5, 10 |
| tri | 5, 10 |
| tri_kyoto | 5, 10 |

## Training Data

`tri` is the tri-lingual (JPN + ENG + CHN) model. The `*_kyoto` models are trained only on the Online Conversation Dataset and are released under the MIT license.

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

## Usage Example

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

Sample scripts:
- [With 2 mic inputs](../example/vap/vap_2mic.py) 🎤
- [With 2 wav file inputs](../example/vap/vap_2wav.py) 🎵
- [With 1 mic input (the second channel is a zero signal)](../example/vap/vap_mic.py) 🎤
- [With 1 mic input, Mimi encoder (`model_type="normal-ver2"`)](../example/vap/vap_mic_ver2.py) 🎤

If only one speaker's audio is available, the dedicated [single-channel model (`vap_mono`)](vap_mono.md) is recommended over feeding a zero signal to the second channel.

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
