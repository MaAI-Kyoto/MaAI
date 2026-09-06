<h1>
<p align="center">
Noise-Robust Turn-Taking (VAP) Model (MC-VAP)
</p>
</h1>
<p align="center">
README: <a href="vap_mc.md">English </a> | <a href="vap_mc_JP.md">Japanese (日本語) </a>
</p>

Please set the `mode` parameter of the `Maai` class to `vap_mc`.

This is the multi-condition version of the standard [VAP model](vap.md): it has been trained on data with various environmental noises added, and the gain of the speech audio was also randomly changed. Therefore, it is expected to operate more robustly in real-world environments than the standard model.

Apart from the training conditions, the model architecture, the inputs, and the outputs are the same as the standard [VAP model](vap.md).

The input requires 2-channel, 16kHz audio data.

## Output

`p_now` and `p_future` are lists of two float values in the range [0.0, 1.0], the probability that each speaker holds the floor over the corresponding time range. The two values are normalized between the speakers, so they sum to 1.0.

- `p_now` covers the next 0 to 600 milliseconds.
- `p_future` covers 600 to 2000 milliseconds ahead.

For general turn-taking purposes, we recommend using `p_now`.

```python
result["p_now"]     # e.g. [0.87, 0.13]  -> speaker 1 is likely to be the next speaker
result["p_future"]  # e.g. [0.62, 0.38]
```

`vad` is a list of two float values, the voice activity probability of each input channel at the current frame.

With `return_p_bins=True`, `p_bins` is a list of per-speaker, per-bin activity probabilities over the four bins (0–200, 200–600, 600–1200, 1200–2000 ms), and `p_bins_now` / `p_bins_future` are their averages over the `p_now` / `p_future` ranges. Unlike `p_now` and `p_future`, these are not normalized between the speakers.

## Supported Languages and Frame Rates

Specify the language with the `lang` parameter of the `Maai` class.

| lang | model_type | frame_rate |
| ---- | ---------- | ---------- |
| jp | `normal` (CPC encoder) | 5, 10, 20 |
| jp | `normal-ver2` (Mimi encoder) | 12.5 |
| jp_kyoto | `normal` (CPC encoder) | 5, 10, 20 |
| jp_kyoto | `normal-ver2` (Mimi encoder) | 12.5 |
| en | `normal` (CPC encoder) | 5, 10, 20 |
| en | `normal-ver2` (Mimi encoder) | 12.5 |
| en_kyoto | `normal` (CPC encoder) | 5, 10 |
| en_kyoto | `normal-ver2` (Mimi encoder) | 12.5 |
| ch | `normal` (CPC encoder) | 5, 10, 20 |
| ch | `normal-ver2` (Mimi encoder) | 12.5 |
| ch_kyoto | `normal` (CPC encoder) | 5, 10 |
| ch_kyoto | `normal-ver2` (Mimi encoder) | 12.5 |
| tri | `normal` (CPC encoder) | 5, 10 |
| tri | `normal-ver2` (Mimi encoder) | 12.5 |
| tri_kyoto | `normal` (CPC encoder) | 5, 10 |
| tri_kyoto | `normal-ver2` (Mimi encoder) | 12.5 |

`model_type` selects the model variant: `"normal"` is the existing variant used in previous releases, and `"normal-ver2"` is the newer variant that uses Mimi as the encoder. Note that the 5/10/20 Hz models are the CPC-based ones (`model_type="normal"`) and the 12.5 Hz model is the Mimi-based one (`model_type="normal-ver2"`).

`frame_rate` specifies the number of samples the VAP model processes per second. Please adjust this value according to your computing environment.

## Training Data

The same data as the standard [VAP model](vap.md), with environmental noise and random gain augmentation applied.
`tri` is the tri-lingual (JPN + ENG + CHN) model. The `*_kyoto` models are trained only on the Online Conversation Dataset and are released under the MIT license.

| lang | Training data | License |
| ---- | ------------- | ------- |
| jp | [Travel Agency Task Dialogue](https://aclanthology.org/2022.lrec-1.619/), [Human-Robot Dialogue](https://aclanthology.org/2025.naacl-long.367/), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | |
| jp_kyoto | [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | MIT |
| en | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62), [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191) | |
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
    mode="vap_mc",
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
- [With 1 mic input (the second channel is a zero signal)](../example/vap_mc/vap_mc_mic.py) 🎤

## 📚 Publication

When publishing results using this model, please cite the following paper. 🙏

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
