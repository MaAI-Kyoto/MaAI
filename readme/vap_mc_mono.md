<h1>
<p align="center">
Noise-Robust Single-Channel Turn-Taking (VAP) Model (MC-Mono-VAP)
</p>
</h1>
<p align="center">
README: <a href="vap_mc_mono.md">English </a> | <a href="vap_mc_mono_JP.md">Japanese (日本語) </a>
</p>

Please set the `mode` parameter of the `Maai` class to `vap_mc_mono`.

This is the multi-condition version of the [single-channel VAP model (`vap_mono`)](vap_mono.md): it has been trained on data with various environmental noises added, and the gain of the speech audio was also randomly changed, in the same way as the [noise-robust VAP model (`vap_mc`)](vap_mc.md). Therefore, it is expected to operate more robustly in real-world environments than `vap_mono`.

Apart from the training conditions, the model architecture, the inputs, and the outputs are the same as [`vap_mono`](vap_mono.md): a dedicated single-channel model that encodes one audio stream and predicts the future activity of that one speaker.
It is intended for use cases where only one speaker's audio is available in a noisy environment (e.g., a single microphone input for a spoken dialogue robot in a public space).

The input requires 1-channel, 16kHz audio data.

## Output

`p_now` and `p_future` are single float values in the range [0.0, 1.0] (not two-element lists):

- `p_now` is the probability that the input speaker is active in the next 0 to 600 milliseconds.
- `p_future` is the same for 600 to 2000 milliseconds ahead.

Because there is no second speaker to compare against, these values are **not** normalized between speakers as in the two-channel `vap` / `vap_mc` models: each is the expected voice-activity ratio of the input speaker over the corresponding time range, already a probability.

`vad` is also a single float value for the input channel.

With `return_p_bins=True`, `p_bins` is a list of four per-bin activity probabilities (0–200, 200–600, 600–1200, 1200–2000 ms), and `p_bins_now` / `p_bins_future` are their averages over the `p_now` / `p_future` ranges.

## Supported Languages and Frame Rates

`vap_mc_mono` is available only for `model_type="normal-ver2"` (Mimi encoder) with a 20-second context (`context_len_sec=20`, the default).

| lang | frame_rate | `vap_mc_mono` |
| ---- | ---------- | ------------- |
| jp | 12.5 | ✅ |
| en | 12.5 | ✅ |
| ch | 12.5 | ✅ |

## Training Data

The same data as [`vap_mono`](vap_mono.md), with environmental noise and random gain augmentation applied.

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
    mode="vap_mc_mono",
    lang="jp",
    frame_rate=12.5,
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
