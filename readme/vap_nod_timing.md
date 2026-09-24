<h1>
<p align="center">
Nod Prediction Model (Timing-only)
</p>
</h1>
<p align="center">
README: <a href="vap_nod_timing.md">English </a> | <a href="vap_nod_timing_JP.md">Japanese (日本語) </a>
</p>

Set the `mode` parameter of the `Maai` class to `nod_timing`.

This model takes 2-channel 16kHz audio data as input, assuming ch1 as user audio and ch2 as system audio.

- `p_nod`: nodding probability
- `p_bc`: backchannel probability

</br>

## Supported Languages

Currently, only Japanese is supported.
Specify this with the `lang` parameter of the `Maai` class.

### Japanese (`lang=jp`)

This model is trained on the following Japanese dataset:
- [Human-Robot Dialogue Corpus]()

</br>

## Example Implementation

```python
from maai import Maai, MaaiInput

mic = MaaiInput.Mic(mic_device_index=0)
zero = MaaiInput.Zero()

maai = Maai(mode="nod_timing", lang="jp", frame_rate=10, audio_ch1=mic, audio_ch2=zero, device="cpu")
maai.start()

while True:
    result = maai.get_result()

    print(result['p_nod'])
    print(result['p_bc'])
```

</br>

## Parameters

The available parameters are summarized below.

`model_type` selects the model variant: `"normal-ver2"` is the newer variant that uses Mimi as the encoder, and `"normal"` is the existing variant used in previous releases, which uses the CPC encoder.

`frame_rate` specifies the number of samples the VAP model processes per second. Please adjust this value according to your computing environment.

### `model_type="normal-ver2"` (Mimi encoder)

| lang | frame_rate |
| ---- | ---------- |
| jp | 12.5 |

### `model_type="normal"` (CPC encoder)

| lang | frame_rate |
| ---- | ---------- |
| jp | 5, 10, 20 |

<br>

## 📚 Papers & References

When publishing results using this model, please cite the following paper. 🙏

Kazushi Kato, Koji Inoue, Divesh Lala, Keiko Ochi, Tatsuya Kawahara<br>
__Real-time Generation of Various Types of Nodding for Avatar Attentive Listening System__<br>
https://www.arxiv.org/abs/2507.23298<br>

```
@inproceedings{kato2025icmi,
    author = {Kazushi Kato and Koji Inoue and Divesh Lala and Keiko Ochi and Tatsuya Kawahara},
    title = {Real-time Generation of Various Types of Nodding for Avatar Attentive Listening System},
    booktitle = {International Conference on Multimodal Interaction (ICMI)},
    year = {2025},
}
```
