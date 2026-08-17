<h1>
<p align="center">
Backchannel Detection (BC-Det) Model
</p>
</h1>
<p align="center">
README: <a href="bc_det.md">English </a> | <a href="bc_det_JP.md">Japanese (日本語) </a>
</p>

Please set the `mode` parameter of the `Maai` class to `bc_det` (two channels) or `bc_det_mono` (single channel).

This model detects **whether the utterance a speaker is producing right now is a backchannel** (相槌, e.g. "うん", "はい", "yeah", "right"). It is a *detection* model: there is no time shift between the audio and the target.

> **`bc_det` vs `bc`** — these solve different tasks and are easy to confuse.
> - [`bc`](vap_bc.md) **predicts** that a backchannel is *about to* happen (target shifted ~0.5 s earlier). Use it to decide *when a system should emit* a backchannel.
> - `bc_det` **detects** that a backchannel *is happening now*. Use it to recognise that the user's short utterance was a backchannel rather than the start of a turn.

The model takes both speaker channels and processes them jointly with cross-channel attention. Both channels matter: the interlocutor's speech is most of the evidence that a short utterance is a backchannel rather than the beginning of a turn.

The input requires 2-channel (or 1-channel for `bc_det_mono`), 16kHz audio data.

## Output

`p_bc_det` is a list of two float values in the range [0.0, 1.0], the backchannel probability of each input channel:

```python
result["p_bc_det"]  # e.g. [0.71, 0.02]  -> speaker 1 is backchanneling, speaker 2 is not
```

For `bc_det_mono`, `p_bc_det` is a single float value for the input channel.

To obtain a binary decision, apply a threshold. Note that **`0.5` is usually not the best operating point** on this task: backchannels cover only about 4% of frames, so the model is trained on heavily imbalanced data. On the Japanese development set the tuned thresholds were about `0.39` for frame-level F1 and `0.45` for event-level F1.

```python
is_backchannel = [v >= 0.45 for v in result["p_bc_det"]]
```

Backchannels are short — the median duration is around 0.25 s — so smoothing the output (filling short gaps, removing short spikes) tends to delete real events rather than tidy their edges. It is off by default and is generally not recommended.

## Supported Languages and Frame Rates

| lang | model_type | frame_rate |
| ---- | ---------- | ---------- |
| jp | `normal-ver2` (Mimi encoder) | 12.5 |
| en | `normal-ver2` (Mimi encoder) | 12.5 |
| ch | `normal-ver2` (Mimi encoder) | 12.5 |

Only the 12.5 Hz Mimi-based models (`model_type="normal-ver2"`) are currently released for this mode.

The models were trained with noise and reverberation (RIR) augmentation, so the noise-robust recipe is the default. For this reason there is no separate `bc_det_mc` mode — `bc_det` *is* the multi-condition model.

## Usage Example

```python
from maai import Maai, MaaiInput, MaaiOutput

mic1 = MaaiInput.Mic(mic_device_index=0)
mic2 = MaaiInput.Mic(mic_device_index=1)

maai = Maai(
    mode="bc_det",
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
    print(result["p_bc_det"])  # [float, float]
```

For the single-channel version, set `mode="bc_det_mono"` and pass only `audio_ch1`; `result["p_bc_det"]` is then a single float. Because the model relies on the interlocutor's speech, the mono variant is less accurate than the two-channel one — prefer `bc_det` when both channels are available.

Sample scripts:
- [With 2 mic inputs](../example/bc_det/bc_det_2mic.py) 🎤
- [With 2 wav file inputs](../example/bc_det/bc_det_2wav.py) 🎵
- [With 1 mic input (mono)](../example/bc_det/bc_det_mono_mic.py) 🎤
- [With 1 wav file input (mono)](../example/bc_det/bc_det_mono_wav.py) 🎵
