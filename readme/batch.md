<h1>
<p align="center">
MaaiBatch - Batch Inference for Recorded Audio
</p>
</h1>
<p align="center">
README: <a href="batch.md">English </a> | <a href="batch_JP.md">Japanese (日本語) </a>
</p>

`Maai` is built for live audio: it runs the audio encoder and the model once per frame (every 80 ms at 12.5 Hz).
When the whole recording is already available (e.g. analysing a dialogue corpus, or scoring synthesized dialogues),
`MaaiBatch` computes the same outputs for the whole recording at once:

- The audio encoder runs over long chunks (`encoder_chunk_frames` frames per call) of several recordings in parallel (`batch_size`), carrying its streaming state across chunks, so the features are those of the frame-by-frame stream.
  - Mimi: the resampler, convolution padding and transformer KV cache are carried over.
  - CPC: the realtime path encodes each frame together with 320 samples of overlap and keeps only the convolution outputs that do not touch the window edges, so a window of several frames gives the same outputs. The LSTM state is carried over, and each model's downsampling layer runs window by window as in realtime.
- Each model's transformer runs over the whole sequence in one forward pass.
  The context-length band mask makes this the same computation as the realtime KV cache, which keeps the last `context_len_sec` seconds.
  Long recordings are split into overlapping windows that are processed as one batch; the overlap covers the receptive field of the transformer, so the split does not change the result.
- With several models (`configs`), the audio is encoded once and shared, as in `MaaiMultiple`.

</br>

## Requirements

- Mimi models (`model_type="normal-ver2"`, the default of `MaaiBatch`) at the Mimi frame rate (`frame_rate=12.5`, the default for Mimi),
  or CPC models (`model_type="normal"`) at any frame rate where 16000 / `frame_rate` is a multiple of 160 samples (5, 10, 20, 50 Hz, ...; 10 Hz by default).
- Modes: `vap`, `vad`, `bc_det`, `bc`, `bc_2type`, `nod`, `nod_timing`, `nod_para` (`vap_prompt` is not supported).
  A pretrained model must exist for the mode, language, encoder, frame rate and context length (see [our HuggingFace repository](https://huggingface.co/maai-kyoto)), or give `local_model`.
- Input: 16 kHz mono float32 waveforms (numpy arrays), one per speaker.

</br>

## Usage

```python
import soundfile as sf
from maai import MaaiBatch

ch1, _ = sf.read("speaker1_16k.wav", dtype="float32")
ch2, _ = sf.read("speaker2_16k.wav", dtype="float32")

batch = MaaiBatch(mode="vap", lang="jp", device="cuda")
result = batch.process(ch1, ch2)

result["time_sec"]   # (n_frames,)    time of each prediction [s]
result["p_now"]      # (n_frames, 2)
result["p_future"]   # (n_frames, 2)
result["vad"]        # (n_frames, 2)
```

Several recordings are processed `batch_size` at a time (similar lengths are grouped together):

```python
results = batch.process_many([(a1, a2), (b1, b2), (c1, c2)])
```

Several models share one encoder pass; the results are nested by label (default: the mode):

```python
batch = MaaiBatch(
    configs=[
        {"mode": "vap", "lang": "jp"},
        {"mode": "bc",  "lang": "jp"},
        {"mode": "nod", "lang": "jp"},
    ],
    device="cuda",
)
result = batch.process(ch1, ch2)
result["vap"]["p_now"], result["bc"]["p_bc"], result["nod"]["p_nod_short"]
```

Each entry of `configs` may contain `"mode"`, `"lang"`, `"mc"`, `"local_model"`, `"return_p_bins"` and `"label"`, as in [MaaiMultiple](multiple.md).

The CPC models are selected with `model_type="normal"` (the same names as in `Maai`):

```python
batch = MaaiBatch(mode="vap", lang="jp", model_type="normal", frame_rate=10, context_len_sec=20)
```

With `mono=True`, only `audio_ch1` is given: `vap`, `vad` and `bc_det` use their single-channel models, and the other models get silence on channel 2 (the same as `MaaiInput.Zero()` in `Maai`).

```python
batch = MaaiBatch(mode="vap", lang="jp", mono=True)
result = batch.process(ch1)
result["p_now"]      # (n_frames,)
```

`MaaiBatch.to_frames(result)` converts a result to one dict per frame, in the same format as `Maai.get_result()` (without `t`, `x1`, `x2`).

An example script is in [example/batch/batch_2wav.py](../example/batch/batch_2wav.py).

</br>

## Output

- `end_sample`, `time_sec`: the input position at which each prediction is made (the end of the audio the model has seen), one per frame as in the realtime `Maai`.
  With Mimi, the realtime default (ONNX Mimi) drops the first encoder output, so the first prediction is at 0.16 s and they follow every 80 ms.
  With CPC, the first prediction is after one frame (0.1 s at 10 Hz).
  An incomplete trailing frame is not scored.
- The model outputs have the same names and values as `Maai.get_result()`, with one row per frame (`p_now`: `(n, 2)`, `p_bc`: `(n,)`, `nod_repetitions`: `(n, 3)`, ...).
  `p_bins`, `p_bins_now` and `p_bins_future` are returned only with `return_p_bins=True`.

</br>

## Accuracy

Comparison with the realtime `Maai` (`context_len_sec=20`, 40 s of the Japanese sample dialogue, CPU):

| Realtime `Maai` | Frames | max \|difference\| |
| --- | --- | --- |
| Mimi 12.5 Hz, `use_mimi_onnx=False` (PyTorch Mimi) vs `MaaiBatch(drop_first_frame=False)` | 500 / 500 | `vap` 2.8e-5, `bc` 1.5e-5 |
| Mimi 12.5 Hz, default (ONNX Mimi) vs `MaaiBatch()` | 499 / 499 | `vap` 5.4e-5, `bc` 2.5e-5 |
| CPC 10 Hz vs `MaaiBatch(model_type="normal")` | 400 / 400 | `vap` 7.5e-6, `bc` 1.3e-6 |

The same holds for the other models (30 s, max difference 4.3e-5 or less, and the same number of frames):
with Mimi, `vad`, `bc_det`, `nod`, `nod_timing`, `nod_para` and the single-channel `vap` / `vad` / `bc_det` (`nod_para`'s `*_pred` agree on every frame);
with CPC, `vap` at 5 / 10 / 20 Hz, `vad`, `bc_2type`, `nod`, `nod_timing`, the single-channel `vap` (10 Hz) and `vad` (50 Hz).
The remaining differences are floating-point rounding (and ONNX Runtime vs PyTorch for the Mimi default).
The tests are in [tests/test_batch.py](../tests/test_batch.py) (`MAAI_TEST_PRETRAINED=1` runs the comparison with the pretrained models).

</br>

## Speed

The 266 s Japanese sample dialogue on a CPU (16 threads):

| | Time |
| --- | --- |
| Realtime `Maai` (`vap`, ONNX Mimi, 12.5 Hz), frame by frame | 32 ms/frame, about 106 s per model |
| `MaaiBatch` with `vap` + `bc` (Mimi, shared encoder) | 12.4 s for both models |
| Realtime `Maai` (`vap`, CPC, 10 Hz), frame by frame | 12.5 ms/frame, about 33 s per model |
| `MaaiBatch` with `vap` + `bc` (CPC, shared encoder) | 9.9 s for both models |

On a GPU the gap is expected to be larger (not measured here), since the per-frame calls of the realtime path cannot fill the GPU while `MaaiBatch` runs large batched calls.

</br>

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `frame_rate` | 12.5 (Mimi) / 10 (CPC) | Frame rate of the models. |
| `cpc_model` | `~/.cache/cpc/60k_epoch4-d0f474de.pt` | Pretrained CPC weights (CPC only; downloaded if missing, as in `Maai`). |
| `batch_size` | `8` | Recordings encoded together, and windows per transformer forward. Lower it if memory runs out. |
| `encoder_chunk_frames` | `50` | Frames per encoder call. Memory grows with it. The result does not depend on it. |
| `fold_context_frames` | receptive field | Context frames before each window when a long recording is split. The default (996 frames at 20 s / 12.5 Hz) makes the split exact. |
| `fold_step_frames` | 2 x context length | New frames per window (500 frames at 20 s / 12.5 Hz). |
| `drop_first_frame` | `True` (Mimi) / `False` (CPC) | Drop the first encoder output like the realtime ONNX Mimi encoder. With Mimi, `False` matches `Maai(..., use_mimi_onnx=False)`. The realtime CPC path does not drop a frame. |

Memory grows with `batch_size` x the length of the recordings (the encoder features of a whole batch are kept until the models have run).
