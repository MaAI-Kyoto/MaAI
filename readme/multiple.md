<h1>
<p align="center">
MaaiMultiple - Sharing the Audio Encoder Across Multiple Models
</p>
</h1>
<p align="center">
README: <a href="multiple.md">English </a> | <a href="multiple_JP.md">Japanese (日本語) </a>
</p>

`MaaiMultiple` runs several Maai models in parallel that share a single audio
encoder. Many Maai models (`vap`, `vap_mc`, `bc`, `bc_2type`, `nod`,
`nod_para`, `vap_prompt`) start their forward pipeline with the same audio
encoder (CPC for `model_type="normal"`, Mimi for `model_type="normal-ver2"`).
Building one independent `Maai` per model would run that encoder N times per
audio frame. `MaaiMultiple` runs the encoder once and feeds the encoded
features into every sub-model, which is significantly more efficient on CPU.

</br>

## When to Use

Use `MaaiMultiple` when you want to combine several behaviors on the same
audio stream, e.g.:

- turn-taking + backchannel + nodding,
- two backchannel variants (`bc` and `bc_2type`) at the same time,
- a comparison of `vap` and `vap_mc` running side by side.

If you only need a single model, use `Maai` directly.

</br>

## Constraints

All sub-models share the encoder, so they must share the encoder
configuration. The following parameters are passed once to `MaaiMultiple` and
applied to every sub-model:

- `model_type` (`"normal"` or `"normal-ver2"`)
- `frame_rate`
- `context_len_sec`
- `device`
- `cpc_model`
- All `mimi_*` parameters (used when `model_type="normal-ver2"`)
- `cache_dir`, `force_download`
- `use_kv_cache`
- `audio_ch1`, `audio_ch2`

Per-model differences are configured through the `configs` list. Each entry
is a `dict` with:

- `"mode"` (required): `"vap"`, `"vap_mc"`, `"vad"`, `"bc"`, `"bc_det"`,
  `"bc_2type"`, `"nod"`, `"nod_para"` or `"vap_prompt"`.
- `"lang"` (required): same value as for `Maai`.
- `"label"` (optional): result-dict key for this sub-model. Defaults to
  `"mode"`. Use it to disambiguate when the same `mode` is registered twice
  (e.g., comparing two languages).
- `"local_model"` (optional): path to a locally trained checkpoint.
- `"return_p_bins"` (optional): same flag as in `Maai` (only meaningful for
  `vap`/`vap_mc`).

</br>

## Example

```python
from maai import MaaiMultiple, MaaiInput

mic = MaaiInput.Mic()
zero = MaaiInput.Zero()

maai_multi = MaaiMultiple(
    configs=[
        {"mode": "vap", "lang": "jp"},
        {"mode": "bc",  "lang": "jp"},
        {"mode": "nod", "lang": "jp"},
    ],
    audio_ch1=mic,
    audio_ch2=zero,
    frame_rate=10,
    device="cpu",
    model_type="normal",
)

maai_multi.start()

while True:
    result = maai_multi.get_result()
    # Shared per-frame fields:
    #   result["t"]   - timestamp
    #   result["x1"]  - audio chunk for channel 1
    #   result["x2"]  - audio chunk for channel 2
    # Per-sub-model outputs are nested under each label (default = mode).
    print(result["vap"]["p_now"])
    print(result["bc"]["p_bc"])
    print(result["nod"]["p_nod_short"])
```

A runnable version of the above lives at
[`example/multiple/multi_mic.py`](../example/multiple/multi_mic.py).

</br>

## Result Format

`get_result()` returns a single `dict` per audio frame:

```text
{
    "t":  <frame timestamp>,
    "x1": <audio chunk for channel 1, np.ndarray>,
    "x2": <audio chunk for channel 2, np.ndarray>,
    "<label-1>": { ...same fields as Maai for that mode... },
    "<label-2>": { ... },
    ...
}
```

Each per-label entry contains exactly the fields that the corresponding
single-model `Maai.get_result()` would have returned (minus the shared `t`,
`x1`, `x2`, which are hoisted to the top level).

</br>

## API

```python
MaaiMultiple(
    configs: list[dict],
    audio_ch1, audio_ch2,
    frame_rate: float = 10,
    context_len_sec: int = 20,
    device: str = "cpu",
    model_type: str = "normal",
    use_kv_cache: bool = True,
    # ... mimi_* / cache_dir / cpc_model / force_download identical to Maai
)
```

Methods mirror `Maai`:

- `start()` - begin the worker thread and the audio sources.
- `stop(wait=True, timeout=2.0)` - stop the worker thread.
- `process(x1, x2)` - feed a chunk of audio manually (skips the mic worker).
- `get_result()` - block until the next combined result dict is available.
- `get_sub_maai(label) -> Maai` - access the underlying `Maai` instance.
- `set_prompt_ch1(prompt, label=None)` /
  `set_prompt_ch2(prompt, label=None)` - update prompts on every
  `vap_prompt` sub-model (or only on `label`).

</br>

## Output: ConsoleBar

`MaaiOutput.ConsoleBar` automatically detects the `MaaiMultiple` result
format. The shared per-frame fields (`t`, `x1`, `x2`) are printed once at
the top, and each sub-model's outputs are rendered below in its own
clearly headed section, so the layout stays readable when several models
are active at the same time.

```python
from maai import MaaiMultiple, MaaiInput, MaaiOutput

mic = MaaiInput.Mic()
zero = MaaiInput.Zero()

maai_multi = MaaiMultiple(
    configs=[
        {"mode": "vap", "lang": "jp"},
        {"mode": "bc",  "lang": "jp"},
        {"mode": "nod", "lang": "jp"},
    ],
    audio_ch1=mic, audio_ch2=zero, frame_rate=10, device="cpu",
)

output = MaaiOutput.ConsoleBar(bar_type="balance")
maai_multi.start()

while True:
    output.update(maai_multi.get_result())
```

The same `ConsoleBar` instance still works unchanged with a single
`Maai`; the multi-section layout only kicks in when nested per-model
dicts are present in the result.

</br>

## Notes

- The encoder modules of all sub-models are replaced with references to the
  primary sub-model's encoders, so encoder weights and Mimi streaming state
  exist only once in memory and on the device.
- The KV cache (`use_kv_cache=True`) is per sub-model and is trimmed to
  `context_len_sec` on every step, identical to `Maai`.
- When `use_kv_cache=False`, the rolling encoded-feature context is shared
  across sub-models (one rolling window of length `context_len_sec`), and
  each sub-model's `decrease_dimension` projection is applied on top of it.
