import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import os
import math
import numpy as np
import json
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path

from .encoder_components import CConv1d, load_CPC, get_cnn_layer
from .util import download_continuous_mimi_onnx

import time
import copy
import threading


def _hf_past_kv_per_layer_tuples(
    past_key_values: Any,
    *,
    clone: bool,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Normalize ``past_key_values`` to per-layer ``(key, value)`` pairs.

    Supports: HF tuple layout ``((k,v), ...)``, and v5 ``StaticCache`` (``layers[*].keys`` / ``values``).
    """
    if past_key_values is None:
        return tuple()

    def _maybe_clone(t: torch.Tensor) -> torch.Tensor:
        return t.detach().clone() if clone else t

    if isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
        el0 = past_key_values[0]
        if (
            isinstance(el0, (list, tuple))
            and len(el0) == 2
            and torch.is_tensor(el0[0])
            and torch.is_tensor(el0[1])
        ):
            return tuple((_maybe_clone(a), _maybe_clone(b)) for a, b in past_key_values)

    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None:
                raise TypeError(
                    f"Unsupported cache layer type {type(layer)} on {type(past_key_values)}"
                )
            out.append((_maybe_clone(keys), _maybe_clone(values)))
        return tuple(out)

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")


def _hf_flatten_past_kv_tensors(past_key_values: Any) -> tuple[torch.Tensor, ...]:
    """Flatten past KV to (k0, v0, k1, v1, ...) for ONNX-style I/O."""
    flat: list[torch.Tensor] = []
    for k, v in _hf_past_kv_per_layer_tuples(past_key_values, clone=False):
        flat.append(k)
        flat.append(v)
    return tuple(flat)


def _hf_sync_static_cache_layer_after_kv_rebind(
    layer: Any,
    *,
    cache_position_start: int | torch.Tensor | None,
    onnx_export_trace: bool = False,
) -> None:
    """
    After replacing ``layer.keys`` / ``layer.values`` on transformers v5 static layers,
    align sequence bookkeeping with the same convention as ``Cache.update`` would maintain.

    ``cache_position_start`` is the global index of the first position in the *current*
    transformer step (i.e. total tokens already processed before this step). Pass a
    0-dim ``torch.Tensor`` (e.g. ``cache_position.reshape(-1)[0]``) so ``torch.onnx.export``
    can connect ``cache_position`` input to ``cumulative_length`` without constant-folding
    ``int(tensor.item())`` to 0. When unknown, use ``None`` and derive a conservative value from buffer width.

    ``onnx_export_trace``: when True (ONNX export wrappers only), skip the final ``cumulative_length`` ``fill_``
    so the tensor keeps the value copied from ``cache_position_start`` (graph edge for export).
    """
    if not hasattr(layer, "cumulative_length_int"):
        return
    max_len = int(getattr(layer, "max_cache_len", 0) or 0)
    if cache_position_start is not None:
        if torch.is_tensor(cache_position_start):
            t0 = cache_position_start.reshape(-1)[0]
            cl = getattr(layer, "cumulative_length", None)
            if isinstance(cl, torch.Tensor) and cl.numel() == 1:
                cl.copy_(t0.to(dtype=cl.dtype))
            # Keep int and tensor bookkeeping aligned. Forcing 0 here (export trace) while
            # ``cumulative_length`` follows ``cache_position`` makes ``StaticSlidingWindowLayer``
            # take the wrong branch vs. index_copy indices and breaks ORT at the sliding edge.
            layer.cumulative_length_int = int(t0.detach().cpu().item())
        else:
            layer.cumulative_length_int = int(cache_position_start)
    else:
        # No global positions: assume a single-chunk alignment consistent with probe/export.
        keys = getattr(layer, "keys", None)
        if keys is None or not torch.is_tensor(keys):
            return
        w = int(keys.shape[-2])
        layer.cumulative_length_int = min(w, max_len) if max_len > 0 else w
    cl = getattr(layer, "cumulative_length", None)
    if isinstance(cl, torch.Tensor) and cl.numel() == 1:
        if (
            onnx_export_trace
            and cache_position_start is not None
            and torch.is_tensor(cache_position_start)
        ):
            pass
        else:
            cl.fill_(int(layer.cumulative_length_int))


def _hf_install_past_kv_layer(
    cache: Any,
    layer_idx: int,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cache_position_start: int | torch.Tensor | None = None,
    onnx_export_trace: bool = False,
) -> None:
    """Rebind one layer's KV buffers on v5 ``StaticCache`` (``layers``) for ONNX export replay."""
    layers = getattr(cache, "layers", None)
    if layers is not None:
        layer = layers[layer_idx]
        layer.keys = k
        layer.values = v
        layer.is_initialized = True
        # ``lazy_initialization`` is skipped when rebinding exported buffers; ``update()`` still needs ``device``.
        if getattr(layer, "device", None) is None and torch.is_tensor(k):
            layer.dtype = k.dtype
            layer.device = k.device
            cl = getattr(layer, "cumulative_length", None)
            if torch.is_tensor(cl):
                layer.cumulative_length = cl.to(device=k.device)
        _hf_sync_static_cache_layer_after_kv_rebind(
            layer,
            cache_position_start=cache_position_start,
            onnx_export_trace=onnx_export_trace,
        )
        return

    raise TypeError(f"Cannot install KV into cache type {type(cache)}")


@dataclass
class _StreamingResamplerState:
    next_t_in_samples: float = 0.0
    in_total_samples: int = 0
    out_total_samples: int = 0
    prev: Optional[torch.Tensor] = None


class _CausalStreamingResampler(nn.Module):
    def __init__(self, orig_freq: float, new_freq: float):
        super().__init__()
        self.orig_freq = float(orig_freq)
        self.new_freq = float(new_freq)
        self._lock = threading.Lock()
        self.reset()

    @property
    def step_in(self) -> float:
        return self.orig_freq / self.new_freq

    def reset(self) -> None:
        with self._lock:
            self.state = _StreamingResamplerState()

    def _empty_like_prev(self) -> torch.Tensor:
        if self.state.prev is None:
            return torch.empty((0, 0, 0), dtype=torch.float32)
        return self.state.prev.unsqueeze(-1)[:, :, :0]

    def process(self, x: torch.Tensor) -> torch.Tensor:
        with self._lock:
            if x.ndim != 3:
                raise ValueError(f"Expected x as [B, C, T], got {tuple(x.shape)}")

            batch_size, channels, frames = x.shape
            if frames == 0:
                return x.new_zeros((batch_size, channels, 0))

            if (
                self.state.prev is None
                or self.state.prev.ndim != 2
                or self.state.prev.shape[0] != batch_size
                or self.state.prev.shape[1] != channels
                or self.state.prev.device != x.device
                or self.state.prev.dtype != x.dtype
            ):
                self.state.prev = x[:, :, 0].detach()
                self.state.next_t_in_samples = float(self.state.in_total_samples)

            in_start = self.state.in_total_samples
            in_end = in_start + frames - 1

            if self.state.next_t_in_samples > in_end:
                self.state.in_total_samples += int(frames)
                self.state.prev = x[:, :, -1].detach()
                return x.new_zeros((batch_size, channels, 0))

            step = self.step_in
            n_out = int(np.floor((in_end - self.state.next_t_in_samples) / step) + 1)
            if n_out <= 0:
                self.state.in_total_samples += int(frames)
                self.state.prev = x[:, :, -1].detach()
                return x.new_zeros((batch_size, channels, 0))

            t = (
                torch.arange(n_out, device=x.device, dtype=torch.float32) * float(step)
                + float(self.state.next_t_in_samples)
            )
            t_floor = torch.floor(t)
            frac = (t - t_floor).to(dtype=x.dtype)
            t_floor = t_floor.to(dtype=torch.long)

            local_i0 = t_floor - int(in_start)
            idx0 = (local_i0 + 1).clamp(min=0, max=frames).to(dtype=torch.long)
            idx1 = (idx0 + 1).clamp(max=frames).to(dtype=torch.long)

            x_ext = torch.cat([self.state.prev.unsqueeze(-1), x], dim=-1)
            gather0 = idx0.view(1, 1, -1).expand(batch_size, channels, -1)
            gather1 = idx1.view(1, 1, -1).expand(batch_size, channels, -1)
            x0 = torch.gather(x_ext, dim=-1, index=gather0)
            x1 = torch.gather(x_ext, dim=-1, index=gather1)

            frac_bc = frac.view(1, 1, -1)
            y = (1.0 - frac_bc) * x0 + frac_bc * x1

            self.state.next_t_in_samples = float(self.state.next_t_in_samples + n_out * step)
            self.state.in_total_samples += int(frames)
            self.state.out_total_samples += int(n_out)
            self.state.prev = x[:, :, -1].detach()
            return y

    def flush(self) -> torch.Tensor:
        with self._lock:
            if self.state.prev is None:
                return self._empty_like_prev()

            out_total_target = int(
                round(self.state.in_total_samples * (self.new_freq / self.orig_freq))
            )
            n_out = out_total_target - self.state.out_total_samples
            if n_out <= 0:
                return self._empty_like_prev()

            y = self.state.prev.unsqueeze(-1).expand(-1, -1, n_out).contiguous()
            self.state.next_t_in_samples = float(
                self.state.next_t_in_samples + n_out * self.step_in
            )
            self.state.out_total_samples = int(out_total_target)
            return y

def mimi_sliding_cache_len(frame_hz_mimi: float) -> int:
    """KV length for Mimi encoder_transformer: ~20s at Mimi frame rate, VAP-aligned (-1 frame)."""
    return max(1, int(round(20.0 * float(frame_hz_mimi))) - 1)


def mimi_encoder_static_cache_max_len(frame_hz_mimi: float) -> int:
    """
    ``StaticCache(..., max_cache_len=...)`` for Mimi encoder streaming.

    Must be **strictly greater** than :func:`mimi_sliding_cache_len` whenever a forward can pass
    ``cache_position`` with length ``T>1``: ``StaticSlidingWindowLayer.update`` uses
    ``arange(T) + cumulative_length`` for ``index_copy_`` into ``[B,H,L,D]`` with
    ``L == min(config.sliding_window, max_cache_len)``. With ``L == mimi_sliding_cache_len`` and
    ``T == 2``, the pair ``(L-2, L-1)`` is valid but ``(L-1, L)`` is not; one extra slot removes
    ORT ``ScatterElements`` out-of-bounds at the sliding boundary without changing the exported
    graph structure (still no Torch fallback at inference).

    For kyutai/mimi (``sliding_window`` 250), ``mimi_sliding_cache_len`` is 249 and this returns 250,
    matching the config sliding cap.
    """
    return mimi_sliding_cache_len(frame_hz_mimi) + 1


def build_mimi_hf_cache_from_flat_past(
    encoder: "EncoderMimi",
    past_group_sizes: list[int],
    state_tensors: tuple[torch.Tensor, ...],
    ref: torch.Tensor,
    *,
    cache_position_start: int | torch.Tensor | None,
    onnx_export_trace: bool = False,
) -> Any:
    """
    Rebuild a HF KV cache from flattened per-layer K/V tensors.

    Matches :class:`MimiStreamingOnnxWrapperV3CachePos` / ORT streaming so PyTorch v5 streaming
    stays numerically aligned with ONNX.

    Set ``onnx_export_trace=True`` only from ONNX export wrappers (``encoder_export``), not from
    runtime :class:`EncoderMimi`.
    """
    device = ref.device
    dtype = ref.dtype
    cache_len = mimi_encoder_static_cache_max_len(encoder.frame_hz_mimi)
    from transformers.cache_utils import StaticCache

    cache = StaticCache(config=encoder.model.config, max_cache_len=cache_len)
    idx = 0
    for layer_idx, group_size in enumerate(past_group_sizes):
        if group_size != 2:
            raise ValueError(f"Expected K/V pair per layer, got group_size={group_size}")
        k = state_tensors[idx].to(device=device, dtype=dtype)
        v = state_tensors[idx + 1].to(device=device, dtype=dtype)
        idx += 2
        _hf_install_past_kv_layer(
            cache,
            layer_idx,
            k,
            v,
            cache_position_start=cache_position_start,
            onnx_export_trace=onnx_export_trace,
        )
    return cache


class EncoderCPC(nn.Module):
    """CPC-based Audio Encoder.
    
    Transforms raw waveform audio into continuous feature representations (h).
    """

    def __init__(self, load_pretrained=True, freeze=True, cpc_model=''):
        """Initialize the CPC Encoder.
        
        Args:
            load_pretrained (bool): Whether to load pre-trained CPC weights.
            freeze (bool): Whether to freeze the encoder weights during training.
            cpc_model (str): Path or identifier for the CPC model checkpoint.
        """
        
        super().__init__()
        
        self.sample_rate = 16000
        
        if load_pretrained:
            self.encoder = load_CPC(checkpoint_cpc=cpc_model, load_state_dict=True)
        else:
            self.encoder = load_CPC(checkpoint_cpc='', load_state_dict=False)
        
        # Keep Hidden layer
        self.encoder.gAR.keepHidden = True
        
        self.output_dim = self.encoder.gEncoder.conv4.out_channels
        self.dim = self.output_dim

        self.downsample_ratio = 160
        self.downsample = get_cnn_layer(
            dim=self.output_dim,
            kernel=[5],
            stride=[2],
            dilation=[1],
            activation="GELU",
        )
        self.downsample_ratio = 320

        if freeze:
            self.freeze()

    def get_default_conf(self):
        return {""}

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def unfreeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        print(f"Trainable {self.__class__.__name__}!")

    def reset_streaming_state(self):
        if hasattr(self.encoder, "gAR") and hasattr(self.encoder.gAR, "hidden"):
            self.encoder.gAR.hidden = None

    def forward_shared(self, waveform):
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)  # channel dim
        z = self.encoder.gEncoder(waveform)
        z = einops.rearrange(z, "b c n -> b n c")
        z = z[:, 1:-1, :]
        z = self.encoder.gAR(z)
        return z

    def forward_specific(self, z):
        return self.downsample(z)

    def forward(self, waveform):
        z = self.forward_shared(waveform)
        return self.forward_specific(z)

    def hash_tensor(self, tensor):
        return hash(tuple(tensor.reshape(-1).tolist()))


class EncoderMimi(nn.Module):
    """Mimi-based Audio Encoder.
    
    Uses Kyutai's Mimi model to extract continuous features from audio.
    Supports streaming and context tracking.
    """
    def __init__(
        self,
        frame_hz: float = 10,
        freeze: bool = True,
        mimi_model_name: str = "kyutai/mimi",
        context_samples: int = 320,
        load_hf_weights: bool = True,
        output_dim: int = 512,
        frame_hz_mimi: float | None = None,
    ):
        """Initialize the Mimi Encoder.
        
        Args:
            frame_hz (float): Target frame rate of the output features.
            freeze (bool): Whether to freeze the encoder weights.
            mimi_model_name (str): Identifier of the Mimi model.
            context_samples (int): Number of overlap context samples.
            load_hf_weights (bool): If False, skip HF ``from_pretrained`` (ONNX runtime).
            output_dim (int): Feature dim when ``load_hf_weights`` is False.
            frame_hz_mimi (float | None): Native Mimi frame rate when skipping HF.
        """
        super().__init__()

        try:
            import torchaudio
        except (ModuleNotFoundError, OSError):
            torchaudio = None

        self._torchaudio = torchaudio
        self.sample_rate = 24000
        self.frame_hz = float(frame_hz)
        self.context_samples = int(context_samples)
        self._audio_resampler = _CausalStreamingResampler(
            orig_freq=16000.0,
            new_freq=float(self.sample_rate),
        )
        self._feature_resampler = None
        self._mimi_padding_cache = None
        self._mimi_transformer_position_next = 0
        self._frame_rate_conv_cache = None
        self._mimi_did_first_24k_leading_zeros = False
        # transformers v5+: same flat K/V contract as ONNX (rebind StaticCache each step).
        self._mimi_streaming_flat_past: Optional[tuple[torch.Tensor, ...]] = None
        self._mimi_streaming_past_group_sizes: Optional[list[int]] = None
        self._mimi_zero_flat_past: Optional[tuple[torch.Tensor, ...]] = None
        self._mimi_onnx_pad_seeded: bool = False
        self.mimi_model_name = str(mimi_model_name)

        if load_hf_weights:
            try:
                from transformers import MimiConfig, MimiModel
                from transformers.models.mimi.modeling_mimi import MimiConv1dPaddingCache
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Mimi encoder requires transformers with Mimi support."
                ) from exc

            self._MimiConv1dPaddingCache = MimiConv1dPaddingCache
            config = MimiConfig.from_pretrained(mimi_model_name)
            config.use_causal_conv = True
            self.model = MimiModel.from_pretrained(mimi_model_name, config=config)
            self.model.eval()

            self.frame_hz_mimi = float(
                getattr(getattr(self.model, "quantizer", None), "frame_rate", 12.5)
            )
            self.output_dim = 512
            if hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
                self.output_dim = self.model.config.hidden_size
            elif hasattr(self.model, "config") and hasattr(self.model.config, "dimension"):
                self.output_dim = self.model.config.dimension
            self._fix_mimi_padding_buffers()
        else:
            # ONNX runtime path: no HF weights; shapes come from ONNX meta.
            self._MimiConv1dPaddingCache = None
            self.model = None
            self.frame_hz_mimi = float(
                frame_hz_mimi if frame_hz_mimi is not None else frame_hz
            )
            self.output_dim = int(output_dim)

        self.dim = self.output_dim
        self.downsample_ratio = int(round(16000 / self.frame_hz)) if self.frame_hz else 0

        self.frame_rate_conv = CConv1d(
            in_channels=self.output_dim,
            out_channels=self.output_dim,
            kernel_size=3,
            padding=0,
            bias=True,
        )

        if freeze:
            self.freeze()

    def freeze(self):
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad_(False)
        for param in self.frame_rate_conv.parameters():
            param.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def unfreeze(self):
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad_(True)
        for param in self.frame_rate_conv.parameters():
            param.requires_grad_(True)
        print(f"Trainable {self.__class__.__name__}!")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.model is not None:
            self.model.eval()
        return self

    def reset_streaming_state(self):
        self._audio_resampler.reset()
        if self._feature_resampler is not None:
            self._feature_resampler.reset()
        self._mimi_padding_cache = None
        self._mimi_transformer_position_next = 0
        self._frame_rate_conv_cache = None
        self._mimi_did_first_24k_leading_zeros = False
        self._mimi_streaming_flat_past = None
        self._mimi_streaming_past_group_sizes = None
        self._mimi_zero_flat_past = None
        self._mimi_onnx_pad_seeded = False

    def _fix_mimi_padding_buffers(self) -> None:
        for module in self.model.modules():
            if hasattr(module, "padding_total"):
                padding_total = getattr(module, "padding_total")
                if torch.is_tensor(padding_total) and padding_total.dtype != torch.int64:
                    setattr(module, "padding_total", padding_total.to(dtype=torch.int64))

            if hasattr(module, "_pad1d") and not hasattr(module, "_pad1d_wrapped"):
                original_pad1d = module._pad1d

                def _pad1d_int(hidden_states, paddings, mode="constant", value=0.0, _orig=original_pad1d):
                    def _to_int(v):
                        if torch.is_tensor(v):
                            return int(v.item())
                        return int(v)

                    if isinstance(paddings, (tuple, list)):
                        paddings = tuple(_to_int(v) for v in paddings)
                    else:
                        paddings = _to_int(paddings)
                    return _orig(hidden_states, paddings, mode=mode, value=value)

                module._pad1d = _pad1d_int
                module._pad1d_wrapped = True

    def _ensure_mimi_padding_cache(self):
        if self._mimi_padding_cache is not None:
            return self._mimi_padding_cache

        per_layer_padding = []
        per_layer_padding_mode = []
        per_layer_in_channels = []

        for layer_name in self.model.encoder._mimiconv1d_layer_names:
            layer = self.model.encoder.get_submodule(layer_name)
            per_layer_padding.append(int(layer.padding_total))
            per_layer_padding_mode.append(layer.pad_mode)
            per_layer_in_channels.append(layer.in_channels)

        if self.model.downsample is not None:
            per_layer_padding.append(int(self.model.downsample.padding_total))
            per_layer_padding_mode.append(self.model.downsample.pad_mode)
            per_layer_in_channels.append(self.model.downsample.in_channels)

        self._mimi_padding_cache = self._MimiConv1dPaddingCache(
            num_layers=len(per_layer_padding),
            per_layer_padding=per_layer_padding,
            per_layer_padding_mode=per_layer_padding_mode,
            per_layer_in_channels=per_layer_in_channels,
        )
        return self._mimi_padding_cache

    def get_streaming_emit_samples_16k(self) -> int:
        """New 16k samples processed in one streaming call."""
        if self.frame_hz <= 0:
            raise ValueError("frame_hz must be > 0")
        return int(round(16000.0 / float(self.frame_hz)))

    def get_streaming_call_window_16k(self) -> int:
        """Current PyTorch call window size (context + new samples) at 16k."""
        return int(self.context_samples + self.get_streaming_emit_samples_16k())

    def get_streaming_mimi_input_24k(self) -> int:
        """Mimi core input size at 24k after stripping overlap context."""
        emit_16k = self.get_streaming_emit_samples_16k()
        return int(round(float(emit_16k) * float(self.sample_rate) / 16000.0))

    def _probe_mimi_cache_templates(
        self,
        batch_size: int = 1,
        num_samples_24k: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[list[torch.Tensor], tuple]:
        """
        Run one Mimi core step and capture cache tensor templates.
        This is used to build fixed ONNX IO signatures for streaming cache IO.
        """
        if num_samples_24k is None:
            num_samples_24k = self.get_streaming_mimi_input_24k()
        if device is None:
            device = next(self.model.parameters()).device

        self._fix_mimi_padding_buffers()
        ref_cache = self._ensure_mimi_padding_cache()
        n_layers = len(ref_cache.per_layer_padding)
        padding_cache = copy.deepcopy(ref_cache)
        padding_cache.padding_cache = [None for _ in range(n_layers)]
        if hasattr(padding_cache, "per_layer_is_init"):
            padding_cache.per_layer_is_init = [False for _ in range(n_layers)]

        x = torch.zeros(
            (batch_size, 1, int(num_samples_24k)),
            device=device,
            dtype=dtype,
        )

        with torch.inference_mode():
            emb = self.model.encoder(x, padding_cache=padding_cache)
            x_seq = emb.transpose(1, 2)
            t_len = int(x_seq.shape[1])
            cache_position = torch.arange(0, t_len, device=device, dtype=torch.long)
            from transformers.cache_utils import StaticCache

            sw = StaticCache(
                config=self.model.config,
                max_cache_len=mimi_encoder_static_cache_max_len(self.frame_hz_mimi),
            )
            position_ids = cache_position.unsqueeze(0).expand(int(batch_size), -1)
            enc_out = self.model.encoder_transformer(
                x_seq,
                past_key_values=sw,
                use_cache=True,
                return_dict=True,
                position_ids=position_ids,
            )
            hidden = enc_out.last_hidden_state.transpose(1, 2)
            if self.model.downsample is not None:
                _ = self.model.downsample(hidden, padding_cache=padding_cache)

        pad_templates: list[torch.Tensor] = []
        for t in padding_cache.padding_cache:
            if t is None:
                pad_templates.append(torch.zeros((batch_size, 1, 0), device=device, dtype=dtype))
            else:
                pad_templates.append(t.detach().clone())

        pkv = enc_out.past_key_values
        past_kv_layers = _hf_past_kv_per_layer_tuples(pkv, clone=True)
        return pad_templates, past_kv_layers

    def get_mimi_streaming_onnx_io_spec(
        self,
        batch_size: int = 1,
        num_samples_24k: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Returns a concrete IO specification for Mimi-core ONNX streaming export.
        """
        dev = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        pad_templates, past = self._probe_mimi_cache_templates(
            batch_size=batch_size,
            num_samples_24k=num_samples_24k,
            device=dev,
            dtype=dtype,
        )

        past_shapes: list[tuple[int, ...]] = []
        for layer in past:
            for t in layer:
                past_shapes.append(tuple(int(v) for v in t.shape))

        return {
            "wave_16k_call_window": self.get_streaming_call_window_16k(),
            "wave_24k_mimi_input": int(
                self.get_streaming_mimi_input_24k()
                if num_samples_24k is None
                else num_samples_24k
            ),
            "padding_cache_shapes": [tuple(int(v) for v in t.shape) for t in pad_templates],
            "past_key_value_shapes": past_shapes,
            "num_padding_tensors": len(pad_templates),
            "num_past_tensors": len(past_shapes),
            "output_dim": int(self.output_dim),
        }

    def _seed_mimi_padding_cache_like_onnx(self, ref: torch.Tensor) -> None:
        """
        Match ``EncoderMimiOnnx`` / ORT init: zero-filled ``padding_cache`` slots with ONNX export shapes.

        A fresh :class:`MimiConv1dPaddingCache` starts with empty buffers; ORT uses explicit zeros of the
        contract shapes, which can diverge on the first streaming step.
        """
        if self._mimi_onnx_pad_seeded:
            return
        spec = self.get_mimi_streaming_onnx_io_spec(batch_size=int(ref.shape[0]))
        pc = self._ensure_mimi_padding_cache()
        for i, shp in enumerate(spec["padding_cache_shapes"]):
            pc.padding_cache[i] = ref.new_zeros(shp)
        if hasattr(pc, "per_layer_is_init"):
            pc.per_layer_is_init = [True] * len(spec["padding_cache_shapes"])
        self._mimi_onnx_pad_seeded = True

    def _ensure_mimi_streaming_flat_templates(self, ref: torch.Tensor) -> None:
        """Lazy-init zero flat K/V tensors (v5 streaming, ONNX-aligned)."""
        if self._mimi_streaming_past_group_sizes is not None:
            return
        _pad, past_templates = self._probe_mimi_cache_templates(
            batch_size=int(ref.shape[0]),
            num_samples_24k=self.get_streaming_mimi_input_24k(),
            device=ref.device,
            dtype=ref.dtype,
        )
        flat: list[torch.Tensor] = []
        for layer in past_templates:
            for t in layer:
                flat.append(torch.zeros(tuple(t.shape), device=ref.device, dtype=ref.dtype))
        self._mimi_streaming_past_group_sizes = [len(layer) for layer in past_templates]
        self._mimi_zero_flat_past = tuple(flat)

    def _encode_continuous_embeddings(self, x: torch.Tensor, streaming: bool) -> torch.Tensor:
        padding_cache = None
        if streaming:
            padding_cache = self._ensure_mimi_padding_cache()
            self._seed_mimi_padding_cache_like_onnx(x)

        embeddings = self.model.encoder(x, padding_cache=padding_cache)
        xseq = embeddings.transpose(1, 2)

        if streaming:
            t = int(xseq.shape[1])
            cache_position = torch.arange(
                self._mimi_transformer_position_next,
                self._mimi_transformer_position_next + t,
                device=xseq.device,
                dtype=torch.long,
            )
            self._ensure_mimi_streaming_flat_templates(x)
            past_in = self._mimi_streaming_flat_past
            if past_in is None:
                past_in = self._mimi_zero_flat_past
            assert past_in is not None and self._mimi_streaming_past_group_sizes is not None
            past_key_values = build_mimi_hf_cache_from_flat_past(
                self,
                self._mimi_streaming_past_group_sizes,
                past_in,
                x,
                cache_position_start=cache_position.reshape(-1)[0],
            )
            position_ids = cache_position.unsqueeze(0).expand(int(xseq.shape[0]), -1)
            encoder_outputs = self.model.encoder_transformer(
                xseq,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                position_ids=position_ids,
            )
            self._mimi_streaming_flat_past = tuple(_hf_flatten_past_kv_tensors(encoder_outputs.past_key_values))
            self._mimi_transformer_position_next = int(encoder_outputs.past_key_values.get_seq_length())
        else:
            encoder_outputs = self.model.encoder_transformer(
                xseq,
                past_key_values=None,
                use_cache=False,
                return_dict=True,
            )

        embeddings = encoder_outputs.last_hidden_state.transpose(1, 2)
        if self.model.downsample is not None:
            embeddings = self.model.downsample(embeddings, padding_cache=padding_cache)

        return embeddings.transpose(1, 2)

    def _resample_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        if self._torchaudio is not None:
            try:
                return self._torchaudio.functional.resample(
                    waveform,
                    orig_freq=16000,
                    new_freq=self.sample_rate,
                )
            except RuntimeError:
                waveform_cpu = waveform.detach().to("cpu") if waveform.device.type != "cpu" else waveform
                waveform_cpu = self._torchaudio.functional.resample(
                    waveform_cpu,
                    orig_freq=16000,
                    new_freq=self.sample_rate,
                )
                return waveform_cpu.to(waveform.device)

        target_len = max(1, int(round(waveform.shape[-1] * self.sample_rate / 16000.0)))
        return F.interpolate(
            waveform,
            size=target_len,
            mode="linear",
            align_corners=False,
        )

    def _resample_audio_streaming(
        self,
        waveform: torch.Tensor,
        finalize_stream: bool,
    ) -> torch.Tensor:
        waveform = self._audio_resampler.process(waveform)

        if finalize_stream:
            waveform_flush = self._audio_resampler.flush()
            if waveform_flush.shape[-1] > 0:
                waveform = torch.cat(
                    [waveform, waveform_flush.to(dtype=waveform.dtype, device=waveform.device)],
                    dim=-1,
                )

        return waveform

    def _apply_frame_rate_conv_streaming(self, emb_t: torch.Tensor) -> torch.Tensor:
        if emb_t.shape[-1] == 0:
            return emb_t

        kernel_size = self.frame_rate_conv.kernel_size[0]
        cache_size = kernel_size - 1

        if self._frame_rate_conv_cache is None:
            self._frame_rate_conv_cache = emb_t.new_zeros(
                emb_t.shape[0],
                emb_t.shape[1],
                cache_size,
            )
        else:
            self._frame_rate_conv_cache = self._frame_rate_conv_cache.to(
                device=emb_t.device,
                dtype=emb_t.dtype,
            )

        conv_input = torch.cat([self._frame_rate_conv_cache, emb_t], dim=-1)
        out = F.conv1d(
            conv_input,
            self.frame_rate_conv.weight,
            bias=self.frame_rate_conv.bias,
            stride=self.frame_rate_conv.stride,
            padding=0,
            dilation=self.frame_rate_conv.dilation,
            groups=self.frame_rate_conv.groups,
        )
        self._frame_rate_conv_cache = conv_input[..., -cache_size:].detach()
        return out

    def _align_frame_rate(
        self,
        embeddings: torch.Tensor,
        input_num_samples: int,
        streaming: bool,
        finalize_stream: bool,
    ) -> torch.Tensor:
        if embeddings.shape[1] == 0:
            return embeddings

        if self.frame_hz is None or self.frame_hz <= 0:
            return embeddings

        if math.isclose(float(self.frame_hz), float(self.frame_hz_mimi), rel_tol=0.0, abs_tol=1e-6):
            return embeddings

        emb_t = embeddings.transpose(1, 2)

        if streaming:
            if (
                self._feature_resampler is None
                or self._feature_resampler.orig_freq != float(self.frame_hz_mimi)
                or self._feature_resampler.new_freq != float(self.frame_hz)
            ):
                self._feature_resampler = _CausalStreamingResampler(
                    orig_freq=float(self.frame_hz_mimi),
                    new_freq=float(self.frame_hz),
                )

            emb_t = self._feature_resampler.process(emb_t)
            if finalize_stream:
                emb_t_flush = self._feature_resampler.flush()
                if emb_t_flush.shape[-1] > 0:
                    emb_t = torch.cat(
                        [emb_t, emb_t_flush.to(dtype=emb_t.dtype, device=emb_t.device)],
                        dim=-1,
                    )

            emb_t = self._apply_frame_rate_conv_streaming(emb_t)
        else:
            target_frames = max(1, int(round(input_num_samples * self.frame_hz / 16000.0)))
            if target_frames != emb_t.shape[-1]:
                emb_t = torch.nn.functional.interpolate(
                    emb_t,
                    size=target_frames,
                    mode="linear",
                    align_corners=False,
                )

            emb_t = self.frame_rate_conv(emb_t)

        return emb_t.transpose(1, 2)

    def _encode_offline(self, waveform: torch.Tensor, resampling: bool = True) -> torch.Tensor:
        input_num_samples = int(waveform.shape[-1])

        if resampling:
            waveform = self._resample_audio(waveform)

        if waveform.shape[-1] == 0:
            return waveform.new_zeros((waveform.shape[0], 0, self.output_dim))

        embeddings = self._encode_continuous_embeddings(waveform, streaming=False)
        return self._align_frame_rate(
            embeddings,
            input_num_samples=input_num_samples,
            streaming=False,
            finalize_stream=True,
        )

    def forward_shared(
        self,
        waveform,
        resampling: bool = True,
        streaming: bool = True,
        finalize_stream: bool = False,
        has_overlap_context: bool = True,
    ):
        if self.model is not None:
            self.model.eval()
            self._fix_mimi_padding_buffers()

        if not torch.is_tensor(waveform):
            raise TypeError(f"Expected waveform to be a torch.Tensor, got {type(waveform)}")

        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)

        waveform = waveform.to(dtype=torch.float32)

        if not streaming:
            if self.model is None:
                raise RuntimeError(
                    "Non-streaming Mimi encode requires HF weights "
                    "(EncoderMimi with load_hf_weights=True)."
                )
            input_num_samples = int(waveform.shape[-1])
            if resampling:
                waveform = self._resample_audio(waveform)
            if waveform.shape[-1] == 0:
                return waveform.new_zeros((waveform.shape[0], 0, self.output_dim)), input_num_samples
            return self._encode_continuous_embeddings(waveform, streaming=False), input_num_samples

        overlap_context = self.context_samples if has_overlap_context else 0
        if overlap_context > 0:
            if waveform.shape[-1] <= overlap_context:
                return waveform.new_zeros((waveform.shape[0], 0, self.output_dim)), int(waveform.shape[-1])
            waveform = waveform[..., overlap_context:]

        input_num_samples = int(waveform.shape[-1])
        if input_num_samples == 0:
            return waveform.new_zeros((waveform.shape[0], 0, self.output_dim)), input_num_samples

        if resampling:
            waveform = self._resample_audio_streaming(
                waveform,
                finalize_stream=finalize_stream,
            )
            # Causal streaming resampler (16 kHz -> 24 kHz): the first chunk is one sample
            # short of Mimi's expected input length (e.g. 1920); prepend a single zero once.
            if (
                not self._mimi_did_first_24k_leading_zeros
                and waveform.shape[-1] > 0
            ):
                pad = waveform.new_zeros(
                    waveform.shape[0],
                    waveform.shape[1],
                    1,
                )
                waveform = torch.cat([pad, waveform], dim=-1)
                self._mimi_did_first_24k_leading_zeros = True

        if waveform.shape[-1] == 0:
            return waveform.new_zeros((waveform.shape[0], 0, self.output_dim)), input_num_samples

        embeddings = self._encode_continuous_embeddings(waveform, streaming=True)
        return embeddings, input_num_samples

    def forward_specific(
        self,
        embeddings: torch.Tensor,
        input_num_samples: int,
        streaming: bool = True,
        finalize_stream: bool = False,
    ):
        if embeddings.shape[1] == 0:
            return embeddings.new_zeros((embeddings.shape[0], 0, self.output_dim))

        return self._align_frame_rate(
            embeddings,
            input_num_samples=input_num_samples,
            streaming=streaming,
            finalize_stream=finalize_stream,
        )

    def forward(
        self,
        waveform,
        resampling: bool = True,
        only_feature_extractor: bool = False,
        streaming: bool = True,
        finalize_stream: bool = False,
        has_overlap_context: bool = True,
    ):
        del only_feature_extractor
        
        embeddings, input_num_samples = self.forward_shared(
            waveform,
            resampling=resampling,
            streaming=streaming,
            finalize_stream=finalize_stream,
            has_overlap_context=has_overlap_context,
        )
        return self.forward_specific(
            embeddings,
            input_num_samples=input_num_samples,
            streaming=streaming,
            finalize_stream=finalize_stream,
        )


class EncoderMimiOnnx(EncoderMimi):
    """Mimi core ONNX backend for optimized inference.
    
    Provides a streaming contract matching `transformers` v5 `StaticCache` with flat K/V I/O.
    Supports CUDA (FP32) and CPU (INT8) execution providers.

    Does not load HF ``MimiModel`` weights. ORT sessions for the same ONNX path are shared
    across encoder instances (e.g. VAP ch1/ch2); per-instance state buffers stay private.
    """

    _ORT_SESSION_CACHE: dict[tuple, Any] = {}

    @staticmethod
    def _onnx_output_shape_for_fixed_bind(shape: list[Any] | tuple[Any, ...]) -> tuple[int, ...]:
        """
        Convert ORT's dynamic dimensions (strings or <= 0) to 1 for fixed IOBinding.
        Streaming Mimi typically uses batch=1 and emb_t=1 per step.
        """
        resolved: list[int] = []
        for d in shape:
            if isinstance(d, int):
                resolved.append(int(d) if d > 0 else 1)
            else:
                resolved.append(1)
        return tuple(resolved)

    def __init__(
        self,
        frame_hz: float = 10,
        freeze: bool = True,
        mimi_model_name: str = "kyutai/mimi",
        context_samples: int = 320,
        onnx_model_path: str = "",
        onnx_meta_path: str = "",
        runtime_device: str = "cpu",
        onnx_cpu_intra_threads: int = 2,
        onnx_cpu_inter_threads: int = 1,
        output_dim: int = 512,
    ):
        """Initialize the ONNX-backed Mimi Encoder.
        
        Args:
            frame_hz (float): Target output frame rate.
            freeze (bool): Whether to freeze weights (mostly for API compatibility here).
            mimi_model_name (str): The identifier of the original model.
            context_samples (int): Overlap context samples.
            onnx_model_path (str): Path to the ONNX model file.
            onnx_meta_path (str): Path to the ONNX metadata JSON file.
            runtime_device (str): Device to run the ONNX model on ('cpu', 'cuda').
            onnx_cpu_intra_threads (int): Number of intra-op threads for CPU execution.
            onnx_cpu_inter_threads (int): Number of inter-op threads for CPU execution.
            output_dim (int): Mimi embedding dim (kyutai/mimi default 512).
        """
        # Skip HF from_pretrained — runtime uses ORT only.
        super().__init__(
            frame_hz=frame_hz,
            freeze=freeze,
            mimi_model_name=mimi_model_name,
            context_samples=context_samples,
            load_hf_weights=False,
            output_dim=int(output_dim),
            frame_hz_mimi=float(frame_hz),
        )

        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "EncoderMimiOnnx requires onnxruntime/onnxruntime-gpu."
            ) from exc

        self._ort = ort
        self._runtime_device = str(runtime_device)
        self._use_cuda = self._runtime_device.startswith("cuda")
        self._onnx_model_path = str(onnx_model_path)
        self._onnx_meta_path = str(onnx_meta_path)
        self._onnx_states: list[np.ndarray] = []
        self._onnx_input_names: list[str] = []
        self._onnx_output_names: list[str] = []
        self._onnx_pad_input_keys: list[str] = []
        self._onnx_past_input_keys: list[str] = []
        self._onnx_has_cache_position_input = False
        self._onnx_has_cache_position_output = False
        self._onnx_cache_position_key = "cache_position"
        self._onnx_cache_position_out_key = "cache_position_out"
        self._onnx_has_position_ids_input = False
        self._onnx_position_ids_key = "position_ids"
        self._onnx_cache_position_start = np.int64(0)
        self._onnx_position_ids_start = np.int64(0)
        self._onnx_cache_position_len = 1
        self._onnx_max_past_len = 1
        self._onnx_cache_position_base = np.zeros((1,), dtype=np.int64)
        self._onnx_cache_position_buffer = np.zeros((1,), dtype=np.int64)
        self._onnx_position_ids_base = np.zeros((1,), dtype=np.int64)
        self._onnx_position_ids_buffer = np.zeros((1,), dtype=np.int64)
        self._onnx_init_past_from_template = True
        self._onnx_num_pad = 0
        self._onnx_wave_shape: tuple[int, ...] = (1, 1, 0)
        self._onnx_inputs: dict[str, np.ndarray] = {}
        self._onnx_cpu_intra_threads = int(onnx_cpu_intra_threads)
        self._onnx_cpu_inter_threads = int(onnx_cpu_inter_threads)
        self._onnx_cuda_pad_slot0: list[Any] = []
        self._onnx_cuda_pad_slot1: list[Any] = []
        self._onnx_cuda_past_slot0: list[Any] = []
        self._onnx_cuda_past_slot1: list[Any] = []
        self._onnx_cuda_kv_ping = False
        self._onnx_cuda_cp_base_t = torch.zeros((1,), dtype=torch.int64, device="cpu")
        self._onnx_cuda_cp_buffer_t = torch.zeros((1,), dtype=torch.int64, device="cpu")
        self._onnx_cuda_pos_base_t = torch.zeros((1,), dtype=torch.int64, device="cpu")
        self._onnx_cuda_pos_buffer_t = torch.zeros((1,), dtype=torch.int64, device="cpu")
        self._onnx_cuda_wave_buffer_t: Optional[torch.Tensor] = None
        self._onnx_cuda_io = None
        self._onnx_cuda_embeddings_ortvalue: Any = None

        self._onnx_sess = self._create_onnx_session()
        self._init_onnx_states()

    def _create_onnx_session(self):
        so = self._ort.SessionOptions()
        so.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self._use_cuda and "CUDAExecutionProvider" in self._ort.get_available_providers():
            providers: list[Any] = [
                (
                    "CUDAExecutionProvider",
                    {
                        "do_copy_in_default_stream": "1",
                        "cudnn_conv_use_max_workspace": "1",
                    },
                ),
                "CPUExecutionProvider",
            ]
            intra = 0
            inter = 0
        else:
            # Realtime-friendly default for CPU EP.
            so.intra_op_num_threads = max(1, self._onnx_cpu_intra_threads)
            so.inter_op_num_threads = max(1, self._onnx_cpu_inter_threads)
            providers = ["CPUExecutionProvider"]
            intra = int(so.intra_op_num_threads)
            inter = int(so.inter_op_num_threads)

        cache_key = (
            os.path.abspath(self._onnx_model_path),
            self._use_cuda,
            intra,
            inter,
            tuple(p[0] if isinstance(p, tuple) else p for p in providers),
        )
        cached = EncoderMimiOnnx._ORT_SESSION_CACHE.get(cache_key)
        if cached is not None:
            return cached
        sess = self._ort.InferenceSession(
            self._onnx_model_path, sess_options=so, providers=providers
        )
        EncoderMimiOnnx._ORT_SESSION_CACHE[cache_key] = sess
        return sess

    def _load_meta(self) -> dict[str, Any]:
        p = Path(self._onnx_meta_path)
        if not p.exists():
            raise FileNotFoundError(f"ONNX meta file not found: {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def _padding_shapes_from_meta(meta: dict[str, Any]) -> list[tuple[int, ...]]:
        """Prefer export contract pad shapes; avoid HF probe at runtime."""
        contract = meta.get("contract") if isinstance(meta.get("contract"), dict) else {}
        spec = contract.get("spec") if isinstance(contract.get("spec"), dict) else {}
        shapes = spec.get("padding_cache_shapes")
        if isinstance(shapes, list) and shapes:
            return [tuple(int(x) for x in s) for s in shapes]
        raise RuntimeError(
            "ONNX meta missing contract.spec.padding_cache_shapes; "
            "re-export Mimi ONNX with export_mimi_streaming_onnx_v2.py."
        )

    def _init_onnx_states(self):
        meta = self._load_meta()
        required_params = meta.get("required_params", None)
        expected_required = {
            "frame_rate": 12.5,
            "context_samples": 320,
            "wave_16k_call_window": 1600,
            "wave_24k_mimi_input": 1920,
        }
        if required_params is None:
            raise RuntimeError(
                "ONNX meta missing required_params. "
                "Legacy meta format is no longer supported."
            )
        for k, v in expected_required.items():
            got = required_params.get(k, None)
            if got != v:
                raise RuntimeError(
                    f"ONNX meta required_params mismatch for {k}: got={got}, expected={v}"
                )

        self._onnx_input_names = list(meta["input_names"])
        self._onnx_output_names = list(meta.get("output_names", []))
        self._onnx_has_cache_position_input = self._onnx_cache_position_key in self._onnx_input_names
        self._onnx_has_cache_position_output = self._onnx_cache_position_out_key in self._onnx_output_names
        self._onnx_has_position_ids_input = self._onnx_position_ids_key in self._onnx_input_names
        self._onnx_cache_position_start = np.int64(0)
        self._onnx_position_ids_start = np.int64(0)
        self._onnx_cache_position_len = int(meta.get("cache_position_len", 1))
        self._onnx_max_past_len = max(1, int(meta.get("max_past_len", 1)))
        cp_len = max(1, int(self._onnx_cache_position_len))
        self._onnx_cache_position_base = np.arange(cp_len, dtype=np.int64)
        self._onnx_cache_position_buffer = np.zeros((cp_len,), dtype=np.int64)
        self._onnx_position_ids_base = np.arange(cp_len, dtype=np.int64)
        self._onnx_position_ids_buffer = np.zeros((cp_len,), dtype=np.int64)
        self._onnx_init_past_from_template = bool(meta.get("onnx_init_past_from_template", True))
        self._onnx_num_pad = int(meta["num_pad_tensors"])
        self._onnx_pad_input_keys = [f"pad_cache_{i}" for i in range(self._onnx_num_pad)]
        n_non_state_inputs = 1
        if self._onnx_has_cache_position_input:
            n_non_state_inputs += 1
        if self._onnx_has_position_ids_input:
            n_non_state_inputs += 1
        n_states_total = len(self._onnx_input_names) - n_non_state_inputs
        n_past = max(0, n_states_total - self._onnx_num_pad)
        self._onnx_past_input_keys = [f"past_{i}" for i in range(n_past)]
        pad_shapes = self._padding_shapes_from_meta(meta)

        wave_len = int(meta["wave_24k_mimi_input"])
        self._onnx_wave_shape = (1, 1, wave_len)
        if "frame_rate" in meta:
            self.frame_hz_mimi = float(meta["frame_rate"])
        dtype = np.float32
        wave = np.zeros(self._onnx_wave_shape, dtype=dtype)

        # ONNX session input metadata: used to derive correct rank & static
        # dims for past KV cache inputs (dynamic seq-len dim set to 0).
        onnx_input_map = {inp.name: inp for inp in self._onnx_sess.get_inputs()}
        missing_inputs = [name for name in self._onnx_input_names if name not in onnx_input_map]
        if missing_inputs:
            raise RuntimeError(
                "ONNX input signature mismatch. Missing inputs in model session: "
                f"{missing_inputs}. meta_input_names={self._onnx_input_names}, "
                f"session_input_names={list(onnx_input_map.keys())}"
            )

        def _zero_from_onnx_input(name: str) -> np.ndarray:
            if name in onnx_input_map:
                shape = []
                for idx, dim in enumerate(onnx_input_map[name].shape):
                    if isinstance(dim, int):
                        shape.append(dim)
                    elif idx == 0:
                        shape.append(1)   # batch dimension must be 1
                    else:
                        shape.append(0)   # dynamic dims (e.g. seq_len) start empty
                return np.zeros(tuple(shape), dtype=dtype)
            return np.zeros((1, 1, 0, 1), dtype=dtype)

        # Padding cache shapes from export meta (no HF probe).
        probe_pad = [np.zeros(tuple(s), dtype=dtype) for s in pad_shapes]
        pad = []
        for i in range(self._onnx_num_pad):
            if i < len(probe_pad):
                pad.append(probe_pad[i])
            else:
                pad.append(_zero_from_onnx_input(self._onnx_pad_input_keys[i]))

        # Past KV cache: always derive from ONNX session input metadata.
        # These start empty (seq_len=0); shapes follow the export contract, not a live probe.
        self._onnx_past_init_time_dims: dict[str, int | None] = {}
        contract = meta.get("contract", {})
        past_template_shapes = contract.get("past_template_shapes", []) if isinstance(contract, dict) else []
        for k in self._onnx_past_input_keys:
            init_t = None
            try:
                idx = int(k.split("_", 1)[1])
            except Exception:
                idx = -1
            if 0 <= idx < len(past_template_shapes):
                shp = past_template_shapes[idx]
                if isinstance(shp, list) and len(shp) >= 3 and isinstance(shp[-2], int):
                    init_t = int(shp[-2])
            self._onnx_past_init_time_dims[k] = init_t
        past = [_zero_from_onnx_input(k) for k in self._onnx_past_input_keys]
        for i, k in enumerate(self._onnx_past_input_keys):
            if past[i].ndim >= 3 and int(past[i].shape[-2]) == 0:
                init_t = self._onnx_past_init_time_dims.get(k, None)
                if self._onnx_init_past_from_template and isinstance(init_t, int) and init_t > 0:
                    new_shape = list(past[i].shape)
                    new_shape[-2] = init_t
                    past[i] = np.zeros(tuple(new_shape), dtype=past[i].dtype)
        self._onnx_states = [*pad, *past]

        input_template = {"wave_24k": wave}
        if self._onnx_has_cache_position_input:
            np.add(self._onnx_cache_position_base, int(self._onnx_cache_position_start), out=self._onnx_cache_position_buffer, casting="unsafe")
            np.mod(self._onnx_cache_position_buffer, int(self._onnx_max_past_len), out=self._onnx_cache_position_buffer)
            input_template[self._onnx_cache_position_key] = self._onnx_cache_position_buffer
        if self._onnx_has_position_ids_input:
            np.add(
                self._onnx_position_ids_base,
                int(self._onnx_position_ids_start),
                out=self._onnx_position_ids_buffer,
                casting="unsafe",
            )
            input_template[self._onnx_position_ids_key] = self._onnx_position_ids_buffer
        for i, s in enumerate(self._onnx_states):
            if i < self._onnx_num_pad:
                input_template[self._onnx_pad_input_keys[i]] = s
            else:
                input_template[self._onnx_past_input_keys[i - self._onnx_num_pad]] = s
        self._onnx_inputs = dict(input_template)
        if self._use_cuda:
            pad_np = self._onnx_states[: self._onnx_num_pad]
            past_np = self._onnx_states[self._onnx_num_pad :]
            # Ping-pong GPU buffers: input and output are never the same OrtValue (matches
            # separate-tensor semantics / avoids in-place races). No CPU KV round-trip.
            self._onnx_cuda_pad_slot0 = [
                self._ort.OrtValue.ortvalue_from_numpy(s, "cuda", 0) for s in pad_np
            ]
            self._onnx_cuda_pad_slot1 = [
                self._ort.OrtValue.ortvalue_from_numpy(np.copy(s), "cuda", 0) for s in pad_np
            ]
            self._onnx_cuda_past_slot0 = [
                self._ort.OrtValue.ortvalue_from_numpy(s, "cuda", 0) for s in past_np
            ]
            self._onnx_cuda_past_slot1 = [
                self._ort.OrtValue.ortvalue_from_numpy(np.copy(s), "cuda", 0) for s in past_np
            ]
            self._onnx_cuda_kv_ping = False
            cp_len = max(1, int(self._onnx_cache_position_len))
            self._onnx_cuda_cp_base_t = torch.arange(cp_len, dtype=torch.int64, device="cuda")
            self._onnx_cuda_cp_buffer_t = torch.zeros((cp_len,), dtype=torch.int64, device="cuda")
            self._onnx_cuda_pos_base_t = torch.arange(cp_len, dtype=torch.int64, device="cuda")
            self._onnx_cuda_pos_buffer_t = torch.zeros((cp_len,), dtype=torch.int64, device="cuda")
            self._onnx_cuda_wave_buffer_t = torch.empty(
                self._onnx_wave_shape,
                dtype=torch.float32,
                device="cuda",
            )
            self._onnx_cuda_io = self._onnx_sess.io_binding()
            emb_name0 = self._onnx_output_names[0] if self._onnx_output_names else None
            if emb_name0:
                emb_meta = next(
                    (o for o in self._onnx_sess.get_outputs() if o.name == emb_name0),
                    None,
                )
                if emb_meta is not None and emb_meta.type == "tensor(float)":
                    emb_shape = self._onnx_output_shape_for_fixed_bind(emb_meta.shape)
                    self._onnx_cuda_embeddings_ortvalue = self._ort.OrtValue.ortvalue_from_shape_and_type(
                        list(emb_shape),
                        np.float32,
                        "cuda",
                        0,
                    )
                else:
                    self._onnx_cuda_embeddings_ortvalue = None
            else:
                self._onnx_cuda_embeddings_ortvalue = None
        else:
            self._onnx_cuda_pad_slot0 = []
            self._onnx_cuda_pad_slot1 = []
            self._onnx_cuda_past_slot0 = []
            self._onnx_cuda_past_slot1 = []
            self._onnx_cuda_kv_ping = False
            self._onnx_cuda_wave_buffer_t = None
            self._onnx_cuda_io = None
            self._onnx_cuda_embeddings_ortvalue = None

    def _advance_cache_position(self, cp_out) -> None:
        step = int(self._onnx_cache_position_len)
        if cp_out is None:
            if self._onnx_has_cache_position_input:
                next_local = int(self._onnx_cache_position_start) + step
                self._onnx_cache_position_start = np.int64(next_local % int(self._onnx_max_past_len))
            if self._onnx_has_position_ids_input:
                self._onnx_position_ids_start = np.int64(int(self._onnx_position_ids_start) + step)
            return
        cp_arr = np.asarray(cp_out, dtype=np.int64).reshape(-1)
        if cp_arr.size > 0:
            if self._onnx_has_cache_position_input:
                self._onnx_cache_position_start = np.int64(int(cp_arr[-1]) % int(self._onnx_max_past_len))
            if self._onnx_has_position_ids_input:
                self._onnx_position_ids_start = np.int64(int(self._onnx_position_ids_start) + step)

    def reset_streaming_state(self):
        super().reset_streaming_state()
        self._init_onnx_states()

    def _encode_continuous_embeddings(self, x: torch.Tensor, streaming: bool) -> torch.Tensor:
        if not streaming:
            raise RuntimeError(
                "EncoderMimiOnnx only supports streaming=True "
                "(HF weights are not loaded on the ONNX path)."
            )

        # ONNX model was exported with B=1 fixed signature.
        if x.shape[0] != 1:
            raise ValueError(f"EncoderMimiOnnx supports batch_size=1, got {x.shape[0]}")

        wave_t = x.detach()
        if self._use_cuda:
            if wave_t.device.type != "cuda":
                wave_t = wave_t.to(device="cuda")
            if wave_t.dtype != torch.float32:
                wave_t = wave_t.to(dtype=torch.float32)
            if not wave_t.is_contiguous():
                wave_t = wave_t.contiguous()
            if tuple(wave_t.shape) != self._onnx_wave_shape:
                raise ValueError(
                    f"ONNX Mimi input shape mismatch: got {tuple(wave_t.shape)}, "
                    f"expected {self._onnx_wave_shape}. "
                    "Adjusted path was removed; please provide fixed-length chunks."
                )
            if self._onnx_cuda_wave_buffer_t is None:
                raise RuntimeError("CUDA wave buffer is not initialized.")
            self._onnx_cuda_wave_buffer_t.copy_(wave_t, non_blocking=False)
            if self._onnx_cuda_io is None:
                self._onnx_cuda_io = self._onnx_sess.io_binding()

            io = self._onnx_cuda_io
            io.clear_binding_inputs()
            io.clear_binding_outputs()
            io.bind_input(
                "wave_24k",
                "cuda",
                0,
                np.float32,
                tuple(self._onnx_cuda_wave_buffer_t.shape),
                int(self._onnx_cuda_wave_buffer_t.data_ptr()),
            )
            if self._onnx_has_cache_position_input:
                torch.add(
                    self._onnx_cuda_cp_base_t,
                    int(self._onnx_cache_position_start),
                    out=self._onnx_cuda_cp_buffer_t,
                )
                torch.remainder(
                    self._onnx_cuda_cp_buffer_t,
                    int(self._onnx_max_past_len),
                    out=self._onnx_cuda_cp_buffer_t,
                )
                io.bind_input(
                    self._onnx_cache_position_key,
                    "cuda",
                    0,
                    np.int64,
                    tuple(self._onnx_cuda_cp_buffer_t.shape),
                    int(self._onnx_cuda_cp_buffer_t.data_ptr()),
                )
            if self._onnx_has_position_ids_input:
                torch.add(
                    self._onnx_cuda_pos_base_t,
                    int(self._onnx_position_ids_start),
                    out=self._onnx_cuda_pos_buffer_t,
                )
                io.bind_input(
                    self._onnx_position_ids_key,
                    "cuda",
                    0,
                    np.int64,
                    tuple(self._onnx_cuda_pos_buffer_t.shape),
                    int(self._onnx_cuda_pos_buffer_t.data_ptr()),
                )
            use_slot0_as_in = not self._onnx_cuda_kv_ping
            pads_in = self._onnx_cuda_pad_slot0 if use_slot0_as_in else self._onnx_cuda_pad_slot1
            pads_out = self._onnx_cuda_pad_slot1 if use_slot0_as_in else self._onnx_cuda_pad_slot0
            pasts_in = self._onnx_cuda_past_slot0 if use_slot0_as_in else self._onnx_cuda_past_slot1
            pasts_out = self._onnx_cuda_past_slot1 if use_slot0_as_in else self._onnx_cuda_past_slot0
            for i in range(self._onnx_num_pad):
                io.bind_ortvalue_input(self._onnx_pad_input_keys[i], pads_in[i])
            for i in range(len(pasts_in)):
                io.bind_ortvalue_input(self._onnx_past_input_keys[i], pasts_in[i])

            emb_name = self._onnx_output_names[0] if self._onnx_output_names else None
            if emb_name is None:
                raise RuntimeError("ONNX output_names missing embedding output.")
            if self._onnx_cuda_embeddings_ortvalue is not None:
                io.bind_ortvalue_output(emb_name, self._onnx_cuda_embeddings_ortvalue)
            else:
                io.bind_output(emb_name, "cuda", 0)
            # cache_position_out のランタイム形状は入力 cache_position と一致しない場合がある
            # （例: メタ cache_position_len=2 でもグラフ出力は長さ 1）。固定バッファ bind は避ける。
            if self._onnx_has_cache_position_output:
                io.bind_output(self._onnx_cache_position_out_key, "cuda", 0)
            for i in range(self._onnx_num_pad):
                io.bind_ortvalue_output(f"pad_cache_out_{i}", pads_out[i])
            for i in range(len(pasts_out)):
                io.bind_ortvalue_output(f"past_out_{i}", pasts_out[i])

            self._onnx_sess.run_with_iobinding(io)
            io.synchronize_outputs()
            out_vals = io.get_outputs()
            self._onnx_cuda_kv_ping = not self._onnx_cuda_kv_ping

            if self._onnx_has_cache_position_output and len(out_vals) > 1:
                self._advance_cache_position(out_vals[1].numpy())
            else:
                self._advance_cache_position(None)

            emb_ov = (
                self._onnx_cuda_embeddings_ortvalue
                if self._onnx_cuda_embeddings_ortvalue is not None
                else out_vals[0]
            )
            emb_t = torch.from_numpy(emb_ov.numpy())
            if x.device.type == "cuda":
                emb_t = emb_t.to(device=x.device, dtype=torch.float32)
            return emb_t

        if wave_t.device.type != "cpu":
            wave_t = wave_t.cpu()
        if wave_t.dtype != torch.float32:
            wave_t = wave_t.to(dtype=torch.float32)
        if not wave_t.is_contiguous():
            wave_t = wave_t.contiguous()
        wave = wave_t.numpy()
        if wave.shape != self._onnx_wave_shape:
            raise ValueError(
                f"ONNX Mimi input shape mismatch: got {tuple(wave.shape)}, "
                f"expected {self._onnx_wave_shape}. "
                "Adjusted path was removed; please provide fixed-length chunks."
            )
        self._onnx_inputs["wave_24k"] = wave
        if self._onnx_has_cache_position_input:
            np.add(
                self._onnx_cache_position_base,
                int(self._onnx_cache_position_start),
                out=self._onnx_cache_position_buffer,
                casting="unsafe",
            )
            np.mod(self._onnx_cache_position_buffer, int(self._onnx_max_past_len), out=self._onnx_cache_position_buffer)
            self._onnx_inputs[self._onnx_cache_position_key] = self._onnx_cache_position_buffer
        if self._onnx_has_position_ids_input:
            np.add(
                self._onnx_position_ids_base,
                int(self._onnx_position_ids_start),
                out=self._onnx_position_ids_buffer,
                casting="unsafe",
            )
            self._onnx_inputs[self._onnx_position_ids_key] = self._onnx_position_ids_buffer
        ort_inputs = self._onnx_inputs
        for i in range(self._onnx_num_pad):
            ort_inputs[self._onnx_pad_input_keys[i]] = self._onnx_states[i]
        for i in range(self._onnx_num_pad, len(self._onnx_states)):
            past_name = self._onnx_past_input_keys[i - self._onnx_num_pad]
            ort_inputs[past_name] = self._onnx_states[i]
        ort_out = self._onnx_sess.run(None, ort_inputs)
        emb_np = ort_out[0]
        state_out_start = 1
        if self._onnx_has_cache_position_output and len(ort_out) > 1:
            self._advance_cache_position(ort_out[1])
            state_out_start = 2
        else:
            self._advance_cache_position(None)
        self._onnx_states = list(ort_out[state_out_start:])
        emb_t = torch.from_numpy(emb_np)
        if x.device.type == "cpu":
            return emb_t
        return emb_t.to(device=x.device, dtype=torch.float32)


def build_audio_encoder(conf, cpc_model: str = ""):
    """Build and return the appropriate audio encoder instance based on the configuration.
    
    Instantiates either EncoderCPC, EncoderMimi, or EncoderMimiOnnx depending on the
    `encoder_type` and related settings specified in `conf`.
    
    Args:
        conf: The configuration object containing encoder settings.
        cpc_model (str): Optional path to a pretrained CPC model.
        
    Returns:
        nn.Module: The configured audio encoder.
    """
    encoder_type = getattr(conf, "encoder_type", "cpc")

    if encoder_type == "cpc":
        return EncoderCPC(
            load_pretrained=True if conf.load_pretrained == 1 else False,
            freeze=conf.freeze_encoder,
            cpc_model=cpc_model,
        )

    if encoder_type == "mimi":
        runtime_device = str(getattr(conf, "runtime_device", "cpu"))
        use_onnx = bool(int(getattr(conf, "mimi_use_onnx", 1)))
        onnx_precision = str(getattr(conf, "mimi_onnx_precision", "fp32")).strip().lower()
        if onnx_precision not in {"fp32", "int8"}:
            raise ValueError(f"Unsupported mimi_onnx_precision: {onnx_precision}")
        if runtime_device.startswith("cuda") and onnx_precision == "int8":
            raise ValueError("mimi_onnx_precision='int8' is not supported with CUDA. Use 'fp32' on CUDA.")

        if not use_onnx:
            return EncoderMimi(
                frame_hz=getattr(conf, "frame_hz", 10),
                freeze=conf.freeze_encoder,
                mimi_model_name=getattr(conf, "mimi_model_name", "kyutai/mimi"),
            )

        def _opt_str(name: str) -> str | None:
            if not hasattr(conf, name):
                return None
            v = getattr(conf, name)
            if v is None:
                return None
            s = str(v).strip()
            return s if s else None

        fp32_path = _opt_str("mimi_onnx_fp32_path")
        fp32_meta = _opt_str("mimi_onnx_fp32_meta_path")
        int8_path = _opt_str("mimi_onnx_int8_path")
        int8_meta = _opt_str("mimi_onnx_int8_meta_path")

        if onnx_precision == "int8":
            local_onnx, local_meta = int8_path, int8_meta
            local_label = "mimi_onnx_int8_path"
        else:
            local_onnx, local_meta = fp32_path, fp32_meta
            local_label = "mimi_onnx_fp32_path"

        hf_cache_dir = getattr(conf, "mimi_onnx_hf_cache_dir", None)
        hf_force = bool(getattr(conf, "mimi_onnx_hf_force_download", False))

        if local_onnx:
            selected_meta = local_meta or f"{local_onnx}.json"
            if not os.path.isfile(local_onnx) or not os.path.isfile(selected_meta):
                raise FileNotFoundError(
                    f"Local Mimi ONNX paths were set ({local_label}) but files are missing: "
                    f"onnx={local_onnx!r} meta={selected_meta!r}"
                )
            selected = local_onnx
            print(f"Using ONNX Mimi backend ({onnx_precision}, local): {selected}")
        else:
            selected, selected_meta = download_continuous_mimi_onnx(
                precision=onnx_precision,
                cache_dir=hf_cache_dir,
                force_download=hf_force,
            )
        return EncoderMimiOnnx(
            frame_hz=getattr(conf, "frame_hz", 12.5),
            freeze=conf.freeze_encoder,
            mimi_model_name=getattr(conf, "mimi_model_name", "kyutai/mimi"),
            onnx_model_path=selected,
            onnx_meta_path=selected_meta,
            runtime_device=runtime_device,
            onnx_cpu_intra_threads=getattr(conf, "mimi_onnx_cpu_intra_threads", 4),
            onnx_cpu_inter_threads=getattr(conf, "mimi_onnx_cpu_inter_threads", 1),
        )

    raise ValueError(f"Unsupported encoder_type: {encoder_type}")