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

import time
import copy


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
        self.reset()

    @property
    def step_in(self) -> float:
        return self.orig_freq / self.new_freq

    def reset(self) -> None:
        self.state = _StreamingResamplerState()

    def _empty_like_prev(self) -> torch.Tensor:
        if self.state.prev is None:
            return torch.empty((0, 0, 0), dtype=torch.float32)
        return self.state.prev.unsqueeze(-1)[:, :, :0]

    def process(self, x: torch.Tensor) -> torch.Tensor:
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

class EncoderCPC(nn.Module):
    """
    Encoder: waveform -> h
    pretrained: default='cpc'

    A simpler version of the Encoder
    check paper (branch) version to see other encoders...
    """

    def __init__(self, load_pretrained=True, freeze=True, cpc_model=''):
        
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

    def forward(self, waveform):
        
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)  # channel dim

        # Backwards using only the encoder encounters:
        # ---------------------------------------------------
        # RuntimeError: one of the variables needed for gradient computation
        # has been modified by an inplace operation:
        # [torch.FloatTensor [4, 256, 1000]], which is output 0 of ReluBackward0, is at version 1;
        # expected version 0 instead. Hint: enable anomaly detection to find
        # the operation that failed to compute its gradient, with
        # torch.autograd.set_detect_anomaly(True).
        # HOWEVER, if we feed through encoder.gAR we do not encounter that problem...

        
        z = self.encoder.gEncoder(waveform)
        z = einops.rearrange(z, "b c n -> b n c")
        z = z[:, 1:-1, :]
        z = self.encoder.gAR(z)
        z = self.downsample(z)
        
        return z

    def hash_tensor(self, tensor):
        return hash(tuple(tensor.reshape(-1).tolist()))


class EncoderMimi(nn.Module):
    def __init__(
        self,
        frame_hz: float = 10,
        freeze: bool = True,
        mimi_model_name: str = "kyutai/mimi",
        context_samples: int = 320,
    ):
        super().__init__()

        try:
            from transformers import MimiConfig, MimiModel
            from transformers.models.mimi.modeling_mimi import MimiConv1dPaddingCache
            from transformers.cache_utils import DynamicCache
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Mimi encoder requires transformers with Mimi support."
            ) from exc

        try:
            import torchaudio
        except (ModuleNotFoundError, OSError):
            torchaudio = None

        self._torchaudio = torchaudio
        self._MimiConv1dPaddingCache = MimiConv1dPaddingCache
        self._DynamicCache = DynamicCache
        self.sample_rate = 24000

        config = MimiConfig.from_pretrained(mimi_model_name)
        config.use_causal_conv = True
        self.model = MimiModel.from_pretrained(mimi_model_name, config=config)
        self.model.eval()

        self.frame_hz_mimi = float(
            getattr(getattr(self.model, "quantizer", None), "frame_rate", 12.5)
        )
        self.frame_hz = float(frame_hz)
        self.context_samples = int(context_samples)
        self._audio_resampler = _CausalStreamingResampler(
            orig_freq=16000.0,
            new_freq=float(self.sample_rate),
        )
        self._feature_resampler = None
        self._mimi_padding_cache = None
        self._mimi_encoder_past_key_values = None
        self._frame_rate_conv_cache = None
        self._mimi_did_first_24k_leading_zeros = False

        self.output_dim = 512
        if hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
            self.output_dim = self.model.config.hidden_size
        elif hasattr(self.model, "config") and hasattr(self.model.config, "dimension"):
            self.output_dim = self.model.config.dimension
        self.dim = self.output_dim
        self.downsample_ratio = int(round(16000 / self.frame_hz)) if self.frame_hz else 0

        self.frame_rate_conv = CConv1d(
            in_channels=self.output_dim,
            out_channels=self.output_dim,
            kernel_size=3,
            padding=0,
            bias=True,
        )

        self._fix_mimi_padding_buffers()

        if freeze:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad_(True)
        print(f"Trainable {self.__class__.__name__}!")

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def reset_streaming_state(self):
        self._audio_resampler.reset()
        if self._feature_resampler is not None:
            self._feature_resampler.reset()
        self._mimi_padding_cache = None
        self._mimi_encoder_past_key_values = None
        self._frame_rate_conv_cache = None
        self._mimi_did_first_24k_leading_zeros = False

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

    def _mimi_padding_layout(self) -> list[tuple[int, int]]:
        """
        Returns list of (in_channels, padding_total) for Mimi conv cache layers.
        """
        layout: list[tuple[int, int]] = []
        for layer_name in self.model.encoder._mimiconv1d_layer_names:
            layer = self.model.encoder.get_submodule(layer_name)
            layout.append((int(layer.in_channels), int(layer.padding_total)))
        if self.model.downsample is not None:
            layout.append(
                (
                    int(self.model.downsample.in_channels),
                    int(self.model.downsample.padding_total),
                )
            )
        return layout

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
            enc_out = self.model.encoder_transformer(
                emb.transpose(1, 2),
                past_key_values=None,
                use_cache=True,
                return_dict=True,
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
        return pad_templates, enc_out.past_key_values

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

    def _encode_continuous_embeddings(self, x: torch.Tensor, streaming: bool) -> torch.Tensor:
        padding_cache = None
        past_key_values = None

        if streaming:
            padding_cache = self._ensure_mimi_padding_cache()
            past_key_values = self._mimi_encoder_past_key_values

        embeddings = self.model.encoder(x, padding_cache=padding_cache)
        encoder_outputs = self.model.encoder_transformer(
            embeddings.transpose(1, 2),
            past_key_values=past_key_values,
            use_cache=streaming,
            return_dict=True,
        )

        if streaming:
            self._mimi_encoder_past_key_values = encoder_outputs.past_key_values

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

        self.model.eval()
        self._fix_mimi_padding_buffers()

        if not torch.is_tensor(waveform):
            raise TypeError(f"Expected waveform to be a torch.Tensor, got {type(waveform)}")

        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)

        waveform = waveform.to(dtype=torch.float32)

        if not streaming:
            return self._encode_offline(waveform, resampling=resampling)

        overlap_context = self.context_samples if has_overlap_context else 0
        if overlap_context > 0:
            if waveform.shape[-1] <= overlap_context:
                return waveform.new_zeros((waveform.shape[0], 0, self.output_dim))
            waveform = waveform[..., overlap_context:]

        input_num_samples = int(waveform.shape[-1])
        if input_num_samples == 0:
            return waveform.new_zeros((waveform.shape[0], 0, self.output_dim))

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
        elif waveform.shape[-1] == 0:
            return waveform.new_zeros((waveform.shape[0], 0, self.output_dim))

        if waveform.shape[-1] == 0:
            return waveform.new_zeros((waveform.shape[0], 0, self.output_dim))

        embeddings = self._encode_continuous_embeddings(waveform, streaming=True)
        if embeddings.shape[1] == 0:
            return waveform.new_zeros((waveform.shape[0], 0, self.output_dim))

        return self._align_frame_rate(
            embeddings,
            input_num_samples=input_num_samples,
            streaming=True,
            finalize_stream=finalize_stream,
        )


class _ReusableOrtRunner:
    def __init__(self, ort, sess, input_template: dict[str, np.ndarray], use_cuda: bool):
        self.sess = sess
        self.io = sess.io_binding()
        self.input_ortvalues = {}

        for name, arr in input_template.items():
            if use_cuda:
                ov = ort.OrtValue.ortvalue_from_numpy(arr, "cuda", 0)
            else:
                ov = ort.OrtValue.ortvalue_from_numpy(arr, "cpu", 0)
            self.io.bind_ortvalue_input(name, ov)
            self.input_ortvalues[name] = ov

        for out in sess.get_outputs():
            self.io.bind_output(out.name, "cuda" if use_cuda else "cpu")

    def run(self, ort_inputs: dict[str, np.ndarray]):
        for name, arr in ort_inputs.items():
            self.input_ortvalues[name].update_inplace(arr)
        self.sess.run_with_iobinding(self.io)
        return self.io.copy_outputs_to_cpu()


class EncoderMimiOnnx(EncoderMimi):
    """
    Mimi core ONNX backend:
    - CUDA: FP32 ONNX + CUDA EP optimized + reusable IOBinding
    - CPU: INT8 ONNX
    """

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
        onnx_reuse_input_dict: bool = True,
        onnx_tune_cpu_threads: bool = True,
    ):
        super().__init__(
            frame_hz=frame_hz,
            freeze=freeze,
            mimi_model_name=mimi_model_name,
            context_samples=context_samples,
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
        self._onnx_runner = None
        self._onnx_states: list[np.ndarray] = []
        self._onnx_input_names: list[str] = []
        self._onnx_pad_input_keys: list[str] = []
        self._onnx_past_input_keys: list[str] = []
        self._onnx_num_pad = 0
        self._onnx_wave_shape: tuple[int, ...] = (1, 1, 0)
        self._onnx_inputs: dict[str, np.ndarray] = {}
        self._onnx_cpu_intra_threads = int(onnx_cpu_intra_threads)
        self._onnx_cpu_inter_threads = int(onnx_cpu_inter_threads)
        self._onnx_reuse_input_dict = bool(onnx_reuse_input_dict)
        self._onnx_tune_cpu_threads = bool(onnx_tune_cpu_threads)
        # Match PyTorch Mimi DynamicCache effective cap (sliding_window - 1).
        sw = getattr(getattr(self, "model", None), "config", None)
        sw_val = getattr(sw, "sliding_window", None) if sw is not None else None
        self._onnx_max_past_len = int(sw_val - 1) if isinstance(sw_val, int) and sw_val > 0 else None

        self._onnx_sess = self._create_onnx_session()
        self._init_onnx_states()

    def _create_onnx_session(self):
        so = self._ort.SessionOptions()
        so.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self._use_cuda and "CUDAExecutionProvider" in self._ort.get_available_providers():
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "do_copy_in_default_stream": "1",
                        "cudnn_conv_use_max_workspace": "1",
                    },
                ),
                "CPUExecutionProvider",
            ]
        else:
            # Realtime-friendly default for CPU EP.
            if self._onnx_tune_cpu_threads:
                so.intra_op_num_threads = max(1, self._onnx_cpu_intra_threads)
                so.inter_op_num_threads = max(1, self._onnx_cpu_inter_threads)
            providers = ["CPUExecutionProvider"]
        return self._ort.InferenceSession(self._onnx_model_path, sess_options=so, providers=providers)

    def _load_meta(self) -> dict[str, Any]:
        p = Path(self._onnx_meta_path)
        if not p.exists():
            raise FileNotFoundError(f"ONNX meta file not found: {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    def _init_onnx_states(self):
        meta = self._load_meta()
        self._onnx_input_names = list(meta["input_names"])
        self._onnx_num_pad = int(meta["num_pad_tensors"])
        self._onnx_pad_input_keys = [f"pad_cache_{i}" for i in range(self._onnx_num_pad)]
        n_states_total = len(self._onnx_input_names) - 1
        n_past = max(0, n_states_total - self._onnx_num_pad)
        self._onnx_past_input_keys = [f"past_{i}" for i in range(n_past)]
        if "max_past_len" in meta and meta["max_past_len"] is not None:
            self._onnx_max_past_len = int(meta["max_past_len"])
        spec = self.get_mimi_streaming_onnx_io_spec(batch_size=1)

        wave_len = int(meta["wave_24k_mimi_input"])
        self._onnx_wave_shape = (1, 1, wave_len)
        dtype = np.float32
        wave = np.zeros(self._onnx_wave_shape, dtype=dtype)
        pad = [np.zeros(tuple(s), dtype=dtype) for s in spec["padding_cache_shapes"]]
        past = [np.zeros(tuple(s), dtype=dtype) for s in spec["past_key_value_shapes"]]
        self._onnx_states = [*pad, *past]

        input_template = {"wave_24k": wave}
        for i, s in enumerate(self._onnx_states):
            if i < self._onnx_num_pad:
                input_template[self._onnx_pad_input_keys[i]] = s
            else:
                input_template[self._onnx_past_input_keys[i - self._onnx_num_pad]] = s
        self._onnx_inputs = dict(input_template)
        self._onnx_runner = _ReusableOrtRunner(self._ort, self._onnx_sess, input_template, self._use_cuda)

    def _run_onnx_cuda_iobinding(self, ort_inputs: dict[str, np.ndarray]):
        # Cache length grows in early streaming steps; rebuild bindings when input shape changes.
        try:
            return self._onnx_runner.run(ort_inputs)
        except Exception:
            self._onnx_runner = _ReusableOrtRunner(self._ort, self._onnx_sess, ort_inputs, use_cuda=True)
            return self._onnx_runner.run(ort_inputs)

    def reset_streaming_state(self):
        super().reset_streaming_state()
        self._init_onnx_states()

    @staticmethod
    def _fit_np_shape(x: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
        y = x
        if y.ndim >= 3 and len(target_shape) >= 3 and target_shape[-2] > 0 and y.shape[-2] > target_shape[-2]:
            start = y.shape[-2] - target_shape[-2]
            y = y.take(indices=range(start, y.shape[-2]), axis=-2)
        if y.shape != target_shape:
            slices = tuple(slice(0, min(y.shape[i], target_shape[i])) for i in range(len(target_shape)))
            y = y[slices]
        if y.shape != target_shape:
            out = np.zeros(target_shape, dtype=y.dtype)
            slices = tuple(slice(0, min(out.shape[i], y.shape[i])) for i in range(len(target_shape)))
            out[slices] = y[slices]
            y = out
        return y

    def _clip_past_len(self, x: np.ndarray) -> np.ndarray:
        if self._onnx_max_past_len is None:
            return x
        if x.ndim < 3:
            return x
        # KV shape: [B, H, T, D], clip T axis (-2)
        t = int(x.shape[-2])
        if t <= self._onnx_max_past_len:
            return x
        return x[..., -self._onnx_max_past_len :, :]

    def _encode_continuous_embeddings(self, x: torch.Tensor, streaming: bool) -> torch.Tensor:
        if not streaming:
            return super()._encode_continuous_embeddings(x, streaming=False)

        # ONNX model was exported with B=1 fixed signature.
        if x.shape[0] != 1:
            raise ValueError(f"EncoderMimiOnnx supports batch_size=1, got {x.shape[0]}")

        wave_t = x.detach()
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
        if self._onnx_reuse_input_dict:
            self._onnx_inputs["wave_24k"] = wave
            ort_inputs = self._onnx_inputs
        else:
            ort_inputs = {"wave_24k": wave}
        for i in range(self._onnx_num_pad):
            ort_inputs[self._onnx_pad_input_keys[i]] = self._onnx_states[i]
        for i in range(self._onnx_num_pad, len(self._onnx_states)):
            ort_inputs[self._onnx_past_input_keys[i - self._onnx_num_pad]] = self._clip_past_len(
                self._onnx_states[i]
            )
        if self._use_cuda:
            ort_out = self._run_onnx_cuda_iobinding(ort_inputs)
        else:
            ort_out = self._onnx_sess.run(None, ort_inputs)
        emb_np = ort_out[0]
        # Keep variable-length cache states, but cap past len to match PyTorch.
        new_states: list[np.ndarray] = []
        for i, o in enumerate(ort_out[1:]):
            if i < self._onnx_num_pad:
                new_states.append(o)
            else:
                new_states.append(self._clip_past_len(o))
        self._onnx_states = new_states
        emb_t = torch.from_numpy(emb_np)
        if x.device.type == "cpu":
            return emb_t
        return emb_t.to(device=x.device, dtype=torch.float32)


def build_audio_encoder(conf, cpc_model: str = ""):
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
        fp32_path = str(
            getattr(
                conf,
                "mimi_onnx_fp32_path",
                "onnx/mimi_streaming_fp32.onnx",
            )
        )
        int8_path = str(
            getattr(
                conf,
                "mimi_onnx_int8_path",
                "onnx/mimi_streaming_int8_matmul.onnx",
            )
        )
        if not use_onnx:
            return EncoderMimi(
                frame_hz=getattr(conf, "frame_hz", 10),
                freeze=conf.freeze_encoder,
                mimi_model_name=getattr(conf, "mimi_model_name", "kyutai/mimi"),
            )

        fp32_meta = str(
            getattr(
                conf,
                "mimi_onnx_fp32_meta_path",
                f"{fp32_path}.json",
            )
        )
        int8_meta = str(
            getattr(
                conf,
                "mimi_onnx_int8_meta_path",
                f"{int8_path}.json",
            )
        )
        if runtime_device.startswith("cuda"):
            selected = fp32_path
            selected_meta = fp32_meta
        else:
            selected = int8_path
            selected_meta = int8_meta

        if not os.path.exists(selected):
            raise FileNotFoundError(f"Mimi ONNX model not found: {selected}")
        if not os.path.exists(selected_meta):
            raise FileNotFoundError(f"Mimi ONNX meta not found: {selected_meta}")

        print(f"Using ONNX Mimi backend: {selected}")
        return EncoderMimiOnnx(
            frame_hz=getattr(conf, "frame_hz", 10),
            freeze=conf.freeze_encoder,
            mimi_model_name=getattr(conf, "mimi_model_name", "kyutai/mimi"),
            onnx_model_path=selected,
            onnx_meta_path=selected_meta,
            runtime_device=runtime_device,
            onnx_cpu_intra_threads=getattr(conf, "mimi_onnx_cpu_intra_threads", 2),
            onnx_cpu_inter_threads=getattr(conf, "mimi_onnx_cpu_inter_threads", 1),
            onnx_reuse_input_dict=bool(int(getattr(conf, "mimi_onnx_reuse_input_dict", 1))),
            onnx_tune_cpu_threads=bool(int(getattr(conf, "mimi_onnx_tune_cpu_threads", 1))),
        )

    raise ValueError(f"Unsupported encoder_type: {encoder_type}")