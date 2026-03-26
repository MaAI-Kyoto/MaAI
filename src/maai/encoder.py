import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import os
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .encoder_components import CConv1d, load_CPC, get_cnn_layer

import time


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
            # First streaming forward after 16 kHz -> 24 kHz: prepend zeros spanning the same
            # duration as overlap stripped at 16 kHz (causal warmup for Mimi).
            if (
                not self._mimi_did_first_24k_leading_zeros
                and waveform.shape[-1] > 0
            ):
                n_pad = int(round(float(overlap_context) * float(self.sample_rate) / 16000.0))
                if n_pad > 0:
                    pad = waveform.new_zeros(
                        waveform.shape[0],
                        waveform.shape[1],
                        n_pad,
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


def build_audio_encoder(conf, cpc_model: str = ""):
    encoder_type = getattr(conf, "encoder_type", "cpc")

    if encoder_type == "cpc":
        return EncoderCPC(
            load_pretrained=True if conf.load_pretrained == 1 else False,
            freeze=conf.freeze_encoder,
            cpc_model=cpc_model,
        )

    if encoder_type == "mimi":
        return EncoderMimi(
            frame_hz=getattr(conf, "frame_hz", 10),
            freeze=conf.freeze_encoder,
            mimi_model_name=getattr(conf, "mimi_model_name", "kyutai/mimi"),
        )

    raise ValueError(f"Unsupported encoder_type: {encoder_type}")