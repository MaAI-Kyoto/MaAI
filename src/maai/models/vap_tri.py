"""Experimental three-party (triadic) VAP model.

Ported from the ``vap-tri`` training code (``VapGPT`` with ``GPTMulti`` cross
channel and the ``query_va`` objective). State-dict keys match the training
checkpoint, so the exported VAP weights load directly.

Architecture (per frame, 12.5 Hz Mimi features):

    encoder{1,2,3} -> decrease_dimension (+relu) -> ar_channel (shared, ALiBi)
      -> GPTMulti (3 channels interleaved time-major, sinusoidal time PE
         + channel embedding, block-causal) -> x, x1, x2, x3
      -> va_classifier(x_c)             : VAD per speaker
      -> query_va_head(x, [x1, x2, x3]) : P(speaker c active in bin k)

``p_now`` / ``p_future`` are the query_va probabilities summed over the "now"
/ "future" bins and normalised over speakers (same as ``tt_metrics_source =
"query_va"`` in vap-tri).

``GPTMulti`` uses absolute (sinusoidal) time positions, so it cannot reuse a KV
cache. The runtime keeps the last ``context_len`` frames of encoder features
and recomputes the whole window every frame, which is exactly the condition the
model was trained on (20 s clips starting at position 0).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..encoder import build_audio_encoder
from ..modules import GPT


@dataclass
class VapTriConfig:
    """Hyper-parameters of the triadic VAP model (stored in the exported file)."""

    frame_hz: float = 12.5
    num_speakers: int = 3
    max_speakers: int = 4
    encoder_dim: int = 512
    dim: int = 256
    dff_k: int = 3
    channel_layers: int = 1
    cross_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    query_va_bin_times: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    query_va_hidden_dim: int = 0
    query_va_head_type: str = "mlp"
    # Inclusive query_va bin ranges for p_now / p_future.
    now_bins: List[int] = field(default_factory=lambda: [0, 1])
    future_bins: Optional[List[int]] = field(default_factory=lambda: [2, 3])
    context_len_sec: float = 20.0
    mimi_onnx_variant: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "VapTriConfig":
        conf = cls()
        for k, v in (d or {}).items():
            if hasattr(conf, k):
                setattr(conf, k, v)
        return conf


class MultiCombinator(nn.Module):
    """Combine N per-speaker representations with a shared projection."""

    def __init__(self, dim: int, activation: str = "GELU"):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim)
        self.activation = getattr(nn, activation)()

    def forward(self, xs: List[Tensor], present: Optional[Tensor] = None) -> Tensor:
        h = torch.stack(xs, dim=2)  # (B, T, C, D)
        h = self.activation(self.ln(self.proj(h)))
        if present is not None:
            h = h * present[:, None, :, None].to(h.dtype)
        return h.sum(dim=2)


class GPTMulti(nn.Module):
    """N-speaker cross-channel Transformer (interleaved, block-causal)."""

    def __init__(
        self,
        dim: int,
        dff_k: int = 3,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "GELU",
        max_speakers: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.channel_emb = nn.Embedding(max_speakers, dim)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=int(dim * dff_k),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.combinator = MultiCombinator(dim=dim, activation=activation)
        self._mask_cache: Dict[tuple, Tensor] = {}
        self._pe_cache: Dict[tuple, Tensor] = {}

    def _time_pe(self, T: int, device, dtype) -> Tensor:
        key = (T, device, dtype)
        pe = self._pe_cache.get(key)
        if pe is None:
            pe = torch.zeros(T, self.dim, device=device, dtype=dtype)
            pos = torch.arange(0, T, device=device, dtype=dtype).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, self.dim, 2, device=device, dtype=dtype)
                * (-math.log(10000.0) / self.dim)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self._pe_cache[key] = pe
        return pe

    def _causal_mask(self, T: int, C: int, device) -> Tensor:
        key = (T, C, device)
        m = self._mask_cache.get(key)
        if m is None:
            idx = torch.arange(T * C, device=device)
            t = idx // C
            m = ~(t[None, :] <= t[:, None])  # True == not allowed
            self._mask_cache[key] = m
        return m

    def forward(self, xs: List[Tensor], present: Optional[Tensor] = None) -> Dict[str, Tensor]:
        B, T, D = xs[0].shape
        C = len(xs)
        device = xs[0].device
        dtype = xs[0].dtype

        h = torch.stack(xs, dim=2).reshape(B, T * C, D)
        time_pe = self._time_pe(T, device, dtype).repeat_interleave(C, dim=0)
        chan_idx = torch.arange(C, device=device).repeat(T)
        h = h + time_pe.unsqueeze(0) + self.channel_emb(chan_idx).unsqueeze(0)

        attn_mask = self._causal_mask(T, C, device)
        key_padding_mask = None
        if present is not None:
            kp = present < 0.5
            key_padding_mask = kp.repeat(1, T).reshape(B, T * C)

        for layer in self.layers:
            h = layer(h, src_mask=attn_mask, src_key_padding_mask=key_padding_mask)

        h = h.reshape(B, T, C, D)
        xs_out = [h[:, :, c] for c in range(C)]
        ret = {"x": self.combinator(xs_out, present=present)}
        for c, xc in enumerate(xs_out):
            ret[f"x{c + 1}"] = xc
        return ret


class QueryVAHead(nn.Module):
    """Shared scalar head ``f(context, speaker, horizon) -> logit`` ("mlp" type)."""

    def __init__(self, d_model: int, num_horizons: int, hidden_dim: int = 0, speaker_dim: int = 0):
        super().__init__()
        h = hidden_dim if hidden_dim and hidden_dim > 0 else d_model
        spk_dim = speaker_dim if speaker_dim and speaker_dim > 0 else d_model
        self.num_horizons = num_horizons
        self.hidden_dim = h
        self.context_proj = nn.Linear(d_model, h)
        self.speaker_proj = nn.Linear(spk_dim, h)
        self.horizon_emb = nn.Embedding(num_horizons, h)
        self.mlp = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

    def forward(self, context_hidden: Tensor, speaker_repr: Tensor, speaker_mask: Optional[Tensor] = None) -> Tensor:
        """context_hidden [B,T,D], speaker_repr [B,T,S,D] -> logits [B,T,S,K]."""
        B, T, _ = context_hidden.shape
        S = speaker_repr.shape[-2]
        K, H = self.num_horizons, self.hidden_dim
        ctx = self.context_proj(context_hidden)[:, :, None, None]
        spk = self.speaker_proj(speaker_repr).unsqueeze(3)
        hor = self.horizon_emb.weight.view(1, 1, 1, K, H)
        logits = self.mlp(ctx + spk + hor).squeeze(-1)
        if speaker_mask is not None:
            absent = speaker_mask.view(B, 1, S, 1).to(logits.dtype) < 0.5
            logits = logits.masked_fill(absent, -1e4)
        return logits


class VapGPT_tri(nn.Module):
    """Three-party VAP model (query_va objective, Mimi encoder)."""

    def __init__(self, tri_conf: VapTriConfig, enc_conf=None):
        """
        Args:
            tri_conf: Model hyper-parameters (see :class:`VapTriConfig`).
            enc_conf: A ``VapConfig``-like object passed to
                :func:`build_audio_encoder` (encoder_type='mimi', ONNX settings).
        """
        super().__init__()
        self.tri_conf = tri_conf
        self.conf = enc_conf
        self.num_speakers = int(tri_conf.num_speakers)
        dim = int(tri_conf.dim)

        self.decrease_dimension = nn.Linear(int(tri_conf.encoder_dim), dim)
        self.absent_embedding = nn.Parameter(torch.zeros(dim))

        self.ar_channel = GPT(
            dim=dim,
            dff_k=tri_conf.dff_k,
            num_layers=tri_conf.channel_layers,
            num_heads=tri_conf.num_heads,
            dropout=tri_conf.dropout,
        )
        self.ar = GPTMulti(
            dim=dim,
            dff_k=tri_conf.dff_k,
            num_layers=tri_conf.cross_layers,
            num_heads=tri_conf.num_heads,
            dropout=tri_conf.dropout,
            max_speakers=max(int(tri_conf.max_speakers), self.num_speakers),
        )
        self.va_classifier = nn.Linear(dim, 1)
        if tri_conf.query_va_head_type != "mlp":
            raise ValueError(
                f"Only query_va_head_type='mlp' is supported, got {tri_conf.query_va_head_type!r}."
            )
        self.query_va_head = QueryVAHead(
            d_model=dim,
            num_horizons=len(tri_conf.query_va_bin_times),
            hidden_dim=tri_conf.query_va_hidden_dim,
            speaker_dim=dim,
        )

    def encoder_names(self) -> List[str]:
        return [f"encoder{c + 1}" for c in range(self.num_speakers)]

    def load_encoder(self):
        """Build one streaming audio encoder per speaker (ONNX sessions are shared)."""
        for name in self.encoder_names():
            setattr(self, name, build_audio_encoder(self.conf).eval())

    def encode_audio(self, audios: List[Tensor]) -> List[Tensor]:
        """Encode each speaker's (B, 1, N) 16 kHz waveform -> list of (B, T, dim)."""
        feats = []
        for name, audio in zip(self.encoder_names(), audios):
            x = getattr(self, name)(audio)
            feats.append(torch.relu(self.decrease_dimension(x)))
        return feats

    def forward(self, xs: List[Tensor], present: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Run the model on full feature sequences.

        Args:
            xs: list of ``num_speakers`` tensors (B, T, dim) (after decrease_dimension).
            present: optional (B, S) presence mask (1 = present, 0 = absent).

        Returns:
            dict with ``vad`` (B,T,S), ``query_va_probs`` (B,T,S,K),
            ``p_now`` (B,T,S) and ``p_future`` (B,T,S) or None.
        """
        xs = list(xs)
        B, T, D = xs[0].shape
        if present is not None:
            present = present.to(device=xs[0].device, dtype=xs[0].dtype)
            absent = present < 0.5
            if bool(absent.any()):
                ae = self.absent_embedding.to(xs[0].dtype).view(1, 1, D)
                for c in range(len(xs)):
                    m = absent[:, c]
                    if bool(m.any()):
                        xs[c] = torch.where(m[:, None, None], ae, xs[c])

        os_ = [self.ar_channel(x)["x"] for x in xs]
        out = self.ar(os_, present=present)
        spk = [out[f"x{c + 1}"] for c in range(len(xs))]

        vad = torch.cat([self.va_classifier(x) for x in spk], dim=-1).sigmoid()
        qv_logits = self.query_va_head(out["x"], torch.stack(spk, dim=2), speaker_mask=present)
        qv = qv_logits.sigmoid()

        p_now = self._next_speaker_probs(qv, self.tri_conf.now_bins, present)
        fb = self.tri_conf.future_bins
        p_future = None if not fb else self._next_speaker_probs(qv, fb, present)
        return {
            "vad": vad,
            "query_va_logits": qv_logits,
            "query_va_probs": qv,
            "p_now": p_now,
            "p_future": p_future,
        }

    @staticmethod
    def _next_speaker_probs(qv: Tensor, bins, present: Optional[Tensor], eps: float = 1e-5) -> Tensor:
        lo, hi = int(bins[0]), min(int(bins[-1]), qv.shape[-1] - 1)
        agg = qv[..., lo : hi + 1].sum(dim=-1)  # (B, T, S)
        if present is not None:
            agg = agg * (present > 0.5).to(agg.dtype).unsqueeze(1)
        return agg / (agg.sum(dim=-1, keepdim=True) + eps)
