"""Offline batch inference for MaAI models over recorded audio.

:class:`~maai.model.Maai` is built for live audio: it runs the audio encoder and
the transformer once per frame, so scoring a recording that is already on disk
costs one pair of forward passes per frame. :class:`MaaiBatch` scores whole
recordings instead:

1. The audio encoder runs over long chunks (``encoder_chunk_frames`` frames per
   call) and over several recordings at once, carrying its streaming state from
   chunk to chunk, so the features are those of the frame-by-frame stream.

   - Mimi (``model_type="normal-ver2"``): the resampler, convolution padding
     and transformer KV cache are carried over.
   - CPC (``model_type="normal"``): the realtime path encodes each frame
     together with 320 samples of overlap and drops the outputs at the window
     edges, so only the valid region of the convolutions is used; a window of
     several frames gives the same outputs. The LSTM state is carried over, and
     each model's downsampling layer runs window by window as in realtime.

2. Each model's transformer runs over the whole sequence in one forward pass.
   The ``context_limit`` band mask makes this the same computation as the
   realtime KV cache, which is trimmed to the last ``context_len_sec`` seconds.
   Long inputs are folded into overlapping windows that are processed as one
   batch (see ``fold_context_frames``).
3. With several models, the audio is encoded once and the features are shared
   (the encoder core is the same for all of them, as in ``MaaiMultiple``).

Timing follows the realtime ``Maai``. With Mimi, the realtime default
(``use_mimi_onnx=True``) drops the first encoder output after a reset, so the
first prediction is made after two frames of audio (see ``drop_first_frame``);
with CPC, after one frame. Small numerical differences remain (ONNX Runtime vs
PyTorch for the Mimi default, and convolution algorithms).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .encoder import EncoderMimi
from .model import DEPRECATED_MODE_ALIASES, _load_vap_model
from .models.config import VapConfig
from .util import MONO_CAPABLE_MODES, resolve_encoder_type

# Modes whose ``encode_audio`` swaps audio1/audio2 before the model forward
# (the model's first channel is the user, i.e. the speaker being responded to).
SWAP_CHANNEL_MODES = frozenset({"bc", "bc_2type", "nod", "nod_timing", "nod_para"})

# Returned only with ``return_p_bins=True`` (same as ``Maai``).
P_BINS_KEYS = ("p_bins", "p_bins_now", "p_bins_future")

DEFAULT_FRAME_RATE = {"mimi": 12.5, "cpc": 10.0}

# Overlap that the realtime path keeps before each frame (Maai.frame_contxt_padding);
# the CPC encoder sees it, Mimi strips it.
CPC_CONTEXT_SAMPLES = 320
# Samples per output frame of the CPC convolutions (CPCEncoder.DOWNSAMPLING).
CPC_HOP_SAMPLES = 160


def _attention_depth(vap: nn.Module) -> int:
    """Attention layers stacked on the deepest path of a model.

    Each layer lets a frame see ``context_limit - 1`` more frames into the past
    (the cross-attention of a stereo layer looks at the other channel at the
    same depth), so the receptive field is depth x (context_limit - 1) frames.
    """
    conf = vap.conf
    depth = conf.channel_layers + conf.cross_layers
    if hasattr(vap, "task_gpts"):  # nod_para's parameter branch
        depth = max(depth, conf.channel_layers + conf.nod_task_gpt_layers)
    return depth


def _per_window(fn, z: torch.Tensor, n_windows: int) -> torch.Tensor:
    """Apply ``fn`` to each of ``n_windows`` equal windows of (N, T, D) separately."""
    n, t, d = z.shape
    y = fn(z.reshape(n * n_windows, t // n_windows, d))
    return y.reshape(n, -1, y.shape[-1])


@dataclass
class _BatchModel:
    label: str
    mode: str
    variant: str
    vap: nn.Module
    return_p_bins: bool

    @property
    def single_tower(self) -> bool:
        return bool(getattr(self.vap, "is_single_tower", False))

    @property
    def swap(self) -> bool:
        return self.mode in SWAP_CHANNEL_MODES


class MaaiBatch:
    """Batch (offline) inference over whole recordings.

    Inputs are 16 kHz mono float32 waveforms, one per speaker. Results are
    numpy arrays with one row per frame, holding the same values that
    ``Maai.get_result()`` would return frame by frame.

    Example:
        >>> batch = MaaiBatch("vap", "jp", device="cuda")
        >>> out = batch.process(wav_ch1, wav_ch2)
        >>> out["p_now"].shape           # (n_frames, 2)
        >>> out["time_sec"][:3]          # [0.16, 0.24, 0.32]
        >>> outs = batch.process_many([(a1, a2), (b1, b2), (c1, c2)])

    Several models share one encoding pass when given as ``configs``; results
    are then nested by label (default: the mode name)::

        >>> batch = MaaiBatch(configs=[{"mode": "vap", "lang": "jp"},
        ...                            {"mode": "bc", "lang": "jp"}])
        >>> out = batch.process(wav_ch1, wav_ch2)
        >>> out["vap"]["p_now"], out["bc"]["p_bc"]

    Both encoders are supported: Mimi (``model_type="normal-ver2"``, the
    default, at 12.5 Hz) and CPC (``model_type="normal"``, 10 Hz by default).
    ``vap_prompt`` is not supported.
    """

    SAMPLE_RATE = 16000

    def __init__(
        self,
        mode: str | None = None,
        lang: str | None = None,
        *,
        configs: list | None = None,
        frame_rate: float | None = None,
        context_len_sec: float = 20,
        device: str = "cpu",
        model_type: str = "normal-ver2",
        mono: bool = False,
        mc: bool = True,
        local_model: str | None = None,
        return_p_bins: bool = False,
        cpc_model: str = os.path.expanduser("~/.cache/cpc/60k_epoch4-d0f474de.pt"),
        mimi_model_name: str = "kyutai/mimi",
        cache_dir: str | None = None,
        force_download: bool = False,
        batch_size: int = 8,
        encoder_chunk_frames: int = 50,
        fold_context_frames: int | None = None,
        fold_step_frames: int | None = None,
        drop_first_frame: bool | None = None,
    ):
        """Load the model(s) and the shared audio encoder.

        Args:
            mode (str | None): Model to run ('vap', 'vad', 'bc_det', 'bc',
                'bc_2type', 'nod', 'nod_timing', 'nod_para'). Give either
                ``mode`` and ``lang``, or ``configs``.
            lang (str | None): Language of the model (e.g. 'jp', 'en').
            configs (list | None): Several models sharing one encoding pass.
                Each entry is a dict with 'mode' and 'lang' and optionally
                'mc', 'local_model', 'return_p_bins' and 'label' (the result
                key; defaults to the mode).
            frame_rate (float | None): Frame rate of the models. Default: 12.5
                for Mimi (the only rate it supports) and 10 for CPC.
            context_len_sec (float): Context length of the models in seconds.
            device (str): 'cpu', 'cuda' or 'cuda:N'.
            model_type (str): 'normal-ver2' (Mimi encoder) or 'normal' (CPC).
            mono (bool): Single-channel input (only ``audio_ch1`` is passed).
                'vap', 'vad' and 'bc_det' then use their single-channel models,
                and the other models get silence on channel 2, as with
                ``MaaiInput.Zero()`` in ``Maai`` / ``MaaiMultiple``.
            mc (bool): Prefer the noise-robust 'vap' model (default for configs).
            local_model (str | None): Local checkpoint for ``mode``.
            return_p_bins (bool): Also return ``p_bins`` / ``p_bins_now`` /
                ``p_bins_future`` for 'vap' (default for configs).
            cpc_model (str): Path of the pretrained CPC weights (CPC only).
            mimi_model_name (str): Hugging Face name of the Mimi model.
            cache_dir (str | None): Cache directory for the model weights.
            force_download (bool): Force re-downloading the model weights.
            batch_size (int): Recordings encoded together, and windows per
                transformer forward. Lower it if memory runs out.
            encoder_chunk_frames (int): Frames per encoder call (4 s at 12.5
                Hz). Memory grows with it; the result does not depend on it.
            fold_context_frames (int | None): Inputs longer than
                ``fold_context_frames + fold_step_frames`` frames are split into
                overlapping windows, each preceded by this many frames of
                context. Default: the receptive field of the transformer,
                (attention layers) x (context frames - 1) = 996 frames for the
                released models at 20 s / 12.5 Hz, which makes the split exact.
                (The band mask is applied in every layer, so a frame depends on
                several context lengths of input, as with the realtime KV cache.)
            fold_step_frames (int | None): New frames per window. Default:
                twice the context length.
            drop_first_frame (bool | None): Drop the first encoder output like
                the realtime ONNX Mimi encoder does. Default: True for Mimi
                (set False to match ``Maai(..., use_mimi_onnx=False)``) and
                False for CPC, whose realtime path does not drop it.
        """
        if (mode is None) == (configs is None):
            raise ValueError("Give either `mode` and `lang`, or `configs`.")
        if mode is not None:
            if lang is None:
                raise ValueError("`lang` is required with `mode`.")
            configs = [
                {
                    "mode": mode,
                    "lang": lang,
                    "local_model": local_model,
                }
            ]
            self._nested = False
        else:
            if not configs:
                raise ValueError("`configs` must contain at least one model.")
            if local_model is not None:
                raise ValueError("With `configs`, give 'local_model' inside each config.")
            self._nested = True

        self.encoder_type = resolve_encoder_type(model_type)
        if self.encoder_type not in DEFAULT_FRAME_RATE:
            raise NotImplementedError(f"MaaiBatch does not support encoder_type '{self.encoder_type}'.")

        try:
            self.device = str(torch.device(device))
        except RuntimeError as exc:
            raise ValueError("Device must be a valid torch device string such as 'cpu', 'cuda', or 'cuda:0'.") from exc
        if not (self.device == "cpu" or self.device.startswith("cuda")):
            raise ValueError("Device must be 'cpu', 'cuda', or 'cuda:N'.")

        if frame_rate is None:
            frame_rate = DEFAULT_FRAME_RATE[self.encoder_type]
        self.frame_rate = float(frame_rate)
        self.context_len_sec = context_len_sec
        self.context_limit = int(round(float(context_len_sec) * self.frame_rate))
        self.samples_per_frame = int(round(self.SAMPLE_RATE / self.frame_rate))
        if self.encoder_type == "cpc" and self.samples_per_frame % CPC_HOP_SAMPLES:
            raise ValueError(
                f"With CPC, 16000 / frame_rate must be a multiple of {CPC_HOP_SAMPLES} samples; "
                f"got frame_rate={frame_rate}."
            )

        if drop_first_frame is None:
            drop_first_frame = self.encoder_type == "mimi"
        if drop_first_frame and self.encoder_type == "cpc":
            raise ValueError("drop_first_frame=True applies to Mimi only; the realtime CPC path does not drop a frame.")
        self.drop_first_frame = bool(drop_first_frame)

        self.mono = bool(mono)
        self.batch_size = int(batch_size)
        self.encoder_chunk_frames = int(encoder_chunk_frames)
        self.fold_step_frames = (
            2 * self.context_limit if fold_step_frames is None else int(fold_step_frames)
        )
        if self.batch_size < 1 or self.encoder_chunk_frames < 1:
            raise ValueError("batch_size and encoder_chunk_frames must be at least 1.")
        if self.fold_step_frames < 1:
            raise ValueError("fold_step_frames must be at least 1.")

        if self.encoder_type == "mimi":
            # One Mimi core shared by every model (the models keep only the
            # layers after it).
            self._core = EncoderMimi(frame_hz=self.frame_rate, mimi_model_name=mimi_model_name)
            self._core = self._core.to(self.device).eval()
            if not math.isclose(self.frame_rate, self._core.frame_hz_mimi, abs_tol=1e-6):
                raise ValueError(
                    f"MaaiBatch runs Mimi at its frame rate ({self._core.frame_hz_mimi:g} Hz); "
                    f"got frame_rate={frame_rate}."
                )

        self.models: list[_BatchModel] = []
        seen_labels: set[str] = set()
        for cfg in configs:
            if "mode" not in cfg or "lang" not in cfg:
                raise ValueError("Each entry of `configs` must contain 'mode' and 'lang'.")
            cfg_mode = cfg["mode"]
            if cfg_mode in DEPRECATED_MODE_ALIASES:
                raise ValueError(
                    f"mode='{cfg_mode}' is a deprecated alias; use mode='{DEPRECATED_MODE_ALIASES[cfg_mode][0]}' "
                    "(with mc= and mono= as needed)."
                )
            if cfg_mode == "vap_prompt":
                raise ValueError("mode='vap_prompt' is not supported by MaaiBatch.")
            label = cfg.get("label", cfg_mode)
            if label in seen_labels:
                raise ValueError(
                    f"Duplicate label '{label}'. Provide a unique 'label' field in each config."
                )
            seen_labels.add(label)

            vap, variant = _load_vap_model(
                self._make_conf(mimi_model_name),
                cfg_mode,
                cfg["lang"],
                # Only these modes have single-channel models; the others are
                # stereo models fed silence on channel 2 (same rule as Maai).
                mono=self.mono and cfg_mode in MONO_CAPABLE_MODES,
                mc=cfg.get("mc", mc),
                frame_rate=self.frame_rate,
                context_len_sec=context_len_sec,
                device=self.device,
                cpc_model=cpc_model,
                model_type=model_type,
                local_model=cfg.get("local_model"),
                cache_dir=cache_dir,
                force_download=force_download,
            )
            if not hasattr(vap, "forward_batch"):
                raise ValueError(f"mode='{cfg_mode}' (variant '{variant}') has no batch forward.")
            self.models.append(
                _BatchModel(
                    label=label,
                    mode=cfg_mode,
                    variant=variant,
                    vap=vap,
                    return_p_bins=bool(cfg.get("return_p_bins", return_p_bins)),
                )
            )

        if self.encoder_type == "cpc":
            # The CPC core (convolutions + LSTM) is the same pretrained network
            # in every model and channel; only the downsampling differs.
            self._core = self.models[0].vap.encoder1
        self._feature_dim = int(self._core.output_dim)

        if fold_context_frames is None:
            depth = max(_attention_depth(m.vap) for m in self.models)
            fold_context_frames = depth * max(0, self.context_limit - 1)
        self.fold_context_frames = int(fold_context_frames)
        if self.fold_context_frames < 0:
            raise ValueError("fold_context_frames must be >= 0.")

        # Channel 2 is encoded only when a model reads it (in mono mode it is
        # silence, as with MaaiInput.Zero).
        self._needs_ch2 = any(not m.single_tower for m in self.models)

    def _make_conf(self, mimi_model_name: str) -> VapConfig:
        conf = VapConfig()
        conf.frame_hz = self.frame_rate
        conf.context_limit = self.context_limit
        conf.encoder_type = self.encoder_type
        if self.encoder_type == "mimi":
            conf.mimi_model_name = mimi_model_name
            # The models keep only their own (frame-rate) layers; the Mimi
            # core is self._core.
            conf.mimi_use_onnx = 0
            conf.mimi_load_hf_weights = 0
        return conf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio_ch1, audio_ch2=None) -> dict:
        """Run the model(s) over one recording.

        Args:
            audio_ch1: 16 kHz mono float32 waveform of speaker 1 (np.ndarray).
            audio_ch2: Waveform of speaker 2 (omit with ``mono=True``). A
                shorter channel is padded with silence.

        Returns:
            dict: ``end_sample`` (int64) and ``time_sec`` give, per frame, the
            input position the prediction is made at (the end of the audio it
            has seen). The model outputs follow as arrays with one row per
            frame, e.g. ``p_now`` (T, 2), ``p_bc`` (T,). With ``configs`` they
            are nested under each label. An incomplete trailing frame is not
            scored, as in the realtime stream.
        """
        return self.process_many([(audio_ch1, audio_ch2)])[0]

    @torch.inference_mode()
    def process_many(self, items: list) -> list:
        """Run the model(s) over several recordings, ``batch_size`` at a time.

        Args:
            items (list): ``(audio_ch1, audio_ch2)`` tuples. With
                ``mono=True`` an item may also be a single waveform.

        Returns:
            list: One result per item, in input order (see :meth:`process`).
        """
        pairs = [self._as_pair(item) for item in items]
        n_samples = [max(len(a), 0 if b is None else len(b)) for a, b in pairs]
        # Similar lengths in one batch keep the padding small.
        order = sorted(range(len(pairs)), key=lambda i: n_samples[i], reverse=True)
        results: list = [None] * len(pairs)
        for s in range(0, len(order), self.batch_size):
            idx = order[s : s + self.batch_size]
            for i, res in zip(idx, self._process_group([pairs[i] for i in idx])):
                results[i] = res
        return results

    @staticmethod
    def to_frames(result: dict) -> list:
        """Convert a :meth:`process` result to one dict per frame.

        The dicts hold the same keys as ``Maai.get_result()`` (without ``t``,
        ``x1`` and ``x2``) plus ``end_sample`` and ``time_sec``.
        """
        flat = {k: v.tolist() for k, v in result.items() if isinstance(v, np.ndarray)}
        nested = {
            label: {k: v.tolist() for k, v in sub.items()}
            for label, sub in result.items()
            if isinstance(sub, dict)
        }
        frames = []
        for t in range(len(result["end_sample"])):
            frame = {k: v[t] for k, v in flat.items()}
            for label, sub in nested.items():
                frame[label] = {k: v[t] for k, v in sub.items()}
            frames.append(frame)
        return frames

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _as_pair(self, item) -> tuple:
        if isinstance(item, tuple):
            if len(item) == 1:
                a, b = item[0], None
            elif len(item) == 2:
                a, b = item
            else:
                raise ValueError("Each item must be (audio_ch1, audio_ch2).")
        elif self.mono:
            a, b = item, None
        else:
            raise ValueError("Each item must be an (audio_ch1, audio_ch2) tuple.")

        if self.mono and b is not None:
            raise ValueError("mono=True takes audio_ch1 only.")
        if not self.mono and b is None:
            raise ValueError("audio_ch2 is required (or create MaaiBatch with mono=True).")
        return self._as_wave(a), None if b is None else self._as_wave(b)

    @staticmethod
    def _as_wave(x) -> np.ndarray:
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError(f"Expected a 1-D 16 kHz mono waveform, got shape {x.shape}.")
        return x

    def _process_group(self, pairs: list) -> list:
        hop = self.samples_per_frame
        off = 1 if self.drop_first_frame else 0
        n_frames = [max(len(a), 0 if b is None else len(b)) // hop for a, b in pairs]
        k_max = max(n_frames)

        n_items = len(pairs)
        if k_max - off <= 0:
            outputs = {m.label: self._empty_outputs(m, n_items) for m in self.models}
        else:
            outputs = {}
            n_pad = k_max * hop
            n_ch = 2 if self._needs_ch2 else 1
            # Rows [0, n_items) are channel 1, [n_items, 2 n_items) channel 2;
            # missing audio (padding, mono channel 2) is silence.
            waves = np.zeros((n_ch * n_items, n_pad), dtype=np.float32)
            for i, (a, b) in enumerate(pairs):
                waves[i, : min(len(a), n_pad)] = a[:n_pad]
                if n_ch == 2 and b is not None:
                    waves[n_items + i, : min(len(b), n_pad)] = b[:n_pad]

            # Encoder features: one row per frame (Mimi) or CPC_HOP_SAMPLES
            # rows per frame (CPC); only Mimi drops a frame.
            feats = self._encode(waves)[:, off:]
            e_a = feats[:n_items]
            e_b = feats[n_items:] if n_ch == 2 else None
            for m in self.models:
                out = self._run_model(m, e_a, e_b, k_max - off)
                outputs[m.label] = {k: v.cpu().numpy() for k, v in out.items()}

        results = []
        for i, k in enumerate(n_frames):
            n_out = max(0, k - off)
            end_sample = (np.arange(n_out, dtype=np.int64) + 1 + off) * hop
            res = {"end_sample": end_sample, "time_sec": end_sample / float(self.SAMPLE_RATE)}
            for m in self.models:
                sub = {
                    key: val[i, :n_out].copy()
                    for key, val in outputs[m.label].items()
                    if m.return_p_bins or key not in P_BINS_KEYS
                }
                if self._nested:
                    res[m.label] = sub
                else:
                    res.update(sub)
            results.append(res)
        return results

    def _empty_outputs(self, m: _BatchModel, n_items: int) -> dict:
        """Zero-frame outputs with the right keys and shapes (too-short input)."""
        dim = self._feature_dim if m.mode == "nod_para" else m.vap.conf.dim
        p = next(m.vap.parameters())
        x = torch.zeros(1, 1, dim, device=p.device, dtype=p.dtype)
        out = m.vap.forward_batch(x) if m.single_tower else m.vap.forward_batch(x, x)
        return {
            k: np.zeros((n_items, 0) + tuple(v.shape[2:]), dtype=v.cpu().numpy().dtype)
            for k, v in out.items()
        }

    def _encode(self, waves: np.ndarray) -> torch.Tensor:
        if self.encoder_type == "cpc":
            return self._encode_cpc(waves)
        return self._encode_mimi(waves)

    def _encode_mimi(self, waves: np.ndarray) -> torch.Tensor:
        """(N, n_samples) 16 kHz -> Mimi features (N, n_frames, 512).

        Same stream as the realtime encoder (resampler, leading 24 kHz zero,
        convolution padding and transformer KV carried across calls), just in
        larger calls.
        """
        hop = self.samples_per_frame
        step = self.encoder_chunk_frames * hop
        x = torch.from_numpy(waves).unsqueeze(1)  # (N, 1, n_samples) on CPU
        self._core.reset_streaming_state()
        parts = []
        for s in range(0, x.shape[-1], step):
            chunk = x[..., s : s + step].to(self.device)
            emb, _ = self._core.forward_shared(chunk, has_overlap_context=False)
            parts.append(emb)
        feats = torch.cat(parts, dim=1)
        if feats.shape[1] != x.shape[-1] // hop:
            raise RuntimeError(
                f"Mimi emitted {feats.shape[1]} frames for {x.shape[-1] // hop} input frames."
            )
        return feats

    def _encode_cpc(self, waves: np.ndarray) -> torch.Tensor:
        """(N, n_samples) 16 kHz -> CPC features before the model-specific
        downsampling, (N, n_frames x samples_per_frame / 160, 256).

        The realtime path feeds [320 samples of overlap + one frame] per call
        (zeros before the start). The convolutions pad both ends of that
        window, and their first and last outputs, the only ones that see the
        padding, are dropped; so a window of several frames gives the same
        outputs. The LSTM state is carried across calls, as in realtime.
        """
        hop = self.samples_per_frame
        ctx = CPC_CONTEXT_SAMPLES
        n_frames = waves.shape[1] // hop
        x = np.zeros((waves.shape[0], ctx + n_frames * hop), dtype=np.float32)
        x[:, ctx:] = waves[:, : n_frames * hop]
        x = torch.from_numpy(x).unsqueeze(1)  # (N, 1, ctx + n_samples) on CPU
        dtype = next(self._core.parameters()).dtype
        self._core.reset_streaming_state()
        parts = []
        for k0 in range(0, n_frames, self.encoder_chunk_frames):
            k1 = min(n_frames, k0 + self.encoder_chunk_frames)
            chunk = x[..., k0 * hop : k1 * hop + ctx].to(device=self.device, dtype=dtype)
            parts.append(self._core.forward_shared(chunk))
        feats = torch.cat(parts, dim=1)
        if feats.shape[1] != n_frames * hop // CPC_HOP_SAMPLES:
            raise RuntimeError(
                f"CPC emitted {feats.shape[1]} frames for {n_frames} input frames "
                f"(expected {n_frames * hop // CPC_HOP_SAMPLES})."
            )
        return feats

    def _run_model(
        self, m: _BatchModel, e_a: torch.Tensor, e_b: torch.Tensor | None, n_frames: int
    ) -> dict:
        e1, e2 = (e_b, e_a) if m.swap else (e_a, e_b)
        if m.single_tower:
            e2 = None
        if self.encoder_type == "cpc":
            # The model's downsampling runs on each realtime window separately.
            e1 = _per_window(m.vap.encoder1.forward_specific, e1, n_frames)
            if e2 is not None:
                e2 = _per_window(m.vap.encoder2.forward_specific, e2, n_frames)
        # Most models project the encoder features in encode_audio; nod_para
        # does it inside its forward (same rule as MaaiMultiple).
        if m.mode != "nod_para" and hasattr(m.vap, "decrease_dimension"):
            e1 = torch.relu(m.vap.decrease_dimension(e1))
            if e2 is not None:
                e2 = torch.relu(m.vap.decrease_dimension(e2))

        out = self._forward_folded(m.vap, e1, e2)
        per_frame = e1.shape[1] // n_frames
        if per_frame > 1:
            # Some CPC models emit several transformer frames per window (e.g.
            # kernel 5 / stride 2 over 10 CPC frames); realtime adds them all to
            # the context but reports only the last one.
            out = {k: v[:, per_frame - 1 :: per_frame] for k, v in out.items()}
        return out

    def _forward_folded(self, vap: nn.Module, x1: torch.Tensor, x2: torch.Tensor | None) -> dict:
        """``vap.forward_batch`` over (B, T, dim), folding long T into windows.

        Window w covers frames [start, end) and contributes [take_from, end):
        the first window is taken whole, the others take their last
        ``fold_step_frames`` frames, each preceded by at least
        ``fold_context_frames`` frames of context.
        """
        n_items, t_len = x1.shape[0], x1.shape[1]
        span = self.fold_context_frames + self.fold_step_frames
        if t_len <= span:
            windows = [(0, 0, t_len)]
        else:
            windows = [(0, 0, span)]
            pos = span
            while pos < t_len:
                end = min(pos + self.fold_step_frames, t_len)
                windows.append((end - span, pos, end))
                pos = end
        length = min(t_len, span)

        per_call = max(1, self.batch_size // n_items)
        pieces = []
        for w0 in range(0, len(windows), per_call):
            group = windows[w0 : w0 + per_call]
            c1 = torch.cat([x1[:, s : s + length] for s, _, _ in group], dim=0)
            if x2 is None:
                out = vap.forward_batch(c1)
            else:
                c2 = torch.cat([x2[:, s : s + length] for s, _, _ in group], dim=0)
                out = vap.forward_batch(c1, c2)
            for j, (s, lo, hi) in enumerate(group):
                pieces.append(
                    {k: v[j * n_items : (j + 1) * n_items, lo - s : hi - s] for k, v in out.items()}
                )
        return {k: torch.cat([p[k] for p in pieces], dim=1) for k in pieces[0]}
