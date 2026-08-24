from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from .config import VapConfig
from ..encoder import build_audio_encoder
from ..modules import GPT, GPTStereo, FiLM, CondPreNorm
from ..objective import ObjectiveVAP

# 推論固定トポロジ（grid_config_12.5hz_taskGPT_mimi_realtime 相当）
_NOD_PARA_ACTIVE_PARAM_TASKS = frozenset({"repetitions", "range", "speed", "swing_binary"})
_GPT_OUTPUT_DROPOUT = 0.2
_TASK_GPT_OUTPUT_DROPOUT = 0.0
_NOD_HEAD_DROPOUT = 0.3


def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_layers: int,
    dropout: float,
) -> nn.Module:
    """Build a Multi-Layer Perceptron (MLP) module.
    
    Args:
        in_dim (int): Input dimension.
        hidden_dim (int): Hidden layer dimension.
        out_dim (int): Output dimension.
        n_layers (int): Number of layers in the MLP.
        dropout (float): Dropout probability.
        
    Returns:
        nn.Module: The constructed MLP module.
    """
    layers: List[nn.Module] = [
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    ]
    for _ in range(n_layers - 1):
        layers += [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class VapGPT_nod_para(nn.Module):
    """Streaming (KV-cached) nod_para model with optional FiLM/AdaLN-Zero
    listener-style conditioning.

    FiLM support mirrors the research VAP-Nodding repo so exported
    ``film_inject_*`` checkpoints load 1:1. When ``conf.nod_film_enable == 0``
    (default) the model is byte-identical to the original non-film nod_para.

    The listener-style conditioning is a fixed per-session vector set via
    :meth:`set_listener_style`. It is z-scored with the checkpoint's train
    stats (``film_cond_mean``/``film_cond_std`` buffers) then passed through
    ``cond_pre_norm`` and distributed to the AdaLN blocks / head FiLMs.
    """

    BINS_P_NOW: List[int] = [0, 1]
    BINS_PFUTURE: List[int] = [2, 3]

    def __init__(self, conf: Optional[VapConfig] = None):
        """Initialize the VapGPT_nod_para model.
        
        Args:
            conf (Optional[VapConfig]): Configuration object.
                If None, default VapConfig is used.
        """
        super().__init__()
        if conf is None:
            conf = VapConfig()
        self.conf = conf
        self.sample_rate = conf.sample_rate
        self.frame_hz = conf.frame_hz

        # range/speed の z-score 逆変換（学習時 stats。必要なら Maai 側で上書き）
        self.nod_param_stats: Dict[str, float] = {
            "range_mean": 0.0,
            "range_std": 1.0,
            "speed_mean": 0.0,
            "speed_std": 1.0,
        }
        # 閾値探索で得た推論時しきい値（.pt に含まれる場合は外部から上書き）
        self.nod_repetitions_thresholds: Dict[str, float] = {"t0": 1.0, "t1": 1.0, "t2": 1.0}
        self.nod_swing_up_threshold: float = 0.5
        # 頷きタイミング閾値（.pt の t_nod。下流のタイミング判定用に保持）
        self.nod_timing_threshold: float = 0.5

        # 回数予測の出力形式: 1=binary(1回 vs 2回以上), 0=3クラス
        self.nod_count_binary = int(getattr(conf, "nod_count_binary", 0)) == 1

        # ---- FiLM (listener-style) conditioning setup ----
        self.use_film = int(getattr(conf, "nod_film_enable", 0)) == 1
        self.film_target_ar_channel = int(getattr(conf, "nod_film_target_ar_channel", 0)) == 1
        self.film_target_ar_channel_x2 = int(getattr(conf, "nod_film_target_ar_channel_x2", 0)) == 1
        self.film_target_ar_cross = int(getattr(conf, "nod_film_target_ar_cross", 0)) == 1
        self.film_target_ar_cross_x2 = int(getattr(conf, "nod_film_target_ar_cross_x2", 0)) == 1
        self.film_target_heads = int(getattr(conf, "nod_film_target_heads", 0)) == 1
        self.film_heads_scope = str(getattr(conf, "nod_film_heads_scope", "all")).lower()
        self.film_cond_dim = int(getattr(conf, "nod_film_cond_dim", 0)) if self.use_film else 0
        self.film_hidden = int(getattr(conf, "nod_film_hidden", 0))
        self.film_block_style = str(getattr(conf, "nod_film_block_style", "post_ffn"))
        self.film_cond_zscore = (
            int(getattr(conf, "nod_film_cond_zscore", 0)) == 1
            and self.use_film
            and self.film_cond_dim > 0
        )
        if self.film_cond_zscore:
            self.register_buffer(
                "film_cond_mean", torch.zeros(self.film_cond_dim), persistent=True
            )
            self.register_buffer(
                "film_cond_std", torch.ones(self.film_cond_dim), persistent=True
            )
        if self.use_film and int(getattr(conf, "nod_film_cond_norm", 0)) == 1 and self.film_cond_dim > 0:
            self.cond_pre_norm = CondPreNorm(self.film_cond_dim)
        else:
            self.cond_pre_norm = None

        # Fixed per-session listener-style vector (raw, pre z-score). None = off.
        self.register_buffer(
            "listener_style", torch.zeros(self.film_cond_dim) if self.film_cond_dim > 0 else torch.zeros(1),
            persistent=False,
        )
        self._listener_style_set = False

        _ar_channel_film = self.film_block_style if (self.use_film and self.film_target_ar_channel) else "none"
        _ar_cross_film = self.film_block_style if (self.use_film and self.film_target_ar_cross) else "none"
        _cond_dim_layers = self.film_cond_dim if self.use_film else 0

        self.objective = ObjectiveVAP(bin_times=conf.bin_times, frame_hz=conf.frame_hz)

        self.ar_channel = GPT(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.channel_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
            film_mode=_ar_channel_film,
            cond_dim=_cond_dim_layers,
            film_hidden=self.film_hidden,
        )
        self.ar = GPTStereo(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.cross_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
            film_mode=_ar_cross_film,
            cond_dim=_cond_dim_layers,
            film_hidden=self.film_hidden,
        )

        self.gpt_output_dropout = (
            nn.Dropout(_GPT_OUTPUT_DROPOUT) if _GPT_OUTPUT_DROPOUT > 0 else None
        )
        self.task_gpt_output_dropout = (
            nn.Dropout(_TASK_GPT_OUTPUT_DROPOUT) if _TASK_GPT_OUTPUT_DROPOUT > 0 else None
        )

        # param 側 A: timing(ar_channel) と同位置の FiLM/AdaLN を独立に持つ
        self.ar_channel_param = GPT(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.channel_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
            film_mode=_ar_channel_film,
            cond_dim=_cond_dim_layers,
            film_hidden=self.film_hidden,
        )
        # param 側 B: timing(ar) と同位置の FiLM/AdaLN を独立に持つ
        self.task_gpts = nn.ModuleDict(
            {
                "0": GPTStereo(
                    dim=conf.dim,
                    dff_k=3,
                    num_layers=conf.nod_task_gpt_layers,
                    num_heads=conf.num_heads,
                    dropout=conf.dropout,
                    context_limit=conf.context_limit,
                    film_mode=_ar_cross_film,
                    cond_dim=_cond_dim_layers,
                    film_hidden=self.film_hidden,
                )
            }
        )
        self.task_gpt_group_map = {
            "repetitions": "0",
            "range": "0",
            "speed": "0",
            "swing_binary": "0",
            "swing_value": "0",
        }

        nod_param_input_dim = conf.dim
        base_head_input_dim = nod_param_input_dim
        head_do = _NOD_HEAD_DROPOUT

        is_baseline = conf.cross_layers == 0 and conf.channel_layers == 0
        timing_in = conf.dim * 2 if is_baseline else conf.dim
        # タイミング系は全 Linear（timing_head_mlp_* なしの固定構成）
        self.va_classifier = nn.Linear(timing_in, 1)
        self.vap_head = nn.Linear(timing_in, self.objective.n_classes)
        self.bc_head = nn.Linear(timing_in, 1)
        self.bc_detect_head = nn.Linear(timing_in, 1)
        self.nod_head = nn.Linear(timing_in, 1)

        mlp_h = conf.nod_head_mlp_hidden
        active = _NOD_PARA_ACTIVE_PARAM_TASKS

        def _para_head(mlp_n: int, in_d: int, out_d: int):
            if mlp_n > 0:
                return _build_mlp(in_d, mlp_h, out_d, mlp_n, head_do)
            return nn.Linear(in_d, out_d)

        nod_repetitions_out = 1 if self.nod_count_binary else 3
        in_d = base_head_input_dim
        self.nod_repetitions_head = (
            _para_head(conf.nod_head_mlp_repetitions, in_d, nod_repetitions_out)
            if "repetitions" in active
            else None
        )
        self.nod_range_head = (
            _para_head(conf.nod_head_mlp_range, in_d, 1) if "range" in active else None
        )
        self.nod_speed_head = (
            _para_head(conf.nod_head_mlp_speed, in_d, 1) if "speed" in active else None
        )
        self.nod_swing_up_binary_head = (
            _para_head(conf.nod_head_mlp_swing_binary, in_d, 1)
            if "swing_binary" in active
            else None
        )
        self.nod_swing_up_value_head = None
        self.nod_swing_up_continuous_head = None

        # ---- C: 各 nod ヘッド直前 FiLM ----
        self.film_heads = nn.ModuleDict()
        if self.use_film and self.film_target_heads and self.film_cond_dim > 0:
            scope = self.film_heads_scope
            timing_targets = {"nod_timing"} if scope in ("all", "timing") else set()
            params_targets = (
                {"count", "range", "speed", "swing_binary"}
                if scope in ("all", "params") else set()
            )
            if "nod_timing" in timing_targets and self.nod_head is not None:
                self.film_heads["nod_timing"] = FiLM(self.film_cond_dim, timing_in, hidden=self.film_hidden)
            head_in_map = {
                "count": (in_d, self.nod_repetitions_head),
                "range": (in_d, self.nod_range_head),
                "speed": (in_d, self.nod_speed_head),
                "swing_binary": (in_d, self.nod_swing_up_binary_head),
            }
            for task_name, (feat_in, head_mod) in head_in_map.items():
                if task_name in params_targets and head_mod is not None:
                    self.film_heads[task_name] = FiLM(self.film_cond_dim, feat_in, hidden=self.film_hidden)

        self.encoder1 = None
        self.encoder2 = None
        self.decrease_dimension = None
        self.decrease_dimension_param = None

    def load_encoder(self, cpc_model: str) -> None:
        """Load and build the audio encoders for both speakers.
        
        Args:
            cpc_model (str): Pre-trained CPC model name or path.
        """
        self.encoder1 = build_audio_encoder(self.conf, cpc_model=cpc_model)
        self.encoder1 = self.encoder1.eval()
        self.encoder2 = build_audio_encoder(self.conf, cpc_model=cpc_model)
        self.encoder2 = self.encoder2.eval()

        if self.encoder1.output_dim != self.conf.dim:
            self.decrease_dimension = nn.Linear(self.encoder1.output_dim, self.conf.dim)
            self.decrease_dimension_param = nn.Linear(
                self.encoder1.output_dim, self.conf.dim
            )
        else:
            self.decrease_dimension = nn.Identity()
            self.decrease_dimension_param = nn.Identity()

        if self.conf.freeze_encoder == 1:
            self.encoder1.freeze()
            self.encoder2.freeze()

    def set_listener_style(self, vec) -> None:
        """Set the fixed per-session listener-style conditioning vector (raw,
        pre z-score). Pass ``None`` to disable conditioning for this session."""
        if not self.use_film or self.film_cond_dim <= 0:
            self._listener_style_set = False
            return
        if vec is None:
            self._listener_style_set = False
            return
        t = torch.as_tensor(vec, dtype=torch.float32).view(-1)
        if t.numel() != self.film_cond_dim:
            raise ValueError(
                f"listener_style must have {self.film_cond_dim} elements, got {t.numel()}"
            )
        self.listener_style = t.to(self.listener_style.device)
        self._listener_style_set = True

    def _build_cond(self, device, dtype) -> Optional[Tensor]:
        """Return the (1, cond_dim) conditioning vector after z-score +
        cond_pre_norm, or None when conditioning is inactive."""
        if not self.use_film or not self._listener_style_set or self.film_cond_dim <= 0:
            return None
        ls = self.listener_style.to(device=device, dtype=dtype).view(1, -1)
        if self.film_cond_zscore:
            mean = self.film_cond_mean.to(device=device, dtype=dtype).view(1, -1)
            std = self.film_cond_std.to(device=device, dtype=dtype).view(1, -1)
            ls = (ls - mean) / torch.clamp(std, min=1e-8)
        if self.cond_pre_norm is not None:
            ls = self.cond_pre_norm(ls)
        return ls

    @property
    def horizon_time(self):
        """Get the horizon time for the projection in seconds.
        
        Returns:
            float: Horizon time for the objective.
        """
        return self.objective.horizon_time

    @staticmethod
    def denormalize(value: float, mean: float, std: float) -> float:
        """Denormalize a z-score value using mean and standard deviation.
        
        Args:
            value (float): The normalized z-score value.
            mean (float): The mean used for normalization.
            std (float): The standard deviation used for normalization.
            
        Returns:
            float: The denormalized original value.
        """
        return value * std + mean

    @staticmethod
    def _apply_repetitions_thresholds(prob: List[float], thresholds: Dict[str, float]) -> int:
        """Apply thresholds to repetition probabilities to determine the predicted class.
        
        Args:
            prob (List[float]): List of probabilities for each repetition class.
            thresholds (Dict[str, float]): Dictionary containing threshold values ('t0', 't1', 't2').
            
        Returns:
            int: The index of the predicted repetition class.
        """
        t = torch.tensor(
            [
                float(thresholds.get("t0", 1.0)),
                float(thresholds.get("t1", 1.0)),
                float(thresholds.get("t2", 1.0)),
            ],
            dtype=torch.float32,
        )
        p = torch.tensor(prob, dtype=torch.float32)
        eps = 1e-8
        ratio = p / torch.clamp(t, min=eps)
        return int(torch.argmax(ratio).item())

    def encode_audio(self, audio1: torch.Tensor, audio2: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """Encode the raw audio inputs into feature representations.
        
        Note: Channel swap is applied for temporal consistency.
        
        Args:
            audio1 (torch.Tensor): Audio waveform for speaker 1 (User).
            audio2 (torch.Tensor): Audio waveform for speaker 2 (System).
            
        Returns:
            Tuple[Tensor, Tensor]: Encoded features for the two speakers.
        """
        
        # Channel swap for temporal consistency
        x1 = self.encoder1(audio2)  # speaker 1 (User)
        x2 = self.encoder2(audio1)  # speaker 2 (System)

        return x1, x2

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        cache: Optional[dict] = None,
        return_all_frames: bool = False,
    ) -> Tuple[dict, dict]:
        """Forward pass for the VapGPT_nod_para model.
        
        Args:
            x1 (Tensor): Input audio embedded tensor for speaker 1.
            x2 (Tensor): Input audio embedded tensor for speaker 2.
            cache (dict, optional): Cache of past keys/values.
            
        Returns:
            Tuple[dict, dict]: Model outputs (predictions for VAD, backchannels, and nodding parameters) 
                               and the updated cache.
        """
        if cache is None:
            cache = {}

        if self.decrease_dimension is None:
            raise RuntimeError("Call load_encoder before forward.")

        # ---- listener-style conditioning ----
        cond = self._build_cond(x1.device, x1.dtype)
        cond_ar_channel = cond if self.film_target_ar_channel else None
        cond_ar_channel_x2 = cond if (self.film_target_ar_channel and self.film_target_ar_channel_x2) else None
        cond_ar_cross = cond if self.film_target_ar_cross else None
        cond_ar_cross_x2 = cond if (self.film_target_ar_cross and self.film_target_ar_cross_x2) else None
        cond_heads = cond if self.film_target_heads else None

        x1_raw, x2_raw = x1, x2
        x1 = torch.relu(self.decrease_dimension(x1_raw))
        x2 = torch.relu(self.decrease_dimension(x2_raw))
        task_x1 = torch.relu(self.decrease_dimension_param(x1_raw))
        task_x2 = torch.relu(self.decrease_dimension_param(x2_raw))

        o1 = self.ar_channel(x1, past_kv=cache.get("ar1"), cond=cond_ar_channel)
        o2 = self.ar_channel(x2, past_kv=cache.get("ar2"), cond=cond_ar_channel_x2)
        out = self.ar(
            o1["x"],
            o2["x"],
            past_kv1=cache.get("cross1"),
            past_kv2=cache.get("cross2"),
            past_kv1_c=cache.get("cross1_c"),
            past_kv2_c=cache.get("cross2_c"),
            cond_x1=cond_ar_cross,
            cond_x2=cond_ar_cross_x2,
        )

        cross_out = out["x"]
        if self.gpt_output_dropout is not None:
            cross_out = self.gpt_output_dropout(cross_out)

        v1 = self.va_classifier(out["x1"])
        v2 = self.va_classifier(out["x2"])
        logits = self.vap_head(cross_out)
        bc = self.bc_head(cross_out)
        # C: nod_head（タイミング）直前 FiLM
        _nod_in = cross_out
        if cond_heads is not None and "nod_timing" in self.film_heads:
            _nod_in = self.film_heads["nod_timing"](_nod_in, cond_heads)
        nod_t = self.nod_head(_nod_in)

        p1 = self.ar_channel_param(task_x1, past_kv=cache.get("arp1"), cond=cond_ar_channel)
        p2 = self.ar_channel_param(task_x2, past_kv=cache.get("arp2"), cond=cond_ar_channel_x2)
        tg = self.task_gpts["0"](
            p1["x"],
            p2["x"],
            past_kv1=cache.get("tg_pkv1"),
            past_kv2=cache.get("tg_pkv2"),
            past_kv1_c=cache.get("tg_pkv1c"),
            past_kv2_c=cache.get("tg_pkv2c"),
            cond_x1=cond_ar_cross,
            cond_x2=cond_ar_cross_x2,
        )
        tg_x = tg["x"]
        if self.task_gpt_output_dropout is not None:
            tg_x = self.task_gpt_output_dropout(tg_x)

        nod_param_bases = {t: tg_x for t in self.task_gpt_group_map}

        def _sel(task: str) -> Tensor:
            return nod_param_bases[task]

        nod_repetitions_out_dim = 1 if self.nod_count_binary else 3

        def _head_or_zeros(module: Optional[nn.Module], task: str, film_key: str, out_dim: int) -> Tensor:
            xb = _sel(task)
            if cond_heads is not None and film_key in self.film_heads:
                xb = self.film_heads[film_key](xb, cond_heads)
            if module is None:
                return xb.new_zeros(*xb.shape[:-1], out_dim)
            return module(xb)

        nod_rep_logits = _head_or_zeros(
            self.nod_repetitions_head, "repetitions", "count", nod_repetitions_out_dim
        )
        nod_range = _head_or_zeros(self.nod_range_head, "range", "range", 1)
        nod_speed = _head_or_zeros(self.nod_speed_head, "speed", "speed", 1)
        nod_swing_bin = _head_or_zeros(
            self.nod_swing_up_binary_head, "swing_binary", "swing_binary", 1
        )

        new_cache = {
            "ar1": (o1["past_k"], o1["past_v"]),
            "ar2": (o2["past_k"], o2["past_v"]),
            "cross1": (out["past_k1"], out["past_v1"]),
            "cross2": (out["past_k2"], out["past_v2"]),
            "cross1_c": (out["past_k1_c"], out["past_v1_c"]),
            "cross2_c": (out["past_k2_c"], out["past_v2_c"]),
            "arp1": (p1["past_k"], p1["past_v"]),
            "arp2": (p2["past_k"], p2["past_v"]),
            "tg_pkv1": (tg["past_k1"], tg["past_v1"]),
            "tg_pkv2": (tg["past_k2"], tg["past_v2"]),
            "tg_pkv1c": (tg["past_k1_c"], tg["past_v1_c"]),
            "tg_pkv2c": (tg["past_k2_c"], tg["past_v2_c"]),
        }

        probs = logits.softmax(dim=-1)
        p_now = self.objective.probs_next_speaker_aggregate(
            probs, from_bin=self.BINS_P_NOW[0], to_bin=self.BINS_P_NOW[-1]
        )
        p_future = self.objective.probs_next_speaker_aggregate(
            probs, from_bin=self.BINS_PFUTURE[0], to_bin=self.BINS_PFUTURE[1]
        )
        p_now_all = p_now.to("cpu").tolist()[0]
        p_future_all = p_future.to("cpu").tolist()[0]
        vad1_all = v1.sigmoid().to("cpu").tolist()[0]
        vad2_all = v2.sigmoid().to("cpu").tolist()[0]
        p_bc_all = bc.sigmoid().to("cpu").tolist()[0]
        p_nod_all = nod_t.sigmoid().to("cpu").tolist()[0]

        if self.nod_count_binary:
            # binary: sigmoid(logit) = P(2回以上)。[P(1回), P(2回以上)] として返す。
            p_multi_all = nod_rep_logits.sigmoid().to("cpu").tolist()[0]
        else:
            nod_repetitions_all = nod_rep_logits.softmax(dim=-1).to("cpu").tolist()[0]

        st = self.nod_param_stats
        nod_range_all = nod_range.to("cpu").tolist()[0]
        nod_speed_all = nod_speed.to("cpu").tolist()[0]
        nod_swing_up_all = nod_swing_bin.sigmoid().to("cpu").tolist()[0]

        def _frame(t: int) -> dict:
            if self.nod_count_binary:
                p_multi = float(p_multi_all[t][0])
                nod_repetitions = [1.0 - p_multi, p_multi]
                nod_repetitions_pred = int(
                    p_multi >= float(self.nod_repetitions_thresholds.get("t0", 0.5))
                )
            else:
                nod_repetitions = [float(value) for value in nod_repetitions_all[t]]
                nod_repetitions_pred = self._apply_repetitions_thresholds(
                    nod_repetitions, self.nod_repetitions_thresholds
                )
            nod_swing_up_prob = float(nod_swing_up_all[t][0])
            return {
                "p_now": [p_now_all[t][1], p_now_all[t][0]],
                "p_future": [p_future_all[t][1], p_future_all[t][0]],
                "vad": [float(vad2_all[t][0]), float(vad1_all[t][0])],
                "p_bc": float(p_bc_all[t][0]),
                "p_nod": float(p_nod_all[t][0]),
                "nod_repetitions": nod_repetitions,
                "nod_repetitions_pred": nod_repetitions_pred,
                "nod_range": self.denormalize(
                    float(nod_range_all[t][0]),
                    float(st.get("range_mean", 0.0)),
                    float(st.get("range_std", 1.0)),
                ),
                "nod_speed": self.denormalize(
                    float(nod_speed_all[t][0]),
                    float(st.get("speed_mean", 0.0)),
                    float(st.get("speed_std", 1.0)),
                ),
                "nod_swing_up": nod_swing_up_prob,
                "nod_swing_up_pred": int(
                    nod_swing_up_prob >= float(self.nod_swing_up_threshold)
                ),
            }

        frames = [_frame(t) for t in range(x1.shape[1])]
        ret = frames if return_all_frames else frames[-1]

        return ret, new_cache
