from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from .config import VapConfig
from ..encoder import build_audio_encoder
from ..modules import GPT, GPTStereo
from ..objective import ObjectiveVAP

# 推論固定トポロジ（grid_config_12.5hz_taskGPT_mimi_realtime 相当）
_NOD_PARA_ACTIVE_PARAM_TASKS = frozenset({"repetitions", "range", "speed", "swing_binary"})
_NOD_REPETITIONS_BINARY = 0  # 0=3-class repetitions, 1=binary
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
    """Voice Activity Projection with parameterized Nodding prediction.
    
    This model predicts detailed parameters for nodding behaviors such as
    repetitions, range, speed, and swing directions, in addition to standard
    voice activity and backchannels.
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

        self.objective = ObjectiveVAP(bin_times=conf.bin_times, frame_hz=conf.frame_hz)

        self.ar_channel = GPT(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.channel_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
        )
        self.ar = GPTStereo(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.cross_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
        )

        self.gpt_output_dropout = (
            nn.Dropout(_GPT_OUTPUT_DROPOUT) if _GPT_OUTPUT_DROPOUT > 0 else None
        )
        self.task_gpt_output_dropout = (
            nn.Dropout(_TASK_GPT_OUTPUT_DROPOUT) if _TASK_GPT_OUTPUT_DROPOUT > 0 else None
        )

        self.ar_channel_param = GPT(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.channel_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
        )
        self.task_gpts = nn.ModuleDict(
            {
                "0": GPTStereo(
                    dim=conf.dim,
                    dff_k=3,
                    num_layers=conf.nod_task_gpt_layers,
                    num_heads=conf.num_heads,
                    dropout=conf.dropout,
                    context_limit=conf.context_limit,
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

        nod_repetitions_out = 1 if _NOD_REPETITIONS_BINARY == 1 else 3
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

        x1_raw, x2_raw = x1, x2
        x1 = torch.relu(self.decrease_dimension(x1_raw))
        x2 = torch.relu(self.decrease_dimension(x2_raw))
        task_x1 = torch.relu(self.decrease_dimension_param(x1_raw))
        task_x2 = torch.relu(self.decrease_dimension_param(x2_raw))

        o1 = self.ar_channel(x1, past_kv=cache.get("ar1"))
        o2 = self.ar_channel(x2, past_kv=cache.get("ar2"))
        out = self.ar(
            o1["x"],
            o2["x"],
            past_kv1=cache.get("cross1"),
            past_kv2=cache.get("cross2"),
            past_kv1_c=cache.get("cross1_c"),
            past_kv2_c=cache.get("cross2_c"),
        )

        cross_out = out["x"]
        if self.gpt_output_dropout is not None:
            cross_out = self.gpt_output_dropout(cross_out)

        v1 = self.va_classifier(out["x1"])
        v2 = self.va_classifier(out["x2"])
        vad = torch.cat((v1, v2), dim=-1)
        logits = self.vap_head(cross_out)
        bc = self.bc_head(cross_out)
        bc_detect = self.bc_detect_head(cross_out)
        nod_t = self.nod_head(cross_out)

        p1 = self.ar_channel_param(task_x1, past_kv=cache.get("arp1"))
        p2 = self.ar_channel_param(task_x2, past_kv=cache.get("arp2"))
        tg = self.task_gpts["0"](
            p1["x"],
            p2["x"],
            past_kv1=cache.get("tg_pkv1"),
            past_kv2=cache.get("tg_pkv2"),
            past_kv1_c=cache.get("tg_pkv1c"),
            past_kv2_c=cache.get("tg_pkv2c"),
        )
        tg_x = tg["x"]
        if self.task_gpt_output_dropout is not None:
            tg_x = self.task_gpt_output_dropout(tg_x)

        nod_param_bases = {t: tg_x for t in self.task_gpt_group_map}

        def _sel(task: str) -> Tensor:
            return nod_param_bases[task]

        nod_repetitions_out_dim = 1 if _NOD_REPETITIONS_BINARY == 1 else 3

        def _head_or_zeros(module: Optional[nn.Module], task: str, out_dim: int) -> Tensor:
            xb = _sel(task)
            if module is None:
                return xb.new_zeros(*xb.shape[:-1], out_dim)
            return module(xb)

        nod_rep_logits = _head_or_zeros(
            self.nod_repetitions_head, "repetitions", nod_repetitions_out_dim
        )
        nod_range = _head_or_zeros(self.nod_range_head, "range", 1)
        nod_speed = _head_or_zeros(self.nod_speed_head, "speed", 1)
        nod_swing_bin = _head_or_zeros(
            self.nod_swing_up_binary_head, "swing_binary", 1
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
        nod_repetitions_all = nod_rep_logits.softmax(dim=-1).to("cpu").tolist()[0]

        st = self.nod_param_stats
        nod_range_all = nod_range.to("cpu").tolist()[0]
        nod_speed_all = nod_speed.to("cpu").tolist()[0]
        nod_swing_up_all = nod_swing_bin.sigmoid().to("cpu").tolist()[0]

        def _frame(t: int) -> dict:
            nod_repetitions = [float(value) for value in nod_repetitions_all[t]]
            nod_swing_up_prob = float(nod_swing_up_all[t][0])
            return {
                "p_now": [p_now_all[t][1], p_now_all[t][0]],
                "p_future": [p_future_all[t][1], p_future_all[t][0]],
                "vad": [float(vad2_all[t][0]), float(vad1_all[t][0])],
                "p_bc": float(p_bc_all[t][0]),
                "p_nod": float(p_nod_all[t][0]),
                "nod_repetitions": nod_repetitions,
                "nod_repetitions_pred": self._apply_repetitions_thresholds(
                    nod_repetitions, self.nod_repetitions_thresholds
                ),
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
