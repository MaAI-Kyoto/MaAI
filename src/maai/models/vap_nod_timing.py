import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import torch.nn.functional as F

from .config import VapConfig
from ..encoder import build_audio_encoder
from ..modules import GPT, GPTStereo
from ..objective import ObjectiveVAP


class VapGPT_nod_timing(nn.Module):
    """Voice Activity Projection with Timing-only nodding prediction.

    Same backbone naming as VapGPT_nod (self_attention / cross_attention /
    va_classifier / vap_head / bc_head), but with a single-output gt_head
    (merged nod occurrence, sigmoid) instead of VapGPT_nod's 4-way softmax
    gt_head. This matches the Timing-stage nod_head convention used by
    VAP_Nodding_para's own model_nod_para.VapGPT (nn.Linear(dim, 1)), and the
    checkpoints produced by VAP_Nodding_old/train.train_VAP_MT (mt / gt_head).
    """

    def __init__(self, conf: Optional[VapConfig] = None):
        super().__init__()
        if conf is None:
            conf = VapConfig()
        self.conf = conf
        self.sample_rate = conf.sample_rate
        self.frame_hz = conf.frame_hz

        self.temp_elapse_time = []

        self.self_attention = GPT(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.channel_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
        )

        self.cross_attention = GPTStereo(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.cross_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
        )

        self.objective = ObjectiveVAP(bin_times=conf.bin_times, frame_hz=conf.frame_hz)

        self.va_classifier = nn.Linear(conf.dim, 1)
        self.vap_head = nn.Linear(conf.dim, self.objective.n_classes)

        # Timing-only: single merged nod-occurrence head (sigmoid), not the
        # 4-way softmax [none, short, long, long_p] used by VapGPT_nod.
        self.gt_head = nn.Linear(conf.dim, 1)
        self.bc_head = nn.Linear(conf.dim, 1)

    def load_encoder(self, cpc_model):
        self.encoder1 = build_audio_encoder(self.conf, cpc_model=cpc_model)
        self.encoder1 = self.encoder1.eval()
        self.encoder2 = build_audio_encoder(self.conf, cpc_model=cpc_model)
        self.encoder2 = self.encoder2.eval()

        encoder_dim = getattr(self.encoder1, "output_dim", self.conf.dim)
        if encoder_dim != self.conf.dim:
            self.decrease_dimension = nn.Linear(encoder_dim, self.conf.dim)

        if self.conf.freeze_encoder == 1:
            print('freeze encoder')
            self.encoder1.freeze()
            self.encoder2.freeze()

    @property
    def horizon_time(self):
        return self.objective.horizon_time

    def encode_audio(self, audio1: torch.Tensor, audio2: torch.Tensor) -> Tuple[Tensor, Tensor]:
        x1 = self.encoder1(audio2)
        x2 = self.encoder2(audio1)

        if hasattr(self, "decrease_dimension"):
            x1 = torch.relu(self.decrease_dimension(x1))
            x2 = torch.relu(self.decrease_dimension(x2))

        return x1, x2

    def vad_loss(self, vad_output, vad):
        return F.binary_cross_entropy_with_logits(vad_output, vad)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        cache: Optional[dict] = None,
        return_all_frames: bool = False,
    ) -> Tuple[dict, dict]:
        """Same calling convention as VapGPT_nod.forward (production/streaming
        interface): pre-encoded x1/x2, optional past-kv cache dict, returns
        (frame(s), new_cache).
        """
        if cache is None:
            cache = {}

        o1 = self.self_attention(x1, past_kv=cache.get("ar1"))
        o2 = self.self_attention(x2, past_kv=cache.get("ar2"))
        out = self.cross_attention(
            o1["x"],
            o2["x"],
            past_kv1=cache.get("cross1"),
            past_kv2=cache.get("cross2"),
            past_kv1_c=cache.get("cross1_c"),
            past_kv2_c=cache.get("cross2_c"),
        )

        new_cache = {
            "ar1": (o1["past_k"], o1["past_v"]),
            "ar2": (o2["past_k"], o2["past_v"]),
            "cross1": (out["past_k1"], out["past_v1"]),
            "cross2": (out["past_k2"], out["past_v2"]),
            "cross1_c": (out["past_k1_c"], out["past_v1_c"]),
            "cross2_c": (out["past_k2_c"], out["past_v2_c"]),
        }

        p_bc = self.bc_head(out["x"])
        p_nod = self.gt_head(out["x"])

        p_bc_all = p_bc.sigmoid().to("cpu").tolist()[0]
        p_nod_all = p_nod.sigmoid().to("cpu").tolist()[0]

        frames = [
            {
                "p_bc": p_bc_all[t][0],
                "p_nod": p_nod_all[t][0],
            }
            for t in range(x1.shape[1])
        ]
        ret = frames if return_all_frames else frames[-1]

        return ret, new_cache

    def forward_batch(self, x1: Tensor, x2: Tensor) -> dict:
        """Non-streaming, full-context forward (no cache), for validating
        streaming-vs-batch numerical equivalence. Equivalent to calling
        forward() with cache=None and return_all_frames=True, but without
        threading a cache dict through (single shot over the whole sequence).
        """
        o1 = self.self_attention(x1)
        o2 = self.self_attention(x2)
        out = self.cross_attention(o1["x"], o2["x"])

        p_bc = self.bc_head(out["x"]).sigmoid()
        p_nod = self.gt_head(out["x"]).sigmoid()

        return {"p_bc": p_bc, "p_nod": p_nod}
