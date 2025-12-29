"""VAP model variant for backchannel prediction."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import torch.nn.functional as F

from .config import VapConfig
from ..encoder import EncoderCPC
from ..modules import GPT, GPTStereo
from ..objective import ObjectiveVAP

class VapGPT_bc(nn.Module):
    """GPT-based VAP model for backchannel detection."""
    def __init__(self, conf: Optional[VapConfig] = None):
        """Initialize the backchannel VAP model.

        Args:
            conf: Optional configuration object.
        """
        super().__init__()
        if conf is None:
            conf = VapConfig()
        self.conf = conf
        self.sample_rate = conf.sample_rate
        self.frame_hz = conf.frame_hz

        self.temp_elapse_time = []

        # Single channel
        self.ar_channel = GPT(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.channel_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
        )

        # Cross channel
        self.ar = GPTStereo(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.cross_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
        )

        self.objective = ObjectiveVAP(bin_times=conf.bin_times, frame_hz=conf.frame_hz)

        # Outputs
        # Voice activity objective -> x1, x2 -> logits ->  BCE
        self.va_classifier = nn.Linear(conf.dim, 1)
        
        self.vap_head = nn.Linear(conf.dim, self.objective.n_classes)

        self.bc_detect_head = nn.Linear(conf.dim, 1)

        # For Backchannel
        self.bc_head = nn.Linear(conf.dim, 1)

    def load_encoder(self, cpc_model):
        """Load CPC encoders and optionally freeze them.

        Args:
            cpc_model: Path to the CPC checkpoint.
        """
        
        # Audio Encoder
        self.encoder1 = EncoderCPC(
            load_pretrained=True if self.conf.load_pretrained == 1 else False,
            freeze=self.conf.freeze_encoder,
            cpc_model=cpc_model
        )
        self.encoder1 = self.encoder1.eval()
        
        self.encoder2 = EncoderCPC(
            load_pretrained=True if self.conf.load_pretrained == 1 else False,
            freeze=self.conf.freeze_encoder,
            cpc_model=cpc_model
        )

        self.encoder2 = self.encoder2.eval()
        
        if self.conf.freeze_encoder == 1:
            print('freeze encoder')
            self.encoder1.freeze()
            self.encoder2.freeze()

    @property
    def horizon_time(self):
        """Return the prediction horizon time in seconds."""
        return self.objective.horizon_time

    def encode_audio(self, audio1: torch.Tensor, audio2: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """Encode paired audio tensors into embeddings.

        Args:
            audio1: Audio tensor for channel 1.
            audio2: Audio tensor for channel 2.

        Returns:
            Tuple of encoded tensors (x1, x2).
        """
        # Channel swap for temporal consistency
        x1 = self.encoder1(audio2)  # speaker 1 (User)
        x2 = self.encoder2(audio1)  # speaker 2 (System)

        return x1, x2

    def vad_loss(self, vad_output, vad):
        """Compute VAD loss for the model outputs.

        Args:
            vad_output: VAD logits.
            vad: VAD labels.

        Returns:
            Loss tensor.
        """
        return F.binary_cross_entropy_with_logits(vad_output, vad)
    
    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        cache: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """
        Forward pass for the VapGPT model.

        Args:
            x1 (Tensor): Input audio embedded tensor for speaker 1.
            x2 (Tensor): Input audio embedded tensor for speaker 2.
            cache (dict, optional): Cache of past keys/values.

        Returns:
            Tuple[dict, dict]: Model outputs and updated cache.
        """

        if cache is None:
            cache = {}

        o1 = self.ar_channel(x1, past_kv=cache.get("ar1"))
        o2 = self.ar_channel(x2, past_kv=cache.get("ar2"))
        out = self.ar(
            o1["x"],
            o2["x"],
            past_kv1=cache.get("cross1"),
            past_kv2=cache.get("cross2"),
        )

        new_cache = {
            "ar1": (o1["past_k"], o1["past_v"]),
            "ar2": (o2["past_k"], o2["past_v"]),
            "cross1": (out["past_k1"], out["past_v1"]),
            "cross2": (out["past_k2"], out["past_v2"]),
        }

        bc = self.bc_head(out["x"])

        p_bc = bc.sigmoid().to("cpu").tolist()[0][-1][0]
        # print(p_bc)

        ret = {"p_bc": p_bc}

        return ret, new_cache
