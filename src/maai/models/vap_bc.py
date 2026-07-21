import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Tuple
import torch.nn.functional as F

from .config import VapConfig
from .vap import vap_max_past_len
from ..encoder import build_audio_encoder
from ..modules import GPT, GPTStereo
from ..objective import ObjectiveVAP

class VapGPT_bc(nn.Module):
    """Voice Activity Projection with Backchannel (VAP-BC) model.
    
    This model extends VapGPT to specifically handle and predict backchannel
    behaviors during conversations.
    """
    

    def __init__(self, conf: Optional[VapConfig] = None):
        """Initialize the VapGPT_bc model.
        
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
        """Load and build the audio encoders for both speakers.
        
        Args:
            cpc_model: Pre-trained CPC model to be used as feature extractor.
        """
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
        """Get the horizon time for the projection in seconds.
        
        Returns:
            float: Horizon time for the objective.
        """
        return self.objective.horizon_time

    def encode_audio(self, audio1: torch.Tensor, audio2: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """Encode the raw audio inputs into feature representations.
        
        Note: Channel swap is applied for temporal consistency.
        
        Args:
            audio1 (torch.Tensor): Audio waveform for speaker 1 (System).
            audio2 (torch.Tensor): Audio waveform for speaker 2 (User).
            
        Returns:
            Tuple[Tensor, Tensor]: Encoded features for the two speakers.
        """
        
        # Channel swap for temporal consistency
        x1 = self.encoder1(audio2)  # speaker 1 (User)
        x2 = self.encoder2(audio1)  # speaker 2 (System)

        if hasattr(self, "decrease_dimension"):
            x1 = torch.relu(self.decrease_dimension(x1))
            x2 = torch.relu(self.decrease_dimension(x2))

        return x1, x2

    def vad_loss(self, vad_output, vad):
        """Compute the Voice Activity Detection (VAD) loss.
        
        Args:
            vad_output: Predicted VAD logits.
            vad: Ground truth VAD labels.
            
        Returns:
            Tensor: Binary cross-entropy loss between predictions and targets.
        """
        return F.binary_cross_entropy_with_logits(vad_output, vad)
    
    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        cache: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """
        Forward pass for the VapGPT_bc model.

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

        bc_logit = self.bc_head(out["x"])
        bc_detect_logit = self.bc_detect_head(out["x"])
        p_bc = float(bc_logit.sigmoid().to("cpu").tolist()[0][-1][0])
        p_bc_detect = float(bc_detect_logit.sigmoid().to("cpu").tolist()[0][-1][0])

        ret = {"p_bc": p_bc, "p_bc_detect": p_bc_detect}

        return ret, new_cache


def bc_outputs_from_logits(bc_logit: Tensor, bc_detect_logit: Tensor) -> dict:
    """Match ``VapGPT_bc.forward`` post-head sigmoid (ONNX runtime path)."""
    p_bc = float(bc_logit.sigmoid().to("cpu").tolist()[0][-1][0])
    p_bc_detect = float(bc_detect_logit.sigmoid().to("cpu").tolist()[0][-1][0])
    return {"p_bc": p_bc, "p_bc_detect": p_bc_detect}


class BcOnnxSession:
    """ORT session for BC transformer step; sigmoid stays in Torch."""

    def __init__(
        self,
        onnx_path: str,
        meta_path: str,
        bc_ref: VapGPT_bc,
        *,
        providers: Optional[List[str]] = None,
        cpu_intra_threads: Optional[int] = None,
        cpu_inter_threads: Optional[int] = None,
    ):
        import json
        import onnxruntime as ort

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.input_names: List[str] = list(self.meta["input_names"])
        self.output_names: List[str] = list(self.meta["output_names"])
        self.channel_layers = int(self.meta["channel_layers"])
        self.cross_layers = int(self.meta["cross_layers"])
        self.num_heads = int(self.meta["num_heads"])
        self.head_dim = int(self.meta["head_dim"])
        self.dim = int(self.meta["dim"])
        self.max_past = int(self.meta.get("max_past_len", vap_max_past_len(10.0, 20.0)))

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if cpu_intra_threads is not None:
            so.intra_op_num_threads = max(1, int(cpu_intra_threads))
        if cpu_inter_threads is not None:
            so.inter_op_num_threads = max(1, int(cpu_inter_threads))
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.bc_ref = bc_ref
        actual_in = [x.name for x in self.session.get_inputs()]
        actual_out = [x.name for x in self.session.get_outputs()]
        if actual_in != self.input_names or actual_out != self.output_names:
            raise RuntimeError(
                f"ONNX IO mismatch: in={actual_in} vs {self.input_names}; "
                f"out={actual_out} vs {self.output_names}"
            )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        cache: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        from .vap import cache_dict_to_flat, flat_to_cache_dict

        device = x1.device
        dtype = x1.dtype
        flat = cache_dict_to_flat(
            cache,
            channel_layers=self.channel_layers,
            cross_layers=self.cross_layers,
            batch=int(x1.shape[0]),
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        feeds = {
            "x1": x1.detach().float().cpu().numpy(),
            "x2": x2.detach().float().cpu().numpy(),
        }
        for name, t in zip(self.input_names[2:], flat):
            feeds[name] = t.detach().float().cpu().numpy()

        outs = self.session.run(self.output_names, feeds)
        bc_logit = torch.from_numpy(outs[0]).to(device=device, dtype=dtype)
        bc_detect_logit = torch.from_numpy(outs[1]).to(device=device, dtype=dtype)
        past_out = [torch.from_numpy(o) for o in outs[2:]]
        new_cache = flat_to_cache_dict(
            past_out,
            channel_layers=self.channel_layers,
            cross_layers=self.cross_layers,
            max_past=self.max_past,
        )
        for key, (ks, vs) in new_cache.items():
            new_cache[key] = (
                [k.to(device=device, dtype=dtype) for k in ks],
                [v.to(device=device, dtype=dtype) for v in vs],
            )
        ret = bc_outputs_from_logits(bc_logit, bc_detect_logit)
        return ret, new_cache