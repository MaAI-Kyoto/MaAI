import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch.nn.functional as F

from .config import VapConfig
from ..encoder import build_audio_encoder
from ..modules import GPT, GPTStereo
from ..objective import ObjectiveVAP

class VapGPT(nn.Module):
    """Voice Activity Projection (VAP) core model using GPT architecture.
    
    This model processes audio features using a GPT-based architecture
    to predict future voice activity, which can be used for turn-taking
    predictions in spoken dialogue systems.
    """
    

    BINS_P_NOW = [0, 1]
    BINS_PFUTURE = [2, 3]
    
    def __init__(self, conf: Optional[VapConfig] = None):
        """Initialize the VapGPT model.
        
        Args:
            conf (Optional[VapConfig]): Configuration object for the model.
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
        
        if self.conf.lid_classify == 1:
            self.lid_classifier = nn.Linear(conf.dim, conf.lid_classify_num_class)
        
        elif self.conf.lid_classify == 2:
            self.lid_classifier_middle = nn.Linear(conf.dim*2, conf.lid_classify_num_class)
        
        if self.conf.lang_cond == 1:
            self.lang_condition = nn.Linear(conf.lid_classify_num_class, conf.dim)
        
        self.vap_head = nn.Linear(conf.dim, self.objective.n_classes)

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
        
        Args:
            audio1 (torch.Tensor): Audio waveform for speaker 1.
            audio2 (torch.Tensor): Audio waveform for speaker 2.
            
        Returns:
            Tuple[Tensor, Tensor]: Encoded features for speaker 1 and speaker 2.
        """
        

        x1 = self.encoder1(audio1)  # speaker 1
        x2 = self.encoder2(audio2)  # speaker 2

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

        # Outputs
        vad1 = self.va_classifier(out["x1"])
        vad2 = self.va_classifier(out["x2"])
        logits = self.vap_head(out["x"])

        probs = logits.softmax(dim=-1)

        p_bins_tensor = self.objective.probs_speaker_bin_aggregate(
            probs, from_bin=0, to_bin=self.objective.n_bins - 1
        )

        # 話者次元は正規化せず、該当ビン（p_now / p_future と同じ範囲）の値を足したもの
        i0, i1 = self.BINS_P_NOW[0], self.BINS_P_NOW[-1]
        j0, j1 = self.BINS_PFUTURE[0], self.BINS_PFUTURE[-1]
        p_bins_now_t = p_bins_tensor[:, :, :, i0 : i1 + 1].sum(dim=-1)
        p_bins_future_t = p_bins_tensor[:, :, :, j0 : j1 + 1].sum(dim=-1)

        p_now = self.objective.probs_next_speaker_aggregate(
            probs,
            from_bin=self.BINS_P_NOW[0],
            to_bin=self.BINS_P_NOW[-1],
        )

        p_future = self.objective.probs_next_speaker_aggregate(
            probs,
            from_bin=self.BINS_PFUTURE[0],
            to_bin=self.BINS_PFUTURE[1],
        )

        # Get back to the CPU（ビン合計は最大 2 なので 2 で割り各話者 [0, 1] に正規化）
        p_bins = p_bins_tensor.to("cpu").tolist()[0][-1]
        p_bins_now = (p_bins_now_t * 0.5).to("cpu").tolist()[0][-1]
        p_bins_future = (p_bins_future_t * 0.5).to("cpu").tolist()[0][-1]
        p_now = p_now.to("cpu").tolist()[0][-1]
        p_future = p_future.to("cpu").tolist()[0][-1]

        vad1 = vad1.sigmoid().to("cpu").tolist()[0][-1][0]
        vad2 = vad2.sigmoid().to("cpu").tolist()[0][-1][0]

        ret = {
            "p_now": p_now,
            "p_future": p_future,
            "vad": [vad1, vad2],
            "p_bins": p_bins,
            "p_bins_now": p_bins_now,
            "p_bins_future": p_bins_future,
        }

        return ret, new_cache

def vap_max_past_len(frame_hz: float, context_len_sec: float) -> int:
    """Match ``Maai.process`` trim: keep last ``audio_context_len - 1`` frames."""
    ctx = int(round(float(context_len_sec) * float(frame_hz)))
    return max(1, ctx - 1)


def _empty_past(
    batch: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    return torch.zeros((batch, num_heads, 0, head_dim), device=device, dtype=dtype)


def _trim_kv(t: Tensor, max_past: int) -> Tensor:
    if t.shape[-2] > max_past:
        return t[..., -max_past:, :].contiguous()
    return t.contiguous()


def cache_dict_to_flat(
    cache: Optional[dict],
    *,
    channel_layers: int,
    cross_layers: int,
    batch: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tensor]:
    """Convert Maai ``vap_cache`` dict to flat past tensor list (T may be 0)."""
    cache = cache or {}
    flat: List[Tensor] = []

    def layer_pair(key: str, n_layers: int) -> None:
        kv = cache.get(key)
        if kv is None:
            ks = [None] * n_layers
            vs = [None] * n_layers
        else:
            ks, vs = kv
        for i in range(n_layers):
            k = ks[i]
            v = vs[i]
            if k is None:
                k = _empty_past(batch, num_heads, head_dim, device, dtype)
            if v is None:
                v = _empty_past(batch, num_heads, head_dim, device, dtype)
            flat.append(k)
            flat.append(v)

    layer_pair("ar1", channel_layers)
    layer_pair("ar2", channel_layers)
    layer_pair("cross1", cross_layers)
    layer_pair("cross2", cross_layers)
    layer_pair("cross1_c", cross_layers)
    layer_pair("cross2_c", cross_layers)
    return flat


def flat_to_cache_dict(
    flat: Sequence[Tensor],
    *,
    channel_layers: int,
    cross_layers: int,
    max_past: Optional[int] = None,
) -> dict:
    """Convert flat past list to Maai cache dict; optional trim like ``Maai.process``."""
    it = iter(flat)

    def take_layers(n: int) -> Tuple[list, list]:
        ks, vs = [], []
        for _ in range(n):
            k = next(it)
            v = next(it)
            if max_past is not None:
                k = _trim_kv(k, max_past)
                v = _trim_kv(v, max_past)
            ks.append(k)
            vs.append(v)
        return ks, vs

    return {
        "ar1": take_layers(channel_layers),
        "ar2": take_layers(channel_layers),
        "cross1": take_layers(cross_layers),
        "cross2": take_layers(cross_layers),
        "cross1_c": take_layers(cross_layers),
        "cross2_c": take_layers(cross_layers),
    }


def vap_outputs_from_logits_vad(
    vap: "VapGPT",
    logits: Tensor,
    vad: Tensor,
) -> dict:
    """Match ``VapGPT.forward`` post-head aggregates (ObjectiveVAP stays in Torch)."""
    probs = logits.softmax(dim=-1)
    p_bins_tensor = vap.objective.probs_speaker_bin_aggregate(
        probs, from_bin=0, to_bin=vap.objective.n_bins - 1
    )
    i0, i1 = vap.BINS_P_NOW[0], vap.BINS_P_NOW[-1]
    j0, j1 = vap.BINS_PFUTURE[0], vap.BINS_PFUTURE[-1]
    p_bins_now_t = p_bins_tensor[:, :, :, i0 : i1 + 1].sum(dim=-1)
    p_bins_future_t = p_bins_tensor[:, :, :, j0 : j1 + 1].sum(dim=-1)
    p_now = vap.objective.probs_next_speaker_aggregate(
        probs, from_bin=vap.BINS_P_NOW[0], to_bin=vap.BINS_P_NOW[-1]
    )
    p_future = vap.objective.probs_next_speaker_aggregate(
        probs, from_bin=vap.BINS_PFUTURE[0], to_bin=vap.BINS_PFUTURE[1]
    )
    vad_sig = vad.sigmoid()
    return {
        "p_now": p_now.to("cpu").tolist()[0][-1],
        "p_future": p_future.to("cpu").tolist()[0][-1],
        "vad": [
            vad_sig.to("cpu").tolist()[0][-1][0],
            vad_sig.to("cpu").tolist()[0][-1][1],
        ],
        "p_bins": p_bins_tensor.to("cpu").tolist()[0][-1],
        "p_bins_now": (p_bins_now_t * 0.5).to("cpu").tolist()[0][-1],
        "p_bins_future": (p_bins_future_t * 0.5).to("cpu").tolist()[0][-1],
    }


def _ort_provider_name(p: Any) -> str:
    return p[0] if isinstance(p, (tuple, list)) else str(p)


def build_ort_providers(
    runtime_device: Optional[str] = None,
    providers: Optional[Sequence[Any]] = None,
) -> List[Any]:
    """Build ORT provider list. Prefers CUDA when ``runtime_device`` starts with ``cuda``."""
    if providers is not None:
        return list(providers)
    import onnxruntime as ort

    avail = set(ort.get_available_providers())
    device = str(runtime_device or "cpu").strip().lower()
    if device.startswith("cuda") and "CUDAExecutionProvider" in avail:
        device_id = 0
        if ":" in device:
            try:
                device_id = int(device.split(":", 1)[1])
            except ValueError:
                device_id = 0
        return [
            ("CUDAExecutionProvider", {"device_id": device_id}),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


def create_ort_inference_session(
    onnx_path: str,
    *,
    runtime_device: Optional[str] = None,
    providers: Optional[Sequence[Any]] = None,
    cpu_intra_threads: Optional[int] = None,
    cpu_inter_threads: Optional[int] = None,
):
    """Create ORT session; fall back to CPU if CUDA session init fails (e.g. int8)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    wanted = build_ort_providers(runtime_device=runtime_device, providers=providers)
    use_cuda = any(_ort_provider_name(p) == "CUDAExecutionProvider" for p in wanted)
    if not use_cuda:
        if cpu_intra_threads is not None:
            so.intra_op_num_threads = max(1, int(cpu_intra_threads))
        if cpu_inter_threads is not None:
            so.inter_op_num_threads = max(1, int(cpu_inter_threads))
    try:
        return ort.InferenceSession(onnx_path, sess_options=so, providers=wanted)
    except Exception as exc:
        if not use_cuda:
            raise
        print(
            f"[ONNX] CUDA EP failed for {onnx_path} ({exc!r}); falling back to CPU. "
            "INT8 MatMul models often need CPU — use fp32 ONNX for CUDA."
        )
        so_cpu = ort.SessionOptions()
        so_cpu.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if cpu_intra_threads is not None:
            so_cpu.intra_op_num_threads = max(1, int(cpu_intra_threads))
        if cpu_inter_threads is not None:
            so_cpu.inter_op_num_threads = max(1, int(cpu_inter_threads))
        return ort.InferenceSession(
            onnx_path, sess_options=so_cpu, providers=["CPUExecutionProvider"]
        )


class VapOnnxSession:
    """ORT session for VAP transformer step; ObjectiveVAP aggregates stay in Torch."""

    def __init__(
        self,
        onnx_path: str,
        meta_path: str,
        vap_ref: VapGPT,
        *,
        runtime_device: Optional[str] = None,
        providers: Optional[List[Any]] = None,
        cpu_intra_threads: Optional[int] = None,
        cpu_inter_threads: Optional[int] = None,
    ):
        import json

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.input_names: List[str] = list(self.meta["input_names"])
        self.output_names: List[str] = list(self.meta["output_names"])
        self.channel_layers = int(self.meta["channel_layers"])
        self.cross_layers = int(self.meta["cross_layers"])
        self.num_heads = int(self.meta["num_heads"])
        self.head_dim = int(self.meta["head_dim"])
        self.dim = int(self.meta["dim"])
        self.max_past = int(self.meta.get("max_past_len", vap_max_past_len(12.5, 20.0)))
        self.runtime_device = str(runtime_device or "cpu")

        self.session = create_ort_inference_session(
            onnx_path,
            runtime_device=self.runtime_device,
            providers=providers,
            cpu_intra_threads=cpu_intra_threads,
            cpu_inter_threads=cpu_inter_threads,
        )
        self.active_providers = list(self.session.get_providers())
        self.use_cuda = "CUDAExecutionProvider" in self.active_providers
        print(
            f"[VapOnnxSession] providers={self.active_providers} "
            f"requested_device={self.runtime_device}"
        )
        self.vap_ref = vap_ref
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
        device = x1.device
        dtype = x1.dtype
        # session.run uses host numpy. Keep KV on CPU between steps; CUDA EP still
        # executes MatMul on GPU after ORT's H2D copy.
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
        logits = torch.from_numpy(outs[0]).to(device=device, dtype=dtype)
        vad = torch.from_numpy(outs[1]).to(device=device, dtype=dtype)
        past_out = [torch.from_numpy(o) for o in outs[2:]]
        new_cache = flat_to_cache_dict(
            past_out,
            channel_layers=self.channel_layers,
            cross_layers=self.cross_layers,
            max_past=self.max_past,
        )
        ret = vap_outputs_from_logits_vad(self.vap_ref, logits, vad)
        return ret, new_cache

