import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from .config import VapConfig
from ..encoder import build_audio_encoder
from ..modules import GPT
from ..objective import ObjectiveVAP


class VapGPT_mono(nn.Module):
    """Single-channel (mono) Voice Activity Projection (VAP) model.

    Unlike the two-speaker :class:`~maai.models.vap.VapGPT`, this is a genuinely
    single-channel model trained on its own ``vap_mono`` checkpoints: it encodes
    one audio stream, replaces the cross-channel ``GPTStereo`` with a plain
    causal :class:`~maai.modules.GPT`, and predicts ``2 ** n_bins`` (16 for the
    default four bins) codebook states describing the future activity of that
    one speaker.

    Because there is no second speaker to normalize against, ``p_now`` and
    ``p_future`` are *not* a two-way split: they are the expected (bin-length
    weighted) activity ratio of the input speaker and are already in [0, 1].
    They are returned as single floats, as are ``vad``, ``p_bins_now`` and
    ``p_bins_future``; ``p_bins`` is a list with one value per bin.

    It is intended for use cases where only one speaker's audio is available,
    e.g. a single microphone input for a spoken dialogue system.
    """

    # Only channel 1 is encoded, so the runtime must not feed a second stream
    # through ``encode_audio`` / ``forward``.
    is_single_tower = True

    BINS_P_NOW = [0, 1]
    BINS_PFUTURE = [2, 3]

    def __init__(self, conf: Optional[VapConfig] = None):
        """Initialize the VapGPT_mono model.

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

        # There is no second channel to cross-attend to, so the stereo tower of
        # the two-speaker model is a plain causal GPT here. The layer counts are
        # kept so the depth matches the two-speaker model.
        self.ar = GPT(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.cross_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
        )

        self.objective = ObjectiveVAP(
            bin_times=conf.bin_times, frame_hz=conf.frame_hz, num_channels=1
        )

        # Outputs
        # Voice activity objective -> x -> logits -> BCE
        self.va_classifier = nn.Linear(conf.dim, 1)
        self.vap_head = nn.Linear(conf.dim, self.objective.n_classes)

    def load_encoder(self, cpc_model):
        """Load and build the audio encoder for the single input channel.

        Args:
            cpc_model: Pre-trained CPC model to be used as feature extractor.
        """
        self.encoder1 = build_audio_encoder(self.conf, cpc_model=cpc_model)
        self.encoder1 = self.encoder1.eval()

        encoder_dim = getattr(self.encoder1, "output_dim", self.conf.dim)
        if encoder_dim != self.conf.dim:
            self.decrease_dimension = nn.Linear(encoder_dim, self.conf.dim)

        if self.conf.freeze_encoder == 1:
            print('freeze encoder')
            self.encoder1.freeze()

    @property
    def horizon_time(self):
        """Get the horizon time for the projection in seconds.

        Returns:
            float: Horizon time for the objective.
        """
        return self.objective.horizon_time

    def encode_audio(self, audio1: torch.Tensor) -> Tensor:
        """Encode the raw audio input into a feature representation.

        Args:
            audio1 (torch.Tensor): Audio waveform for the input speaker.

        Returns:
            Tensor: Encoded features for the input speaker.
        """
        x1 = self.encoder1(audio1)

        if hasattr(self, "decrease_dimension"):
            x1 = torch.relu(self.decrease_dimension(x1))

        return x1

    def forward(
        self,
        x1: Tensor,
        cache: Optional[dict] = None,
        return_all_frames: bool = False,
    ) -> Tuple[dict, dict]:
        """
        Forward pass for the mono VapGPT model.

        Args:
            x1 (Tensor): Input audio embedded tensor for the input speaker.
            cache (dict, optional): Cache of past keys/values.
            return_all_frames: Return one result dictionary for every input
                time step instead of only the final step.

        Returns:
            Tuple[dict, dict]: Model outputs and updated cache.
        """

        if cache is None:
            cache = {}

        o1 = self.ar_channel(x1, past_kv=cache.get("ar1"))
        out = self.ar(o1["x"], past_kv=cache.get("ar2"))

        new_cache = {
            "ar1": (o1["past_k"], o1["past_v"]),
            "ar2": (out["past_k"], out["past_v"]),
        }

        # Outputs
        vad = self.va_classifier(out["x"])
        logits = self.vap_head(out["x"])

        probs = logits.softmax(dim=-1)

        # 話者間正規化はしない（比較対象の話者がいないため）。
        # 各ビン長で重み付けした発話期待値がそのまま [0, 1] の確率になる。
        p_now = self.objective.probs_self_activity(
            probs,
            from_bin=self.BINS_P_NOW[0],
            to_bin=self.BINS_P_NOW[-1],
        )
        p_future = self.objective.probs_self_activity(
            probs,
            from_bin=self.BINS_PFUTURE[0],
            to_bin=self.BINS_PFUTURE[-1],
        )

        p_bins_tensor = self.objective.probs_bins(probs)

        # p_now / p_future と同じビン範囲の周辺確率を平均して [0, 1] に収める
        i0, i1 = self.BINS_P_NOW[0], self.BINS_P_NOW[-1]
        j0, j1 = self.BINS_PFUTURE[0], self.BINS_PFUTURE[-1]
        p_bins_now_t = p_bins_tensor[:, :, i0 : i1 + 1].mean(dim=-1)
        p_bins_future_t = p_bins_tensor[:, :, j0 : j1 + 1].mean(dim=-1)

        # Get back to the CPU
        p_bins_all = p_bins_tensor.to("cpu").tolist()[0]
        p_bins_now_all = p_bins_now_t.to("cpu").tolist()[0]
        p_bins_future_all = p_bins_future_t.to("cpu").tolist()[0]
        p_now_all = p_now.to("cpu").tolist()[0]
        p_future_all = p_future.to("cpu").tolist()[0]
        vad_all = vad.sigmoid().to("cpu").tolist()[0]

        frames = [
            {
                "p_now": p_now_all[t],
                "p_future": p_future_all[t],
                "vad": vad_all[t][0],
                "p_bins": p_bins_all[t],
                "p_bins_now": p_bins_now_all[t],
                "p_bins_future": p_bins_future_all[t],
            }
            for t in range(x1.shape[1])
        ]
        ret = frames if return_all_frames else frames[-1]

        return ret, new_cache
