import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from .config import VapConfig
from ..encoder import build_audio_encoder
from ..modules import GPT, GPTStereo

class VadGPT(nn.Module):
    """Voice Activity Detection (VAD) model.

    Detects *current* voice activity for each of the two input channels.
    The architecture is the VAP model without the projection head: the
    per-channel and cross-channel transformers are followed by a single
    ``va_classifier`` that is shared between both channels.

    Unlike a per-channel energy based VAD, the cross-channel attention lets
    the model decide which speaker is actually talking even when the voice
    of the other speaker leaks into the channel.
    """

    def __init__(self, conf: Optional[VapConfig] = None):
        """Initialize the VadGPT model.

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
        # The VAD model only uses the per-channel outputs (x1 / x2), so the
        # combinator is not built (it has no weights in the checkpoints).
        self.ar = GPTStereo(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.cross_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
            combinator=False,
        )

        # Outputs
        # Voice activity objective -> x1, x2 -> logits -> BCE
        self.va_classifier = nn.Linear(conf.dim, 1)

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

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        cache: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """
        Forward pass for the VadGPT model.

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

        # Get back to the CPU
        vad1 = vad1.sigmoid().to("cpu").tolist()[0][-1][0]
        vad2 = vad2.sigmoid().to("cpu").tolist()[0][-1][0]

        ret = {
            "vad": [vad1, vad2],
        }

        return ret, new_cache


class VadGPT_mono(VadGPT):
    """Single-channel (mono) variant of the VAD model.

    The underlying model is identical to :class:`VadGPT` (the same
    pretrained ``vad`` checkpoints are loaded), but the interface is
    mono: only channel 1 carries speech while channel 2 is fed silence
    (``MaaiInput.Zero``). The output ``vad`` is therefore a single float
    for channel 1 instead of a value per speaker.
    """

    def forward(self, x1: Tensor, x2: Tensor, cache: Optional[dict] = None) -> Tuple[dict, dict]:
        """
        Forward pass for the mono VAD model.

        Args:
            x1 (Tensor): Input audio embedded tensor for the single speaker.
            x2 (Tensor): Input audio embedded tensor for the silent channel.
            cache (dict, optional): Cache of past keys/values.

        Returns:
            Tuple[dict, dict]: Model outputs (scalar ``vad``) and updated cache.
        """
        ret, new_cache = super().forward(x1, x2, cache)

        # ch1 のみを返す
        ret["vad"] = ret["vad"][0]

        return ret, new_cache
