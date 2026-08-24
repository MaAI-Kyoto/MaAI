import torch
from torch import Tensor
from typing import Optional, Tuple

from .vap import VapGPT

class VapGPT_mono(VapGPT):
    """Single-channel (mono) variant of the VAP model.

    The underlying model is identical to :class:`VapGPT` (the same
    pretrained ``vap`` checkpoints are loaded), but the interface is
    mono: only channel 1 carries speech while channel 2 is fed silence
    (``MaaiInput.Zero``). The outputs are converted accordingly:

    - ``p_now`` / ``p_future``: the speaker-normalized channel-1 value
      is linearly stretched from [0.5, 1.0] to [0.0, 1.0]; values at or
      below 0.5 map to 0.0. Returned as a single float.
    - ``vad``: channel-1 value only, as a single float.
    - ``p_bins`` / ``p_bins_now`` / ``p_bins_future``: channel-1 entries only.
    """

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        cache: Optional[dict] = None,
        return_all_frames: bool = False,
    ) -> Tuple[dict, dict]:
        """
        Forward pass for the mono VAP model.

        Args:
            x1 (Tensor): Input audio embedded tensor for the input speaker.
            x2 (Tensor): Input audio embedded tensor for the silent channel.
            cache (dict, optional): Cache of past keys/values.

        Returns:
            Tuple[dict, dict]: Mono-converted model outputs and updated cache.
        """
        ret, new_cache = super().forward(
            x1, x2, cache, return_all_frames=return_all_frames
        )

        if return_all_frames:
            for frame in ret:
                frame["p_now"] = max(0.0, (frame["p_now"][0] - 0.5) * 2.0)
                frame["p_future"] = max(0.0, (frame["p_future"][0] - 0.5) * 2.0)
                frame["vad"] = frame["vad"][0]
                frame["p_bins"] = frame["p_bins"][0]
                frame["p_bins_now"] = frame["p_bins_now"][0]
                frame["p_bins_future"] = frame["p_bins_future"][0]
            return ret, new_cache

        # 話者間正規化済み ch1 の [0.5, 1.0] を [0, 1] に線形伸長(0.5 以下は 0)
        ret["p_now"] = max(0.0, (ret["p_now"][0] - 0.5) * 2.0)
        ret["p_future"] = max(0.0, (ret["p_future"][0] - 0.5) * 2.0)

        # ch1 のみを返す
        ret["vad"] = ret["vad"][0]
        ret["p_bins"] = ret["p_bins"][0]
        ret["p_bins_now"] = ret["p_bins_now"][0]
        ret["p_bins_future"] = ret["p_bins_future"][0]

        return ret, new_cache
