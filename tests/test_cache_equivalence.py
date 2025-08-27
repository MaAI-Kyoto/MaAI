import sys
from pathlib import Path

import torch

# Ensure src directory is on the Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from maai.models.vap import VapGPT
from maai.models.config import VapConfig


def test_cache_equivalence():
    """Ensure cached step-by-step inference matches full-sequence output."""
    conf = VapConfig(
        dim=8,
        channel_layers=1,
        cross_layers=1,
        num_heads=1,
        dropout=0.0,
    )
    model = VapGPT(conf).eval()

    seq_len = 4
    x1 = torch.randn(1, seq_len, conf.dim)
    x2 = torch.randn(1, seq_len, conf.dim)

    # Single forward pass without cache
    full_out = model(x1, x2)

    # Step-by-step forward passes using a simple cache of inputs
    cache_x1, cache_x2 = None, None
    cached_out = None
    for t in range(seq_len):
        new_x1 = x1[:, t : t + 1]
        new_x2 = x2[:, t : t + 1]
        cache_x1 = new_x1 if cache_x1 is None else torch.cat([cache_x1, new_x1], dim=1)
        cache_x2 = new_x2 if cache_x2 is None else torch.cat([cache_x2, new_x2], dim=1)
        cached_out = model(cache_x1, cache_x2)

    torch.testing.assert_close(
        torch.tensor(full_out["p_now"]), torch.tensor(cached_out["p_now"])
    )
    torch.testing.assert_close(
        torch.tensor(full_out["p_future"]), torch.tensor(cached_out["p_future"])
    )
    torch.testing.assert_close(
        torch.tensor(full_out["vad"]), torch.tensor(cached_out["vad"])
    )
