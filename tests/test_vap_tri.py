"""Tests of the experimental three-party VAP model (maai.models.vap_tri / MaaiTri).

Always run (random weights, no files needed):
  - causality: outputs at frame t do not change when future frames are appended
  - presence mask: an absent speaker gets ~0 p_now / p_future
  - ConsoleBar renders a three-party result

With an exported model and reference outputs (export-maai-tri.py in the vap-tri
repository; set the environment variables):
  MAAI_TRI_MODEL      exported .pt
  MAAI_TRI_REF        *_ref.npz written with --ref-wav
  MAAI_TRI_MIMI_ONNX  fine-tuned Mimi ONNX (sidecar <path>.json), optional
  - parity of the VAP transformer with the original vap-tri model, fed with the
    reference Mimi features (strict)
  - end-to-end streaming through MaaiTri.process with the ONNX Mimi (reported;
    streaming Mimi and the dropped first frame make it differ slightly)

Run: python tests/test_vap_tri.py   (or pytest tests/test_vap_tri.py)
"""
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from maai import MaaiOutput  # noqa: E402
from maai.models.vap_tri import VapGPT_tri, VapTriConfig  # noqa: E402

MODEL = os.environ.get("MAAI_TRI_MODEL")
REF = os.environ.get("MAAI_TRI_REF")
MIMI_ONNX = os.environ.get("MAAI_TRI_MIMI_ONNX")


def _random_model(seed=0):
    torch.manual_seed(seed)
    return VapGPT_tri(VapTriConfig()).eval()


def _random_feats(T, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.relu(torch.randn(1, T, 256, generator=g)) for _ in range(3)]


@torch.no_grad()
def test_causal():
    model = _random_model()
    xs = _random_feats(40)
    full = model(xs)
    part = model([x[:, :25] for x in xs])
    for k in ("p_now", "p_future", "vad", "query_va_probs"):
        assert torch.allclose(full[k][:, :25], part[k], atol=1e-5), k


@torch.no_grad()
def test_present_mask():
    model = _random_model()
    xs = _random_feats(20)
    out = model(xs, present=torch.tensor([[1.0, 1.0, 0.0]]))
    assert out["p_now"][..., 2].abs().max() < 1e-3
    assert out["p_future"][..., 2].abs().max() < 1e-3
    assert torch.allclose(out["p_now"].sum(-1), torch.ones(1, 20), atol=1e-3)


def test_console_bar():
    result = {
        "t": 0.0,
        "x1": np.zeros(1280, dtype=np.float32),
        "x2": np.zeros(1280, dtype=np.float32),
        "x3": np.zeros(1280, dtype=np.float32),
        "p_now": [0.2, 0.3, 0.5],
        "p_future": [0.1, 0.6, 0.3],
        "vad": [0.9, 0.1, 0.0],
        "p_bins": [[0.1] * 4, [0.2] * 4, [0.3] * 4],
    }
    buf = io.StringIO()
    with redirect_stdout(buf):
        MaaiOutput.ConsoleBar().update(result)
    text = buf.getvalue()
    for label in ("x3", "p_now(spk3)", "p_future(spk2)", "vad(spk1)", "p_bins(spk3)"):
        assert label in text, label


def _load_exported():
    raw = torch.load(MODEL, map_location="cpu")
    model = VapGPT_tri(VapTriConfig.from_dict(raw["maai_tri_config"])).eval()
    model.load_state_dict(raw["state_dict"], strict=True)
    return model


@torch.no_grad()
def test_parity_with_reference():
    if not (MODEL and REF):
        print("skip: set MAAI_TRI_MODEL and MAAI_TRI_REF")
        return
    model = _load_exported()
    ref = np.load(REF)
    feats = torch.from_numpy(ref["enc_feats"]).float()  # (3, T, 512)
    xs = [torch.relu(model.decrease_dimension(feats[c : c + 1])) for c in range(3)]
    out = model(xs, present=torch.ones(1, 3))
    for k in ("p_now", "p_future", "vad", "query_va_probs"):
        if k not in ref.files:
            continue
        diff = np.abs(out[k][0].numpy() - ref[k]).max()
        print(f"parity {k}: max abs diff = {diff:.2e}")
        assert diff < 1e-4, k


@torch.no_grad()
def test_streaming_end_to_end():
    if not (MODEL and REF and MIMI_ONNX):
        print("skip: set MAAI_TRI_MODEL, MAAI_TRI_REF and MAAI_TRI_MIMI_ONNX")
        return
    from maai import MaaiInput, MaaiTri

    maai = MaaiTri(
        MaaiInput.Zero(), MaaiInput.Zero(), MaaiInput.Zero(),
        local_model=MODEL, mimi_local_onnx_fp32_path=MIMI_ONNX,
    )
    ref = np.load(REF)
    wav = ref["waveform"].astype(np.float32)  # (3, N)
    for i in range(0, wav.shape[1] - 160 + 1, 160):
        maai.process([wav[c, i : i + 160] for c in range(3)])
    results = []
    while not maai.result_dict_queue.empty():
        results.append(maai.get_result())
    # The first Mimi output is dropped, so streaming frame i ~ reference frame i + 1.
    n = min(len(results), ref["p_now"].shape[0] - 1)
    assert n > 0
    for k in ("p_now", "p_future", "vad"):
        if k not in ref.files:
            continue
        s = np.array([r[k] for r in results[:n]])
        r = ref[k][1 : n + 1]
        corr = np.corrcoef(s.ravel(), r.ravel())[0, 1]
        print(f"streaming {k}: frames={n}, mean abs diff={np.abs(s - r).mean():.4f}, corr={corr:.4f}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
