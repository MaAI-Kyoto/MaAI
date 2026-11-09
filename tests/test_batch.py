"""Tests of offline batch inference (maai.batch.MaaiBatch / forward_batch).

Always run (random weights, no files needed):
  - the vectorized band mask equals the original element-wise loop
  - forward_batch == forward(return_all_frames=True) for every model class
  - forward_batch with the band mask == the realtime KV-cache stream
    (cache trimmed to context_limit - 1 after every frame, as in Maai.process)
  - folding a long sequence into windows == one full forward
  - MaaiBatch.process_many (grouping, padding, order, timing, mono, several
    models) with a stand-in for the Mimi encoder
  - the CPC path == Maai.process frame by frame (random CPC weights; 1 and 3
    transformer frames per call, channel swap, mono)

With pretrained weights (downloads kyutai/mimi, the CPC weights and the MaAI
models; set MAAI_TEST_PRETRAINED=1):
  - Mimi features do not depend on encoder_chunk_frames or on batching
  - MaaiBatch == Maai frame by frame with the PyTorch Mimi (strict)
  - MaaiBatch vs Maai with the default ONNX Mimi (reported; small differences)
  - MaaiBatch == Maai frame by frame with CPC (model_type="normal")

Run: python tests/test_batch.py   (or pytest tests/test_batch.py)
"""
import copy
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from maai.batch import MaaiBatch, _BatchModel  # noqa: E402
from maai.encoder import EncoderCPC  # noqa: E402
from maai.encoder_components import CPCAR, CPCEncoder, CPCModel, get_cnn_layer  # noqa: E402
from maai.model import Maai  # noqa: E402
from maai.models.bc_det import BcDetGPT, BcDetGPT_mono  # noqa: E402
from maai.models.config import VapConfig  # noqa: E402
from maai.models.vad import VadGPT, VadGPT_mono  # noqa: E402
from maai.models.vap import VapGPT  # noqa: E402
from maai.models.vap_bc import VapGPT_bc  # noqa: E402
from maai.models.vap_bc_2type import VapGPT_bc_2type  # noqa: E402
from maai.models.vap_mono import VapGPT_mono  # noqa: E402
from maai.models.vap_nod import VapGPT_nod  # noqa: E402
from maai.models.vap_nod_para import VapGPT_nod_para  # noqa: E402
from maai.models.vap_nod_timing import VapGPT_nod_timing  # noqa: E402
from maai.modules import MultiHeadAttentionAlibi  # noqa: E402

PRETRAINED = os.environ.get("MAAI_TEST_PRETRAINED") == "1"
WAV1 = os.path.join(ROOT, "example", "wav_sample", "jpn_inoue_16k.wav")
WAV2 = os.path.join(ROOT, "example", "wav_sample", "jpn_sumida_16k.wav")

MODEL_CLASSES = [
    VapGPT,
    VapGPT_mono,
    VadGPT,
    VadGPT_mono,
    BcDetGPT,
    BcDetGPT_mono,
    VapGPT_bc,
    VapGPT_bc_2type,
    VapGPT_nod,
    VapGPT_nod_timing,
    VapGPT_nod_para,
]


def _random_model(cls, context_limit=6, seed=0):
    torch.manual_seed(seed)
    conf = VapConfig()
    conf.frame_hz = 12.5
    conf.context_limit = context_limit
    model = cls(conf)
    if cls is VapGPT_nod_para:
        # nod_para projects the raw 512-d encoder features itself.
        model.decrease_dimension = nn.Linear(512, conf.dim)
        model.decrease_dimension_param = nn.Linear(512, conf.dim)
    # Larger weights than the GPT init so that the outputs are not all ~0.5.
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            mod.weight.data.normal_(0.0, 0.1)
    # float64: batch-size dependent float32 rounding would hide small logic errors.
    return model.double().eval()


def _inputs(model, batch, frames, seed=1):
    g = torch.Generator().manual_seed(seed)
    dim = 512 if isinstance(model, VapGPT_nod_para) else model.conf.dim
    xs = [torch.relu(torch.randn(batch, frames, dim, generator=g, dtype=torch.float64)) for _ in range(2)]
    return xs[:1] if getattr(model, "is_single_tower", False) else xs


def _assert_frames_close(frames, batch_out, b, atol=1e-9, label=""):
    for key, tensor in batch_out.items():
        got = tensor[b].double().numpy()
        want = np.asarray([f[key] for f in frames], dtype=np.float64)
        assert got.shape == want.shape, f"{label}{key}: {got.shape} vs {want.shape}"
        diff = np.abs(got - want).max() if got.size else 0.0
        assert diff <= atol, f"{label}{key}: max abs diff {diff:.2e}"


def test_band_mask_matches_loop():
    for context_limit, T in ((1, 5), (4, 11), (7, 7), (3, 20)):
        attn = MultiHeadAttentionAlibi(dim=16, num_heads=4, dropout=0.0, context_limit=context_limit)
        got = attn.mask_scores(torch.zeros(1, 4, T, T))

        ref = attn.get_alibi_mask(T)
        for j in range(T):
            for n in range(max(0, j - context_limit + 1)):
                ref[..., j, n] = float("-inf")
        assert torch.equal(got, ref), (context_limit, T)


@torch.no_grad()
def test_forward_batch_matches_forward():
    for cls in MODEL_CLASSES:
        model = _random_model(cls)
        xs = _inputs(model, batch=2, frames=15)
        out = model.forward_batch(*xs)
        for b in range(2):
            with redirect_stdout(io.StringIO()):  # VapGPT_bc prints every call
                frames, _ = model.forward(*[x[b : b + 1] for x in xs], cache=None, return_all_frames=True)
            _assert_frames_close(frames, out, b, label=f"{cls.__name__}: ")


@torch.no_grad()
def test_forward_batch_matches_kv_stream():
    context_limit = 6
    for cls in MODEL_CLASSES:
        model = _random_model(cls, context_limit=context_limit)
        xs = _inputs(model, batch=1, frames=4 * context_limit + 9)
        out = model.forward_batch(*xs)

        cache, frames = None, []
        for t in range(xs[0].shape[1]):
            with redirect_stdout(io.StringIO()):
                frame, cache = model.forward(*[x[:, t : t + 1] for x in xs], cache=cache)
            cache = Maai._trim_vap_cache(cache, context_limit - 1)
            frames.append(frame)
        _assert_frames_close(frames, out, 0, label=f"{cls.__name__} (stream): ")


@torch.no_grad()
def test_fold_matches_full_forward():
    context_limit = 5
    for cls in (VapGPT, VapGPT_mono, VapGPT_nod_para):
        model = _random_model(cls, context_limit=context_limit)
        xs = _inputs(model, batch=2, frames=97)
        full = model.forward_batch(*xs)

        batch = MaaiBatch.__new__(MaaiBatch)
        # Receptive field: 4 attention layers x (context_limit - 1) frames.
        batch.fold_context_frames = 4 * (context_limit - 1)
        for step, batch_size in ((7, 3), (11, 8), (200, 1)):
            batch.fold_step_frames = step
            batch.batch_size = batch_size
            folded = batch._forward_folded(model, xs[0], xs[1] if len(xs) > 1 else None)
            for key in full:
                diff = (folded[key].double() - full[key].double()).abs().max().item()
                assert diff <= 1e-9, f"{cls.__name__} {key} step={step}: {diff:.2e}"


def _fake_batch(specs, mono=False, batch_size=2, drop_first_frame=True):
    """MaaiBatch with random models and a causal stand-in for the Mimi encoder.

    specs: list of (label, model class, mode).
    """
    batch = MaaiBatch.__new__(MaaiBatch)
    batch.samples_per_frame = 1280
    batch.mono = mono
    batch.batch_size = batch_size
    batch.drop_first_frame = drop_first_frame
    batch.fold_context_frames = 4 * (6 - 1)
    batch.fold_step_frames = 7
    batch._nested = len(specs) > 1
    batch.models = []
    for seed, (label, cls, mode) in enumerate(specs):
        model = _random_model(cls, context_limit=6, seed=seed)
        if cls is not VapGPT_nod_para:
            model.decrease_dimension = nn.Linear(512, model.conf.dim).double()
        batch.models.append(_BatchModel(label, mode, cls.__name__, model, return_p_bins=False))
    batch._needs_ch2 = any(not m.single_tower for m in batch.models)

    w = torch.randn(1280, 512, generator=torch.Generator().manual_seed(7), dtype=torch.float64)

    def encode(waves):  # (N, K * 1280) -> (N, K, 512), frame k depends on frame k only
        x = torch.from_numpy(waves).double()
        return torch.tanh(x.view(x.shape[0], -1, 1280) @ w * 10.0)

    batch._encode = encode
    batch.encoder_type = "mimi"
    batch._feature_dim = 512
    return batch


def _random_wave(seconds, seed):
    return np.random.default_rng(seed).standard_normal(int(seconds * 16000)).astype(np.float32) * 0.1


@torch.no_grad()
def test_process_many_matches_process():
    specs = [("vap", VapGPT, "vap"), ("bc", VapGPT_bc, "bc"), ("nod_para", VapGPT_nod_para, "nod_para")]
    multi = _fake_batch(specs)
    # Different lengths (incl. a trailing partial frame and a too-short item) so
    # that grouping, padding and the restored order are exercised.
    lengths = [5.3, 2.0, 7.95, 0.1, 4.0]
    items = [(_random_wave(s, 2 * i), _random_wave(s * 0.9, 2 * i + 1)) for i, s in enumerate(lengths)]
    many = multi.process_many(items)

    for (a, b), got, sec in zip(items, many, lengths):
        k = int(sec * 16000) // 1280
        n = max(0, k - 1)
        assert np.array_equal(got["end_sample"], (np.arange(n) + 2) * 1280), sec
        assert np.allclose(got["time_sec"], got["end_sample"] / 16000)
        alone = multi.process(a, b)
        for label, cls, _ in specs:
            single = _fake_batch([(label, cls, label)], batch_size=1)
            single.models[0].vap = multi.models[[s[0] for s in specs].index(label)].vap
            ref = single.process(a, b)
            for key, val in ref.items():
                if key in ("end_sample", "time_sec"):
                    continue
                assert val.shape[0] == n, (label, key, val.shape)
                assert np.allclose(got[label][key], val, atol=1e-9), (label, key, sec)
                assert np.allclose(alone[label][key], val, atol=1e-9), (label, key, sec)


@torch.no_grad()
def test_process_mono_and_drop_first():
    batch = _fake_batch([("vap", VapGPT_mono, "vap")], mono=True, drop_first_frame=False)
    a = _random_wave(3.0, 0)
    out = batch.process(a)
    k = len(a) // 1280
    assert np.array_equal(out["end_sample"], (np.arange(k) + 1) * 1280)
    assert out["p_now"].shape == (k,) and out["vad"].shape == (k,)
    assert "p_bins" not in out  # return_p_bins=False
    frames = MaaiBatch.to_frames(out)
    assert len(frames) == k and isinstance(frames[0]["p_now"], float)

    try:
        batch.process(a, a)
    except ValueError:
        pass
    else:
        raise AssertionError("mono=True must reject audio_ch2")


def _random_cpc_encoder(kernel, seed=3):
    """EncoderCPC with random weights (the real one loads the CPC checkpoint)."""
    torch.manual_seed(seed)
    enc = EncoderCPC.__new__(EncoderCPC)
    nn.Module.__init__(enc)
    enc.encoder = CPCModel(CPCEncoder(256), CPCAR(256, 256, True, 1, mode="LSTM"))
    enc.output_dim = enc.dim = 256
    enc.downsample = get_cnn_layer(dim=256, kernel=[kernel], stride=[2], dilation=[1], activation="GELU")
    return enc.double().eval()


def _cpc_realtime(model, a, b, hop, context_limit):
    """Maai.process with the CPC encoder: [320 samples of overlap + one frame]
    per call, KV cache trimmed to context_limit - 1, the last frame reported."""
    pad = np.zeros(320, dtype=np.float32)
    sa = np.concatenate([pad, a])
    sb = None if b is None else np.concatenate([pad, b])
    for enc in (model.encoder1, getattr(model, "encoder2", None)):
        if enc is not None:
            enc.reset_streaming_state()
    cache, frames = None, []
    for k in range(len(a) // hop):
        w1 = torch.from_numpy(sa[k * hop : k * hop + hop + 320]).double().view(1, 1, -1)
        if b is None:
            e = model.encode_audio(w1)
            frame, cache = model.forward(e, cache=cache)
        else:
            w2 = torch.from_numpy(sb[k * hop : k * hop + hop + 320]).double().view(1, 1, -1)
            e1, e2 = model.encode_audio(w1, w2)  # applies the bc/nod channel swap
            frame, cache = model.forward(e1, e2, cache=cache)
        cache = Maai._trim_vap_cache(cache, context_limit - 1)
        frames.append(frame)
    return frames


@torch.no_grad()
def test_cpc_matches_realtime_stream():
    context_limit = 6
    cases = [
        # (model class, mode, samples per frame, downsampling kernel)
        (VapGPT, "vap", 1600, 10),       # 10 CPC frames -> 1 frame per call (vap 10 Hz)
        (VapGPT_bc, "bc", 1600, 5),      # 10 CPC frames -> 3 frames per call (bc 10 Hz)
        (VapGPT_mono, "vap", 320, 2),    # 2 CPC frames -> 1 frame per call (50 Hz)
    ]
    for cls, mode, hop, kernel in cases:
        model = _random_model(cls, context_limit=context_limit)
        model.encoder1 = _random_cpc_encoder(kernel)
        mono = getattr(model, "is_single_tower", False)
        if not mono:
            model.encoder2 = copy.deepcopy(model.encoder1)

        a = _random_wave(30 * hop / 16000 + 0.03, 0)
        b = None if mono else _random_wave(30 * hop / 16000 + 0.03, 1)
        ref = _cpc_realtime(model, a, b, hop, context_limit)

        batch = MaaiBatch.__new__(MaaiBatch)
        batch.encoder_type = "cpc"
        batch.device = "cpu"
        batch.samples_per_frame = hop
        batch.mono = mono
        batch.batch_size = 3
        batch.encoder_chunk_frames = 7
        batch.drop_first_frame = False
        batch.fold_context_frames = 4 * (context_limit - 1)
        batch.fold_step_frames = 5
        batch._nested = False
        batch.models = [_BatchModel(mode, mode, cls.__name__, model, return_p_bins=True)]
        batch._needs_ch2 = not mono
        batch._core = model.encoder1
        batch._feature_dim = 256

        out = batch.process(a) if mono else batch.process(a, b)
        assert len(ref) == len(out["end_sample"]) == len(a) // hop
        assert np.array_equal(out["end_sample"], (np.arange(len(ref)) + 1) * hop)
        got = {k: torch.from_numpy(v).unsqueeze(0) for k, v in out.items() if k not in ("end_sample", "time_sec")}
        _assert_frames_close(ref, got, 0, label=f"CPC {cls.__name__} hop={hop} kernel={kernel}: ")


# ----------------------------------------------------------------------
# Pretrained models
# ----------------------------------------------------------------------

def _load_pair(seconds):
    import soundfile as sf

    a, sr1 = sf.read(WAV1, dtype="float32")
    b, sr2 = sf.read(WAV2, dtype="float32")
    assert sr1 == sr2 == 16000
    n = int(seconds * 16000)
    return a[:n], b[:n]


def _run_realtime(mode, a, b, model_type="normal-ver2", frame_rate=12.5, **kwargs):
    from maai import MaaiInput

    with redirect_stdout(io.StringIO()):
        maai = Maai(
            mode=mode,
            lang="jp",
            frame_rate=frame_rate,
            context_len_sec=20,
            model_type=model_type,
            audio_ch1=MaaiInput.Chunk(),
            audio_ch2=MaaiInput.Chunk(),
            **kwargs,
        )
        for s in range(0, len(a) - 159, 160):
            maai.process(a[s : s + 160], b[s : s + 160])
    results = []
    while not maai.result_dict_queue.empty():
        results.append(maai.get_result())
    return results


def _compare(results, out, keys):
    n = min(len(results), len(out["end_sample"]))
    assert n > 0
    stats = {}
    for key in keys:
        rt = np.asarray([r[key] for r in results[:n]], dtype=np.float64)
        bt = out[key][:n].astype(np.float64)
        stats[key] = (np.abs(rt - bt).max(), np.corrcoef(rt.ravel(), bt.ravel())[0, 1])
    return n, stats


@torch.no_grad()
def test_mimi_chunk_invariance():
    if not PRETRAINED:
        print("skip: set MAAI_TEST_PRETRAINED=1")
        return
    a, b = _load_pair(30)
    batch = MaaiBatch("vap", "jp")
    waves = np.stack([a, b])
    ref = None
    for chunk in (1, 7, 125, 400):
        batch.encoder_chunk_frames = chunk
        feats = batch._encode(waves)
        single = batch._encode(waves[1:])
        assert (feats[1:] - single).abs().max().item() < 1e-4, "batching changed the features"
        if ref is None:
            ref = feats
            continue
        diff = (feats - ref).abs().max().item()
        print(f"mimi encoder_chunk_frames={chunk}: max abs diff vs 1 frame/call = {diff:.2e}")
        assert diff < 1e-4, chunk


@torch.no_grad()
def test_batch_vs_realtime_pytorch_mimi():
    if not PRETRAINED:
        print("skip: set MAAI_TEST_PRETRAINED=1")
        return
    a, b = _load_pair(40)
    for mode, keys in (("vap", ("p_now", "p_future", "vad")), ("bc", ("p_bc", "p_bc_detect"))):
        results = _run_realtime(mode, a, b, use_mimi_onnx=False)
        with redirect_stdout(io.StringIO()):
            out = MaaiBatch(mode, "jp", drop_first_frame=False).process(a, b)
        assert len(results) == len(out["end_sample"]), (len(results), len(out["end_sample"]))
        n, stats = _compare(results, out, keys)
        for key, (diff, corr) in stats.items():
            print(f"[{mode}] pytorch-mimi realtime vs batch {key}: frames={n} max abs diff={diff:.2e} corr={corr:.6f}")
            assert diff < 1e-3, (mode, key, diff)


@torch.no_grad()
def test_batch_vs_realtime_onnx_mimi():
    if not PRETRAINED:
        print("skip: set MAAI_TEST_PRETRAINED=1")
        return
    a, b = _load_pair(40)
    for mode, keys in (("vap", ("p_now", "p_future", "vad")), ("bc", ("p_bc", "p_bc_detect"))):
        results = _run_realtime(mode, a, b)
        with redirect_stdout(io.StringIO()):
            out = MaaiBatch(mode, "jp").process(a, b)
        assert len(results) == len(out["end_sample"]), (len(results), len(out["end_sample"]))
        n, stats = _compare(results, out, keys)
        for key, (diff, corr) in stats.items():
            print(f"[{mode}] onnx-mimi realtime vs batch {key}: frames={n} max abs diff={diff:.2e} corr={corr:.6f}")
            # ONNX Runtime vs PyTorch Mimi: ~1e-4 on CPU; leave room for GPU kernels.
            assert diff < 1e-2 and corr > 0.999, (mode, key, diff, corr)


@torch.no_grad()
def test_batch_vs_realtime_cpc():
    if not PRETRAINED:
        print("skip: set MAAI_TEST_PRETRAINED=1")
        return
    a, b = _load_pair(40)
    # bc at 10 Hz adds three transformer frames per call (kernel 5 / stride 2).
    for mode, keys in (("vap", ("p_now", "p_future", "vad")), ("bc", ("p_bc", "p_bc_detect"))):
        results = _run_realtime(mode, a, b, model_type="normal", frame_rate=10)
        with redirect_stdout(io.StringIO()):
            out = MaaiBatch(mode, "jp", model_type="normal").process(a, b)
        assert len(results) == len(out["end_sample"]), (len(results), len(out["end_sample"]))
        n, stats = _compare(results, out, keys)
        for key, (diff, corr) in stats.items():
            print(f"[{mode}] cpc realtime vs batch {key}: frames={n} max abs diff={diff:.2e} corr={corr:.6f}")
            assert diff < 1e-3, (mode, key, diff)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
