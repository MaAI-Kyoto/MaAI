"""Experimental three-party VAP runtime (``MaaiTri``).

Kept separate from :class:`maai.model.Maai`, which is built around two
channels. The public API mirrors ``Maai`` (``start`` / ``get_result`` /
``stop``); results carry ``x1``-``x3`` and three-element ``p_now``,
``p_future`` and ``vad`` lists.
"""

import queue
import threading
import time

import numpy as np
import torch

from .input import Base
from .models.config import VapConfig
from .models.vap_tri import VapGPT_tri, VapTriConfig
from .util import load_vap_tri_model


class MaaiTri:
    """Runtime for the experimental three-party (triadic) VAP model.

    Each speaker's 16 kHz audio is encoded by its own streaming Mimi encoder
    (12.5 Hz). Encoder features of the last ``context_len_sec`` seconds are
    kept and the VAP transformer is re-run over that window every frame (the
    cross-channel transformer uses absolute time positions, so no KV cache).

    The fine-tuned Mimi that the model was trained with is resolved from the
    ``mimi_onnx_variant`` recorded in the checkpoint, unless a local ONNX file
    is given with ``mimi_local_onnx_fp32_path``.
    """

    NUM_SPEAKERS = 3
    CALC_PROCESS_TIME_INTERVAL = 100

    def __init__(
        self,
        audio_ch1: Base,
        audio_ch2: Base,
        audio_ch3: Base,
        lang: str = "jp",
        frame_rate: float = 12.5,
        context_len_sec: float = 20,
        device: str = "cpu",
        local_model: str | None = None,
        use_mimi_onnx: bool = True,
        mimi_onnx_precision: str = "fp32",
        mimi_local_onnx_fp32_path: str | None = None,
        mimi_local_onnx_fp32_meta_path: str | None = None,
        mimi_local_onnx_int8_path: str | None = None,
        mimi_local_onnx_int8_meta_path: str | None = None,
        mimi_onnx_cpu_intra_threads: int | None = None,
        mimi_onnx_cpu_inter_threads: int | None = None,
        cache_dir: str | None = None,
        force_download: bool = False,
        return_p_bins: bool = False,
        present: list | None = None,
    ):
        """Initialize the MaaiTri instance.

        Args:
            audio_ch1 (Base): Audio input source for speaker 1.
            audio_ch2 (Base): Audio input source for speaker 2.
            audio_ch3 (Base): Audio input source for speaker 3.
            lang (str): Language of the pretrained model on the hub.
            frame_rate (float): Frame rate. Only 12.5 Hz is available.
            context_len_sec (float): Context length in seconds (trained with 20).
            device (str): 'cpu', 'cuda' or 'cuda:N'.
            local_model (str | None): Path to a local model file (vap_3party_mimi_state_dict_*.pt) instead of downloading it.
                When omitted the model is downloaded from the hub.
            use_mimi_onnx (bool): Use the ONNX Mimi encoder (recommended).
                The PyTorch encoder uses the fine-tuned Mimi weights only when
                they were exported with ``--include-mimi``.
            mimi_onnx_precision (str): 'fp32' or 'int8'.
            mimi_local_onnx_fp32_path (str | None): Local fine-tuned Mimi ONNX file.
            mimi_local_onnx_fp32_meta_path (str | None): Its JSON sidecar
                (defaults to ``<onnx path>.json``).
            mimi_local_onnx_int8_path (str | None): Local INT8 Mimi ONNX file.
            mimi_local_onnx_int8_meta_path (str | None): Its JSON sidecar.
            mimi_onnx_cpu_intra_threads (int | None): ONNX CPU intra-op threads.
            mimi_onnx_cpu_inter_threads (int | None): ONNX CPU inter-op threads.
            cache_dir (str | None): Cache directory for downloaded files.
            force_download (bool): Force re-download of hub files.
            return_p_bins (bool): Also return ``p_bins`` ([speaker][bin] probabilities).
            present (list | None): Which speakers are present, e.g. [1, 1, 0]
                for a two-party conversation on the three-party model.
        """
        if float(frame_rate) != 12.5:
            raise ValueError("MaaiTri supports frame_rate=12.5 only.")

        try:
            self.device = str(torch.device(device))
        except RuntimeError as exc:
            raise ValueError("Device must be a valid torch device string such as 'cpu', 'cuda', or 'cuda:0'.") from exc
        if not (self.device == "cpu" or self.device.startswith("cuda")):
            raise ValueError("Device must be 'cpu', 'cuda', or 'cuda:N'.")

        precision = str(mimi_onnx_precision).strip().lower()
        if precision not in {"fp32", "int8"}:
            raise ValueError("mimi_onnx_precision must be 'fp32' or 'int8'.")
        if self.device.startswith("cuda") and precision == "int8":
            raise ValueError("mimi_onnx_precision='int8' is not supported with CUDA. Use 'fp32' on CUDA.")

        # Load the exported checkpoint
        if local_model is None:
            raw = load_vap_tri_model(lang, cache_dir=cache_dir, force_download=force_download)
        else:
            print("Loading model from local file:", local_model)
            raw = torch.load(local_model, map_location="cpu")
        if not isinstance(raw, dict) or "state_dict" not in raw:
            raise ValueError("Unexpected MaaiTri checkpoint format (missing 'state_dict').")
        tri_conf = VapTriConfig.from_dict(raw.get("maai_tri_config"))
        if float(tri_conf.frame_hz) != float(frame_rate):
            raise ValueError(f"Checkpoint frame rate {tri_conf.frame_hz} != requested {frame_rate}.")
        self.source_info = raw.get("source", {})

        # Encoder settings (read by build_audio_encoder)
        enc_conf = VapConfig()
        enc_conf.frame_hz = float(frame_rate)
        enc_conf.encoder_type = "mimi"
        enc_conf.runtime_device = self.device
        enc_conf.mimi_use_onnx = 1 if use_mimi_onnx else 0
        enc_conf.mimi_onnx_precision = precision
        enc_conf.mimi_onnx_frames_per_call = 1
        enc_conf.mimi_onnx_variant = tri_conf.mimi_onnx_variant
        if mimi_onnx_cpu_intra_threads is not None:
            enc_conf.mimi_onnx_cpu_intra_threads = int(mimi_onnx_cpu_intra_threads)
        if mimi_onnx_cpu_inter_threads is not None:
            enc_conf.mimi_onnx_cpu_inter_threads = int(mimi_onnx_cpu_inter_threads)
        for name, value in (
            ("mimi_onnx_fp32_path", mimi_local_onnx_fp32_path),
            ("mimi_onnx_fp32_meta_path", mimi_local_onnx_fp32_meta_path),
            ("mimi_onnx_int8_path", mimi_local_onnx_int8_path),
            ("mimi_onnx_int8_meta_path", mimi_local_onnx_int8_meta_path),
        ):
            if value is not None:
                setattr(enc_conf, name, str(value))
        if cache_dir is not None:
            enc_conf.mimi_onnx_hf_cache_dir = cache_dir
        enc_conf.mimi_onnx_hf_force_download = bool(force_download)

        self.vap = VapGPT_tri(tri_conf, enc_conf)
        missing, unexpected = self.vap.load_state_dict(raw["state_dict"], strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"MaaiTri checkpoint does not match the model: missing={missing}, unexpected={unexpected}"
            )
        self.vap.load_encoder()

        if not use_mimi_onnx:
            mimi_sd = raw.get("mimi_state_dict")
            if mimi_sd:
                for name in self.vap.encoder_names():
                    getattr(self.vap, name).model.load_state_dict(mimi_sd, strict=False)
                print("Loaded fine-tuned Mimi weights into the PyTorch encoder.")
            else:
                print(
                    "[Warning] The checkpoint has no fine-tuned Mimi weights; the PyTorch encoder "
                    "uses the original kyutai/mimi. Use use_mimi_onnx=True for the fine-tuned Mimi."
                )

        self.vap.to(self.device)
        self.vap = self.vap.eval()

        self.return_p_bins = bool(return_p_bins)
        self._use_mimi_onnx = bool(use_mimi_onnx)
        self.frame_rate = float(frame_rate)
        self.audio_context_len = int(round(float(context_len_sec) * self.frame_rate))

        self.mics = [audio_ch1, audio_ch2, audio_ch3]
        self._mic_queues = [m.subscribe() for m in self.mics]

        self.sampling_rate = 16000
        self.frame_contxt_padding = 320
        # 12.5Hz -> 320 + 1280 samples
        self.audio_samples_per_frame = int(round(self.sampling_rate / self.frame_rate))
        self.audio_frame_size = self.audio_samples_per_frame + self.frame_contxt_padding

        self.set_present(present)

        self.result_dict_queue = queue.Queue()
        self.list_process_time_context = []
        self.last_interval_time = time.time()
        self.process_time_abs = -1

        self._stop_event = threading.Event()
        self._worker_thread = None

        self.reset_runtime_state()

    def set_present(self, present: list | None = None):
        """Set which speakers are present (1) or absent (0), e.g. [1, 1, 0]."""
        if present is None:
            present = [1.0] * self.NUM_SPEAKERS
        if len(present) != self.NUM_SPEAKERS:
            raise ValueError(f"present must have {self.NUM_SPEAKERS} elements.")
        self._present = torch.tensor([present], dtype=torch.float32, device=self.device)

    def reset_runtime_state(self):
        """Reset the internal audio buffers, feature window and encoder states."""
        self.current_audio = [[] for _ in range(self.NUM_SPEAKERS)]
        self.e_full = [None] * self.NUM_SPEAKERS
        self._skip_first_encoder_output = self._use_mimi_onnx
        for name in self.vap.encoder_names():
            encoder = getattr(self.vap, name, None)
            if encoder is not None and hasattr(encoder, "reset_streaming_state"):
                encoder.reset_streaming_state()

    def _keep_padding(self):
        self.current_audio = [a[-self.frame_contxt_padding:].copy() for a in self.current_audio]

    def worker(self):
        """Background loop to fetch audio from queues and run inference."""
        for q in self._mic_queues:
            q.queue.clear()

        while not self._stop_event.is_set():
            xs = [m.get_audio_data(q) for m, q in zip(self.mics, self._mic_queues)]
            if self._stop_event.is_set() or any(x is None for x in xs):
                break

            self.process(xs)

            for i, q in enumerate(self._mic_queues):
                if q.qsize() > 100:
                    q.queue.clear()
                    print(f"[Warning] Audio queue (channel {i + 1}) overflow detected. Clearing audio queues.")

    def start(self):
        """Start the background audio fetching and processing thread."""
        self.reset_runtime_state()
        for m in self.mics:
            m.start()
        self._stop_event.clear()
        for q in self._mic_queues:
            q.queue.clear()
        self._worker_thread = threading.Thread(target=self.worker, daemon=True)
        self._worker_thread.start()

    def stop(self, wait: bool = True, timeout: float = 2.0):
        """Safely stop the background processing thread."""
        self._stop_event.set()
        try:
            for q in self._mic_queues:
                q.put(None)
        except Exception:
            pass
        if wait and self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        try:
            for q in self._mic_queues:
                q.queue.clear()
        except Exception:
            pass
        self.reset_runtime_state()

    def process(self, xs):
        """Process one chunk of audio (a list with one array per speaker)."""
        time_start = time.time()

        for i, x in enumerate(xs):
            if len(self.current_audio[i]) == 0:
                self.current_audio[i] = np.zeros(self.frame_contxt_padding, dtype=np.float32)
            self.current_audio[i] = np.concatenate([self.current_audio[i], np.asarray(x, dtype=np.float32)])

        if len(self.current_audio[0]) < self.audio_frame_size:
            return

        with torch.inference_mode():
            audios = [
                torch.from_numpy(a).float().view(1, 1, -1).to(self.device, non_blocking=True)
                for a in self.current_audio
            ]
            es = self.vap.encode_audio(audios)

            if any(e.shape[1] == 0 for e in es):
                self._keep_padding()
                print("[Warning] No audio features extracted. Skipping this frame.")
                return

            # Skip the first Mimi encoder output to avoid the startup-only mismatch
            # between ONNX and PyTorch cache warmup behavior (same as Maai).
            if self._skip_first_encoder_output:
                self._skip_first_encoder_output = False
                self.process_time_abs = time.time()
                self._keep_padding()
                return

            for i, e in enumerate(es):
                full = e if self.e_full[i] is None else torch.cat([self.e_full[i], e], dim=1)
                self.e_full[i] = full[:, -self.audio_context_len:]

            out = self.vap(self.e_full, present=self._present)

        result_dict = {"t": time.time()}
        for i, a in enumerate(self.current_audio):
            result_dict[f"x{i + 1}"] = a[self.frame_contxt_padding:].copy()
        result_dict["p_now"] = out["p_now"][0, -1].tolist()
        if out["p_future"] is not None:
            result_dict["p_future"] = out["p_future"][0, -1].tolist()
        # An absent speaker's input is replaced by the absent embedding, so its VAD is meaningless.
        result_dict["vad"] = (out["vad"][0, -1] * self._present[0]).tolist()
        if self.return_p_bins:
            result_dict["p_bins"] = out["query_va_probs"][0, -1].tolist()
        self.result_dict_queue.put(result_dict)

        self.list_process_time_context.append(time.time() - time_start)
        if len(self.list_process_time_context) > self.CALC_PROCESS_TIME_INTERVAL:
            ave_proc_time = np.mean(self.list_process_time_context)
            num_process_frame = len(self.list_process_time_context) / (time.time() - self.last_interval_time)
            self.last_interval_time = time.time()
            rtf = ave_proc_time / (1.0 / self.frame_rate)
            perf_message = (
                f"[vap_tri] Average processing time: {ave_proc_time:.5f} [sec], "
                f"#process/sec: {num_process_frame:.3f}, RTF: {rtf:.2f}"
            )
            if rtf > 1.0:
                perf_message += "  [WARNING] HIGH LOAD (RTF > 1.0)"
            print(perf_message)
            self.list_process_time_context.clear()

        self.process_time_abs = time.time()
        self._keep_padding()

    def get_result(self):
        """Retrieve the next inference result (blocks until available)."""
        return self.result_dict_queue.get()
