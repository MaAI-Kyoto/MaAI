import torch
import torch.nn as nn
import time
import numpy as np
import threading
import queue
import copy
import os
import sys

from .input import Base, Zero
from .util import load_vap_model, resolve_encoder_type
from .models.vap import VapGPT
from .models.vap_mono import VapGPT_mono
from .models.vad import VadGPT, VadGPT_mono
from .models.bc_det import BcDetGPT, BcDetGPT_mono
from .models.vap_bc import VapGPT_bc
from .models.vap_bc_2type import VapGPT_bc_2type
from .models.vap_nod import VapGPT_nod
from .models.vap_nod_para import VapGPT_nod_para
from .models.config import VapConfig
# from .models.vap_prompt import VapGPT_prompt

# 2-channel modes that have a dedicated single-channel (monaural) counterpart.
MONO_ALTERNATIVE_MODES = {
    "vad": "vad_mono",
    "bc_det": "bc_det_mono",
}


def _confirm_zero_second_channel(mode: str, mono_mode: str) -> None:
    """Warn that a 2-channel mode is being fed a silent second channel.

    Asks on stdin whether to keep going with `mode` or to quit so that the
    dedicated monaural mode can be used instead. If stdin is not interactive
    the warning is printed and processing continues.

    Args:
        mode (str): The 2-channel mode that was requested.
        mono_mode (str): The recommended single-channel mode.

    Raises:
        SystemExit: If the user chooses not to continue.
    """
    print(
        f"[WARNING] mode='{mode}' is a 2-channel model, but the second channel is "
        "MaaiInput.Zero (silence). For single-channel (monaural) input, "
        f"mode='{mono_mode}' is recommended."
    )

    if not sys.stdin or not sys.stdin.isatty():
        print(f"[WARNING] stdin is not interactive; continuing with mode='{mode}'.")
        return

    while True:
        try:
            answer = input(f"Continue with mode='{mode}'? [y/N]: ").strip().lower()
        except EOFError:
            answer = ""
        if answer in ("y", "yes"):
            return
        if answer in ("", "n", "no"):
            raise SystemExit(f"Aborted. Please use mode='{mono_mode}' instead.")
        print("Please answer 'y' or 'n'.")


class Maai():
    """Main wrapper class for running the MaAI model.
    
    Handles audio input streams, model loading, audio processing,
    feature extraction, and VAP outputs in a background thread.
    """
    
    BINS_P_NOW = [0, 1]
    BINS_PFUTURE = [2, 3]
    
    CALC_PROCESS_TIME_INTERVAL = 100

    def __init__(
        self,
        mode,
        lang: str,
        audio_ch1: Base,
        audio_ch2: Base = None,
        frame_rate: float = 10,
        context_len_sec: int = 20,
        device: str = "cpu",
        # num_channels: int = 2,
        cpc_model: str = os.path.expanduser("~/.cache/cpc/60k_epoch4-d0f474de.pt"),
        model_type: str = "normal",
        mimi_model_name: str = "kyutai/mimi",
        use_mimi_onnx: bool = True,
        mimi_onnx_precision: str = "fp32",
        mimi_onnx_fp32_path: str | None = None,
        mimi_onnx_fp32_meta_path: str | None = None,
        mimi_onnx_int8_path: str | None = None,
        mimi_onnx_int8_meta_path: str | None = None,
        mimi_local_onnx_fp32_path: str | None = None,
        mimi_local_onnx_fp32_meta_path: str | None = None,
        mimi_local_onnx_int8_path: str | None = None,
        mimi_local_onnx_int8_meta_path: str | None = None,
        mimi_onnx_cpu_intra_threads: int | None = None,
        mimi_onnx_cpu_inter_threads: int | None = None,
        cache_dir: str = None,
        force_download: bool = False,
        use_kv_cache: bool = True,
        local_model = None,
        return_p_bins: bool = False,
        inference_chunk_frames: int = 1,
    ):
        """Initialize the Maai instance.
        
        Args:
            mode (str): Operational mode (e.g., 'vap', 'bc', 'nod').
            lang (str): Language setting (e.g., 'jp', 'en').
            audio_ch1 (Base): Audio input source for channel 1.
            audio_ch2 (Base): Audio input source for channel 2.
                Not used in 'vap_mono' mode (silence is fed internally).
            frame_rate (float): Frame rate for processing audio.
            context_len_sec (int): Audio context length in seconds.
            device (str): Device to run the model on ('cpu', 'cuda').
            cpc_model (str): Path to CPC model weights.
            model_type (str): General model type (e.g., 'normal').
            mimi_model_name (str): Hugging Face model name for Mimi.
            use_mimi_onnx (bool): Whether to use ONNX backend for Mimi.
            mimi_onnx_precision (str): Precision for ONNX model ('fp32', 'int8').
            mimi_onnx_fp32_path (str | None): Path to FP32 ONNX model.
            mimi_onnx_fp32_meta_path (str | None): Path to FP32 ONNX meta.
            mimi_onnx_int8_path (str | None): Path to INT8 ONNX model.
            mimi_onnx_int8_meta_path (str | None): Path to INT8 ONNX meta.
            mimi_local_onnx_fp32_path (str | None): Path to local FP32 ONNX model.
            mimi_local_onnx_fp32_meta_path (str | None): Path to local FP32 ONNX meta.
            mimi_local_onnx_int8_path (str | None): Path to local INT8 ONNX model.
            mimi_local_onnx_int8_meta_path (str | None): Path to local INT8 ONNX meta.
            mimi_onnx_cpu_intra_threads (int | None): ONNX CPU intra-op threads.
            mimi_onnx_cpu_inter_threads (int | None): ONNX CPU inter-op threads.
            cache_dir (str): Cache directory for model weights.
            force_download (bool): Force download of model weights.
            use_kv_cache (bool): Whether to use KV caching during inference.
            local_model (str | None): Path to a local custom model file.
            return_p_bins (bool): Whether to return probability bins in 'vap' mode.
            inference_chunk_frames (int): Number of Mimi/VAP frames evaluated
                in one causal forward. When greater than one, only the final
                frame of each batch is emitted through ``get_result()``.
        """

        if mode in ("vap_mono", "vad_mono", "bc_det_mono"):
            if audio_ch2 is None:
                audio_ch2 = Zero()
            elif not isinstance(audio_ch2, Zero):
                raise ValueError(
                    f"mode='{mode}' takes a single input channel; "
                    "audio_ch2 must be omitted (or a MaaiInput.Zero)."
                )
        elif audio_ch2 is None:
            raise ValueError(f"audio_ch2 is required for mode '{mode}'.")
        elif mode in MONO_ALTERNATIVE_MODES and isinstance(audio_ch2, Zero):
            _confirm_zero_second_channel(mode, MONO_ALTERNATIVE_MODES[mode])

        self.return_p_bins = bool(return_p_bins)
        inference_chunk_frames = int(inference_chunk_frames)
        if inference_chunk_frames < 1:
            raise ValueError("inference_chunk_frames must be at least 1.")

        encoder_type = resolve_encoder_type(model_type)

        conf = VapConfig()
        conf.frame_hz = float(frame_rate)
        # The cache is retained as context_len-1 history frames.  This mask
        # also makes each query in a multi-frame forward observe the same
        # sliding context that it would have received in separate forwards.
        conf.context_limit = int(round(float(context_len_sec) * float(frame_rate)))
        conf.encoder_type = encoder_type
        conf.mimi_model_name = mimi_model_name
        conf.mimi_use_onnx = 1 if bool(use_mimi_onnx) else 0
        conf.mimi_onnx_frames_per_call = int(inference_chunk_frames)
        precision = str(mimi_onnx_precision).strip().lower()
        if precision not in {"fp32", "int8"}:
            raise ValueError("mimi_onnx_precision must be 'fp32' or 'int8'.")
        if str(device).startswith("cuda") and precision == "int8":
            raise ValueError("mimi_onnx_precision='int8' is not supported with CUDA. Use 'fp32' on CUDA.")
        conf.mimi_onnx_precision = precision
        if mimi_onnx_cpu_intra_threads is not None:
            conf.mimi_onnx_cpu_intra_threads = int(mimi_onnx_cpu_intra_threads)
        if mimi_onnx_cpu_inter_threads is not None:
            conf.mimi_onnx_cpu_inter_threads = int(mimi_onnx_cpu_inter_threads)
        fp32_onnx = mimi_local_onnx_fp32_path or mimi_onnx_fp32_path
        fp32_meta = mimi_local_onnx_fp32_meta_path or mimi_onnx_fp32_meta_path
        int8_onnx = mimi_local_onnx_int8_path or mimi_onnx_int8_path
        int8_meta = mimi_local_onnx_int8_meta_path or mimi_onnx_int8_meta_path
        if fp32_onnx is not None:
            conf.mimi_onnx_fp32_path = str(fp32_onnx)
        if fp32_meta is not None:
            conf.mimi_onnx_fp32_meta_path = str(fp32_meta)
        if int8_onnx is not None:
            conf.mimi_onnx_int8_path = str(int8_onnx)
        if int8_meta is not None:
            conf.mimi_onnx_int8_meta_path = str(int8_meta)
        if cache_dir is not None:
            conf.mimi_onnx_hf_cache_dir = cache_dir
        conf.mimi_onnx_hf_force_download = bool(force_download)

        # # Middle size model
        # if "middle" in lang:
        #     conf.dim = 256
        #     conf.channel_layers = 2
        #     conf.cross_layers = 6
        #     conf.num_heads = 8
        
        if mode in ["vap", "vap_mc"]:
            self.vap = VapGPT(conf)

        elif mode == "vap_mono":
            self.vap = VapGPT_mono(conf)

        elif mode == "vad":
            self.vap = VadGPT(conf)

        elif mode == "vad_mono":
            self.vap = VadGPT_mono(conf)

        elif mode == "bc_det":
            self.vap = BcDetGPT(conf)

        elif mode == "bc_det_mono":
            self.vap = BcDetGPT_mono(conf)

        elif mode == "bc":
            self.vap = VapGPT_bc(conf)
        
        elif mode == "bc_2type":
            self.vap = VapGPT_bc_2type(conf)
        
        elif mode == "nod":
            self.vap = VapGPT_nod(conf)

        elif mode == "nod_para":
            conf.dropout = 0.2
            self.vap = VapGPT_nod_para(conf)
        
        elif mode == "vap_prompt":
            from .models.vap_prompt import VapGPT_prompt
            self.vap = VapGPT_prompt(conf)
        
        try:
            self.device = str(torch.device(device))
        except RuntimeError as exc:
            raise ValueError("Device must be a valid torch device string such as 'cpu', 'cuda', or 'cuda:0'.") from exc

        if not (self.device == "cpu" or self.device.startswith("cuda")):
            raise ValueError("Device must be 'cpu', 'cuda', or 'cuda:N'.")
        
        # Store the initial state of the model to check for unchanged parameters
        initial_state_dict = {name: param.clone() for name, param in self.vap.named_parameters()}

        nod_param_stats_from_file = None
        nod_count_thresholds_from_file = None
        if local_model is None:
            sd = load_vap_model(
                mode,
                frame_rate,
                context_len_sec,
                lang,
                device,
                cache_dir,
                force_download,
                model_type=model_type,
            )
            if (
                mode == "nod_para"
                and isinstance(sd, dict)
                and "state_dict" in sd
            ):
                nod_param_stats_from_file = sd.get("nod_param_stats")
                nod_count_thresholds_from_file = sd.get("nod_count_thresholds") or sd.get(
                    "nod_repetitions_thresholds"
                )
                sd = sd["state_dict"]
        else:
            print("Loading model from local file:", local_model)
            raw = torch.load(local_model, map_location="cpu")
            if isinstance(raw, dict):
                nod_param_stats_from_file = raw.get("nod_param_stats")
                nod_count_thresholds_from_file = raw.get(
                    "nod_count_thresholds"
                ) or raw.get("nod_repetitions_thresholds")
                if "state_dict" in raw:
                    sd = raw["state_dict"]
                else:
                    sd = raw
            else:
                sd = raw

        if hasattr(self.vap, "conf"):
            setattr(self.vap.conf, "runtime_device", self.device)
        self.vap.load_encoder(cpc_model=cpc_model)
        if mode == "nod_para" and isinstance(sd, dict):
            remapped_sd: dict = {}
            for _k, _v in sd.items():
                if _k.startswith("nod_count_head."):
                    remapped_sd[
                        "nod_repetitions_head." + _k[len("nod_count_head.") :]
                    ] = _v
                else:
                    remapped_sd[_k] = _v
            sd = remapped_sd
        self.vap.load_state_dict(sd, strict=False)

        if (
            mode == "nod_para"
            and nod_param_stats_from_file is not None
            and isinstance(nod_param_stats_from_file, dict)
        ):
            for _k in ("range_mean", "range_std", "speed_mean", "speed_std"):
                if _k in nod_param_stats_from_file:
                    self.vap.nod_param_stats[_k] = float(nod_param_stats_from_file[_k])
        if (
            mode == "nod_para"
            and nod_count_thresholds_from_file is not None
            and isinstance(nod_count_thresholds_from_file, dict)
        ):
            for _k in ("t0", "t1", "t2"):
                if _k in nod_count_thresholds_from_file:
                    self.vap.nod_repetitions_thresholds[_k] = float(
                        nod_count_thresholds_from_file[_k]
                    )
            if "t_swing" in nod_count_thresholds_from_file:
                self.vap.nod_swing_up_threshold = float(nod_count_thresholds_from_file["t_swing"])

        if conf.encoder_type == "cpc" and 'encoder.downsample.1.weight' in sd:
            self.vap.encoder1.downsample[1].weight = nn.Parameter(sd['encoder.downsample.1.weight'])
            self.vap.encoder1.downsample[1].bias = nn.Parameter(sd['encoder.downsample.1.bias'])
            self.vap.encoder1.downsample[2].ln.weight = nn.Parameter(sd['encoder.downsample.2.ln.weight'])
            self.vap.encoder1.downsample[2].ln.bias = nn.Parameter(sd['encoder.downsample.2.ln.bias'])
            
            self.vap.encoder2.downsample[1].weight = nn.Parameter(sd['encoder.downsample.1.weight'])
            self.vap.encoder2.downsample[1].bias = nn.Parameter(sd['encoder.downsample.1.bias'])
            self.vap.encoder2.downsample[2].ln.weight = nn.Parameter(sd['encoder.downsample.2.ln.weight'])
            self.vap.encoder2.downsample[2].ln.bias = nn.Parameter(sd['encoder.downsample.2.ln.bias'])
        
        # print(sd.keys())
        # input("Model loaded. Press Enter to continue...")
        if conf.encoder_type == "mimi" and 'encoder.frame_rate_conv.weight' in sd:
            self.vap.encoder1.frame_rate_conv.weight = nn.Parameter(sd['encoder.frame_rate_conv.weight'])
            self.vap.encoder1.frame_rate_conv.bias = nn.Parameter(sd['encoder.frame_rate_conv.bias'])
            
            self.vap.encoder2.frame_rate_conv.weight = nn.Parameter(sd['encoder.frame_rate_conv.weight'])
            self.vap.encoder2.frame_rate_conv.bias = nn.Parameter(sd['encoder.frame_rate_conv.bias'])

        # Check for parameters that were not updated from their initial values
        for name, param in self.vap.named_parameters():
            if name in initial_state_dict:
                if torch.equal(param.data, initial_state_dict[name].data):
                    # Exclude encoder parameters that are loaded separately
                    if not name.startswith('encoder.'):
                        print(f"Warning: Parameter '{name}' was not updated from its initial value.")

        self.vap.to(self.device)
        self.vap = self.vap.eval()
        
        self.mode = mode
        self.model_type = model_type
        self.encoder_type = encoder_type
        self._use_mimi_onnx = bool(use_mimi_onnx)
        self.inference_chunk_frames = inference_chunk_frames
        if self.inference_chunk_frames > 1:
            if self.encoder_type != "mimi":
                raise ValueError(
                    "inference_chunk_frames > 1 currently supports the Mimi encoder only."
                )
        self.mic1 = audio_ch1
        self.mic2 = audio_ch2

        # Always subscribe a dedicated queue for each mic if possible
        self._mic1_queue = self.mic1.subscribe()
        self._mic2_queue = self.mic2.subscribe()

        self.audio_contenxt_lim_sec = context_len_sec
        self.frame_rate = float(frame_rate)
        
        # Context length of the audio embeddings (depends on frame rate)
        self.audio_context_len = int(round(self.audio_contenxt_lim_sec * self.frame_rate))
        
        self.sampling_rate = 16000
        self.frame_contxt_padding = 320
        
        # Frame size
        # 10Hz -> 320 + 1600 samples
        # 12.5Hz -> 320 + 1280 samples
        # 20Hz -> 320 + 800 samples
        # 50Hz -> 320 + 320 samples
        self.audio_samples_per_frame = int(round(self.sampling_rate / self.frame_rate))
        self.audio_frame_size = (
            self.audio_samples_per_frame * self.inference_chunk_frames
            + self.frame_contxt_padding
        )
        
        self.current_x1_audio = []
        self.current_x2_audio = []
        
        self.result_p_now = 0.
        self.result_p_future = 0.
        self.result_p_bc_react = 0.
        self.result_p_bc_emo = 0.
        self.result_p_bc = 0.
        self.result_p_nod_short = 0.
        self.result_p_nod_long = 0.
        self.result_p_nod_long_p = 0.
        self.result_last_time = -1
        
        self.result_vad = [0., 0.]

        self.process_time_abs = -1

        self.e1_full = []
        self.e2_full = []

        self.list_process_time_context = []
        self.last_interval_time = time.time()

        self.result_dict_queue = queue.Queue()

        self.use_kv_cache = use_kv_cache
        self.vap_cache = None
        
        # Thread control
        self._stop_event = threading.Event()
        self._worker_thread = None

        self.reset_runtime_state()

    def reset_runtime_state(self):
        """Reset the internal audio buffers and cache states."""
        self.current_x1_audio = []
        self.current_x2_audio = []
        self.e1_full = []
        self.e2_full = []
        self.vap_cache = None
        self._skip_first_encoder_output = bool(self.encoder_type == "mimi" and self._use_mimi_onnx)

        for encoder_name in ["encoder1", "encoder2"]:
            encoder = getattr(self.vap, encoder_name, None)
            if encoder is not None and hasattr(encoder, "reset_streaming_state"):
                encoder.reset_streaming_state()

    @staticmethod
    def _trim_vap_cache(cache: dict, max_frames: int) -> dict:
        """Keep only the newest ``max_frames`` entries of every VAP KV cache."""
        trimmed = {}
        for key, (k_list, v_list) in cache.items():
            trimmed[key] = (
                [
                    tensor[..., -max_frames:, :]
                    if isinstance(tensor, torch.Tensor) and tensor.dim() >= 3
                    else tensor
                    for tensor in k_list
                ],
                [
                    tensor[..., -max_frames:, :]
                    if isinstance(tensor, torch.Tensor) and tensor.dim() >= 3
                    else tensor
                    for tensor in v_list
                ],
            )
        return trimmed

    # def _increase_mimi_chunk_threshold(self, attempted_num_samples: int):
    #     if self.encoder_type != "mimi":
    #         return

    #     previous_threshold = int(self.audio_frame_size)
    #     next_threshold = int(attempted_num_samples) + int(Base.FRAME_SIZE)
    #     if next_threshold <= self.audio_frame_size:
    #         next_threshold = self.audio_frame_size + int(Base.FRAME_SIZE)

    #     self.audio_frame_size = next_threshold
    #     if self.audio_frame_size != previous_threshold:
    #         print(
    #             f"[Info] Mimi streaming chunk threshold adjusted: {previous_threshold} -> {self.audio_frame_size} samples "
    #             f"({self.audio_frame_size / self.sampling_rate:.3f} sec)."
    #         )
    
    def worker(self):
        """Background loop to fetch audio from queues and run inference."""
        
        # Clear the queues at the start
        # This is to ensure that the queues are empty before starting the processing loop
        self._mic1_queue.queue.clear()
        self._mic2_queue.queue.clear()
        
        while not self._stop_event.is_set():
            x1 = self.mic1.get_audio_data(self._mic1_queue)
            x2 = self.mic2.get_audio_data(self._mic2_queue)

            if self._stop_event.is_set() or x1 is None or x2 is None:
                break

            self.process(x1, x2)

            # Clear the queues if they are too large
            if self._mic1_queue.qsize() > 100:
                self._mic1_queue.queue.clear()
                print("[Warning] Audio queue (channel 1) overflow detected. Clearing audio queues.")
            if self._mic2_queue.qsize() > 100:
                self._mic2_queue.queue.clear()
                print("[Warning] Audio queue (channel 2) overflow detected. Clearing audio queues.")

            # print(self._mic1_queue.qsize(), self._mic2_queue.qsize())

            # self._mic1_queue.queue.clear()
            # self._mic2_queue.queue.clear()

    def start(self):
        """Start the background audio fetching and processing thread."""

        self.reset_runtime_state()

        self.mic1.start()
        self.mic2.start()
        self._stop_event.clear()
        # Queue を空にしてからスレッド起動（スレッドに古いデータが届かないよう先にクリア）
        self._mic1_queue.queue.clear()
        self._mic2_queue.queue.clear()
        self._worker_thread = threading.Thread(target=self.worker, daemon=True)
        self._worker_thread.start()
    
    def stop(self, wait: bool = True, timeout: float = 2.0):
        """
        Safely stop the background processing thread.
        Args:
            wait (bool): If True, wait for the thread to finish.
            timeout (float): Max seconds to wait when joining.
        """
        self._stop_event.set()
        # Unblock blocking gets by pushing sentinels
        try:
            self._mic1_queue.put(None)
            self._mic2_queue.put(None)
        except Exception:
            pass
        if wait and self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

        # Best-effort queue cleanup
        try:
            self._mic1_queue.queue.clear()
            self._mic2_queue.queue.clear()
        except Exception:
            pass

        self.reset_runtime_state()

    def process(self, x1, x2):
        """Process a chunk of audio for both channels.
        
        Args:
            x1 (np.ndarray): Audio data chunk for channel 1.
            x2 (np.ndarray): Audio data chunk for channel 2.
        """
        
        time_start = time.time()

        # Initialize buffer if empty
        if len(self.current_x1_audio) == 0:
            self.current_x1_audio = np.zeros(self.frame_contxt_padding, dtype=np.float32)
        if len(self.current_x2_audio) == 0:
            self.current_x2_audio = np.zeros(self.frame_contxt_padding, dtype=np.float32)
        
        # x1 = x1.astype(np.float32, copy=False)
        # x2 = x2.astype(np.float32, copy=False)

        # Add to buffer
        self.current_x1_audio = np.concatenate([self.current_x1_audio, x1])
        self.current_x2_audio = np.concatenate([self.current_x2_audio, x2])

        # Return if the buffer does not have enough length
        if len(self.current_x1_audio) < self.audio_frame_size:
            return

        # Extract data for inference
        x1_proc = self.current_x1_audio
        x2_proc = self.current_x2_audio

        x1_dist = x1_proc[self.frame_contxt_padding:]
        x2_dist = x2_proc[self.frame_contxt_padding:]

        with torch.inference_mode():
            # Create tensors more efficiently with specified dtype and device
            x1_ = torch.from_numpy(x1_proc).float().unsqueeze(0).unsqueeze(0)
            x2_ = torch.from_numpy(x2_proc).float().unsqueeze(0).unsqueeze(0)
            
            # Move to device only once
            if self.device != 'cpu':
                x1_ = x1_.to(self.device, non_blocking=True)
                x2_ = x2_.to(self.device, non_blocking=True)

            # try:
            e1, e2 = self.vap.encode_audio(x1_, x2_)
            # except RuntimeError as exc:
            #     short_chunk_error = (
            #         self.encoder_type == "mimi"
            #         and "Calculated padded input size per channel" in str(exc)
            #         and "Kernel size can't be greater than actual input size" in str(exc)
            #     )
            #     if short_chunk_error:
            #         self._increase_mimi_chunk_threshold(len(self.current_x1_audio))
            #         self.process_time_abs = time.time()
            #         return
            #     raise

            if e1.shape[1] == 0 or e2.shape[1] == 0:
                # if self.encoder_type == "mimi":
                #     self._increase_mimi_chunk_threshold(len(self.current_x1_audio))
                # self.process_time_abs = time.time()
                if self.frame_contxt_padding > 0:
                    self.current_x1_audio = self.current_x1_audio[-self.frame_contxt_padding:].copy()
                    self.current_x2_audio = self.current_x2_audio[-self.frame_contxt_padding:].copy()
                else:
                    self.current_x1_audio = np.empty(0, dtype=np.float32)
                    self.current_x2_audio = np.empty(0, dtype=np.float32)
                print("[Warning] No audio features extracted. Skipping this frame.")
                return

            # Skip the first Mimi encoder output to avoid the startup-only mismatch
            # between ONNX and PyTorch cache warmup behavior.
            if self._skip_first_encoder_output:
                self._skip_first_encoder_output = False
                self.process_time_abs = time.time()
                if self.frame_contxt_padding > 0:
                    self.current_x1_audio = self.current_x1_audio[-self.frame_contxt_padding:].copy()
                    self.current_x2_audio = self.current_x2_audio[-self.frame_contxt_padding:].copy()
                else:
                    self.current_x1_audio = np.empty(0, dtype=np.float32)
                    self.current_x2_audio = np.empty(0, dtype=np.float32)
                return

            # Full model
            if not self.use_kv_cache:
                
                self.e1_full.append(e1)
                self.e2_full.append(e2)
            
                x1_full_ = torch.cat(self.e1_full, dim=1)
                x2_full_ = torch.cat(self.e2_full, dim=1)
                # A list item may contain multiple frames.  Retain the
                # rolling context by time dimension, not list length.
                if x1_full_.shape[1] > self.audio_context_len:
                    x1_full_ = x1_full_[:, -self.audio_context_len :]
                    x2_full_ = x2_full_[:, -self.audio_context_len :]
                self.e1_full = [x1_full_]
                self.e2_full = [x2_full_]
                
                # Move to device only if necessary
                if self.device != 'cpu':
                    x1_full_ = x1_full_.to(self.device, non_blocking=True)
                    x2_full_ = x2_full_.to(self.device, non_blocking=True)

                out, _ = self.vap.forward(
                    x1_full_,
                    x2_full_,
                    cache=None,
                    return_all_frames=self.inference_chunk_frames > 1,
                )

            # User KV cache
            elif self.use_kv_cache:

                out, self.vap_cache = self.vap.forward(
                    e1,
                    e2,
                    cache=self.vap_cache,
                    return_all_frames=self.inference_chunk_frames > 1,
                )

                if self.vap_cache is not None:
                    self.vap_cache = self._trim_vap_cache(
                        self.vap_cache, self.audio_context_len - 1
                    )

            if self.inference_chunk_frames > 1 and (
                e1.shape[1] != self.inference_chunk_frames
                or e2.shape[1] != self.inference_chunk_frames
            ):
                raise RuntimeError(
                    "Mimi emitted an unexpected number of VAP frames: "
                    f"ch1={e1.shape[1]}, ch2={e2.shape[1]}, "
                    f"expected={self.inference_chunk_frames}."
                )

            emitted_at = time.time()
            if self.inference_chunk_frames > 1:
                # No-cache inference returns the complete context, whereas
                # cached inference returns only this chunk.  The public
                # realtime queue emits only the final frame of each batch.
                out_frames = out[-self.inference_chunk_frames :]
                out = out_frames[-1]
                x1_result = x1_dist[-self.audio_samples_per_frame :].copy()
                x2_result = x2_dist[-self.audio_samples_per_frame :].copy()
            else:
                x1_result = x1_dist.copy()
                x2_result = x2_dist.copy()

            # Pre-create result dict structure to avoid repeated key creation
            result_dict = {
                "t": emitted_at,
                "x1": x1_result,
                "x2": x2_result,
            }
            
            # Use dictionary mapping for mode-specific outputs (faster than if-elif chain)
            mode_outputs = {
                "vap": lambda: {
                    "p_now": out['p_now'],
                    "p_future": out['p_future'],
                    "vad": out['vad'],
                    "p_bins": out['p_bins'],
                    "p_bins_now": out['p_bins_now'],
                    "p_bins_future": out['p_bins_future'],
                },
                "vap_mc": lambda: {
                    "p_now": out['p_now'],
                    "p_future": out['p_future'],
                    "vad": out['vad'],
                    "p_bins": out['p_bins'],
                    "p_bins_now": out['p_bins_now'],
                    "p_bins_future": out['p_bins_future'],
                },
                "vap_mono": lambda: {
                    "p_now": out['p_now'],
                    "p_future": out['p_future'],
                    "vad": out['vad'],
                    "p_bins": out['p_bins'],
                    "p_bins_now": out['p_bins_now'],
                    "p_bins_future": out['p_bins_future'],
                },
                "vad": lambda: {
                    "vad": out['vad'],
                },
                "vad_mono": lambda: {
                    "vad": out['vad'],
                },
                "bc_det": lambda: {
                    "p_bc_det": out['p_bc_det'],
                },
                "bc_det_mono": lambda: {
                    "p_bc_det": out['p_bc_det'],
                },
                "vap_prompt": lambda: {
                    "p_now": out['p_now'],
                    "p_future": out['p_future'],
                    "vad": out['vad']
                },
                "bc": lambda: {
                    "p_bc": out['p_bc'],
                    "p_bc_detect": out['p_bc_detect']
                },
                "bc_2type": lambda: {
                    "p_bc_react": out['p_bc_react'],
                    "p_bc_emo": out['p_bc_emo']
                },
                "nod": lambda: {
                    "p_bc": out['p_bc'],
                    "p_nod_short": out['p_nod_short'],
                    "p_nod_long": out['p_nod_long'],
                    "p_nod_long_p": out['p_nod_long_p']
                },
                "nod_para": lambda: {
                    "p_nod": out["p_nod"],
                    "nod_repetitions": out["nod_repetitions"],
                    "nod_repetitions_pred": out["nod_repetitions_pred"],
                    "nod_range": out["nod_range"],
                    "nod_speed": out["nod_speed"],
                    "nod_swing_up": out["nod_swing_up"],
                    "nod_swing_up_pred": out["nod_swing_up_pred"],
                },
            }
            
            # Get mode-specific outputs
            if self.mode in mode_outputs:
                _out = mode_outputs[self.mode]()
                if not self.return_p_bins and self.mode in ("vap", "vap_mc", "vap_mono"):
                    for _k in ("p_bins", "p_bins_now", "p_bins_future"):
                        _out.pop(_k, None)
                result_dict.update(_out)
            
            self.result_dict_queue.put(result_dict)
            
            time_process = time.time() - time_start
            self.list_process_time_context.append(time_process)
            
            # Performance monitoring (unchanged for clarity)
            if len(self.list_process_time_context) > self.CALC_PROCESS_TIME_INTERVAL:
                ave_proc_time = np.mean(self.list_process_time_context)  # np.mean is faster than np.average
                num_process_frame = len(self.list_process_time_context) / (time.time() - self.last_interval_time)
                self.last_interval_time = time.time()

                perf_message = f'[{self.mode}] Average processing time: {ave_proc_time:.5f} [sec], #process/sec: {num_process_frame:.3f}'
                if self.encoder_type == "mimi":
                    perf_message += f', chunk_samples: {self.audio_frame_size}'
                print(perf_message)
                self.list_process_time_context.clear()  # clear() is faster than = []
            
            self.process_time_abs = time.time()

        # Keep only the last samples in the buffer (use views for efficiency)
        if self.frame_contxt_padding > 0:
            self.current_x1_audio = self.current_x1_audio[-self.frame_contxt_padding:].copy()
            self.current_x2_audio = self.current_x2_audio[-self.frame_contxt_padding:].copy()
        else:
            self.current_x1_audio = np.empty(0, dtype=np.float32)
            self.current_x2_audio = np.empty(0, dtype=np.float32)
    
    def get_result(self):
        """Retrieve the latest inference result from the queue.
        
        Returns:
            dict: The latest result containing predictions and raw audio data.
        """
        return self.result_dict_queue.get()
    
    def set_prompt_ch1(self, prompt: str):
        """
        Set the prompt text for speaker 1. This method is only available for the 'vap_prompt' mode.
        
        Args:
            prompt (str): The prompt text for speaker 1.
        """
        
        if self.mode != "vap_prompt":
            raise ValueError("This method is only available for the 'vap_prompt' mode.")
        
        self.vap.set_prompt_ch1(prompt, self.device)

    def set_prompt_ch2(self, prompt: str):
        """
        Set the prompt text for speaker 2. This method is only available for the 'vap_prompt' mode.

        Args:
            prompt (str): The prompt text for speaker 2.
        """

        if self.mode != "vap_prompt":
            raise ValueError("This method is only available for the 'vap_prompt' mode.")

        self.vap.set_prompt_ch2(prompt, self.device)


class MaaiMultiple:
    """
    Run several Maai models in parallel that share a single audio encoder.

    This is useful when you want to combine, for example, the turn-taking
    (``vap``) model with the backchannel (``bc``) model and the nodding
    (``nod``) model on the same input audio. A naive approach would build
    one ``Maai`` instance per model and run the (relatively expensive) audio
    encoder once for each instance. ``MaaiMultiple`` instead encodes the
    audio once per frame and feeds the encoded features into every sub-model.

    All sub-models must share the same encoder configuration:
    ``model_type``, ``frame_rate``, ``context_len_sec``, ``device``,
    ``inference_chunk_frames`` and the ``mimi_*`` parameters. Per-model
    differences allowed in ``configs`` are
    ``mode``, ``lang``, ``local_model``, ``return_p_bins`` and an optional
    ``label`` used as the result key.

    Each call to :meth:`get_result` returns a single ``dict`` whose top level
    contains shared fields ``t``, ``x1``, ``x2`` plus one nested ``dict``
    per sub-model keyed by its label (or its mode if no label is given).
    """

    CALC_PROCESS_TIME_INTERVAL = 100

    # Modes whose ``encode_audio`` swaps audio1/audio2 before model forward.
    _SWAP_MODES = {"bc", "bc_2type", "nod", "nod_para"}

    def __init__(
        self,
        configs: list,
        audio_ch1: Base,
        audio_ch2: Base,
        frame_rate: float = 10,
        context_len_sec: int = 20,
        device: str = "cpu",
        cpc_model: str = os.path.expanduser("~/.cache/cpc/60k_epoch4-d0f474de.pt"),
        model_type: str = "normal",
        mimi_model_name: str = "kyutai/mimi",
        use_mimi_onnx: bool = True,
        mimi_onnx_precision: str = "fp32",
        mimi_onnx_fp32_path: str | None = None,
        mimi_onnx_fp32_meta_path: str | None = None,
        mimi_onnx_int8_path: str | None = None,
        mimi_onnx_int8_meta_path: str | None = None,
        mimi_local_onnx_fp32_path: str | None = None,
        mimi_local_onnx_fp32_meta_path: str | None = None,
        mimi_local_onnx_int8_path: str | None = None,
        mimi_local_onnx_int8_meta_path: str | None = None,
        mimi_onnx_cpu_intra_threads: int | None = None,
        mimi_onnx_cpu_inter_threads: int | None = None,
        cache_dir: str = None,
        force_download: bool = False,
        use_kv_cache: bool = True,
        inference_chunk_frames: int = 1,
    ):
        if not configs:
            raise ValueError("MaaiMultiple requires at least one model config.")

        inference_chunk_frames = int(inference_chunk_frames)
        if inference_chunk_frames < 1:
            raise ValueError("inference_chunk_frames must be at least 1.")

        shared_kwargs = dict(
            audio_ch1=audio_ch1,
            audio_ch2=audio_ch2,
            frame_rate=frame_rate,
            context_len_sec=context_len_sec,
            device=device,
            cpc_model=cpc_model,
            model_type=model_type,
            mimi_model_name=mimi_model_name,
            use_mimi_onnx=use_mimi_onnx,
            mimi_onnx_precision=mimi_onnx_precision,
            mimi_onnx_fp32_path=mimi_onnx_fp32_path,
            mimi_onnx_fp32_meta_path=mimi_onnx_fp32_meta_path,
            mimi_onnx_int8_path=mimi_onnx_int8_path,
            mimi_onnx_int8_meta_path=mimi_onnx_int8_meta_path,
            mimi_local_onnx_fp32_path=mimi_local_onnx_fp32_path,
            mimi_local_onnx_fp32_meta_path=mimi_local_onnx_fp32_meta_path,
            mimi_local_onnx_int8_path=mimi_local_onnx_int8_path,
            mimi_local_onnx_int8_meta_path=mimi_local_onnx_int8_meta_path,
            mimi_onnx_cpu_intra_threads=mimi_onnx_cpu_intra_threads,
            mimi_onnx_cpu_inter_threads=mimi_onnx_cpu_inter_threads,
            cache_dir=cache_dir,
            force_download=force_download,
            use_kv_cache=use_kv_cache,
            inference_chunk_frames=inference_chunk_frames,
        )

        self.sub_maais: list[Maai] = []
        self.labels: list[str] = []
        seen_labels: set[str] = set()
        for cfg in configs:
            if "mode" not in cfg or "lang" not in cfg:
                raise ValueError("Each entry of `configs` must contain 'mode' and 'lang'.")
            label = cfg.get("label", cfg["mode"])
            if label in seen_labels:
                raise ValueError(
                    f"Duplicate label '{label}'. Provide a unique 'label' field in each config."
                )
            seen_labels.add(label)
            self.labels.append(label)
            sub = Maai(
                mode=cfg["mode"],
                lang=cfg["lang"],
                local_model=cfg.get("local_model"),
                return_p_bins=cfg.get("return_p_bins", False),
                **shared_kwargs,
            )
            self.sub_maais.append(sub)

        primary = self.sub_maais[0]

        # Mic configuration. Each Maai sub-instance subscribes a queue from
        # the input source in __init__; we keep only the primary's queues and
        # detach the rest so the audio source does not push frames into queues
        # that nobody drains.
        self.mic1 = audio_ch1
        self.mic2 = audio_ch2
        self._mic1_queue = primary._mic1_queue
        self._mic2_queue = primary._mic2_queue
        for sub in self.sub_maais[1:]:
            self._unsubscribe(self.mic1, sub._mic1_queue)
            self._unsubscribe(self.mic2, sub._mic2_queue)

        # Shared frame parameters (validated via primary; all sub-Maais use
        # the same encoder configuration so these match across instances).
        self.device = primary.device
        self.encoder_type = primary.encoder_type
        self._use_mimi_onnx = primary._use_mimi_onnx
        self.frame_rate = float(frame_rate)
        self.audio_contenxt_lim_sec = context_len_sec
        self.audio_context_len = primary.audio_context_len
        self.sampling_rate = primary.sampling_rate
        self.frame_contxt_padding = primary.frame_contxt_padding
        self.audio_frame_size = primary.audio_frame_size
        self.audio_samples_per_frame = primary.audio_samples_per_frame
        self.use_kv_cache = bool(use_kv_cache)
        self.inference_chunk_frames = inference_chunk_frames
        if self.inference_chunk_frames > 1 and self.encoder_type != "mimi":
            raise ValueError(
                "inference_chunk_frames > 1 currently supports the Mimi encoder only."
            )

        # Shared audio buffers and (when KV cache is disabled) shared encoded
        # context. These are populated by the worker on each frame.
        self.current_x1_audio = []
        self.current_x2_audio = []
        self.eA_full: list[torch.Tensor] = []
        self.eB_full: list[torch.Tensor] = []

        # One result queue, populated with a combined dict per frame.
        self.result_dict_queue: queue.Queue = queue.Queue()

        # Performance monitoring.
        self.list_process_time_context: list[float] = []
        self.last_interval_time = time.time()
        self.process_time_abs = -1

        # Threading.
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._sub_executor = None
        self._ensure_sub_executor()

        self.reset_runtime_state()

    @staticmethod
    def _unsubscribe(source: Base, q):
        try:
            with source._lock:
                if q in source._subscriber_queues:
                    source._subscriber_queues.remove(q)
        except Exception:
            pass

    def _ensure_sub_executor(self) -> None:
        """Create the sub-model pool if missing. Recreated after ``stop()`` on ``start()``."""
        if len(self.sub_maais) <= 1:
            self._sub_executor = None
            return
        if self._sub_executor is not None:
            return
        from concurrent.futures import ThreadPoolExecutor

        self._sub_executor = ThreadPoolExecutor(
            max_workers=len(self.sub_maais),
            thread_name_prefix="maai-multi-sub",
        )

    def reset_runtime_state(self):
        self.current_x1_audio = []
        self.current_x2_audio = []
        self.eA_full = []
        self.eB_full = []
        self._skip_first_encoder_output = bool(
            self.encoder_type == "mimi" and self._use_mimi_onnx
        )

        primary_vap = self.sub_maais[0].vap
        for encoder_name in ["encoder1", "encoder2"]:
            encoder = getattr(primary_vap, encoder_name, None)
            if encoder is not None and hasattr(encoder, "reset_streaming_state"):
                encoder.reset_streaming_state()

        for sub in self.sub_maais:
            sub.vap_cache = None
            sub.e1_full = []
            sub.e2_full = []

    def worker(self):
        self._mic1_queue.queue.clear()
        self._mic2_queue.queue.clear()

        while not self._stop_event.is_set():
            x1 = self.mic1.get_audio_data(self._mic1_queue)
            x2 = self.mic2.get_audio_data(self._mic2_queue)

            if self._stop_event.is_set() or x1 is None or x2 is None:
                break

            self.process(x1, x2)

            if self._mic1_queue.qsize() > 100:
                self._mic1_queue.queue.clear()
                print("[Warning] Audio queue (channel 1) overflow detected. Clearing audio queues.")
            if self._mic2_queue.qsize() > 100:
                self._mic2_queue.queue.clear()
                print("[Warning] Audio queue (channel 2) overflow detected. Clearing audio queues.")

    def start(self):
        self.reset_runtime_state()
        self._ensure_sub_executor()

        self.mic1.start()
        self.mic2.start()
        self._stop_event.clear()
        # Queue を空にしてからスレッド起動（スレッドに古いデータが届かないよう先にクリア）
        self._mic1_queue.queue.clear()
        self._mic2_queue.queue.clear()
        self._worker_thread = threading.Thread(target=self.worker, daemon=True)
        self._worker_thread.start()

    def stop(self, wait: bool = True, timeout: float = 2.0):
        self._stop_event.set()
        try:
            self._mic1_queue.put(None)
            self._mic2_queue.put(None)
        except Exception:
            pass
        if wait and self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

        try:
            self._mic1_queue.queue.clear()
            self._mic2_queue.queue.clear()
        except Exception:
            pass

        if getattr(self, "_sub_executor", None) is not None:
            self._sub_executor.shutdown(wait=False)
            self._sub_executor = None

        self.reset_runtime_state()

    def _trim_audio_buffers(self):
        if self.frame_contxt_padding > 0:
            self.current_x1_audio = self.current_x1_audio[-self.frame_contxt_padding:].copy()
            self.current_x2_audio = self.current_x2_audio[-self.frame_contxt_padding:].copy()
        else:
            self.current_x1_audio = np.empty(0, dtype=np.float32)
            self.current_x2_audio = np.empty(0, dtype=np.float32)

    @staticmethod
    def _trim_kv_cache(cache: dict, ctx_len: int) -> dict:
        new_cache = {}
        for key, (k_list, v_list) in cache.items():
            new_k_list = []
            new_v_list = []
            for t in k_list:
                if isinstance(t, torch.Tensor) and t.dim() >= 3:
                    new_k_list.append(t[..., -(ctx_len - 1):, :])
                else:
                    new_k_list.append(t)
            for t in v_list:
                if isinstance(t, torch.Tensor) and t.dim() >= 3:
                    new_v_list.append(t[..., -(ctx_len - 1):, :])
                else:
                    new_v_list.append(t)
            new_cache[key] = (new_k_list, new_v_list)
        return new_cache

    def process(self, x1, x2):
        time_start = time.time()

        if len(self.current_x1_audio) == 0:
            self.current_x1_audio = np.zeros(self.frame_contxt_padding, dtype=np.float32)
        if len(self.current_x2_audio) == 0:
            self.current_x2_audio = np.zeros(self.frame_contxt_padding, dtype=np.float32)

        self.current_x1_audio = np.concatenate([self.current_x1_audio, x1])
        self.current_x2_audio = np.concatenate([self.current_x2_audio, x2])

        if len(self.current_x1_audio) < self.audio_frame_size:
            return

        x1_proc = self.current_x1_audio
        x2_proc = self.current_x2_audio
        x1_dist = x1_proc[self.frame_contxt_padding:]
        x2_dist = x2_proc[self.frame_contxt_padding:]

        with torch.inference_mode():
            x1_t = torch.from_numpy(x1_proc).float().unsqueeze(0).unsqueeze(0)
            x2_t = torch.from_numpy(x2_proc).float().unsqueeze(0).unsqueeze(0)
            if self.device != "cpu":
                x1_t = x1_t.to(self.device, non_blocking=True)
                x2_t = x2_t.to(self.device, non_blocking=True)

            # Shared encoding step. We extract the shared features (before model-specific downsample)
            # using the primary model's encoder. The state/KV cache is maintained here.
            primary_vap = self.sub_maais[0].vap
            
            if self.encoder_type == "mimi":
                eA_shared, input_num_samples_A = primary_vap.encoder1.forward_shared(x1_t)
                eB_shared, input_num_samples_B = primary_vap.encoder2.forward_shared(x2_t)
            else:
                eA_shared = primary_vap.encoder1.forward_shared(x1_t)
                eB_shared = primary_vap.encoder2.forward_shared(x2_t)

            if eA_shared.shape[1] == 0 or eB_shared.shape[1] == 0:
                self._trim_audio_buffers()
                print("[Warning] No audio features extracted. Skipping this frame.")
                return

            # Skip the first Mimi encoder output to avoid the startup-only
            # mismatch between ONNX and PyTorch cache warmup behavior, just
            # like Maai.process does.
            if self._skip_first_encoder_output:
                self._skip_first_encoder_output = False
                self.process_time_abs = time.time()
                self._trim_audio_buffers()
                return

            # If KV cache is disabled, maintain a shared rolling context of
            # encoded features and feed each sub-model with the full window.
            if not self.use_kv_cache:
                self.eA_full.append(eA_shared)
                self.eB_full.append(eB_shared)
                eA_in = torch.cat(self.eA_full, dim=1)
                eB_in = torch.cat(self.eB_full, dim=1)
                # A list item may contain multiple frames (inference_chunk_frames > 1).
                # Retain the rolling context by time dimension, not list length.
                if eA_in.shape[1] > self.audio_context_len:
                    eA_in = eA_in[:, -self.audio_context_len:]
                    eB_in = eB_in[:, -self.audio_context_len:]
                self.eA_full = [eA_in]
                self.eB_full = [eB_in]
            else:
                eA_in = eA_shared
                eB_in = eB_shared

            if self.inference_chunk_frames > 1 and (
                eA_in.shape[1] != self.inference_chunk_frames
                or eB_in.shape[1] != self.inference_chunk_frames
            ) and self.use_kv_cache:
                raise RuntimeError(
                    "Mimi emitted an unexpected number of VAP frames: "
                    f"chA={eA_in.shape[1]}, chB={eB_in.shape[1]}, "
                    f"expected={self.inference_chunk_frames}."
                )

            if self.inference_chunk_frames > 1:
                # One combined result per process() call, representing only
                # the newest frame -- same convention as Maai.process().
                x1_result = x1_dist[-self.audio_samples_per_frame:].copy()
                x2_result = x2_dist[-self.audio_samples_per_frame:].copy()
            else:
                x1_result = x1_dist.copy()
                x2_result = x2_dist.copy()

            results_combined: dict = {
                "t": time.time(),
                "x1": x1_result,
                "x2": x2_result,
            }

            def _run_sub(label: str, sub: "Maai") -> tuple[str, dict]:
                # inference_mode is thread-local, so re-enter it in workers.
                with torch.inference_mode():
                    # Reproduce the per-mode swap that lives in each model's
                    # encode_audio: bc/bc_2type/nod/nod_para want the swapped
                    # (user, system) ordering, the others want the natural one.
                    if sub.mode in self._SWAP_MODES:
                        e1_shared, e2_shared = eB_in, eA_in
                    else:
                        e1_shared, e2_shared = eA_in, eB_in

                    # Apply the model-specific downsampling layer
                    if self.encoder_type == "mimi":
                        if sub.mode in self._SWAP_MODES:
                            n1, n2 = input_num_samples_B, input_num_samples_A
                        else:
                            n1, n2 = input_num_samples_A, input_num_samples_B
                        e1 = sub.vap.encoder1.forward_specific(e1_shared, input_num_samples=n1)
                        e2 = sub.vap.encoder2.forward_specific(e2_shared, input_num_samples=n2)
                    else:
                        e1 = sub.vap.encoder1.forward_specific(e1_shared)
                        e2 = sub.vap.encoder2.forward_specific(e2_shared)

                    # Apply each sub-model's decrease_dimension here (most models
                    # apply it inside encode_audio). nod_para applies projections
                    # internally inside its forward, so leave it alone.
                    if sub.mode != "nod_para" and hasattr(sub.vap, "decrease_dimension"):
                        e1 = torch.relu(sub.vap.decrease_dimension(e1))
                        e2 = torch.relu(sub.vap.decrease_dimension(e2))

                    if self.use_kv_cache:
                        out, sub.vap_cache = sub.vap.forward(e1, e2, cache=sub.vap_cache)
                        if sub.vap_cache is not None:
                            sub.vap_cache = self._trim_kv_cache(
                                sub.vap_cache, self.audio_context_len
                            )
                    else:
                        out, _ = sub.vap.forward(e1, e2, cache=None)

                    return label, self._extract_outputs(sub.mode, out, sub.return_p_bins)

            if self._sub_executor is not None:
                futures = [
                    self._sub_executor.submit(_run_sub, label, sub)
                    for label, sub in zip(self.labels, self.sub_maais)
                ]
                for fut in futures:
                    label, extracted = fut.result()
                    results_combined[label] = extracted
            else:
                for label, sub in zip(self.labels, self.sub_maais):
                    _, extracted = _run_sub(label, sub)
                    results_combined[label] = extracted

            self.result_dict_queue.put(results_combined)

            time_process = time.time() - time_start
            self.list_process_time_context.append(time_process)

            if len(self.list_process_time_context) > self.CALC_PROCESS_TIME_INTERVAL:
                ave_proc_time = np.mean(self.list_process_time_context)
                num_process_frame = (
                    len(self.list_process_time_context)
                    / (time.time() - self.last_interval_time)
                )
                self.last_interval_time = time.time()

                modes = ",".join(self.labels)
                msg = (
                    f"[multi:{modes}] Average processing time: {ave_proc_time:.5f} [sec], "
                    f"#process/sec: {num_process_frame:.3f}"
                )
                if self.encoder_type == "mimi":
                    msg += f", chunk_samples: {self.audio_frame_size}"
                print(msg)
                self.list_process_time_context.clear()

            self.process_time_abs = time.time()

        self._trim_audio_buffers()

    @staticmethod
    def _extract_outputs(mode: str, out: dict, return_p_bins: bool) -> dict:
        if mode in ("vap", "vap_mc", "vap_mono"):
            d = {
                "p_now": out["p_now"],
                "p_future": out["p_future"],
                "vad": out["vad"],
                "p_bins": out["p_bins"],
                "p_bins_now": out["p_bins_now"],
                "p_bins_future": out["p_bins_future"],
            }
            if not return_p_bins:
                for k in ("p_bins", "p_bins_now", "p_bins_future"):
                    d.pop(k, None)
            return d
        if mode in ("vad", "vad_mono"):
            return {"vad": out["vad"]}
        if mode in ("bc_det", "bc_det_mono"):
            return {"p_bc_det": out["p_bc_det"]}
        if mode == "vap_prompt":
            return {
                "p_now": out["p_now"],
                "p_future": out["p_future"],
                "vad": out["vad"],
            }
        if mode == "bc":
            return {
                "p_bc": out["p_bc"],
                "p_bc_detect": out["p_bc_detect"],
            }
        if mode == "bc_2type":
            return {
                "p_bc_react": out["p_bc_react"],
                "p_bc_emo": out["p_bc_emo"],
            }
        if mode == "nod":
            return {
                "p_bc": out["p_bc"],
                "p_nod_short": out["p_nod_short"],
                "p_nod_long": out["p_nod_long"],
                "p_nod_long_p": out["p_nod_long_p"],
            }
        if mode == "nod_para":
            return {
                "p_nod": out["p_nod"],
                "nod_repetitions": out["nod_repetitions"],
                "nod_repetitions_pred": out["nod_repetitions_pred"],
                "nod_range": out["nod_range"],
                "nod_speed": out["nod_speed"],
                "nod_swing_up": out["nod_swing_up"],
                "nod_swing_up_pred": out["nod_swing_up_pred"],
            }
        return {}

    def get_result(self):
        return self.result_dict_queue.get()

    def get_sub_maai(self, label: str) -> Maai:
        """Return the underlying ``Maai`` instance registered under ``label``."""
        for lbl, sub in zip(self.labels, self.sub_maais):
            if lbl == label:
                return sub
        raise KeyError(f"No sub-model with label '{label}'. Available: {self.labels}")

    def set_prompt_ch1(self, prompt: str, label: str | None = None):
        """Set channel-1 prompt on every ``vap_prompt`` sub-model (or only on ``label``)."""
        applied = False
        for lbl, sub in zip(self.labels, self.sub_maais):
            if sub.mode == "vap_prompt" and (label is None or lbl == label):
                sub.set_prompt_ch1(prompt)
                applied = True
        if label is not None and not applied:
            raise ValueError(f"No 'vap_prompt' sub-model found for label '{label}'.")

    def set_prompt_ch2(self, prompt: str, label: str | None = None):
        """Set channel-2 prompt on every ``vap_prompt`` sub-model (or only on ``label``)."""
        applied = False
        for lbl, sub in zip(self.labels, self.sub_maais):
            if sub.mode == "vap_prompt" and (label is None or lbl == label):
                sub.set_prompt_ch2(prompt)
                applied = True
        if label is not None and not applied:
            raise ValueError(f"No 'vap_prompt' sub-model found for label '{label}'.")
