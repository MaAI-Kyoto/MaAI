import torch
import torch.nn as nn
import time
import numpy as np
import threading
import queue
import copy
import os

from .input import Base
from .util import load_vap_model
from .models.vap import VapGPT
from .models.vap_bc import VapGPT_bc
from .models.vap_bc_2type import VapGPT_bc_2type
from .models.vap_nod import VapGPT_nod
from .models.config import VapConfig
# from .models.vap_prompt import VapGPT_prompt

class Maai():
    
    BINS_P_NOW = [0, 1]
    BINS_PFUTURE = [2, 3]
    
    CALC_PROCESS_TIME_INTERVAL = 100

    def __init__(
        self,
        mode,
        frame_rate: int = 10,
        context_len_sec: int = 20,
        lang: str = "jp",
        audio_ch1: Base = None,
        audio_ch2: Base = None,
        num_channels: int = 2,
        cpc_model: str = os.path.expanduser("~/.cache/cpc/60k_epoch4-d0f474de.pt"),
        device: str = "cpu",
        cache_dir: str = None,
        force_download: bool = False,
        use_kv_cache: bool = True
    ):

        conf = VapConfig()
        
        if mode in ["vap", "vap_mc"]:
            self.vap = VapGPT(conf)
        
        elif mode == "bc":
            self.vap = VapGPT_bc(conf)

        elif mode == "bc_2type":
            self.vap = VapGPT_bc_2type(conf)
        
        elif mode == "nod":
            self.vap = VapGPT_nod(conf)
        
        elif mode == "vap_prompt":
            from .models.vap_prompt import VapGPT_prompt
            self.vap = VapGPT_prompt(conf)
        
        self.device = device

        if self.device not in ["cpu", "cuda"]:
            raise ValueError("Device must be 'cpu' or 'cuda'.")
        
        # Store the initial state of the model to check for unchanged parameters
        initial_state_dict = {name: param.clone() for name, param in self.vap.named_parameters()}

        sd = load_vap_model(mode, frame_rate, context_len_sec, lang, device, cache_dir, force_download)
        self.vap.load_encoder(cpc_model=cpc_model)
        self.vap.load_state_dict(sd, strict=False)

        # The downsampling parameters are not loaded by "load_state_dict"
        self.vap.encoder1.downsample[1].weight = nn.Parameter(sd['encoder.downsample.1.weight'])
        self.vap.encoder1.downsample[1].bias = nn.Parameter(sd['encoder.downsample.1.bias'])
        self.vap.encoder1.downsample[2].ln.weight = nn.Parameter(sd['encoder.downsample.2.ln.weight'])
        self.vap.encoder1.downsample[2].ln.bias = nn.Parameter(sd['encoder.downsample.2.ln.bias'])
        
        self.vap.encoder2.downsample[1].weight = nn.Parameter(sd['encoder.downsample.1.weight'])
        self.vap.encoder2.downsample[1].bias = nn.Parameter(sd['encoder.downsample.1.bias'])
        self.vap.encoder2.downsample[2].ln.weight = nn.Parameter(sd['encoder.downsample.2.ln.weight'])
        self.vap.encoder2.downsample[2].ln.bias = nn.Parameter(sd['encoder.downsample.2.ln.bias'])

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
        self.mic1 = audio_ch1
        self.mic2 = audio_ch2

        # Always subscribe a dedicated queue for each mic if possible
        self._mic1_queue = self.mic1.subscribe()
        self._mic2_queue = self.mic2.subscribe()

        self.audio_contenxt_lim_sec = context_len_sec
        self.frame_rate = frame_rate
        
        # Context length of the audio embeddings (depends on frame rate)
        self.audio_context_len = int(self.audio_contenxt_lim_sec * self.frame_rate)
        
        self.sampling_rate = 16000
        self.frame_contxt_padding = 320 # Independe from frame size
        
        # Frame size
        # 10Hz -> 320 + 1600 samples
        # 20Hz -> 320 + 800 samples
        # 50Hz -> 320 + 320 samples
        self.audio_frame_size = self.sampling_rate // self.frame_rate + self.frame_contxt_padding
        
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
    
    def worker(self):
        
        # Clear the queues at the start
        # This is to ensure that the queues are empty before starting the processing loop
        self._mic1_queue.queue.clear()
        self._mic2_queue.queue.clear()
        
        while True:
            x1 = self.mic1.get_audio_data(self._mic1_queue)
            x2 = self.mic2.get_audio_data(self._mic2_queue)
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

        self.mic1.start()
        self.mic2.start()
        threading.Thread(target=self.worker, daemon=True).start()

        # Queueを空にする
        self._mic1_queue.queue.clear()
        self._mic2_queue.queue.clear()
    
    def process(self, x1, x2):
        
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

        with torch.no_grad():
            # Create tensors more efficiently with specified dtype and device
            x1_ = torch.from_numpy(x1_proc).float().unsqueeze(0).unsqueeze(0)
            x2_ = torch.from_numpy(x2_proc).float().unsqueeze(0).unsqueeze(0)
            
            # Move to device only once
            if self.device != 'cpu':
                x1_ = x1_.to(self.device, non_blocking=True)
                x2_ = x2_.to(self.device, non_blocking=True)

            e1, e2 = self.vap.encode_audio(x1_, x2_)

            # Full model
            if not self.use_kv_cache:
                
                self.e1_full.append(e1)
                self.e2_full.append(e2)
            
                # More efficient context management
                if len(self.e1_full) > self.audio_context_len:
                    self.e1_full.pop(0)  # Remove from front instead of slicing
                if len(self.e2_full) > self.audio_context_len:
                    self.e2_full.pop(0)
                
                x1_full_ = torch.cat(self.e1_full, dim=1)
                x2_full_ = torch.cat(self.e2_full, dim=1)
                
                # Move to device only if necessary
                if self.device != 'cpu':
                    x1_full_ = x1_full_.to(self.device, non_blocking=True)
                    x2_full_ = x2_full_.to(self.device, non_blocking=True)

                out, _ = self.vap.forward(x1_full_, x2_full_, cache=None)

            # User KV cache
            elif self.use_kv_cache:

                out, self.vap_cache = self.vap.forward(e1, e2, cache=self.vap_cache)

                ## Trim all cache data in self.vap_cache so that the second-to-last dimension is self.audio_context_len - 1
                if self.vap_cache is not None:
                    new_cache = {}
                    for key, (k_list, v_list) in self.vap_cache.items():
                        new_k_list = []
                        new_v_list = []
                        for t in k_list:
                            if isinstance(t, torch.Tensor) and t.dim() >= 3:
                                new_k_list.append(t[..., -(self.audio_context_len - 1) :, :])
                            else:
                                new_k_list.append(t)
                        for t in v_list:
                            if isinstance(t, torch.Tensor) and t.dim() >= 3:
                                new_v_list.append(t[..., -(self.audio_context_len - 1) :, :])
                            else:
                                new_v_list.append(t)
                        new_cache[key] = (new_k_list, new_v_list)
                    self.vap_cache = new_cache

            # Pre-create result dict structure to avoid repeated key creation
            result_dict = {
                "t": time.time(),
                "x1": x1_dist.copy(),  # Only copy when necessary
                "x2": x2_dist.copy(),
            }
            
            # Use dictionary mapping for mode-specific outputs (faster than if-elif chain)
            mode_outputs = {
                "vap": lambda: {
                    "p_now": out['p_now'],
                    "p_future": out['p_future'],
                    "vad": out['vad']
                },
                "vap_mc": lambda: {
                    "p_now": out['p_now'],
                    "p_future": out['p_future'],
                    "vad": out['vad']
                },
                "vap_prompt": lambda: {
                    "p_now": out['p_now'],
                    "p_future": out['p_future'],
                    "vad": out['vad']
                },
                "bc": lambda: {
                    "p_bc": out['p_bc']
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
                }
            }
            
            # Get mode-specific outputs
            if self.mode in mode_outputs:
                result_dict.update(mode_outputs[self.mode]())
            
            self.result_dict_queue.put(result_dict)
            
            time_process = time.time() - time_start
            self.list_process_time_context.append(time_process)
            
            # Performance monitoring (unchanged for clarity)
            if len(self.list_process_time_context) > self.CALC_PROCESS_TIME_INTERVAL:
                ave_proc_time = np.mean(self.list_process_time_context)  # np.mean is faster than np.average
                num_process_frame = len(self.list_process_time_context) / (time.time() - self.last_interval_time)
                self.last_interval_time = time.time()

                print(f'[{self.mode}] Average processing time: {ave_proc_time:.5f} [sec], #process/sec: {num_process_frame:.3f}')
                self.list_process_time_context.clear()  # clear() is faster than = []
            
            self.process_time_abs = time.time()

        # Keep only the last samples in the buffer (use views for efficiency)
        self.current_x1_audio = self.current_x1_audio[-self.frame_contxt_padding:].copy()
        self.current_x2_audio = self.current_x2_audio[-self.frame_contxt_padding:].copy()
    
    def get_result(self):
        return self.result_dict_queue.get()
    
    def set_prompt_ch1(self, prompt: str):
        """
        Set the prompt text for speaker 1. This method is only available for the 'vap_prompt' mode.
        
        Args:
            prompt (str): The prompt text for speaker 1.
        """
        
        if self.mode != "vap_prompt":
            raise ValueError("This method is only available for the 'vap_prompt' mode.")
        
        self.vap.set_prompt_ch1(prompt)
    
    def set_prompt_ch2(self, prompt: str):
        """
        Set the prompt text for speaker 2. This method is only available for the 'vap_prompt' mode.
        
        Args:
            prompt (str): The prompt text for speaker 2.
        """
        
        if self.mode != "vap_prompt":
            raise ValueError("This method is only available for the 'vap_prompt' mode.")
        
        self.vap.set_prompt_ch2(prompt)
    