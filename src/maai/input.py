import socket
import pyaudio
import queue
import threading
import soundfile as sf
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
import time
from . import util
import numpy as np
import sys
import locale

def available_mic_devices(print_out=True):
    """Retrieve and optionally print a list of available microphone devices.
    
    Args:
        print_out (bool): Whether to print the device list to the console.
        
    Returns:
        dict: A dictionary mapping device index to device information.
    """
    p = pyaudio.PyAudio()
    device_info = {}
    
    encoding = locale.getpreferredencoding()
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            # Avoid mojibake by encoding/decoding device name
            name = info['name']

            if isinstance(name, bytes):
                try:
                    name = name.decode(encoding, errors='replace')
                except Exception:
                    name = name.decode('utf-8', errors='replace')
            
            # str型ならそのまま
            elif isinstance(name, str):
                try:
                    name = name.encode(encoding).decode('utf-8')
                except Exception:
                    name = name.encode('utf-8').decode('utf-8')
            
            # 改行文字をスペースに置換
            name = str(name).replace('\r', ' ').replace('\n', ' ')
            device_info[info['index']] = {
                'name': name,
                'maxInputChannels': info['maxInputChannels'],
                'maxOutputChannels': info['maxOutputChannels']
            }

    if print_out:
        print("Available microphone devices:")
        for index, info in device_info.items():
            print(f"Device {index}: {info['name']} (In: {info['maxInputChannels']}, Out: {info['maxOutputChannels']})")

    return device_info

class Base:
    """Base class for all audio input sources.
    
    Provides a publish-subscribe mechanism for audio data streams.
    """
    FRAME_SIZE = 160
    SAMPLING_RATE = 16000
    def __init__(self):
        self._subscriber_queues = []  # List of subscriber queues
        self._lock = threading.Lock()
        self._is_thread_started = False
        self.channels = 1

    def subscribe(self):
        q = queue.Queue()
        with self._lock:
            self._subscriber_queues.append(q)
        return q

    def _put_to_all_queues(self, data):
        # Put data into all subscriber queues and the default queue
        with self._lock:
            for q in self._subscriber_queues:
                q.put(data)

    def get_audio_data(self, q=None):
        return q.get()
    
    def _get_queue_size(self):
        with self._lock:
            return sum([len(q.queue) for q in self._subscriber_queues])

    def get_channel(self, channel_index):
        return ChannelSelector(self, channel_index)

class ChannelSelector(Base):
    """Selects and isolates a specific audio channel from a multi-channel source."""
    
    def __init__(self, source, channel_index):
        super().__init__()
        self.source = source
        self.channel_index = channel_index
        self.channels = 1

    def subscribe(self):
        return self.source.subscribe()

    def get_audio_data(self, q=None):
        data = self.source.get_audio_data(q)
        source_channels = getattr(self.source, 'channels', 1)
        if source_channels > 1:
            data_np = np.array(data)
            if data_np.ndim == 1:
                data_np = data_np.reshape(-1, source_channels)[:, self.channel_index]
            elif data_np.ndim == 2:
                data_np = data_np[:, self.channel_index]
            return data_np.tolist()
        return data

    def start(self):
        self.source.start()

class Mic(Base):
    """Audio input source that reads directly from a local microphone."""

    def __init__(self, audio_gain=1.0, mic_device_index=-1, device_name=None, channels=1, use_channel=None):
        """Initialize the Mic input source.
        
        Args:
            audio_gain (float): Multiplier for the audio signal amplitude.
            mic_device_index (int): PyAudio device index to use.
            device_name (str, optional): Substring of the device name to search for.
            channels (int): Number of audio channels to capture.
            use_channel (int, optional): Specific channel to extract if capturing multiple channels.
        """
        
        super().__init__()
        
        self.p = pyaudio.PyAudio()
        self.audio_gain = audio_gain
        self.channels = channels
        self.use_channel = use_channel

        if device_name is not None and mic_device_index == -1:
            # If a specific device name is provided, find the index of that device
            device_info = available_mic_devices(print_out=False)
            found = False
            for idx, info in device_info.items():
                if device_name in info['name']:
                    mic_device_index = idx
                    found = True
                    break
            if found:
                print(f"Using microphone device: {device_name} (Index: {mic_device_index})")
            else:
                print(f"Device with name '{device_name}' not found. Using default device.")
                mic_device_index = -1
            
        self.mic_device_index = mic_device_index if mic_device_index >= 0 else None
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=self.channels,
                                  rate=self.SAMPLING_RATE,
                                  input=True,
                                  output=False,
                                  input_device_index=self.mic_device_index,
                                  start=False)

    def _read_mic(self):

        self.stream.start_stream()
        
        while True:
            d = self.stream.read(self.FRAME_SIZE, exception_on_overflow=False)
            d = np.frombuffer(d, dtype=np.float32)
            
            if self.channels > 1 and self.use_channel is not None:
                d = d.reshape(-1, self.channels)[:, self.use_channel]
                
            d = [float(a) for a in d]
            
            if self.audio_gain != 1.0:
                d = [a * self.audio_gain for a in d]

            self._put_to_all_queues(d)

    def start(self):
        if not self._is_thread_started:
            threading.Thread(target=self._read_mic, daemon=True).start()
            self._is_thread_started = True

class Wav(Base):
    """Audio input source that reads from a WAV file and plays it back."""
    
    def __init__(self, wav_file_path, audio_gain=1.0, use_channel=None):
        """Initialize the Wav input source.
        
        Args:
            wav_file_path (str): Path to the WAV file.
            audio_gain (float): Multiplier for the audio signal amplitude.
            use_channel (int, optional): Specific channel to extract if the file has multiple channels.
        """
        super().__init__()
        self.wav_file_path = wav_file_path
        self.audio_gain = audio_gain
        self.use_channel = use_channel
        self.raw_wav_queue = queue.Queue()

        if not os.path.exists(self.wav_file_path):
            raise FileNotFoundError(f"WAV file not found: {self.wav_file_path}")
        
        # Check the frame rate of the WAV file
        info = sf.info(self.wav_file_path)
        self.SAMPLING_RATE = info.samplerate
        self.channels = info.channels
        if self.SAMPLING_RATE != 16000:
            raise ValueError(f"Unsupported sample rate: {self.SAMPLING_RATE}. Expected 16000 Hz.")
        
        data, _ = sf.read(file=self.wav_file_path, dtype='float32')
        
        if self.channels > 1 and self.use_channel is not None:
            data = data[:, self.use_channel]
            
        for i in range(0, len(data), self.FRAME_SIZE):
            if i + self.FRAME_SIZE > len(data):
                break
            d = data[i:i+self.FRAME_SIZE]
            self.raw_wav_queue.put(d)

    def _read_wav(self):
        start_time = time.time()
        frame_duration = self.FRAME_SIZE / self.SAMPLING_RATE
        pygame.mixer.init(frequency=16000, size=-16, channels=1, buffer=512)
        sound = pygame.mixer.Sound(self.wav_file_path)
        sound.play()
        frame_count = 0
        while True:
            if self.raw_wav_queue.empty():
                break
            expected_time = start_time + frame_count * frame_duration
            current_time = time.time()
            if current_time < expected_time:
                time.sleep(0.001)
                continue
            data = self.raw_wav_queue.get()
            frame_count += 1
            if data is None:
                continue
            self._put_to_all_queues(data)

    def start(self):
        if not self._is_thread_started:
            threading.Thread(target=self._read_wav, daemon=True).start()
            self._is_thread_started = True

class Tcp(Base):
    """Audio input source that receives data over a TCP connection."""
    
    def __init__(self, ip='127.0.0.1', port=8501, audio_gain=1.0,recv_float32=False, client_mode=False):
        """Initialize the Tcp input source.
        
        Args:
            ip (str): IP address to bind or connect to.
            port (int): Port number.
            audio_gain (float): Multiplier for the audio signal amplitude.
            recv_float32 (bool): If True, expects 4-byte float data. Otherwise expects 8-byte float.
            client_mode (bool): If True, acts as a TCP client. If False, acts as a TCP server.
        """
        super().__init__()
        self.ip = ip
        self.port = port
        self.conn = None
        self.addr = None
        self._is_thread_started_process = False
        self._is_thread_started_server = False
        self.audio_gain = audio_gain
        self.recv_float32 = recv_float32  # 4byte float受信オプション
        self.client_mode = client_mode  # クライアントモードオプション

    def _server(self):
        while True:
            if self.conn is not None:
                time.sleep(0.1)
                continue
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind((self.ip, self.port))
                s.listen(1)
                print('[IN] Waiting for connection of audio input...')
                self.conn, self.addr = s.accept()
                print('[IN] Connected by', self.addr)
            except Exception as e:
                print('[IN] Connection failed:', e)
                time.sleep(1)
                continue

    def _client(self):
        while True:
            if self.conn is not None:
                time.sleep(0.1)
                continue
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.ip, self.port))
                self.conn = s
                self.addr = (self.ip, self.port)
                print('[IN] Connected to server', self.addr)
            except Exception as e:
                print('[IN] Client connect failed:', e)
                time.sleep(1)
                continue

    def _process(self):
        import struct
        while True:
            try:
                if self.conn is None:
                    time.sleep(0.1)
                    continue
                
                # Float32受信オプションが有効な場合、4byte floatを受信
                if self.recv_float32:
                    size_recv = 4 * self.FRAME_SIZE
                    # print(f"[IN] Receiving {size_recv} bytes of float32 data")
                    data = self.conn.recv(size_recv)
                    # print(f"[IN] Received {len(data)} bytes of float32 data")
                    if len(data) < size_recv:
                        while len(data) < size_recv:
                            data_ = self.conn.recv(size_recv - len(data))
                            if len(data_) == 0:
                                break
                            data += data_
                    if len(data) == 0:
                        raise ConnectionError("Connection closed")
                    # 4byte float -> 8byte float変換
                    x1_short = util.conv_bytearray_2_floatarray_short(data)
                    x1 = [float(a) for a in x1_short]  # Convert to float list
                
                # 通常は8byte float受信
                else:
                    size_recv = 8 * self.FRAME_SIZE
                    data = self.conn.recv(size_recv)
                    if len(data) < size_recv:
                        while len(data) < size_recv:
                            data_ = self.conn.recv(size_recv - len(data))
                            if len(data_) == 0:
                                break
                            data += data_
                    if len(data) == 0:
                        raise ConnectionError("Connection closed")
                    x1 = util.conv_bytearray_2_floatarray(data)

                if self.audio_gain != 1.0:
                    x1 = [a * self.audio_gain for a in x1]
                    
                self._put_to_all_queues(x1)

            except Exception as e:
                if self.addr is not None:
                    print('[IN] Disconnected by', self.addr)
                else:
                    print('[IN] Disconnected (no connection established)')
                print(e)
                self.conn = None
                self.addr = None
                continue

    def start(self):
        if not self._is_thread_started_process:
            threading.Thread(target=self._process, daemon=True).start()
            self._is_thread_started_process = True

    def start_server(self):
        if not self._is_thread_started_server:
            if self.client_mode:
                threading.Thread(target=self._client, daemon=True).start()
            else:
                threading.Thread(target=self._server, daemon=True).start()
            self._is_thread_started_server = True
    
    def _send_data_manual(self, data):
        if self.conn is None:
            raise ConnectionError("No connection established. Call start_server() first.")
        self.conn.send(data)
    
    def is_connected(self):
        return self.conn is not None and self.addr is not None


class TcpMic(Base):
    """Audio input source that reads from a microphone and sends the data to a TCP server."""
    
    def __init__(self, server_ip='127.0.0.1', port=8501, audio_gain=1.0, mic_device_index=0, channels=1, use_channel=None):
        """Initialize the TcpMic source.
        
        Args:
            server_ip (str): IP address of the TCP server.
            port (int): Port of the TCP server.
            audio_gain (float): Multiplier for the audio signal amplitude.
            mic_device_index (int): PyAudio device index to use.
            channels (int): Number of audio channels.
            use_channel (int, optional): Specific channel to extract.
        """
        self.ip = server_ip
        self.port = port
        self.p = pyaudio.PyAudio()
        self.audio_gain = audio_gain
        self.mic_device_index = mic_device_index
        self.channels = channels
        self.use_channel = use_channel
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=self.channels,
                                  rate=self.SAMPLING_RATE,
                                  input=True,
                                  output=False,
                                  input_device_index=self.mic_device_index)
    
    def connect_server(self):    
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.ip, self.port))
        print('[CLIENT] Connected to the server')

    def _start_client(self):
        while True:
            try:
                self.connect_server()
                while True:
                    try:
                        d = self.stream.read(self.FRAME_SIZE, exception_on_overflow=False)
                        d = np.frombuffer(d, dtype=np.float32)
                        
                        if self.channels > 1 and self.use_channel is not None:
                            d = d.reshape(-1, self.channels)[:, self.use_channel]

                        if self.audio_gain != 1.0:
                            d = d * self.audio_gain
                            
                        d = [float(a) for a in d]
                        data_sent = util.conv_floatarray_2_byte(d)
                        self.sock.sendall(data_sent)
                    except Exception as e:
                        print('[CLIENT] Send error:', e)
                        break  # 送信エラー時は再接続ループへ
            except Exception as e:
                print('[CLIENT] Connect error:', e)
                time.sleep(0.5)
                continue
            # 切断時はソケットを閉じて再接続ループへ
            try:
                if hasattr(self, 'sock') and self.sock is not None:
                    self.sock.close()
            except Exception:
                pass
            self.sock = None
            print('[CLIENT] Disconnected. Reconnecting...')
            time.sleep(0.5)

    def start(self):
        threading.Thread(target=self._start_client, daemon=True).start()


class TcpChunk(Base):
    """Audio input source that receives arbitrary sized chunks over TCP."""
    
    def __init__(self, server_ip='127.0.0.1', port=8501):
        """Initialize the TcpChunk source.
        
        Args:
            server_ip (str): IP address of the TCP server.
            port (int): Port of the TCP server.
        """
        self.ip = server_ip
        self.port = port
        self.chunk_size = 1024
        self.sock = None

    def connect_server(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.ip, self.port))
        print('[CLIENT] Connected to the server')

    def _start_client(self):
        while True:
            try:
                self.connect_server()
                while True:
                    try:
                        data = self.sock.recv(self.chunk_size)
                        if not data:
                            break
                        self._process_chunk(data)
                    except Exception as e:
                        print('[CLIENT] Receive error:', e)
                        break
            except Exception as e:
                print('[CLIENT] Connect error:', e)
                time.sleep(0.5)
                continue
            # 切断時はソケットを閉じて再接続ループへ
            try:
                if hasattr(self, 'sock') and self.sock is not None:
                    self.sock.close()
            except Exception:
                pass
            self.sock = None
            print('[CLIENT] Disconnected. Reconnecting...')
            time.sleep(0.5)

    def start(self):
        threading.Thread(target=self._start_client, daemon=True).start()
    
    def put_chunk(self, chunk_data):
        if self.sock is not None:
            data_sent = util.conv_floatarray_2_byte(chunk_data)
            self.sock.sendall(data_sent)

class Zero(Base):
    """Audio input source that generates continuous zeros or white noise."""
    
    def __init__(self, white_noise=False):
        """Initialize the Zero input source.
        
        Args:
            white_noise (bool): If True, generates low-amplitude white noise instead of zeros.
        """
        super().__init__()
        self.max_queue_size = 10
        self._is_thread_started_process = False
        self.data_added = [0.] * self.FRAME_SIZE  # Initialize with zeros
        self.white_noise = white_noise

        if self.white_noise:
            print("Zero input with white noise is enabled.")
            self.data_added = np.random.normal(0, 0.00001, self.FRAME_SIZE).astype(np.float32).tolist()

        # self.subscribe()  # Subscribe to the queue to ensure it exists

        # while self._get_queue_size() < self.max_queue_size:
        #     self._put_to_all_queues(self.data_added)
        #     # print(self._get_queue_size())

    def _process(self):

        # data_added = [0.] * self.FRAME_SIZE  # Initialize with zeros
        count = 0
        while True:
            try:
                # print(self._get_queue_size())
                if self._get_queue_size() >= self.max_queue_size:
                    time.sleep(0.01)
                    continue
                # print([0.] * self.FRAME_SIZE)
                # print(len([0.] * self.FRAME_SIZE))
                self._put_to_all_queues(self.data_added)
                # print('added zero data')

                if self.white_noise:
                    count += 1
                    if count % 10 == 0:
                        self.data_added = np.random.normal(0, 0.00001, self.FRAME_SIZE).astype(np.float32).tolist()
                        count = 0

            except Exception as e:
                print('[ZERO] Error:', e)
                #time.sleep(0.001)
                continue

    def start(self):
        if not self._is_thread_started_process:
            threading.Thread(target=self._process, daemon=True).start()
            self._is_thread_started_process = True


class Chunk(Base):
    """Audio input source for manually providing chunked data."""
    
    def __init__(self):
        super().__init__()

    def put_chunk(self, chunk_data):
        chunk_list = [float(x) for x in chunk_data]
        self._put_to_all_queues(chunk_list)

    def start(self):
        pass


class BleedCancel(Base):
    """Real-time speaker-bleed removal for two face-to-face lapel microphones.

    When two people talk face to face with lapel mics, each mic also picks up the partner's voice.
    This wraps a 2-channel source (or two mono sources) and removes the bleed using the partner's
    mic as the reference (2-pass linear AEC + residual suppression, see :mod:`maai.bleed`).
    The output is a 2-channel source in the same format as ``Mic(channels=2)``; use
    ``get_channel(0)`` / ``get_channel(1)`` to feed it to ``Maai``.

    The canceller keeps running while disabled (so it stays adapted), and the disabled output is
    the raw input delayed by the same latency, so toggling never shifts the timeline.

    Notes:
        - The filters need about 3 s (in total) of speech where only the partner talks before
          they converge; bleed remains until then.
        - Both mics should share the same clock (e.g. two inputs of one audio interface).
        - Algorithmic latency is about 44 ms (see :attr:`latency_ms`).

    Example:
        >>> mic = MaaiInput.Mic(mic_device_index=0, channels=2)
        >>> bc = MaaiInput.BleedCancel(mic)          # or BleedCancel(mic1, mic2)
        >>> maai = Maai(mode="vap", audio_ch1=bc.get_channel(0), audio_ch2=bc.get_channel(1), ...)
        >>> bc.set_enabled(False)                    # switch off at runtime
    """

    QUEUE_WARN_SIZE = 100

    def __init__(self, source, source_ch2=None, enabled=True, **params):
        """Initialize the bleed canceller.

        Args:
            source (Base): A 2-channel source (L = speaker 1, R = speaker 2), or the mono source
                of speaker 1 when ``source_ch2`` is given.
            source_ch2 (Base, optional): Mono source of speaker 2.
            enabled (bool): Whether bleed removal is applied to the output initially.
            **params: Parameters of :class:`maai.bleed.BleedCanceller`
                (e.g. ``aes_beta``, ``aes=False``, ``hpf_hz``, ``margin``).
        """
        super().__init__()
        if source_ch2 is None and getattr(source, 'channels', 1) != 2:
            raise ValueError(
                "BleedCancel needs a 2-channel source (e.g. Mic(channels=2)) or two mono sources "
                "(BleedCancel(source_ch1, source_ch2))."
            )
        self.source = source
        self.source_ch2 = source_ch2
        self.channels = 2
        self._params = dict(params)
        self._enabled = bool(enabled)
        self._reset_requested = False
        self._monitor_queues = []
        self._last_queue_warn = 0.0

        # Subscribe now so that no data is missed once the sources start
        self._q1 = source.subscribe()
        self._q2 = source_ch2.subscribe() if source_ch2 is not None else None
        self._in_pending = [np.zeros(0), np.zeros(0)]  # per-channel buffers (two mono sources)

        self._init_engine()

    def _init_engine(self):
        from scipy.signal import butter, sosfilt_zi
        from .bleed import BleedCanceller
        from .bleed.realtime.engine import DelayLine

        self._engine = BleedCanceller(**dict(self._params, sr=self.SAMPLING_RATE))
        p = self._engine.p
        if p["hpf_hz"]:
            self._sos = butter(2, p["hpf_hz"], "highpass", fs=self.SAMPLING_RATE, output="sos")
            self._zi = np.stack([sosfilt_zi(self._sos) * 0.0] * 2, axis=-1)
        else:
            self._sos = None
        self._raw_delay = [DelayLine(self._engine.latency) for _ in range(2)]
        self._pending = np.zeros((0, 2))       # raw input waiting for a full block
        self._buf_raw = np.zeros((0, 2))       # latency-aligned raw output not yet emitted
        self._buf_clean = np.zeros((0, 2))     # processed output not yet emitted

    def _feed(self, x):
        """Process raw stereo samples (n, 2) and return the completed output frames.

        Returns:
            list: dicts with ``raw`` (latency-aligned input), ``clean`` (processed), ``output``
            (what is published) as (FRAME_SIZE, 2) arrays, and ``enabled``.
        """
        from scipy.signal import sosfilt

        if self._reset_requested:
            self._reset_requested = False
            self._init_engine()
        self._pending = np.concatenate([self._pending, np.asarray(x, np.float64).reshape(-1, 2)], 0)
        B = self._engine.block
        nb = len(self._pending) // B
        if nb:
            raw = self._pending[:nb * B]
            self._pending = self._pending[nb * B:]
            filt = raw
            if self._sos is not None:
                filt, self._zi = sosfilt(self._sos, raw, axis=0, zi=self._zi)
            clean, raw_d = [], []
            for i in range(nb):
                blk = filt[i * B:(i + 1) * B]
                oa, ob = self._engine.process(blk[:, 0], blk[:, 1])
                clean.append(np.stack([oa, ob], 1))
                rblk = raw[i * B:(i + 1) * B]
                raw_d.append(np.stack([self._raw_delay[c].process(rblk[:, c]) for c in range(2)], 1))
            self._buf_clean = np.concatenate([self._buf_clean] + clean, 0)
            self._buf_raw = np.concatenate([self._buf_raw] + raw_d, 0)

        frames = []
        F = self.FRAME_SIZE
        nf = len(self._buf_clean) // F
        for i in range(nf):
            r = self._buf_raw[i * F:(i + 1) * F]
            c = self._buf_clean[i * F:(i + 1) * F]
            enabled = self._enabled
            frames.append(dict(raw=r, clean=c, output=c if enabled else r, enabled=enabled))
        self._buf_clean = self._buf_clean[nf * F:]
        self._buf_raw = self._buf_raw[nf * F:]
        return frames

    @staticmethod
    def _to_stereo(d):
        x = np.asarray(d, np.float64)
        return x.reshape(-1, 2) if x.ndim == 1 else x[:, :2]

    @staticmethod
    def _to_mono(d):
        x = np.asarray(d, np.float64)
        return x if x.ndim == 1 else x[:, 0]

    def _read_input(self):
        """Block until new input is available and return it as (n, 2)."""
        if self._q2 is None:
            while True:
                d = self._q1.get()
                if d is not None:
                    return self._to_stereo(d)
        # Two mono sources: read from the channel that is behind, so that arbitrary chunk sizes
        # stay sample-synchronous
        while True:
            c = 0 if len(self._in_pending[0]) <= len(self._in_pending[1]) else 1
            d = (self._q1 if c == 0 else self._q2).get()
            if d is None:
                continue
            self._in_pending[c] = np.concatenate([self._in_pending[c], self._to_mono(d)])
            n = min(len(self._in_pending[0]), len(self._in_pending[1]))
            if n:
                x = np.stack([self._in_pending[0][:n], self._in_pending[1][:n]], 1)
                self._in_pending = [p[n:] for p in self._in_pending]
                return x

    def _check_queue(self):
        size = self._q1.qsize() + (self._q2.qsize() if self._q2 is not None else 0)
        if size > self.QUEUE_WARN_SIZE and time.time() - self._last_queue_warn > 5.0:
            self._last_queue_warn = time.time()
            print(f"[Warning] BleedCancel cannot keep up with the input ({size} chunks queued).")

    def _process(self):
        while True:
            try:
                x = self._read_input()
                for fr in self._feed(x):
                    self._put_to_all_queues(fr["output"].reshape(-1).tolist())
                    if self._monitor_queues:
                        with self._lock:
                            for q in self._monitor_queues:
                                q.put(fr)
                self._check_queue()
            except Exception as e:
                print('[BleedCancel] Error:', e)
                time.sleep(0.01)

    def start(self):
        self.source.start()
        if self.source_ch2 is not None:
            self.source_ch2.start()
        if not self._is_thread_started:
            threading.Thread(target=self._process, daemon=True).start()
            self._is_thread_started = True

    def set_enabled(self, enabled):
        """Turn bleed removal on/off at runtime (processing keeps running either way)."""
        self._enabled = bool(enabled)

    @property
    def enabled(self):
        """bool: Whether bleed removal is currently applied to the output."""
        return self._enabled

    @property
    def latency_ms(self):
        """float: Algorithmic latency in ms (processing + block buffering, excluding device buffers)."""
        return 1000.0 * (self._engine.latency + self._engine.block) / self.SAMPLING_RATE

    def status(self):
        """Return the adaptation state of the canceller (delays, AEC convergence, AES gain, warnings)."""
        return self._engine.status()

    def reset(self):
        """Discard the adapted filters and start adapting from scratch (e.g. after moving the mics)."""
        self._reset_requested = True

    def subscribe_monitor(self):
        """Subscribe to per-frame monitoring data (for visualization / recording).

        Returns:
            queue.Queue: receives dicts with ``raw`` (input aligned to the output latency),
            ``clean`` (processed, regardless of the on/off state), ``output`` (what is published)
            as (FRAME_SIZE, 2) float arrays, and ``enabled`` (bool).
        """
        q = queue.Queue()
        with self._lock:
            self._monitor_queues.append(q)
        return q
