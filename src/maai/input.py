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
