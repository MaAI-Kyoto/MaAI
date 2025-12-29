"""Output utilities for rendering and transmitting model results."""

import sys
import math
import time
from typing import Dict, Any, List, Union
import numpy as np
import socket
import threading
import time
import pickle
import queue
from . import util
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def _draw_bar(value: float, length: int = 30) -> str:
    """Draw a basic bar graph representation.

    Args:
        value: Normalized value between 0 and 1.
        length: Total bar length in characters.

    Returns:
        Rendered bar string.
    """
    bar_len = min(length, max(0, int(value * length)))
    return '█' * bar_len + '-' * (length - bar_len)


def _draw_symmetric_bar(value: float, length: int = 30) -> str:
    """Draw a symmetric bar graph in the range -1.0 to 1.0.

    Args:
        value: Value in the range -1.0 to 1.0.
        length: Total bar length in characters.

    Returns:
        Rendered bar string.
    """
    max_len = length // 2
    value = max(-1.0, min(1.0, value))
    if value >= 0:
        pos = int(value * max_len)
        return ' ' * max_len + '│' + '█' * pos + ' ' * (max_len - pos)
    else:
        neg = int(-value * max_len)
        return ' ' * (max_len - neg) + '█' * neg + '│' + ' ' * max_len


def _draw_balance_bar(value: float, length: int = 30) -> str:
    """Draw a balance bar centered at 0.5.

    Args:
        value: Value in the range 0.0 to 1.0.
        length: Total bar length in characters.

    Returns:
        Rendered bar string.
    """
    # 2チャネルの場合は1チャネル目のデータを渡すこと
    max_len = length // 2
    value = max(0.0, min(1.0, value))
    diff = value - 0.5
    if diff > 0:
        # value > 0.5: ch1が話す確率が高い → 左側にバー
        left = int(diff * 2 * max_len + 0.5)
        return ' ' * (max_len - left) + '█' * left + '│' + ' ' * max_len
    else:
        # value < 0.5: ch2が話す確率が高い → 右側にバー
        right = int(-diff * 2 * max_len + 0.5)
        return ' ' * max_len + '│' + '█' * right + ' ' * (max_len - right)


def _rms(values: Union[List[float], tuple]) -> float:
    """Compute the RMS value of a sequence.

    Args:
        values: Sequence of numeric values.

    Returns:
        RMS value.
    """
    if not values:
        return 0.0
    return math.sqrt(sum(x * x for x in values) / len(values))


def _format_value(value: Any, max_length: int = 50) -> str:
    """Format a value for display in a compact form.

    Args:
        value: Value to format.
        max_length: Maximum string length before truncation.

    Returns:
        Formatted string representation.
    """
    if isinstance(value, (list, tuple)):
        if len(value) > 0:
            if isinstance(value[0], (int, float)):
                # 数値のリストの場合、最初の数個の値を表示
                preview = value[:5]
                if len(value) > 5:
                    return f"[{', '.join(f'{v:.3f}' for v in preview)}...] ({len(value)} items)"
                else:
                    return f"[{', '.join(f'{v:.3f}' for v in preview)}]"
            else:
                return f"[{', '.join(str(v) for v in value[:3])}...]" if len(value) > 3 else str(value)
        else:
            return "[]"
    elif isinstance(value, float):
        return f"{value:.4f}"
    elif isinstance(value, int):
        return str(value)
    else:
        str_val = str(value)
        if len(str_val) > max_length:
            return str_val[:max_length-3] + "..."
        return str_val


def _get_bar_for_value(key: str, value: Any, bar_length: int = 30, bar_type: str = "normal") -> tuple[str, float]:
    """Select the appropriate bar visualization for a value.

    Args:
        key: Result key name.
        value: Value to visualize.
        bar_length: Total bar length in characters.
        bar_type: Visualization style ("normal" or "balance").

    Returns:
        Tuple of (bar string, numeric value used).
    """
    if isinstance(value, (list, tuple)):
        if len(value) > 2:
            if isinstance(value[0], (int, float)):
                # 数値のリストの場合、RMS値を計算
                rms_val = _rms(value) * 3  # type: ignore
                return _draw_bar(rms_val, bar_length), rms_val
            else:
                return "N/A", 0.0
        elif len(value) == 2:
            _value = value[0] / (value[0] + value[1])
            return _draw_balance_bar(float(_value), bar_length), value
        else:
            return "N/A", 0.0
    elif isinstance(value, (int, float)):
        return _draw_bar(float(value), bar_length), float(value)
    else:
        return "N/A", 0.0


class ConsoleBar:
    """
    Render maai.get_result() output as a console bar chart.
    """
    def __init__(self, bar_length: int = 30, bar_type: str = "normal"):
        self.bar_length = bar_length
        self.bar_type = bar_type
        self._first = True

    def update(self, result: Dict[str, Any]):
        """Render an updated bar chart for a result dictionary.

        Args:
            result: Result dictionary from the model.
        """
        if self._first:
            sys.stdout.write("\x1b[2J")  # 初期クリア
            self._first = False
        sys.stdout.write("\x1b[H")  # カーソルを左上に移動
        
        # 時刻の表示
        if 't' in result:
            dt = time.localtime(result['t'])
            ms = int((result['t'] - int(result['t'])) * 1000)
            print(f"Time: {dt.tm_year:04d}/{dt.tm_mon:02d}/{dt.tm_mday:02d} {dt.tm_hour:02d}:{dt.tm_min:02d}:{dt.tm_sec:02d}.{ms:03d}")
            print("-" * (self.bar_length + 30))
        
        # bar_typeがbalanceのとき、x1/x2の値とバーを2行で横並び表示
        if self.bar_type == "balance" and 'x1' in result and 'x2' in result:
            x1 = np.squeeze(np.array(result['x1'])).tolist()
            x2 = np.squeeze(np.array(result['x2'])).tolist()
            bar1, val1 = _get_bar_for_value('x1', x1, self.bar_length // 2 - 1, "normal")
            bar1 = bar1[::-1]
            bar2, val2 = _get_bar_for_value('x2', x2, self.bar_length // 2 - 1, "normal")
            print(f"x1 │ x2{' ' * 8}: {bar1} │ {bar2} ({val1:.4f}, {val2:.4f})")
        
        # vadがある場合はvad[0]をvad(x1)、vad[1]をvad(x2)として表示
        if 'vad' in result:
            vad1 = result['vad'][0]
            vad2 = result['vad'][1]
            result['vad(x1)'] = vad1
            result['vad(x2)'] = vad2
    
        # 各キーを動的に処理
        for key, value in result.items():
            if key == 't' or key == 'vad':
                continue
            # x1/x2は既に横並びで表示したのでスキップ
            if self.bar_type == "balance" and key in ['x1', 'x2']:
                continue
            if not isinstance(value, (float, int)):
                value = np.squeeze(np.array(value)).tolist()
            bar, _value = _get_bar_for_value(key, value, self.bar_length, self.bar_type)
            if type(_value) is float:
                print(f"{key:15}: {bar} ({_value:.3f})")
            elif type(_value) is list:
                print(f"{key:15}: {bar} ({', '.join(f'{v:.3f}' for v in _value)})")
        print("-" * (self.bar_length + 30))

class TcpReceiver:
    """Receive VAP results over TCP and expose a queue interface."""
    def __init__(self, ip, port, mode):
        self.ip = ip
        self.port = port
        self.mode = mode
        self.sock = None
        self.result_queue = queue.Queue()
    
    def _bytearray_2_vapresult(self, data: bytes) -> Dict[str, Any]:
        """Decode a byte payload into a VAP result dict.

        Args:
            data: Serialized result payload.

        Returns:
            Decoded result dictionary.
        """
        if self.mode in ['vap', 'vap_mc', 'vap_prompt']:
            vap_result = util.conv_bytearray_2_vapresult(data)
        elif self.mode == 'bc_2type':
            vap_result = util.conv_bytearray_2_vapresult_bc_2type(data)
        elif self.mode == 'nod':
            vap_result = util.conv_bytearray_2_vapresult_nod(data)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        return vap_result
    
    def connect_server(self):    
        """Connect to the TCP result server."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.ip, self.port))
        print('[CLIENT] Connected to the server')

    def _start_client(self):
        """Continuously receive results and enqueue them."""
        while True:
            try:
                self.connect_server()
                while True:
                    try:
                        size = int.from_bytes(self.sock.recv(4), 'little')
                        data = b''
                        while len(data) < size:
                            data += self.sock.recv(size - len(data))
                        vap_result = self._bytearray_2_vapresult(data)
                        self.result_queue.put(vap_result)

                    except Exception as e:
                        print('[CLIENT] Receive error:', e)
                        break  # 受信エラー時は再接続ループへ
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
        """Start the TCP receiver thread."""
        threading.Thread(target=self._start_client, daemon=True).start()
    
    def get_result(self):
        """Return the next received result."""
        return self.result_queue.get()

class TcpTransmitter:
    """Transmit VAP results over TCP to a client."""
    def __init__(self, ip, port, mode):
        self.ip = ip
        self.port = port
        self.mode = mode
        self.result_queue = queue.Queue()
    
    def _vapresult_2_bytearray(self, result_dict: Dict[str, Any]) -> bytes:
        """Encode a result dictionary to bytes for transmission.

        Args:
            result_dict: Result dictionary to encode.

        Returns:
            Serialized byte buffer.
        """
        if self.mode in ['vap', 'vap_mc']:
            data_sent = util.conv_vapresult_2_bytearray(result_dict)
        elif self.mode == 'bc_2type':
            data_sent = util.conv_vapresult_2_bytearray_bc_2type(result_dict)
        elif self.mode == 'nod':
            data_sent = util.conv_vapresult_2_bytearray_nod(result_dict)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        return data_sent
        
    def _start_server(self):
        """Start the TCP server loop for sending results."""
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind((self.ip, self.port))
                s.listen(1)
                print('[OUT] Waiting for connection...')
                conn, addr = s.accept()
                print('[OUT] Connected by', addr)
                while True:
                    try:
                        result_dict = self.result_queue.get()
                        data_sent = self._vapresult_2_bytearray(result_dict)
                        data_sent_all = len(data_sent).to_bytes(4, 'little') + data_sent
                        conn.sendall(data_sent_all)
                    except Exception as e:
                        print('[OUT] Send error:', e)
                        break
            except Exception as e:
                print('[OUT] Disconnected by', addr)
                print(e)
                continue
            
    def start_server(self):
        """Launch the TCP server thread."""
        threading.Thread(target=self._start_server, daemon=True).start()
        
    def update(self, result: Dict[str, Any]):
        """Queue a result for transmission.

        Args:
            result: Result dictionary to send.
        """
        self.result_queue.put(result)

# 新規追加: GUIでバーグラフを表示するクラス
class GuiBar:
    """Show results as a GUI bar chart using matplotlib."""
    def __init__(self, bar_type: str = "normal"):
        self.bar_type = bar_type
        self.plt = plt
        self.fig, self.ax = plt.subplots()
        plt.ion()
        plt.show()
        # バーアーティストを保持してリアルタイム更新
        self.bars = None

        import seaborn as sns
        sns.set_theme(style="whitegrid")

    def update(self, result: Dict[str, Any]):
        """Update the bar chart using a result dictionary.

        Args:
            result: Result dictionary from the model.
        """
        labels = []
        values = []
        for key, value in result.items():
            if key == 't':
                continue
            # 配列やリストは適切にスカラー化
            if not isinstance(value, (int, float)):
                value = np.squeeze(np.array(value)).tolist()
            if isinstance(value, (list, tuple)):
                if len(value) > 2 and isinstance(value[0], (int, float)):
                    val = _rms(value) * 3
                elif len(value) == 2:
                    total = value[0] + value[1]
                    val = (value[1] / total) if total != 0 else 0.0
                else:
                    val = 0.0
            else:
                try:
                    val = float(value)
                except Exception:
                    continue
            labels.append(key)
            values.append(val)
        # 初回描画またはラベル数が変わった場合は新規描画
        if self.bars is None or len(self.bars) != len(values):
            self.ax.clear()
            self.bars = self.ax.bar(labels, values, color='skyblue')
            self.ax.set_ylim(0, 1)
            self.ax.set_xticks(range(len(labels)))
            self.ax.set_xticklabels(labels)
            self.ax.set_title('Result Bar Graph')
        else:
            # 既存のバーを更新
            for bar, v in zip(self.bars, values):
                bar.set_height(v)
        # 描画を反映
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)
        
class GuiPlot:
    """Plot waveform and probability traces in a GUI.

    Args:
        shown_context_sec: Seconds of history to display.
        frame_rate: Frame rate used for model outputs.
        sample_rate: Sample rate of waveform inputs.
        figsize: Matplotlib figure size.
        use_fixed_draw_rate: Whether to throttle redraws.
    """
    def __init__(self, shown_context_sec: int = 10, frame_rate: int = 10, sample_rate: int = 16000, figsize=(14, 10), use_fixed_draw_rate: bool = True):
        self.figsize = figsize
        self.shown_context_sec = shown_context_sec
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.MAX_CONTEXT_LEN = frame_rate * shown_context_sec
        self.MAX_CONTEXT_WAV_LEN = sample_rate * shown_context_sec
        self.plt = plt
        self.fig = None
        self.axes = dict()
        self.lines = dict()
        self.fills = dict()
        self.keys = []
        self.initialized = False
        self.data_buffer = dict()  # key: list or float
        self.use_fixed_draw_rate = use_fixed_draw_rate
        self._last_draw_time = 0.0

    def _init_fig(self, result: Dict[str, any]):
        """Initialize the matplotlib figure layout for result keys.

        Args:
            result: Initial result dictionary.
        """
        special_keys = ['x1', 'x2', 'p_now', 'p_future', 'vad']
        self.keys = [k for k in special_keys if k in result] + [k for k in result.keys() if k not in special_keys and k != 't']
        n = len(self.keys)
        self.fig, axs = self.plt.subplots(n, 1, figsize=self.figsize, squeeze=False, tight_layout=True)
        axs = axs.flatten()
        self.axes = {}
        self.lines = {}
        self.fills = {}
        self.data_buffer = {}
        tab_colors = list(mcolors.TABLEAU_COLORS.values())
        p_keys = [k for k in self.keys if k.startswith('p_') and k not in ('p_now', 'p_future')]
        color_map = {k: tab_colors[i % len(tab_colors)] for i, k in enumerate(p_keys)}
        th_map = {k: 0.5 for k in p_keys}
        for i, key in enumerate(self.keys):
            ax = axs[i]
            self.axes[key] = ax
            val = result[key]
            if not isinstance(val, (int, float)):
                val = np.squeeze(np.array(val))
            if key == 'x1':
                time_x1 = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_WAV_LEN)
                buf = np.zeros(len(time_x1))
                line, = ax.plot(time_x1, buf, c='y')
                self.lines[key] = line
                self.data_buffer[key] = list(buf)
                ax.set_title('Input waveform 1')
                ax.set_xlabel('Time [s]')
                ax.set_ylim(-1, 1)
                ax.set_xlim(-self.shown_context_sec, 0)
            elif key == 'x2':
                time_x2 = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_WAV_LEN)
                buf = np.zeros(len(time_x2))
                line, = ax.plot(time_x2, buf, c='b')
                self.lines[key] = line
                self.data_buffer[key] = list(buf)
                ax.set_title('Input waveform 2')
                ax.set_xlabel('Time [s]')
                ax.set_ylim(-1, 1)
                ax.set_xlim(-self.shown_context_sec, 0)
            elif key == 'p_now':
                time_x3 = np.linspace(0, self.MAX_CONTEXT_LEN, self.MAX_CONTEXT_LEN)
                buf = np.ones(len(time_x3)) * 0.5
                fill1 = ax.fill_between(time_x3, y1=0.5, y2=buf, where=buf > 0.5, color='y', interpolate=True)
                fill2 = ax.fill_between(time_x3, y1=buf, y2=0.5, where=buf < 0.5, color='b', interpolate=True)
                self.fills[key] = (fill1, fill2)
                self.data_buffer[key] = list(buf)
                ax.set_title('Output p_now (short-term turn-taking prediction)')
                ax.set_xlabel('Sample')
                ax.set_xlim(0, self.MAX_CONTEXT_LEN)
                ax.set_ylim(0, 1)
            elif key == 'p_future':
                time_x4 = np.linspace(0, self.MAX_CONTEXT_LEN, self.MAX_CONTEXT_LEN)
                buf = np.ones(len(time_x4)) * 0.5
                fill1 = ax.fill_between(time_x4, y1=0.5, y2=buf, where=buf > 0.5, color='y', interpolate=True)
                fill2 = ax.fill_between(time_x4, y1=buf, y2=0.5, where=buf < 0.5, color='b', interpolate=True)
                self.fills[key] = (fill1, fill2)
                self.data_buffer[key] = list(buf)
                ax.set_title('Output p_future (long-term turn-taking prediction)')
                ax.set_xlabel('Sample')
                ax.set_xlim(0, self.MAX_CONTEXT_LEN)
                ax.set_ylim(0, 1)
            elif key.startswith('p_'):
                color = color_map.get(key, 'green')
                th = th_map.get(key, 0.5)
                time_x = np.linspace(0, self.MAX_CONTEXT_LEN, self.MAX_CONTEXT_LEN)
                buf = np.zeros(len(time_x))
                fill = ax.fill_between(time_x, y1=0.0, y2=buf, color=color, interpolate=True)
                self.fills[key] = fill
                self.data_buffer[key] = list(buf)
                ax.set_title(key)
                ax.set_xlabel('Sample')
                ax.set_xlim(0, self.MAX_CONTEXT_LEN)
                ax.set_ylim(0, th*2)
                ax.axhline(y=th, color='black', linestyle='--')
            elif key == 'vad':
                time_x = np.arange(self.MAX_CONTEXT_LEN)
                buf1 = np.zeros(self.MAX_CONTEXT_LEN)
                buf2 = np.zeros(self.MAX_CONTEXT_LEN)
                fill1 = ax.fill_between(time_x, y1=0, y2=buf1, where=buf1>0, color='y', alpha=0.7, label='VAD1', interpolate=True)
                fill2 = ax.fill_between(time_x, y1=0, y2=-buf2, where=buf2>0, color='b', alpha=0.7, label='VAD2', interpolate=True)
                self.fills[key] = (fill1, fill2)
                self.data_buffer[key] = [list(buf1), list(buf2)]
                ax.set_title('Voice Activity Detection (VAD)')
                ax.set_xlabel('Frame')
                ax.set_ylabel('VAD2  VAD1')
                ax.set_ylim(-1, 1)
                ax.set_xlim(0, self.MAX_CONTEXT_LEN)
                ax.axhline(0, color='black', linestyle='--', linewidth=1)
                ax.legend(loc='upper right')
            elif isinstance(val, (np.ndarray, list, tuple)) and np.array(val).ndim == 1 and len(val) > 1:
                x = np.arange(len(val))
                line, = ax.plot(x, val, c='g')
                self.lines[key] = line
                self.data_buffer[key] = list(val)
                ax.set_title(key)
                ax.set_xlim(0, len(val))
            elif isinstance(val, (int, float, np.floating, np.integer)):
                bar = ax.bar([key], [val], color='orange')
                self.lines[key] = bar
                self.data_buffer[key] = float(val)
                ax.set_title(key)
                ax.set_ylim(0, 1)
            else:
                ax.text(0.5, 0.5, str(val), ha='center', va='center')
                ax.set_title(key)
        self.fig.tight_layout()
        self.plt.ion()
        self.plt.show()
        self.initialized = True

    def update(self, result: Dict[str, any]):
        """Update all plots with the latest result values.

        Args:
            result: Result dictionary from the model.
        """
        import time
        draw = True
        if self.use_fixed_draw_rate:
            now = time.time()
            if now - self._last_draw_time < 0.2:
                draw = False
            else:
                self._last_draw_time = now
        if not self.initialized:
            self._init_fig(result)
        tab_colors = list(mcolors.TABLEAU_COLORS.values())
        p_keys = [k for k in self.keys if k.startswith('p_') and k not in ('p_now', 'p_future')]
        color_map = {k: tab_colors[i % len(tab_colors)] for i, k in enumerate(p_keys)}
        for key in self.keys:
            if key not in result:
                continue
            val = result[key]
            if key in ['x1', 'x2'] and key in self.lines:
                buf = self.data_buffer[key]
                val = val[-self.sample_rate // self.frame_rate:]
                buf = buf + list(val)
                if len(buf) > self.MAX_CONTEXT_WAV_LEN:
                    buf = buf[-self.MAX_CONTEXT_WAV_LEN:]
                self.data_buffer[key] = buf
                if draw:
                    time_x = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_WAV_LEN)
                    self.lines[key].set_data(time_x, buf)
            elif (key == 'p_now' or key == 'p_future') and key in self.fills:
                buf = self.data_buffer[key]
                buf = buf + [float(val[0])]
                if len(buf) > self.MAX_CONTEXT_LEN:
                    buf = buf[-self.MAX_CONTEXT_LEN:]
                self.data_buffer[key] = buf
                if draw:
                    time_x = np.linspace(0, self.MAX_CONTEXT_LEN, self.MAX_CONTEXT_LEN)
                    ax = self.axes[key]
                    arr = np.array(buf)
                    fills = self.fills[key]
                    for f in fills:
                        if f is not None:
                            f.remove()
                    fill1 = ax.fill_between(time_x, y1=0.5, y2=arr, where=arr > 0.5, color='y', interpolate=True)
                    fill2 = ax.fill_between(time_x, y1=arr, y2=0.5, where=arr < 0.5, color='b', interpolate=True)
                    self.fills[key] = [fill1, fill2]
            elif key.startswith('p_') and key in self.fills:
                buf = self.data_buffer[key]
                color = color_map.get(key, 'green')
                buf = buf + [float(val)]
                if len(buf) > self.MAX_CONTEXT_LEN:
                    buf = buf[-self.MAX_CONTEXT_LEN:]
                self.data_buffer[key] = buf
                if draw:
                    time_x = np.linspace(0, self.MAX_CONTEXT_LEN, self.MAX_CONTEXT_LEN)
                    ax = self.axes[key]
                    arr = np.array(buf)
                    f = self.fills[key]
                    if f is not None:
                        f.remove()
                    fill = ax.fill_between(time_x, y1=0.0, y2=arr, color=color, interpolate=True)
                    self.fills[key] = fill
            elif key == 'vad' and key in self.fills:
                buf1, buf2 = self.data_buffer[key]
                vad1, vad2 = result[key]
                buf1 = list(buf1) + [float(vad1)]
                buf2 = list(buf2) + [float(vad2)]
                if len(buf1) > self.MAX_CONTEXT_LEN:
                    buf1 = buf1[-self.MAX_CONTEXT_LEN:]
                if len(buf2) > self.MAX_CONTEXT_LEN:
                    buf2 = buf2[-self.MAX_CONTEXT_LEN:]
                self.data_buffer[key] = [buf1, buf2]
                if draw:
                    time_x = np.arange(self.MAX_CONTEXT_LEN)
                    ax = self.axes[key]
                    arr1 = np.array(buf1)
                    arr2 = np.array(buf2)
                    fills = self.fills[key]
                    for f in fills:
                        if f is not None:
                            f.remove()
                    fill1 = ax.fill_between(time_x, y1=0, y2=arr1, where=arr1>0, color='y', alpha=0.7, label='VAD1', interpolate=True)
                    fill2 = ax.fill_between(time_x, y1=0, y2=-arr2, where=arr2>0, color='b', alpha=0.7, label='VAD2', interpolate=True)
                    self.fills[key] = [fill1, fill2]
            elif key in self.lines:
                buf = self.data_buffer[key]
                if isinstance(val, (np.ndarray, list, tuple)) and len(val) > 1:
                    buf = list(buf) + list(val)
                    if len(buf) > self.MAX_CONTEXT_WAV_LEN: # Changed from self.maxlen to self.MAX_CONTEXT_WAV_LEN
                        buf = buf[-self.MAX_CONTEXT_WAV_LEN:]
                    self.data_buffer[key] = buf
                    if draw:
                        x = np.arange(len(buf))
                        self.lines[key].set_data(x, buf)
                        self.axes[key].set_xlim(0, len(buf))
                        self.axes[key].relim()
                        self.axes[key].autoscale_view()
                else:
                    v = float(val)
                    self.data_buffer[key] = v
                    if draw:
                        self.lines[key][0].set_height(v)
        if draw:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
