#!/usr/bin/env python3
"""
GUI viewer of the experimental three-party VAP model (MaaiTri).

Scrolling time-series of each speaker's waveform and voice activity, and the
next-speaker probabilities p_now (0-0.6 s) / p_future (0.6-2.0 s) as stacked
areas, with a live card per speaker. The power button on a card marks the speaker
as absent (MaaiTri.set_present). Play / Pause (or Space) controls playback; it
starts paused unless --autoplay is given.

    python vap_tri_gui.py                         # J01_city_MaAI.wav on the desktop
    python vap_tri_gui.py --wav other_3ch.wav
    python vap_tri_gui.py --mic                   # 3-channel audio interface

Requires PyQt5 and pyqtgraph (pip install PyQt5 pyqtgraph).
"""

import argparse
import os
import sys
import time
from collections import deque

import numpy as np

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# This allows you to import modules from the src directory without pip installing the package.
# Uncomment the line below if you need to run this script directly without installing the package.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from PyQt5 import QtCore, QtGui, QtWidgets  # before pyqtgraph so that it uses PyQt5
import pyqtgraph as pg

from maai import MaaiTri, MaaiInput

DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")

FRAME_HZ = 12.5
ENV_POINTS = 16  # waveform envelope points per 80 ms frame

BG = "#0D0F14"
PANEL = "#151821"
BORDER = "#222736"
GRID = "#2A3042"
TEXT = "#E8EAF1"
MUTED = "#7F869A"
SPK_COLORS = ["#4FD1C5", "#F687B3", "#F6C453"]
SPK_NAMES = ["Speaker 1", "Speaker 2", "Speaker 3"]
FONT = "Segoe UI"

STYLE = f"""
QWidget#root {{ background: {BG}; }}
QLabel {{ color: {TEXT}; font-family: "{FONT}", "Yu Gothic UI"; }}
QLabel#title {{ font-size: 21px; font-weight: 600; letter-spacing: 0.5px; }}
QLabel#subtitle {{ color: {MUTED}; font-size: 12px; }}
QLabel#status {{ color: {MUTED}; font-size: 12px; font-family: "Consolas", "{FONT}"; }}
QFrame#panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 16px; }}
QLabel#cardName {{ font-size: 14px; font-weight: 600; }}
QLabel#big {{ font-size: 40px; font-weight: 300; }}
QLabel#caption {{ color: {MUTED}; font-size: 11px; letter-spacing: 1px; }}
QLabel#rowLabel {{ color: {MUTED}; font-size: 11px; }}
QLabel#rowValue {{ color: {TEXT}; font-size: 11px; font-family: "Consolas", "{FONT}"; }}
QPushButton#toggle {{
    color: {MUTED}; background: transparent; border: 1px solid {BORDER};
    border-radius: 10px; min-height: 20px; padding: 0px 12px; font-size: 11px; font-family: "{FONT}";
}}
QPushButton#toggle:checked {{ color: {TEXT}; border-color: {MUTED}; }}
QPushButton#play {{
    color: {BG}; background: {TEXT}; border: none; border-radius: 17px;
    min-height: 34px; min-width: 112px; padding: 0px 18px;
    font-size: 13px; font-weight: 600; font-family: "{FONT}";
}}
QPushButton#play:hover {{ background: #FFFFFF; }}
QPushButton#play[playing="true"] {{ color: {TEXT}; background: transparent; border: 1px solid {MUTED}; }}
QPushButton#play[playing="true"]:hover {{ border-color: {TEXT}; }}
QPushButton#play:disabled {{ color: {MUTED}; background: transparent; border: 1px solid {BORDER}; }}
"""


def _rgba(hex_color: str, alpha: int) -> QtGui.QColor:
    c = QtGui.QColor(hex_color)
    c.setAlpha(alpha)
    return c


class Meter(QtWidgets.QWidget):
    """Thin rounded horizontal meter."""

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._color = QtGui.QColor(color)
        self._value = 0.0
        self.setFixedHeight(6)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_value(self, v: float):
        v = float(min(max(v, 0.0), 1.0))
        if abs(v - self._value) > 1e-3:
            self._value = v
            self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = QtCore.QRectF(self.rect())
        rad = r.height() / 2
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(GRID))
        p.drawRoundedRect(r, rad, rad)
        if self._value > 0:
            w = max(r.width() * self._value, r.height())
            p.setBrush(self._color)
            p.drawRoundedRect(QtCore.QRectF(r.left(), r.top(), w, r.height()), rad, rad)


class SpeakerCard(QtWidgets.QFrame):
    presence_changed = QtCore.pyqtSignal(int, bool)

    def __init__(self, idx: int, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.color = SPK_COLORS[idx]
        self._active = None
        self._speaking = None
        self._present = True
        self.setObjectName("panel")

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 16)
        lay.setSpacing(4)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        self.dot = QtWidgets.QLabel()
        self.dot.setFixedSize(10, 10)
        name = QtWidgets.QLabel(SPK_NAMES[idx])
        name.setObjectName("cardName")
        self.toggle = QtWidgets.QPushButton("present")
        self.toggle.setObjectName("toggle")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setCursor(QtCore.Qt.PointingHandCursor)
        self.toggle.toggled.connect(self._on_toggle)
        top.addWidget(self.dot)
        top.addWidget(name)
        top.addStretch(1)
        top.addWidget(self.toggle)
        lay.addLayout(top)

        self.big = QtWidgets.QLabel("—")
        self.big.setObjectName("big")
        lay.addWidget(self.big)
        cap = QtWidgets.QLabel("NEXT-SPEAKER PROBABILITY · NOW")
        cap.setObjectName("caption")
        lay.addWidget(cap)
        lay.addSpacing(8)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(7)
        self.meters, self.values = {}, {}
        for row, key in enumerate(("now", "future", "voice")):
            lab = QtWidgets.QLabel(key)
            lab.setObjectName("rowLabel")
            m = Meter(self.color)
            val = QtWidgets.QLabel("0.00")
            val.setObjectName("rowValue")
            val.setFixedWidth(34)
            val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            grid.addWidget(lab, row, 0)
            grid.addWidget(m, row, 1)
            grid.addWidget(val, row, 2)
            self.meters[key], self.values[key] = m, val
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)

        self._set_active(False)
        self._set_speaking(False)

    def _on_toggle(self, checked: bool):
        self._present = checked
        self.toggle.setText("present" if checked else "absent")
        self.setGraphicsEffect(None)
        if not checked:
            eff = QtWidgets.QGraphicsOpacityEffect(self)
            eff.setOpacity(0.4)
            self.setGraphicsEffect(eff)
        self.presence_changed.emit(self.idx, checked)

    def _set_active(self, active: bool):
        if active == self._active:
            return
        self._active = active
        if active:
            tint = _rgba(self.color, 22).name(QtGui.QColor.HexArgb)
            self.setStyleSheet(
                f"QFrame#panel {{ background: {tint}; border: 1px solid {self.color}; }}"
                f"QLabel#big {{ color: {self.color}; }}"
            )
        else:
            self.setStyleSheet("")

    def _set_speaking(self, speaking: bool):
        if speaking == self._speaking:
            return
        self._speaking = speaking
        fill = self.color if speaking else GRID
        self.dot.setStyleSheet(f"background: {fill}; border-radius: 5px;")

    def set_values(self, p_now: float, p_future: float, vad: float, is_next: bool):
        self.big.setText(f"{p_now * 100:.0f}%")
        for key, v in (("now", p_now), ("future", p_future), ("voice", vad)):
            self.meters[key].set_value(v)
            self.values[key].setText(f"{v:.2f}")
        self._set_active(is_next and self._present)
        self._set_speaking(vad >= 0.5)


def _style_plot(plot: pg.PlotItem, title: str, show_x: bool):
    plot.setTitle(f"<span style='color:{MUTED}; font-size:9pt; letter-spacing:1px'>{title}</span>", justify="left")
    plot.setMenuEnabled(False)
    plot.setMouseEnabled(x=False, y=False)
    plot.hideButtons()
    plot.showGrid(x=True, y=False, alpha=0.12)
    font = QtGui.QFont(FONT, 8)
    for name in ("left", "bottom"):
        ax = plot.getAxis(name)
        ax.setPen(pg.mkPen(GRID))
        ax.setTextPen(pg.mkPen(MUTED))
        ax.setStyle(tickFont=font, tickLength=0)
    plot.getAxis("left").setWidth(34)
    if show_x:
        plot.getAxis("bottom").setLabel("seconds", color=MUTED)
    else:
        plot.getAxis("bottom").setStyle(showValues=False)
    plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(MUTED, width=1, style=QtCore.Qt.DashLine)))


class TriVapWindow(QtWidgets.QWidget):
    def __init__(self, maai: MaaiTri, source, source_name: str, window_sec: float):
        super().__init__()
        self.maai = maai
        self.source = source
        self.started = False
        self.playing = False
        self.finished = False
        self.n_frames = int(round(window_sec * FRAME_HZ))
        self.setObjectName("root")
        self.setWindowTitle("MaAI · Three-party VAP")
        self.resize(1360, 820)

        n = self.n_frames
        self.env_min = [deque(np.zeros((n, ENV_POINTS)), maxlen=n) for _ in range(3)]
        self.env_max = [deque(np.zeros((n, ENV_POINTS)), maxlen=n) for _ in range(3)]
        self.vad = deque(np.zeros((n, 3)), maxlen=n)
        self.p_now = deque(np.full((n, 3), 1 / 3), maxlen=n)
        self.p_future = deque(np.full((n, 3), 1 / 3), maxlen=n)
        self.frames = 0
        self.t_first = None
        self.last_latency = 0.0
        self.fps_hist = deque(maxlen=50)

        # x: frame i covers the 80 ms ending at x_frame[i]; 0 = now
        self.x_frame = (np.arange(n) - (n - 1)) / FRAME_HZ
        sub = (np.arange(ENV_POINTS) + 0.5) / ENV_POINTS - 1.0
        self.x_env = (self.x_frame[:, None] + sub[None, :] / FRAME_HZ).ravel()
        self.wave_scale = 0.4 / 0.05

        self._build_ui(source_name)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)

    # ------------------------------------------------------------------ UI
    def _build_ui(self, source_name: str):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)

        header = QtWidgets.QHBoxLayout()
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(2)
        title = QtWidgets.QLabel("Three-party Voice Activity Projection")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(f"MaAI · MaaiTri (experimental)  —  {source_name}")
        subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch(1)
        self.status = QtWidgets.QLabel("ready  ·  press Play (Space)")
        self.status.setObjectName("status")
        self.status.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        header.addWidget(self.status)
        header.addSpacing(18)
        self.play_btn = QtWidgets.QPushButton()
        self.play_btn.setObjectName("play")
        self.play_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.play_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.play_btn.clicked.connect(self.toggle_play)
        header.addWidget(self.play_btn)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self, activated=self.toggle_play)
        self._update_play_button()
        root.addLayout(header)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, 1)

        # --- plots
        plot_panel = QtWidgets.QFrame()
        plot_panel.setObjectName("panel")
        pl = QtWidgets.QVBoxLayout(plot_panel)
        pl.setContentsMargins(10, 10, 14, 8)
        glw = pg.GraphicsLayoutWidget()
        glw.setBackground(PANEL)
        glw.ci.layout.setVerticalSpacing(6)
        pl.addWidget(glw)
        body.addWidget(plot_panel, 1)

        window = self.n_frames / FRAME_HZ
        self.p_wave = glw.addPlot(row=0, col=0)
        self.p_pnow = glw.addPlot(row=1, col=0)
        self.p_pfut = glw.addPlot(row=2, col=0)
        glw.ci.layout.setRowStretchFactor(0, 5)
        glw.ci.layout.setRowStretchFactor(1, 3)
        glw.ci.layout.setRowStretchFactor(2, 3)
        _style_plot(self.p_wave, "WAVEFORM  ·  VOICE ACTIVITY", show_x=False)
        _style_plot(self.p_pnow, "NEXT SPEAKER  ·  p_now  (0 – 0.6 s)", show_x=False)
        _style_plot(self.p_pfut, "NEXT SPEAKER  ·  p_future  (0.6 – 2.0 s)", show_x=True)
        for p in (self.p_wave, self.p_pnow, self.p_pfut):
            p.setXRange(-window, 0.15, padding=0)
        self.p_pnow.setXLink(self.p_wave)
        self.p_pfut.setXLink(self.p_wave)

        # waveform lanes (speaker 1 on top) with the VAD as a soft band behind
        self.p_wave.setYRange(0, 3, padding=0.02)
        self.p_wave.getAxis("left").setTicks([[(2.5 - k, f"S{k + 1}") for k in range(3)]])
        self.wave_curves, self.vad_curves = [], []
        for k in range(3):
            base = 2 - k
            c = SPK_COLORS[k]
            vad_lo = pg.PlotCurveItem(self.x_frame, np.full(self.n_frames, base + 0.04), pen=pg.mkPen(None))
            vad_hi = pg.PlotCurveItem(self.x_frame, np.full(self.n_frames, base + 0.04),
                                      pen=pg.mkPen(_rgba(c, 110), width=1))
            self.p_wave.addItem(vad_lo)
            self.p_wave.addItem(vad_hi)
            self.p_wave.addItem(pg.FillBetweenItem(vad_lo, vad_hi, brush=pg.mkBrush(_rgba(c, 34))))
            mid = np.full(self.x_env.size, base + 0.5)
            lo = pg.PlotCurveItem(self.x_env, mid, pen=pg.mkPen(_rgba(c, 200), width=1))
            hi = pg.PlotCurveItem(self.x_env, mid, pen=pg.mkPen(_rgba(c, 200), width=1))
            self.p_wave.addItem(lo)
            self.p_wave.addItem(hi)
            self.p_wave.addItem(pg.FillBetweenItem(lo, hi, brush=pg.mkBrush(_rgba(c, 170))))
            self.wave_curves.append((lo, hi, base + 0.5))
            self.vad_curves.append((vad_hi, base + 0.04))
            if k < 2:
                self.p_wave.addItem(pg.InfiniteLine(pos=base, angle=0, pen=pg.mkPen(GRID)))

        # stacked next-speaker probabilities
        self.stacks = {}
        for name, plot in (("p_now", self.p_pnow), ("p_future", self.p_pfut)):
            plot.setYRange(0, 1, padding=0.02)
            plot.getAxis("left").setTicks([[(0, "0"), (0.5, ".5"), (1, "1")]])
            curves = [pg.PlotCurveItem(self.x_frame, np.zeros(self.n_frames), pen=pg.mkPen(None))]
            plot.addItem(curves[0])
            for k in range(3):
                cv = pg.PlotCurveItem(self.x_frame, np.zeros(self.n_frames), pen=pg.mkPen(PANEL, width=1))
                plot.addItem(cv)
                plot.addItem(pg.FillBetweenItem(curves[-1], cv, brush=pg.mkBrush(_rgba(SPK_COLORS[k], 200))))
                curves.append(cv)
            plot.addItem(pg.InfiniteLine(pos=0.5, angle=0, pen=pg.mkPen(_rgba(TEXT, 40), style=QtCore.Qt.DotLine)))
            self.stacks[name] = curves

        # --- speaker cards
        side = QtWidgets.QVBoxLayout()
        side.setSpacing(16)
        self.cards = []
        for k in range(3):
            card = SpeakerCard(k)
            card.presence_changed.connect(self._on_presence)
            side.addWidget(card)
            self.cards.append(card)
        side.addStretch(1)
        side_w = QtWidgets.QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(290)
        body.addWidget(side_w)

    # ------------------------------------------------------------ playback
    def toggle_play(self):
        if self.finished:
            return
        if not self.started:
            self.started = True
            self.maai.start()
        elif self.playing:
            if hasattr(self.source, "pause"):
                self.source.pause()
        elif hasattr(self.source, "resume"):
            self.source.resume()
        self.playing = not self.playing
        self._update_play_button()
        if not self.playing:
            self.status.setText(self.status.text().split("  ·  ")[0] + "  ·  paused")

    def _update_play_button(self):
        if self.finished:
            text = "Finished"
        elif self.playing:
            text = "❚❚   Pause"
        else:
            text = "▶   Play" if not self.started else "▶   Resume"
        self.play_btn.setText(text)
        self.play_btn.setEnabled(not self.finished)
        self.play_btn.setProperty("playing", self.playing)
        self.play_btn.style().unpolish(self.play_btn)
        self.play_btn.style().polish(self.play_btn)

    # ------------------------------------------------------------ updates
    def _on_presence(self, _idx: int, _present: bool):
        self.maai.set_present([1.0 if c.toggle.isChecked() else 0.0 for c in self.cards])

    def _push(self, r: dict):
        for k in range(3):
            x = np.asarray(r[f"x{k + 1}"], dtype=np.float32)
            x = x[: (x.size // ENV_POINTS) * ENV_POINTS].reshape(ENV_POINTS, -1)
            self.env_min[k].append(x.min(axis=1))
            self.env_max[k].append(x.max(axis=1))
        self.vad.append(np.asarray(r["vad"], dtype=np.float32))
        self.p_now.append(np.asarray(r["p_now"], dtype=np.float32))
        self.p_future.append(np.asarray(r.get("p_future", r["p_now"]), dtype=np.float32))
        now = time.time()
        self.last_latency = now - r["t"]
        if self.t_first is None:
            self.t_first = now
        self.fps_hist.append(now)
        self.frames += 1

    def _tick(self):
        q = self.maai.result_dict_queue
        got = False
        while True:
            try:
                r = q.get_nowait()
            except Exception:
                break
            if self.playing:  # a live microphone keeps running while paused; drop its results
                self._push(r)
                got = True
        if not got:
            if self.playing and getattr(self.source, "finished", False) and q.empty():
                self.finished, self.playing = True, False
                self._update_play_button()
                self.status.setText(self.status.text().split("  ·  ")[0] + "  ·  end of file")
            return
        self._redraw()

    def _redraw(self):
        # waveform envelopes, one common scale so relative loudness is visible
        mins = [np.asarray(self.env_min[k]).ravel() for k in range(3)]
        maxs = [np.asarray(self.env_max[k]).ravel() for k in range(3)]
        peak = max(float(np.max(np.abs(m))) for m in mins + maxs)
        target = 0.42 / max(peak, 0.03)
        self.wave_scale += 0.15 * (target - self.wave_scale)
        for k, (lo, hi, mid) in enumerate(self.wave_curves):
            lo.setData(self.x_env, mid + np.clip(mins[k] * self.wave_scale, -0.46, 0.46))
            hi.setData(self.x_env, mid + np.clip(maxs[k] * self.wave_scale, -0.46, 0.46))

        vad = np.asarray(self.vad)
        for k, (curve, base) in enumerate(self.vad_curves):
            curve.setData(self.x_frame, base + vad[:, k] * 0.92)

        for name, buf in (("p_now", self.p_now), ("p_future", self.p_future)):
            cum = np.cumsum(np.asarray(buf), axis=1)
            for k in range(3):
                self.stacks[name][k + 1].setData(self.x_frame, cum[:, k])

        pn, pf, v = self.p_now[-1], self.p_future[-1], self.vad[-1]
        nxt = int(np.argmax(pn))
        for k, card in enumerate(self.cards):
            card.set_values(float(pn[k]), float(pf[k]), float(v[k]), k == nxt)

        fps = 0.0
        if len(self.fps_hist) > 1:
            fps = (len(self.fps_hist) - 1) / max(self.fps_hist[-1] - self.fps_hist[0], 1e-6)
        self.status.setText(
            f"t {self.frames / FRAME_HZ:6.1f} s   ·   {fps:4.1f} frames/s   ·   "
            f"display latency {self.last_latency * 1000:3.0f} ms"
        )

    def closeEvent(self, event):
        self.timer.stop()
        try:
            self.maai.stop(wait=False)
        except Exception:
            pass
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.stop()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", default=os.path.join(DESKTOP, "J01_city_MaAI.wav"), help="3-channel 16 kHz wav")
    parser.add_argument("--mic", action="store_true", help="use a 3-channel microphone input instead of --wav")
    parser.add_argument("--mic-device-index", type=int, default=-1)
    parser.add_argument("--model", default=None, help="local model file (default: hub)")
    parser.add_argument("--mimi-onnx", default=None, help="local fine-tuned Mimi ONNX (sidecar: <path>.json)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window-sec", type=float, default=10.0, help="seconds shown in the graphs")
    parser.add_argument("--autoplay", action="store_true", help="start playing without pressing Play")
    args = parser.parse_args()

    if args.mic:
        src = MaaiInput.Mic(mic_device_index=args.mic_device_index, channels=3)
        source_name = "3-channel microphone"
    else:
        src = MaaiInput.Wav(wav_file_path=args.wav)
        if src.channels != 3:
            raise ValueError(f"{args.wav} has {src.channels} channels; expected 3")
        source_name = os.path.basename(args.wav)

    maai = MaaiTri(
        audio_ch1=src.get_channel(0),
        audio_ch2=src.get_channel(1),
        audio_ch3=src.get_channel(2),
        lang="jp",
        device=args.device,
        local_model=args.model,
        mimi_local_onnx_fp32_path=args.mimi_onnx,
    )

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    pg.setConfigOptions(antialias=True)

    win = TriVapWindow(maai, src, source_name, args.window_sec)
    win.show()
    if args.autoplay:
        win.toggle_play()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
