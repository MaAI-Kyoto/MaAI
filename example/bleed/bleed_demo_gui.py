"""
GUI demo of the speaker-bleed removal (MaaiInput.BleedCancel) for two face-to-face lapel mics.

- Scrolling waveforms of both channels (gray: raw input, color: output)
- Toggle bleed removal ON/OFF while watching the waveforms
- Record button: saves {timestamp}_raw.wav (input), {timestamp}_clean.wav (processed, regardless
  of the toggle) and {timestamp}_output.wav (what was output, follows the toggle) at 16 kHz, 2ch
  (L = speaker 1, R = speaker 2). All three files are time-aligned.

Usage:
  python bleed_demo_gui.py --list_devices
  python bleed_demo_gui.py --device 2                     # one 2ch device (L = speaker 1, R = speaker 2)
  python bleed_demo_gui.py --device1 1 --device2 3        # two mono mics
  python bleed_demo_gui.py --wav1 s1.wav --wav2 s2.wav    # recorded 16 kHz mono WAVs (no mics needed)

Note: the filters converge after about 3 s (in total) of speech where only the partner talks.
Both mics should share the same clock (two inputs of one audio interface work best).
"""

import argparse
import html
import os
import queue
import sys
import time

import numpy as np
import soundfile as sf

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# This allows you to import modules from the src directory without pip installing the package.
# Uncomment the line below if you need to run this script directly without installing the package.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

# PyQt5 must be imported before pyqtgraph: otherwise pyqtgraph picks PyQt6 (if installed), and mixing
# it with the PyQt5 QApplication crashes silently when the window is created
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from maai import MaaiInput

SR = MaaiInput.Base.SAMPLING_RATE
COLORS = [(31, 119, 180), (214, 39, 40)]


def build_input(args):
    """Create the MaaiInput source(s) and wrap them with BleedCancel."""
    params = {}
    if args.no_aes:
        params["aes"] = False
    if args.aes_beta is not None:
        params["aes_beta"] = args.aes_beta

    def mic(dev, channels):
        if dev is not None and not str(dev).isdigit():
            return MaaiInput.Mic(device_name=dev, channels=channels)
        return MaaiInput.Mic(mic_device_index=int(dev) if dev is not None else -1, channels=channels)

    if args.wav1 or args.wav2:
        if not (args.wav1 and args.wav2):
            sys.exit("Specify both --wav1 and --wav2")
        return MaaiInput.BleedCancel(MaaiInput.Wav(wav_file_path=args.wav1),
                                     MaaiInput.Wav(wav_file_path=args.wav2), **params)
    if args.device1 is not None or args.device2 is not None:
        if args.device1 is None or args.device2 is None:
            sys.exit("Specify both --device1 and --device2")
        return MaaiInput.BleedCancel(mic(args.device1, 1), mic(args.device2, 1), **params)
    return MaaiInput.BleedCancel(mic(args.device, 2), **params)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list_devices", action="store_true", help="list microphone devices and exit")
    ap.add_argument("--device", default=None, help="2ch input device (index or part of the name)")
    ap.add_argument("--device1", default=None, help="mono input device of speaker 1")
    ap.add_argument("--device2", default=None, help="mono input device of speaker 2")
    ap.add_argument("--wav1", default=None, help="16 kHz mono WAV of speaker 1")
    ap.add_argument("--wav2", default=None, help="16 kHz mono WAV of speaker 2")
    ap.add_argument("--out_dir", default="rec", help="directory for recordings")
    ap.add_argument("--window_sec", type=float, default=5.0, help="length of the waveform display")
    ap.add_argument("--ylim", type=float, default=0.5, help="initial y-axis range (+/-)")
    ap.add_argument("--off", action="store_true", help="start with bleed removal OFF")
    ap.add_argument("--no_aes", action="store_true", help="linear AEC only (no residual suppression)")
    ap.add_argument("--aes_beta", type=float, default=None, help="residual suppression strength (default 0.7)")
    args = ap.parse_args(argv)

    if args.list_devices:
        MaaiInput.available_mic_devices()
        return

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    bc = build_input(args)
    bc.set_enabled(not args.off)
    win = BleedDemoWindow(bc, args)
    win.show()
    bc.start()
    app.exec_()


def _db(x):
    return 10 * np.log10(np.mean(np.square(x)) + 1e-12)


class BleedDemoWindow(QtWidgets.QWidget):
    def __init__(self, bc, args):
        super().__init__()
        self.bc = bc
        self.args = args
        self.mon_q = bc.subscribe_monitor()
        n = int(args.window_sec * SR)
        self.buf_raw = np.zeros((n, 2), np.float32)
        self.buf_out = np.zeros((n, 2), np.float32)
        self.t_axis = (np.arange(n) - n) / SR
        self.writers = None
        self.rec_start = None
        self.rec_samples = 0
        self.last_saved = ""

        self.setWindowTitle("MaAI - Bleed Removal Demo")
        self.resize(1100, 700)
        layout = QtWidgets.QVBoxLayout(self)

        # --- controls ---
        ctrl = QtWidgets.QHBoxLayout()
        self.btn_toggle = QtWidgets.QPushButton()
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(bc.enabled)
        self.btn_toggle.setMinimumHeight(40)
        self.btn_toggle.toggled.connect(self.on_toggle)
        self.btn_rec = QtWidgets.QPushButton()
        self.btn_rec.setMinimumHeight(40)
        self.btn_rec.clicked.connect(self.on_record)
        self.btn_reset = QtWidgets.QPushButton("Reset filters")
        self.btn_reset.setMinimumHeight(40)
        self.btn_reset.clicked.connect(self.on_reset)
        self.chk_raw = QtWidgets.QCheckBox("Show raw input (gray)")
        self.chk_raw.setChecked(True)
        self.chk_raw.toggled.connect(self.on_show_raw)
        for w in (self.btn_toggle, self.btn_rec, self.btn_reset):
            ctrl.addWidget(w)
        ctrl.addWidget(self.chk_raw)
        ctrl.addStretch(1)
        layout.addLayout(ctrl)

        # --- waveforms ---
        pg.setConfigOptions(antialias=False)
        self.plots, self.curves_raw, self.curves_out = [], [], []
        for c in range(2):
            pw = pg.PlotWidget()
            pw.setBackground("w")
            pw.setTitle(f"ch{c + 1} (speaker {c + 1})", color=COLORS[c])
            pw.setLabel("bottom", "time [s]")
            pw.setYRange(-args.ylim, args.ylim)
            pw.setXRange(-args.window_sec, 0, padding=0)
            pw.showGrid(x=True, y=True, alpha=0.3)
            pw.setDownsampling(auto=True, mode="peak")
            pw.setClipToView(True)
            pw.addLegend(offset=(10, 5))
            self.curves_raw.append(pw.plot(pen=pg.mkPen((160, 160, 160), width=1), name="raw input"))
            self.curves_out.append(pw.plot(pen=pg.mkPen(COLORS[c], width=1), name="output"))
            if self.plots:
                pw.setXLink(self.plots[0])
            self.plots.append(pw)
            layout.addWidget(pw, 1)

        self.lbl_level = QtWidgets.QLabel()
        self.lbl_status = QtWidgets.QLabel()
        self.lbl_status.setWordWrap(True)
        for lbl in (self.lbl_level, self.lbl_status):
            lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            layout.addWidget(lbl)

        self._update_buttons()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(33)
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(500)

    # --- UI callbacks ---
    def on_toggle(self, checked):
        self.bc.set_enabled(checked)
        self._update_buttons()

    def on_reset(self):
        self.bc.reset()

    def on_show_raw(self, checked):
        for cv in self.curves_raw:
            cv.setVisible(checked)

    def on_record(self):
        if self.writers is None:
            os.makedirs(self.args.out_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.writers = {k: sf.SoundFile(os.path.join(self.args.out_dir, f"{ts}_{k}.wav"), "w", SR, 2, "PCM_16")
                            for k in ("raw", "clean", "output")}
            self.rec_start = time.time()
            self.rec_samples = 0
        else:
            self._stop_recording()
        self._update_buttons()

    def _stop_recording(self):
        if self.writers is None:
            return
        names = [w.name for w in self.writers.values()]
        for w in self.writers.values():
            w.close()
        self.writers = None
        self.last_saved = f"Saved {self.rec_samples / SR:.1f} s: " + ", ".join(names)
        print(self.last_saved)

    def _update_buttons(self):
        # A border in the style sheet makes Qt draw the button itself (the native style would
        # otherwise override the background of a checked button)
        style = "QPushButton {{ background-color: {}; color: {}; font-weight: bold; padding: 0 16px; " \
                "border: 1px solid #757575; border-radius: 4px; }}"
        on = self.bc.enabled
        self.btn_toggle.setText("Bleed removal: ON" if on else "Bleed removal: OFF")
        self.btn_toggle.setStyleSheet(style.format("#2e7d32" if on else "#9e9e9e", "white"))
        rec = self.writers is not None
        self.btn_rec.setText("■ Stop recording" if rec else "● Record")
        self.btn_rec.setStyleSheet(style.format("#c62828", "white") if rec else style.format("#fafafa", "#c62828"))
        self.btn_reset.setStyleSheet(style.format("#fafafa", "black"))

    # --- periodic updates ---
    def update_plot(self):
        frames = []
        while True:
            try:
                frames.append(self.mon_q.get_nowait())
            except queue.Empty:
                break
        if not frames:
            return
        raw = np.concatenate([f["raw"] for f in frames]).astype(np.float32)
        out = np.concatenate([f["output"] for f in frames]).astype(np.float32)
        if self.writers is not None:
            clean = np.concatenate([f["clean"] for f in frames])
            self.writers["raw"].write(np.clip(raw, -1, 1))
            self.writers["clean"].write(np.clip(clean, -1, 1))
            self.writers["output"].write(np.clip(out, -1, 1))
            self.rec_samples += len(raw)
        n = len(self.buf_raw)
        m = min(len(raw), n)
        self.buf_raw = np.concatenate([self.buf_raw[m:], raw[-m:]])
        self.buf_out = np.concatenate([self.buf_out[m:], out[-m:]])
        for c in range(2):
            self.curves_raw[c].setData(self.t_axis, self.buf_raw[:, c])
            self.curves_out[c].setData(self.t_axis, self.buf_out[:, c])

    def update_status(self):
        k = SR // 2  # last 0.5 s
        lv = [f"ch{c + 1}: raw {_db(self.buf_raw[-k:, c]):6.1f} dB / output {_db(self.buf_out[-k:, c]):6.1f} dB"
              for c in range(2)]
        self.lbl_level.setText("Level (last 0.5 s)   " + "    ".join(lv))

        st = self.bc.status()
        fmt = lambda d: "-" if d is None else f"{1000 * d / SR:.1f} ms"
        adapted = ["yes" if a["adapted"] else "no" for a in st["aec"][-1]]
        parts = [f"Latency {self.bc.latency_ms:.0f} ms",
                 f"AEC converged ch1/ch2: {adapted[0]}/{adapted[1]}",
                 f"Bleed delay 2->1 {fmt(st['delay_b_to_a'])}, 1->2 {fmt(st['delay_a_to_b'])}"]
        if "aes_gain" in st:
            parts.append(f"AES gain {st['aes_gain'][0]:.2f}/{st['aes_gain'][1]:.2f}")
        if self.writers is not None:
            parts.append(f"<b style='color:#c62828'>REC {time.time() - self.rec_start:.0f} s</b>")
        elif self.last_saved:
            parts.append(html.escape(self.last_saved))
        text = " | ".join(parts)
        for w in st["warnings"]:
            text += f"<br><b style='color:#e65100'>WARNING: {html.escape(w)}</b>"
        self.lbl_status.setText(text)

    def closeEvent(self, event):
        self._stop_recording()
        super().closeEvent(event)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
