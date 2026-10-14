<h1>
<p align="center">
How to use the <b>MaaiInput</b> module
</p>
</h1>
<p align="center">
README: <a href="input.md">English </a> | <a href="input_JP.md">Japanese (日本語) </a>
</p>

MaaiInput (`input.py`) is a module for flexible handling of audio input in MaAI.
It supports audio input via WAV files, microphone, and TCP communication.

</br>

## Class List

- `Mic`: Microphone input (real-time)
- `Wav`: WAV file input
- `Tcp`: Receive audio data via TCP as a server
- `TcpMic`: Transmit microphone audio data via TCP as a client
- `Zero`: Generates silent (zero-filled) audio data (intended for temporarily filling system-side audio)
- `Chunk`: Chunk input
- `TcpChunk`: Transmit audio data via TCP in chunks as a client
- `BleedCancel`: Removes the bleed (cross-talk) between two face-to-face lapel mics in real time (wraps other inputs)

</br>

## Basic Usage

### WAV File Input
```python
from maai import MaaiInput

wav1 = MaaiInput.Wav(wav_file_path="user.wav")
wav2 = MaaiInput.Wav(wav_file_path="system.wav")
```

### Microphone Input
```python
mic1 = MaaiInput.Mic(mic_device_index=0)
mic2 = MaaiInput.Mic(mic_device_index=1)
```

### TCP Input (Server)
```python
tcp_receiver = MaaiInput.Tcp(ip="0.0.0.0", port=12345)
tcp_receiver.start_server()
```

### TCP Mic Transmission (Client)
```python
tcp_mic = MaaiInput.TcpMic(server_ip="Destination IP", port=12345, mic_device_index=0)
tcp_mic.start()
```

### Silent (Zero-filled) Input
```python
zero_input = MaaiInput.Zero()
```

### Chunk Input
```python
mic_input = MaaiInput.Chunk()
mic_input.put_chunk(mic_chunk)
```

### TCP Chunk Transmission (Client)
```python
tcp_chunk = MaaiInput.TcpChunk(server_ip="Destination IP", port=12345, mic_device_index=0)
tcp_chunk.start()
```

</br>

## Bleed Removal for Two Lapel Mics (`BleedCancel`)

When two people talk face to face with lapel (pin) mics, each mic also picks up the partner's voice.
`BleedCancel` removes this bleed in real time. It uses the partner's mic as the reference
(2-pass linear AEC + residual suppression), so the own voice is kept even during overlapping speech.

```python
from maai import Maai, MaaiInput

# Two lapel mics on one 2ch audio interface (L = speaker 1, R = speaker 2)
mic = MaaiInput.Mic(mic_device_index=0, channels=2)
bc = MaaiInput.BleedCancel(mic)
# or two mono sources: MaaiInput.BleedCancel(MaaiInput.Mic(mic_device_index=0), MaaiInput.Mic(mic_device_index=1))

maai = Maai(mode="vap", lang="jp", audio_ch1=bc.get_channel(0), audio_ch2=bc.get_channel(1), device="cpu")
maai.start()

bc.set_enabled(False)   # switch OFF / ON at runtime
print(bc.status())      # adaptation state (AEC convergence, bleed delay, warnings)
```

- Any MaaiInput source works: one 2ch source (`Mic(channels=2)`), or two mono sources (`Mic`, `Wav`, `Tcp`, `Chunk`, ...)
- Processing keeps running while disabled, so the filters stay adapted; the disabled output is the raw input delayed by the same latency (toggling does not shift the timeline)
- Latency: about 44 ms (`bc.latency_ms`)
- The filters converge after about 3 s (in total) of speech where only the partner talks; bleed remains until then. `bc.reset()` restarts the adaptation (e.g. after moving the mics)
- Both mics should share the same clock (two inputs of one audio interface work best)
- Main parameters (keyword arguments): `aes_beta` (residual suppression strength, default 0.7), `aes=False` (linear AEC only, least distortion), `margin` (increase it if the input-path delay warning appears)
- `bc.subscribe_monitor()` gives per-frame `raw` / `clean` / `output` arrays for visualization and recording

### Demo and examples (`example/bleed/`)

```bash
# GUI: waveforms of both channels, ON/OFF toggle, recording button
python example/bleed/bleed_demo_gui.py --list_devices
python example/bleed/bleed_demo_gui.py --device 2                  # one 2ch device
python example/bleed/bleed_demo_gui.py --device1 1 --device2 3     # two mono mics
python example/bleed/bleed_demo_gui.py --wav1 s1.wav --wav2 s2.wav # 16 kHz recordings

# VAP with bleed removal
python example/bleed/vap_2mic_bleed.py
```

The GUI saves `{timestamp}_raw.wav` (input), `{timestamp}_clean.wav` (processed) and `{timestamp}_output.wav` (what was output, follows the toggle), all 16 kHz 2ch and time-aligned.

### Offline (recorded files)

The bleed can also be removed from recordings afterwards. The offline version uses the whole file for the delay / clock-drift estimate and is slightly better than the real-time one. Any input sample rate is accepted; the output is 16 kHz.

```bash
python example/bleed/bleed_offline.py --s1 speaker1.wav --s2 speaker2.wav --out_dir out/
python example/bleed/bleed_offline.py --stereo session.wav --out_dir out/   # L = speaker 1, R = speaker 2
```

```python
from maai.bleed import process_files, process_pair

process_files(s1="speaker1.wav", s2="speaker2.wav", out_dir="out")   # writes *_s1.wav, *_s2.wav, *_stereo.wav
a_clean, b_clean, log = process_pair(a, b, sr_in=44100)              # numpy arrays (output: 16 kHz)
```

</br>

## TCP Audio Data Format

When connecting with your own original system, send audio data to an instance of the `TCPReceiver` class. The data format is as follows:

- Sampling rate: 16,000 Hz
- 1 frame: 160 samples
- Each sample: 8-byte double-precision floating point (double, little endian, -1.0 to +1.0)
- 1 frame data size: 1,280 bytes

### Data Packet Example

| Byte Offset | Data Type | Description |
| ---- | ---- | --- |
| 0 - 7 | Double | Audio Data - Sample 1 |
| 8 - 15 | Double | Audio Data - Sample 2 |
| ... | ... | ... |
| 1264 - 1271 | Double | Audio Data - Sample 159 |
| 1272 - 1279 | Double | Audio Data - Sample 160 |

</br>

## Get Device List

You can get a list of microphone devices with `available_mic_devices()`.

```python
from maai import MaaiInput

MaaiInput.available_mic_devices()
```

</br>

## Notes

- Only 16,000Hz WAV files are supported
- TCP communication sends/receives audio data for two people per frame
- Each class starts asynchronous processing with `start()`

---

For more details, please refer to [README.md](../README.md).
