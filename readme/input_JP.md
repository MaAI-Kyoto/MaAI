<h1>
<p align="center">
<b>MaaiInput</b> モジュールの使い方
</p>
</h1>
<p align="center">
README: <a href="input.md">English </a> | <a href="input_JP.md">Japanese (日本語) </a>
</p>

MaaiInput (`input.py`) は MaAI の音声入力を柔軟に扱うためのモジュールです。
WAVファイル、マイク、TCP通信による音声入力をサポートしています。

</br>

## クラス一覧

- `Mic` : マイク入力（リアルタイム）
- `Wav` : WAVファイル入力
- `Tcp` : TCP経由で音声データを受信（サーバ）
- `TcpMic` : TCP経由でマイク入力される音声データを送信（クライアント）
- `Zero` : 無音（ゼロ埋め）データを生成（システム側の音声を一時的に埋めることを想定）
- `Chunk` : チャンク入力
- `TcpChunk` : TCP経由で音声データをチャンク送信（クライアント）
- `BleedCancel` : 対面2者のピンマイク間の回り込み音声をリアルタイムで除去（他の入力をラップして使用）

</br>

## 基本的な使い方

### WAVファイル入力
```python
from maai import MaaiInput

wav1 = MaaiInput.Wav(wav_file_path="user.wav")
wav2 = MaaiInput.Wav(wav_file_path="system.wav")
```

### マイク入力
```python
mic1 = MaaiInput.Mic(mic_device_index=0)
mic2 = MaaiInput.Mic(mic_device_index=1)
```

### TCP入力
```python
tcp_receiver = MaaiInput.Tcp(ip="0.0.0.0", port=12345)
tcp_receiver.start_server()
```

### TCPマイク送信
```python
tcp_mic = MaaiInput.TcpMic(server_ip="送信先IP", port=12345, mic_device_index=0)
tcp_mic.start()
```

### 無音（ゼロ埋め）入力
```python
zero_input = MaaiInput.Zero()
```

### チャンク入力
```python
mic_input = MaaiInput.Chunk()
mic_input.put_chunk(mic_chunk)
```

### TCPチャンク送信

```python
tcp_chunk = MaaiInput.TcpChunk(server_ip="送信先IP", port=12345, mic_device_index=0)
tcp_chunk.start()
```

</br>

## ピンマイク2本の回り込み除去（`BleedCancel`）

2人が対面でピンマイクを使うと、それぞれのマイクに相手の声が回り込んで入ります。
`BleedCancel` はこの回り込みをリアルタイムで除去します。相手のマイクを参照信号とするAEC（線形AEC×2パス＋残留抑圧）を用いるため、
同時発話中も自分の声は保持されます。

```python
from maai import Maai, MaaiInput

# 2本のピンマイクを1台のオーディオインタフェースの2chに接続（L=話者1, R=話者2）
mic = MaaiInput.Mic(mic_device_index=0, channels=2)
bc = MaaiInput.BleedCancel(mic)
# モノラル入力2つでも可: MaaiInput.BleedCancel(MaaiInput.Mic(mic_device_index=0), MaaiInput.Mic(mic_device_index=1))

maai = Maai(mode="vap", lang="jp", audio_ch1=bc.get_channel(0), audio_ch2=bc.get_channel(1), device="cpu")
maai.start()

bc.set_enabled(False)   # 実行中に OFF / ON を切り替え
print(bc.status())      # 適応状態（AECの収束、回り込み遅延、警告）
```

- 任意の MaaiInput と組み合わせ可能: 2chの入力1つ（`Mic(channels=2)`）、またはモノラル入力2つ（`Mic`, `Wav`, `Tcp`, `Chunk` など）
- OFF の間も内部の処理は動き続けるため、フィルタは適応したまま保たれます。OFF 時の出力は同じ遅延だけ遅らせた生の入力なので、切り替えても時間軸はずれません
- 遅延: 約 44 ms（`bc.latency_ms`）
- 起動後、相手だけが話す区間が合計 3 秒ほどたまるまでフィルタが収束せず、その間は回り込みが残ります。マイクの位置を変えたときなどは `bc.reset()` で適応をやり直せます
- 2本のマイクは同じクロックで録る必要があります（同じオーディオインタフェースの2入力を推奨）
- 主なパラメータ（キーワード引数）: `aes_beta`（残留抑圧の強さ、既定 0.7）、`aes=False`（線形AECのみ、歪みが最小）、`margin`（入力系の遅延差の警告が出たら増やす）
- `bc.subscribe_monitor()` で、フレームごとの `raw` / `clean` / `output` を取得できます（可視化・録音用）

### デモ・サンプル（`example/bleed/`）

```bash
# GUI: 両チャネルの波形表示、ON/OFF 切り替え、録音ボタン
python example/bleed/bleed_demo_gui.py --list_devices
python example/bleed/bleed_demo_gui.py --device 2                  # 2chデバイス1つ
python example/bleed/bleed_demo_gui.py --device1 1 --device2 3     # モノラルマイク2本
python example/bleed/bleed_demo_gui.py --wav1 s1.wav --wav2 s2.wav # 16kHzの録音ファイル

# 回り込み除去 + VAP
python example/bleed/vap_2mic_bleed.py
```

GUI の録音では `{timestamp}_raw.wav`（入力）、`{timestamp}_clean.wav`（処理後）、`{timestamp}_output.wav`（実際の出力、ON/OFF に従う）を保存します（いずれも 16kHz・2ch・時刻同期済み）。

### オフライン処理（録音済みファイル）

ピンマイクで録音済みの音声に対して、後から回り込み除去を適用できます。オフライン版はファイル全体から遅延・クロックずれを推定するため、リアルタイム版よりやや高性能です。入力のサンプリングレートは任意で、出力は 16kHz です。

```bash
python example/bleed/bleed_offline.py --s1 speaker1.wav --s2 speaker2.wav --out_dir out/
python example/bleed/bleed_offline.py --stereo session.wav --out_dir out/   # L=話者1, R=話者2
```

```python
from maai.bleed import process_files, process_pair

process_files(s1="speaker1.wav", s2="speaker2.wav", out_dir="out")   # *_s1.wav, *_s2.wav, *_stereo.wav を出力
a_clean, b_clean, log = process_pair(a, b, sr_in=44100)              # numpy 配列（出力は 16kHz）
```

</br>

## TCP通信による音声データフォーマット

各自のオリジナルのシステムと接続する際には `TCPReceiver` クラスのインスタンスに対して、音声データを送信するとよいでしょう。
その際のデータのフォーマットは下記となります。

- サンプリングレート: 16,000 Hz
- 1フレーム: 160サンプル
- 各サンプル: 8バイトの倍精度浮動小数点（double, little endian, -1.0～+1.0）
- 1フレームのデータサイズ: 1,280バイト

### データパケット例

| バイトオフセット | データ型 | 説明 |
| ---- | ---- | --- |
| 0 - 7 | Double | 音声データ - サンプル 1 |
| 8 - 15 | Double | 音声データ - サンプル 2 |
| ... | ... | ... |
| 1264 - 1271 | Double | 音声データ - サンプル 159 |
| 1272 - 1279 | Double | 音声データ - サンプル 160 |

</br>

## デバイス一覧取得

マイクデバイスの一覧は `available_mic_devices()` で取得できます。

```python
from maai import MaaiInput

MaaiInput.available_mic_devices()
```

## 注意事項

- WAVファイルは16,000Hzのみ対応
- TCP通信は2人分の音声データを1フレームで送受信
- 各クラスは `start()` で非同期処理を開始

---

詳細は [README.md](../README_JP.md) も参照してください。
