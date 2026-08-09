<h1>
<p align="center">
音声区間検出 (VAD) モデル
</p>
</h1>
<p align="center">
README: <a href="vad.md">English </a> | <a href="vad_JP.md">Japanese (日本語) </a>
</p>

`Maai` クラスの `mode` パラメータに `vad` (2チャネル) または `vad_mono` (1チャネル) を指定してください。

ターンテイキングのモデルとは異なり、このモデルは未来を予測しません。**各話者が今まさに発話しているかどうか**を検出します。

このモデルは2話者のチャネルを同時に受け取り、チャネル間のアテンションを用いて処理します。これが通常のチャネルごとの VAD との大きな違いです。一方の話者の声が他方のマイクに回り込んでいる場合(クロストーク)でも、どちらが実際に話しているのかを判定できます。

学習時に雑音・残響 (RIR) の付与を行っているため、雑音に頑健な条件がデフォルトになっています。そのため `vad_mc` という別モードは用意しておらず、`vad` 自体がマルチコンディションのモデルです。

入力は 2 チャネル(`vad_mono` の場合は 1 チャネル)・16kHz の音声データです。

## 出力

`vad` は各入力チャネルの音声活動の確率を表す、[0.0, 1.0] の範囲の2要素の float のリストです:

```python
result["vad"]  # 例: [0.93, 0.02]  -> 話者1は発話中、話者2は発話していない
```

`vad_mono` の場合、`vad` は入力チャネルに対する単一の float 値です。

2値の判定を得るにはしきい値を適用してください。既定値は `0.5` です。学習時の開発データでは `0.54` が最も F1 スコアが高くなりました。

```python
is_speaking = [v >= 0.5 for v in result["vad"]]
```

## 対応言語・フレームレート

| lang | model_type | frame_rate |
| ---- | ---------- | ---------- |
| jp | `normal` (CPC エンコーダ) | 50 |
| jp | `normal-ver2` (Mimi エンコーダ) | 12.5 |

VAP モデルとは異なり、50 Hz のモデルが CPC ベース (`model_type="normal"`)、12.5 Hz のモデルが Mimi ベース (`model_type="normal-ver2"`) である点にご注意ください。

## 使用例

```python
from maai import Maai, MaaiInput, MaaiOutput

mic1 = MaaiInput.Mic(mic_device_index=0)
mic2 = MaaiInput.Mic(mic_device_index=1)

maai = Maai(
    mode="vad",
    lang="jp",
    frame_rate=12.5,
    audio_ch1=mic1,
    audio_ch2=mic2,
    device="cpu",
    model_type="normal-ver2",
    use_mimi_onnx=True,
    mimi_onnx_precision="fp32",
)
maai.start()

while True:
    result = maai.get_result()
    print(result["vad"])  # [float, float]
```

1チャネル版を使う場合は `mode="vad_mono"` を指定し、`audio_ch1` のみを渡してください。このとき `result["vad"]` は単一の float になります。

サンプルスクリプト:
- [マイク2本の入力](../example/vad/vad_2mic.py) 🎤
- [wav ファイル2本の入力](../example/vad/vad_2wav.py) 🎵
- [マイク1本の入力 (mono)](../example/vad/vad_mono_mic.py) 🎤
- [wav ファイル1本の入力 (mono)](../example/vad/vad_mono_wav.py) 🎵
