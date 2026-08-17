<h1>
<p align="center">
相槌検出 (BC-Det) モデル
</p>
</h1>
<p align="center">
README: <a href="bc_det.md">English </a> | <a href="bc_det_JP.md">Japanese (日本語) </a>
</p>

`Maai` クラスの `mode` パラメータに `bc_det` (2チャネル) または `bc_det_mono` (1チャネル) を指定してください。

このモデルは、**話者が今まさに発話している内容が相槌かどうか**を検出します (例:「うん」「はい」「なるほど」)。音声と正解ラベルの間に時間的なずれを設けていない、純粋な*検出*モデルです。

> **`bc_det` と `bc` の違い** — 名前は似ていますが解いている課題が異なります。
> - [`bc`](vap_bc_JP.md) は相槌が*これから*起きることを**予測**します (正解ラベルを約 0.5 秒前にずらして学習)。システムが*いつ相槌を打つべきか*を決めるために使います。
> - `bc_det` は相槌が*今まさに*起きていることを**検出**します。ユーザの短い発話が、ターンの開始ではなく相槌であったと認識するために使います。

このモデルは2話者のチャネルを同時に受け取り、チャネル間のアテンションを用いて処理します。両方のチャネルが重要です。短い発話がターンの開始ではなく相槌であると判断するための手がかりの多くは、対話相手の発話にあるためです。

入力は 2 チャネル(`bc_det_mono` の場合は 1 チャネル)・16kHz の音声データです。

## 出力

`p_bc_det` は各入力チャネルの相槌の確率を表す、[0.0, 1.0] の範囲の2要素の float のリストです:

```python
result["p_bc_det"]  # 例: [0.71, 0.02]  -> 話者1は相槌中、話者2は相槌していない
```

`bc_det_mono` の場合、`p_bc_det` は入力チャネルに対する単一の float 値です。

2値の判定を得るにはしきい値を適用してください。ただし、**このタスクでは `0.5` が最適とは限りません**。相槌はフレーム全体の 4% 程度しか占めず、学習データが大きく偏っているためです。日本語の開発データでチューニングした結果、フレーム単位の F1 では約 `0.39`、イベント単位の F1 では約 `0.45` が最適でした。

```python
is_backchannel = [v >= 0.45 for v in result["p_bc_det"]]
```

相槌は短く、継続長の中央値は 0.25 秒程度です。そのため出力の平滑化 (短い無音の穴埋めや短いスパイクの除去) は、境界を整えるよりも実際のイベントを消してしまう傾向があります。既定では無効であり、基本的に推奨されません。

## 対応言語・フレームレート

| lang | model_type | frame_rate |
| ---- | ---------- | ---------- |
| jp | `normal-ver2` (Mimi エンコーダ) | 12.5 |
| en | `normal-ver2` (Mimi エンコーダ) | 12.5 |
| ch | `normal-ver2` (Mimi エンコーダ) | 12.5 |

現時点でこのモードは 12.5 Hz の Mimi ベースのモデル (`model_type="normal-ver2"`) のみを公開しています。

学習時に雑音・残響 (RIR) の付与を行っているため、雑音に頑健な条件がデフォルトになっています。そのため `bc_det_mc` という別モードは用意しておらず、`bc_det` 自体がマルチコンディションのモデルです。

## 使用例

```python
from maai import Maai, MaaiInput, MaaiOutput

mic1 = MaaiInput.Mic(mic_device_index=0)
mic2 = MaaiInput.Mic(mic_device_index=1)

maai = Maai(
    mode="bc_det",
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
    print(result["p_bc_det"])  # [float, float]
```

1チャネル版を使う場合は `mode="bc_det_mono"` を指定し、`audio_ch1` のみを渡してください。このとき `result["p_bc_det"]` は単一の float になります。なお、このモデルは対話相手の発話を手がかりにしているため、1チャネル版は2チャネル版よりも精度が下がります。両方のチャネルが利用できる場合は `bc_det` を使用してください。

サンプルスクリプト:
- [マイク2本の入力](../example/bc_det/bc_det_2mic.py) 🎤
- [wav ファイル2本の入力](../example/bc_det/bc_det_2wav.py) 🎵
- [マイク1本の入力 (mono)](../example/bc_det/bc_det_mono_mic.py) 🎤
- [wav ファイル1本の入力 (mono)](../example/bc_det/bc_det_mono_wav.py) 🎵
