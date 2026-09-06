<h1>
<p align="center">
1チャネル音声用ターンテイキング (VAP) モデル (Mono-VAP)
</p>
</h1>
<p align="center">
README: <a href="vap_mono.md">English </a> | <a href="vap_mono_JP.md">Japanese (日本語) </a>
</p>

`Maai` クラスの `mode` パラメータに `vap_mono` を指定してください。

このモデルは、標準の2話者 VAP モデルに無音チャネルを与えたものではなく、**1チャネル専用に学習された独立のモデル**です。音声を1本だけエンコードし、標準モデルのチャネル間 Transformer を通常の因果 Transformer に置き換えて、その話者の将来の音声活動を直接予測します。
片方の話者の音声しか得られないユースケース(例: 音声対話システムでのマイク1本の入力)を想定しています。

入力は 1 チャネル・16kHz の音声データです。

## 出力

`p_now` と `p_future` は [0.0, 1.0] の範囲の単一の float 値です(2要素リストではありません):

- `p_now` は入力話者が 0〜600 ミリ秒先に発話している確率を表します。
- `p_future` は 600〜2000 ミリ秒先について同様の確率を表します。

比較対象となる第2話者が存在しないため、標準の `vap` モデルのような話者間の正規化は行いません。それぞれの値は、該当区間における入力話者の音声活動の期待値そのものであり、すでに確率になっています。

`vad` も入力チャネルに対する単一の float 値です。

`return_p_bins=True` を指定した場合、`p_bins` は4つのビン(0〜200, 200〜600, 600〜1200, 1200〜2000 ミリ秒)ごとの音声活動確率のリストになり、`p_bins_now` / `p_bins_future` はそれぞれ `p_now` / `p_future` の範囲における平均値になります。

## 対応言語・フレームレート・コンテキスト長

| lang | model_type | frame_rate | context_len_sec | `vap_mono` |
| ---- | ---------- | ---------- | --------------- | ---------- |
| jp | `normal` (CPC エンコーダ) | 50 | 20 | ✅ |
| jp | `normal-ver2` (Mimi エンコーダ) | 12.5 | 20 | ✅ |
| en | `normal` (CPC エンコーダ) | 50 | 20 | 準備中 |
| en | `normal-ver2` (Mimi エンコーダ) | 12.5 | 20 | 準備中 |
| ch | `normal` (CPC エンコーダ) | 50 | 20 | 準備中 |
| ch | `normal-ver2` (Mimi エンコーダ) | 12.5 | 20 | ✅ |

現在公開しているのは日本語・中国語の Mimi エンコーダのモデルです。その他の組み合わせは順次公開予定です。

50 Hz のモデルが CPC ベース (`model_type="normal"`)、12.5 Hz のモデルが Mimi ベース (`model_type="normal-ver2"`) である点にご注意ください。標準の [VAP モデル](vap_JP.md) とは異なり、`vap_mono` はエンコーダごとにフレームレートが1種類、コンテキスト長は 20 秒固定です。

## 学習データ

[VAD モデル](vad_JP.md) と同一のデータで学習しています。

| lang | 学習データ |
| ---- | ---------- |
| jp | [旅行代理店タスク対話](https://aclanthology.org/2022.lrec-1.619/)、[人間ロボット対話](https://aclanthology.org/2025.naacl-long.367/)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) |
| en | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62)、[Seamless Interaction](https://ai.meta.com/research/seamless-interaction/)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) |
| ch | [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) |

## 使用例

```python
from maai import Maai, MaaiInput, MaaiOutput

mic = MaaiInput.Mic()

maai = Maai(
    mode="vap_mono",
    lang="jp",
    frame_rate=12.5,
    context_len_sec=20,
    audio_ch1=mic,   # audio_ch2 は不要
    device="cpu",
    model_type="normal-ver2",
    use_mimi_onnx=True,
    mimi_onnx_precision="fp32",
)
maai.start()

while True:
    result = maai.get_result()
    print(result["p_now"], result["p_future"], result["vad"])  # すべて単一の float
```

サンプルスクリプト:
- [マイク1本の入力](../example/vap_mono/vap_mono_mic.py) 🎤
- [wav ファイル1本の入力](../example/vap_mono/vap_mono_wav.py) 🎵
