<h1>
<p align="center">
ノイズロバスト1チャネル音声用ターンテイキング (VAP) モデル (MC-Mono-VAP)
</p>
</h1>
<p align="center">
README: <a href="vap_mc_mono.md">English </a> | <a href="vap_mc_mono_JP.md">Japanese (日本語) </a>
</p>

`Maai` クラスの `mode` パラメータに `vap_mc_mono` を指定してください。

これは [1チャネル VAP モデル (`vap_mono`)](vap_mono_JP.md) のマルチコンディション版です。[ノイズロバスト VAP モデル (`vap_mc`)](vap_mc_JP.md) と同様に、学習データに様々な環境雑音を重畳し、さらに発話音声のゲインもランダムに変更させています。そのため実環境で `vap_mono` より頑健に動作することが期待されます。

学習条件以外は、モデル構造・入力・出力とも [`vap_mono`](vap_mono_JP.md) と同じです。音声を1本だけエンコードし、その話者の将来の音声活動を予測する1チャネル専用のモデルです。
雑音のある環境で片方の話者の音声しか得られないユースケース(例: 公共空間で動作する対話ロボットでのマイク1本の入力)を想定しています。

入力は 1 チャネル・16kHz の音声データです。

## 出力

`p_now` と `p_future` は [0.0, 1.0] の範囲の単一の float 値です(2要素リストではありません):

- `p_now` は入力話者が 0〜600 ミリ秒先に発話している確率を表します。
- `p_future` は 600〜2000 ミリ秒先について同様の確率を表します。

比較対象となる第2話者が存在しないため、2チャネルの `vap` / `vap_mc` モデルのような話者間の正規化は行いません。それぞれの値は、該当区間における入力話者の音声活動の期待値そのものであり、すでに確率になっています。

`vad` も入力チャネルに対する単一の float 値です。

`return_p_bins=True` を指定した場合、`p_bins` は4つのビン(0〜200, 200〜600, 600〜1200, 1200〜2000 ミリ秒)ごとの音声活動確率のリストになり、`p_bins_now` / `p_bins_future` はそれぞれ `p_now` / `p_future` の範囲における平均値になります。

## 対応言語・フレームレート

`vap_mc_mono` は `model_type="normal-ver2"`（Mimi エンコーダ）、コンテキスト長 20 秒（`context_len_sec=20`、デフォルト値）のみ提供しています。

| lang | frame_rate | `vap_mc_mono` |
| ---- | ---------- | ------------- |
| jp | 12.5 | ✅ |
| en | 12.5 | ✅ |
| ch | 12.5 | ✅ |

## 学習データ

[`vap_mono`](vap_mono_JP.md) と同じデータに、環境雑音の重畳とゲインのランダム変更を適用しています。

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
    mode="vap_mc_mono",
    lang="jp",
    frame_rate=12.5,
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

## 📚 論文・参考文献

このモデルを利用した成果を発表する際は、以下の論文を引用してください。🙏

Koji Inoue, Yuki Okafuji, Jun Baba, Yoshiki Ohira, Katsuya Hyodo, Tatsuya Kawahara<br>
__A Noise-Robust Turn-Taking System for Real-World Dialogue Robots: A Field Experiment__<br>
https://www.arxiv.org/abs/2503.06241<br>

```
@misc{inoue2025noisevap,
    author = {Koji Inoue and Yuki Okafuji and Jun Baba and Yoshiki Ohira and Katsuya Hyodo and Tatsuya Kawahara},
    title = {A Noise-Robust Turn-Taking System for Real-World Dialogue Robots: A Field Experiment},
    year = {2025},
    note = {arXiv:2503.06241},
    url = {https://www.arxiv.org/abs/2503.06241},
}
```
