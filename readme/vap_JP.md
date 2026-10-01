<h1>
<p align="center">
ターンテイキング (VAP) モデル
</p>
</h1>
<p align="center">
README: <a href="vap.md">English </a> | <a href="vap_JP.md">Japanese (日本語) </a>
</p>

`Maai` クラスの `mode` パラメータに `vap` を指定してください。

このモデルは**話者の近い未来の音声活動**を予測します。これが音声対話システムにおけるターンテイキング判断の基礎になります。

## モデルの選択

`mode="vap"` には4種類のモデルがあり、`mc` パラメータと音声入力の与え方から自動的に選択されます。

| | `mc=True`（デフォルト） | `mc=False` |
| --- | --- | --- |
| `audio_ch1` + `audio_ch2`（2チャネル） | ノイズロバスト2チャネルモデル (`vap_mc`) | 通常の2チャネルモデル (`vap`) |
| `audio_ch1` のみ、または `audio_ch2=MaaiInput.Zero()`（1チャネル） | ノイズロバスト1チャネルモデル (`vap_mc_mono`) | 通常の1チャネルモデル (`vap_mono`) |

- **ノイズロバスト（マルチコンディション）モデル**（`mc=True`）は、学習データに様々な環境雑音を重畳し、さらに発話音声のゲインもランダムに変更させて学習しています。そのため実環境で通常のモデルより頑健に動作することが期待されます。指定した `lang` / `frame_rate` / `model_type` にノイズロバストモデルが存在しない場合は、通常のモデルが使われます。
- **1チャネルモデル**は、2チャネルモデルに無音チャネルを与えたものではなく、**1チャネル専用に学習された独立のモデル**です。音声を1本だけエンコードし、チャネル間 Transformer を通常の因果 Transformer に置き換えて、その話者の将来の音声活動を直接予測します。片方の話者の音声しか得られないユースケース（例: 音声対話システムでのマイク1本の入力）を想定しています。指定した組み合わせに1チャネルモデルが存在しない場合は、第2チャネルを無音として2チャネルモデルが使われ、警告が表示されます。

実際に読み込まれたモデルは `maai.variant`（例: `"vap_mc"`, `"vap_mc_mono"`）で確認できます。

> **注意:** これまで `mode="vap"` は常に通常のモデルを使用していました。現在は `mc=True` がデフォルトのため、従来と同じ動作にするには `mc=False` を指定してください。旧 mode 名 `vap_mc` / `vap_mono` / `vap_mc_mono` も引き続き動作しますが、非推奨です。

入力は 16kHz の音声データ（2チャネルまたは1チャネル）です。

## 出力

### 2チャネルモデル

`p_now` と `p_future` は、各話者が該当の時間範囲で発話権を持つ確率を表す、[0.0, 1.0] の範囲の2要素の float のリストです。2つの値は話者間で正規化されており、合計が 1.0 になります。

- `p_now` は次の 0〜600 ミリ秒を対象とします。
- `p_future` は 600〜2000 ミリ秒先を対象とします。

一般的なターンテイキング用途では `p_now` の利用を推奨します。

```python
result["p_now"]     # 例: [0.87, 0.13]  -> 話者1が次の話者になる可能性が高い
result["p_future"]  # 例: [0.62, 0.38]
```

`vad` は各入力チャネルの現フレームにおける音声活動の確率を表す、2要素の float のリストです（[VAD モデル](vad_JP.md)と同じ量を VAP モデル内部で計算したものです）。

`return_p_bins=True` を指定すると、`p_bins` として4つのビン (0〜200, 200〜600, 600〜1200, 1200〜2000 ミリ秒) ごと・話者ごとの活動確率が得られ、`p_bins_now` / `p_bins_future` はそれぞれ `p_now` / `p_future` の範囲での平均です。`p_now` や `p_future` とは異なり、これらは話者間で正規化されていません。

### 1チャネルモデル

`p_now` と `p_future` は [0.0, 1.0] の範囲の単一の float 値です（2要素リストではありません）:

- `p_now` は入力話者が 0〜600 ミリ秒先に発話している確率を表します。
- `p_future` は 600〜2000 ミリ秒先について同様の確率を表します。

比較対象となる第2話者が存在しないため、話者間の正規化は行いません。それぞれの値は、該当区間における入力話者の音声活動の期待値そのものであり、すでに確率になっています。

`vad` も入力チャネルに対する単一の float 値です。

`return_p_bins=True` を指定した場合、`p_bins` は4つのビン（0〜200, 200〜600, 600〜1200, 1200〜2000 ミリ秒）ごとの音声活動確率のリストになり、`p_bins_now` / `p_bins_future` はそれぞれ `p_now` / `p_future` の範囲における平均値になります。

## 対応言語・フレームレート

`Maai` クラスの `lang` パラメータで言語を指定してください。

`model_type` はモデル種別を指定します。`"normal-ver2"` は Mimi をエンコーダとして使用する新しいモデル、`"normal"` はこれまでのリリースで使っていた既存モデル（CPC エンコーダ）です。

`frame_rate` は VAP モデルが1秒あたりに処理するサンプル数を指定します。ご利用の計算環境に合わせて調整してください。

各セルは利用可能な `frame_rate` を表します。「–」はそのモデルが存在しないことを表し、その場合は[モデルの選択](#モデルの選択)で説明したフォールバックが適用されます。

### `model_type="normal-ver2"`（Mimi エンコーダ）

1チャネルモデルは 20 秒のコンテキスト（`context_len_sec=20`、デフォルト）でのみ利用できます。

| lang | 2ch (`vap`) | 2ch・mc (`vap_mc`) | 1ch (`vap_mono`) | 1ch・mc (`vap_mc_mono`) |
| ---- | ---- | ---- | ---- | ---- |
| jp | 12.5 | 12.5 | 12.5 | 12.5 |
| jp_kyoto | 12.5 | 12.5 | – | – |
| en | 12.5 | 12.5 | 12.5 | 12.5 |
| en_kyoto | 12.5 | 12.5 | – | – |
| ch | 12.5 | 12.5 | 12.5 | 12.5 |
| ch_kyoto | 準備中 | 12.5 | – | – |
| tri | 12.5 | 12.5 | – | – |
| tri_kyoto | 12.5 | 12.5 | – | – |

### `model_type="normal"`（CPC エンコーダ）

| lang | 2ch (`vap`) | 2ch・mc (`vap_mc`) | 1ch (`vap_mono`) | 1ch・mc (`vap_mc_mono`) |
| ---- | ---- | ---- | ---- | ---- |
| jp | 5, 10, 20 | 5, 10, 20 | 10, 20, 50 | – |
| jp_kyoto | 5, 10, 20 | 5, 10, 20 | – | – |
| en | 5, 10, 20 | 5, 10, 20 | 10, 20, 50 | – |
| en_kyoto | 5, 10 | 5, 10 | – | – |
| ch | 5, 10, 20 | 5, 10, 20 | 10, 20, 50 | – |
| ch_kyoto | 5, 10 | 5, 10 | – | – |
| tri | 5, 10 | 5, 10 | – | – |
| tri_kyoto | 5, 10 | 5, 10 | – | – |

## 学習データ

`tri` は3言語対応（日本語＋英語＋中国語）のモデルです。`*_kyoto` のモデルはオンライン会話データセットのみで学習されており、MIT ライセンスで公開されています。

### 2チャネルモデル

| lang | 学習データ | ライセンス |
| ---- | ---------- | ---------- |
| jp | [旅行代理店タスク対話コーパス](https://aclanthology.org/2022.lrec-1.619/)、[ヒューマンロボット対話コーパス](https://aclanthology.org/2025.naacl-long.367/)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | |
| jp_kyoto | [オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | MIT |
| en | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62)、[Seamless Interaction](https://ai.meta.com/research/seamless-interaction/)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | |
| en_kyoto | [オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | MIT |
| ch | [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | |
| ch_kyoto | [オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | MIT |
| tri | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62)、[HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15)、[旅行代理店タスク対話コーパス](https://aclanthology.org/2022.lrec-1.619/)、[ヒューマンロボット対話コーパス](https://aclanthology.org/2025.naacl-long.367/)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | |
| tri_kyoto | [オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | MIT |

ノイズロバスト2チャネルモデル (`vap_mc`) は、同じデータに環境雑音とランダムなゲイン変更を加えて学習しています（`en` は Switchboard corpus とオンライン会話データセットのみ）。

### 1チャネルモデル

[VAD モデル](vad_JP.md) と同一のデータで学習しています。ノイズロバスト1チャネルモデル (`vap_mc_mono`) は、同じデータに環境雑音とランダムなゲイン変更を加えて学習しています。

| lang | 学習データ |
| ---- | ---------- |
| jp | [旅行代理店タスク対話コーパス](https://aclanthology.org/2022.lrec-1.619/)、[ヒューマンロボット対話コーパス](https://aclanthology.org/2025.naacl-long.367/)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) |
| en | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62)、[Seamless Interaction](https://ai.meta.com/research/seamless-interaction/)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) |
| ch | [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) |

## 使用例

### 2チャネル入力

```python
from maai import Maai, MaaiInput

wav1 = MaaiInput.Wav(wav_file_path="path_to_your_user_wav_file")
wav2 = MaaiInput.Wav(wav_file_path="path_to_your_system_wav_file")

maai = Maai(
    mode="vap",
    lang="jp",
    frame_rate=10,
    audio_ch1=wav1,
    audio_ch2=wav2,
    device="cpu",
    # mc=False,  # 通常（非ノイズロバスト）のモデルを使う場合
)
maai.start()

while True:
    result = maai.get_result()
    print(result["p_now"])     # [float, float]
    print(result["p_future"])  # [float, float]
```

### 1チャネル入力

```python
from maai import Maai, MaaiInput

mic = MaaiInput.Mic()

maai = Maai(
    mode="vap",
    lang="jp",
    frame_rate=12.5,
    audio_ch1=mic,   # audio_ch2 を省略 -> 1チャネルモデル
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
- [マイク2本の入力](../example/vap/vap_2mic.py) 🎤
- [wav ファイル2本の入力](../example/vap/vap_2wav.py) 🎵
- [マイク1本の入力（第2チャネルはゼロ信号）](../example/vap/vap_mic.py) 🎤
- [マイク1本の入力・Mimi エンコーダ (`model_type="normal-ver2"`)](../example/vap/vap_mic_ver2.py) 🎤
- [マイク1本の入力・ノイズロバストモデル](../example/vap_mc/vap_mc_mic.py) 🎤
- [マイク1本の入力・1チャネルモデル](../example/vap_mono/vap_mono_mic.py) 🎤
- [wav ファイル1本の入力・1チャネルモデル](../example/vap_mono/vap_mono_wav.py) 🎵

## 📚 論文・参考文献

このモデルを利用した成果を発表する際は、以下の論文を引用してください。🙏

Koji Inoue, Bing'er Jiang, Erik Ekstedt, Tatsuya Kawahara, Gabriel Skantze<br>
__Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection__<br>
International Workshop on Spoken Dialogue Systems Technology (IWSDS), 2024<br>
https://arxiv.org/abs/2401.04868<br>

```
@inproceedings{inoue2024iwsds,
    author = {Koji Inoue and Bing'er Jiang and Erik Ekstedt and Tatsuya Kawahara and Gabriel Skantze},
    title = {Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection},
    booktitle = {International Workshop on Spoken Dialogue Systems Technology (IWSDS)},
    year = {2024},
    url = {https://arxiv.org/abs/2401.04868},
}
```

トリリンガルVAPモデルを利用する場合は、以下も引用してください。

Koji Inoue, Bing'er Jiang, Erik Ekstedt, Tatsuya Kawahara, Gabriel Skantze<br>
__Multilingual Turn-taking Prediction Using Voice Activity Projection__<br>
Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING), pages 11873-11883, 2024<br>
https://aclanthology.org/2024.lrec-main.1036/<br>

```
@inproceedings{inoue2024lreccoling,
    author = {Koji Inoue and Bing'er Jiang and Erik Ekstedt and Tatsuya Kawahara and Gabriel Skantze},
    title = {Multilingual Turn-taking Prediction Using Voice Activity Projection},
    booktitle = {Proceedings of the Joint International Conference on Computational Linguistics and Language Resources and Evaluation (LREC-COLING)},
    pages = {11873--11883},
    year = {2024},
    url = {https://aclanthology.org/2024.lrec-main.1036/},
}
```

ノイズロバスト (`mc`) モデルを利用する場合は、以下も引用してください。

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
