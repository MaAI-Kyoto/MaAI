<h1>
<p align="center">
ターンテイキング (VAP) モデル
</p>
</h1>
<p align="center">
README: <a href="vap.md">English </a> | <a href="vap_JP.md">Japanese (日本語) </a>
</p>

このモデルは**話者の近い未来の音声活動**を予測します。これが音声対話システムにおけるターンテイキング判断の基礎になります。

- `Maai` クラスの `mode` に `"vap"` を指定してください。
- 入力は 16kHz の音声で、2チャネル（2話者）または1チャネル（1話者）に対応しています。
- 入力に応じて最適なモデルが自動的に選択されます（[モデルの選択](#モデルの選択)を参照）。

> **注意:** `mode="vap"` はデフォルトでノイズロバストモデルを使用するようになりました（`mc=True`）。従来の動作（通常モデル）にするには `mc=False` を指定してください。
> 旧 mode 名 `vap_mc` / `vap_mono` / `vap_mc_mono` も引き続き動作しますが、非推奨です。

<br>

## クイックスタート

### 2チャネル入力（2話者）

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
)
maai.start()

while True:
    result = maai.get_result()
    print(result["p_now"])     # [float, float]
    print(result["p_future"])  # [float, float]
```

### 1チャネル入力（1話者）

`audio_ch2` を省略するだけです。

```python
from maai import Maai, MaaiInput

mic = MaaiInput.Mic()

maai = Maai(
    mode="vap",
    lang="jp",
    frame_rate=12.5,
    audio_ch1=mic,
    device="cpu",
    model_type="normal-ver2",
)
maai.start()

while True:
    result = maai.get_result()
    print(result["p_now"], result["p_future"], result["vad"])  # 単一の float
```

<br>

## モデルの選択

`mode="vap"` には4種類のモデルがあり、`mc` パラメータと音声入力の与え方から自動的に選択されます。

| 入力 | `mc=True`（デフォルト） | `mc=False` |
| --- | --- | --- |
| **2チャネル**: `audio_ch1` + `audio_ch2` | `vap_mc`（ノイズロバスト） | `vap`（通常） |
| **1チャネル**: `audio_ch1` のみ、または `audio_ch2=MaaiInput.Zero()` | `vap_mc_mono`（ノイズロバスト） | `vap_mono`（通常） |

実際に読み込まれたモデルは `maai.variant` で確認できます。

### ノイズロバストモデル（`mc=True`）

- 学習データに様々な環境雑音を重畳し、さらに発話音声のゲインもランダムに変更して学習しています（マルチコンディション学習）。
- 実環境で通常のモデルより頑健に動作することが期待されます。

### 1チャネルモデル

- 2チャネルモデルに無音チャネルを与えたものでは**なく**、1チャネル専用に学習された独立のモデルです。
- 音声を1本だけエンコードし、その話者の将来の音声活動を直接予測します。
- 片方の話者の音声しか得られないユースケース（例: 音声対話システムでのマイク1本の入力）を想定しています。

### フォールバック

すべての `lang` / `frame_rate` / `model_type` の組み合わせに4種類すべてのモデルがあるわけではありません（[対応言語・フレームレート](#対応言語フレームレート)を参照）。希望のモデルが存在しない場合は次のように動作します。

- ノイズロバストモデルがない → 通常のモデルを使用します。
- 1チャネルモデルがない → 第2チャネルを無音として2チャネルモデルを使用し、警告を表示します。

<br>

## 出力

2チャネルモデルと1チャネルモデルで出力の形式が異なります。

| キー | 2チャネルモデル | 1チャネルモデル |
| --- | --- | --- |
| `p_now` | `[float, float]`: 次の 0〜600 ミリ秒で各話者が発話権を持つ確率 | `float`: 次の 0〜600 ミリ秒で入力話者が発話している確率 |
| `p_future` | `[float, float]`: 600〜2000 ミリ秒先について同様 | `float`: 600〜2000 ミリ秒先について同様 |
| `vad` | `[float, float]`: 現フレームにおける各チャネルの音声活動の確率 | `float`: 入力チャネルの音声活動の確率 |

- すべての値は [0.0, 1.0] の範囲です。
- 一般的なターンテイキング用途では `p_now` の利用を推奨します。
- **2チャネル**: `p_now` と `p_future` は話者間で正規化されており、2つの値の合計は 1.0 になります。
- **1チャネル**: 比較対象となる第2話者がいないため、正規化は**行いません**。それぞれの値は入力話者の音声活動の期待値そのものです。
- `vad` は [VAD モデル](vad_JP.md)と同じ量を VAP モデル内部で計算したものです。

```python
# 2チャネル
result["p_now"]     # 例: [0.87, 0.13]  -> 話者1が次の話者になる可能性が高い
result["p_future"]  # 例: [0.62, 0.38]

# 1チャネル
result["p_now"]     # 例: 0.87  -> 入力話者が次の 600 ミリ秒で発話している可能性が高い
```

### ビンごとの確率（`return_p_bins=True`）

`return_p_bins=True` を指定すると、以下のキーも返されます。

- `p_bins`: 4つのビン（0〜200, 200〜600, 600〜1200, 1200〜2000 ミリ秒）ごとの音声活動確率。2チャネルモデルでは話者ごと、1チャネルモデルでは1つのリストです。
- `p_bins_now` / `p_bins_future`: それぞれ `p_now` / `p_future` の範囲での平均。

`p_now` や `p_future` とは異なり、これらは話者間で正規化されていません。

<br>

## 対応言語・フレームレート

- `lang`: モデルの言語です。
- `model_type`: `"normal-ver2"` は Mimi エンコーダを使用する新しいモデル、`"normal"` はこれまでのリリースで使っていた既存モデル（CPC エンコーダ）です。
- `frame_rate`: 1秒あたりに処理するフレーム数です。ご利用の計算環境に合わせて調整してください。

### `model_type="normal-ver2"`（Mimi エンコーダ・`frame_rate=12.5`）

<table>
<thead>
<tr>
<th rowspan="2" align="left">言語</th>
<th rowspan="2" align="left"><code>lang</code></th>
<th colspan="2">🎧🎧 2チャネル</th>
<th colspan="2">🎧 1チャネル</th>
</tr>
<tr>
<th>通常<br><sub><code>vap</code></sub></th>
<th>ノイズロバスト<br><sub><code>vap_mc</code></sub></th>
<th>通常<br><sub><code>vap_mono</code></sub></th>
<th>ノイズロバスト<br><sub><code>vap_mc_mono</code></sub></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2"><b>日本語</b></td>
<td><code>jp</code></td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">✅</td>
</tr>
<tr>
<td><code>jp_kyoto</code> <sub>MIT</sub></td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td rowspan="2"><b>英語</b></td>
<td><code>en</code></td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">✅</td>
</tr>
<tr>
<td><code>en_kyoto</code> <sub>MIT</sub></td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td rowspan="2"><b>中国語</b></td>
<td><code>ch</code></td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">✅</td>
</tr>
<tr>
<td><code>ch_kyoto</code> <sub>MIT</sub></td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td rowspan="2"><b>3言語</b></td>
<td><code>tri</code></td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td><code>tri_kyoto</code> <sub>MIT</sub></td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
</tbody>
</table>

### `model_type="normal"`（CPC エンコーダ）

各セルは利用可能な `frame_rate` を表します。

<table>
<thead>
<tr>
<th rowspan="2" align="left">言語</th>
<th rowspan="2" align="left"><code>lang</code></th>
<th colspan="2">🎧🎧 2チャネル</th>
<th colspan="2">🎧 1チャネル</th>
</tr>
<tr>
<th>通常<br><sub><code>vap</code></sub></th>
<th>ノイズロバスト<br><sub><code>vap_mc</code></sub></th>
<th>通常<br><sub><code>vap_mono</code></sub></th>
<th>ノイズロバスト<br><sub><code>vap_mc_mono</code></sub></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2"><b>日本語</b></td>
<td><code>jp</code></td>
<td align="center">5, 10, 20</td>
<td align="center">5, 10, 20</td>
<td align="center">10, 20, 50</td>
<td align="center">—</td>
</tr>
<tr>
<td><code>jp_kyoto</code> <sub>MIT</sub></td>
<td align="center">5, 10, 20</td>
<td align="center">5, 10, 20</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td rowspan="2"><b>英語</b></td>
<td><code>en</code></td>
<td align="center">5, 10, 20</td>
<td align="center">5, 10, 20</td>
<td align="center">10, 20, 50</td>
<td align="center">—</td>
</tr>
<tr>
<td><code>en_kyoto</code> <sub>MIT</sub></td>
<td align="center">5, 10</td>
<td align="center">5, 10</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td rowspan="2"><b>中国語</b></td>
<td><code>ch</code></td>
<td align="center">5, 10, 20</td>
<td align="center">5, 10, 20</td>
<td align="center">10, 20, 50</td>
<td align="center">—</td>
</tr>
<tr>
<td><code>ch_kyoto</code> <sub>MIT</sub></td>
<td align="center">5, 10</td>
<td align="center">5, 10</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td rowspan="2"><b>3言語</b></td>
<td><code>tri</code></td>
<td align="center">5, 10</td>
<td align="center">5, 10</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td><code>tri_kyoto</code> <sub>MIT</sub></td>
<td align="center">5, 10</td>
<td align="center">5, 10</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
</tbody>
</table>

✅ 利用可能 ／ — 未提供（[フォールバック](#フォールバック)が適用されます）／ <sub>MIT</sub> MIT ライセンスで公開

<br>

## 学習データ

- `tri` は3言語対応（日本語＋英語＋中国語）のモデルです。
- `*_kyoto` のモデルはオンライン会話データセットのみで学習されており、MIT ライセンスで公開されています。
- ノイズロバストモデル（`mc`）は、対応する通常モデルと同じデータに環境雑音とランダムなゲイン変更を加えて学習しています。

### 2チャネルモデル

| lang | 学習データ | ライセンス |
| ---- | ---------- | ---------- |
| jp | [旅行代理店タスク対話コーパス](https://aclanthology.org/2022.lrec-1.619/)、[ヒューマンロボット対話コーパス](https://aclanthology.org/2025.naacl-long.367/)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | |
| jp_kyoto | [オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | MIT |
| en | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62)、[Seamless Interaction](https://ai.meta.com/research/seamless-interaction/)\*、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | |
| en_kyoto | [オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | MIT |
| ch | [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | |
| ch_kyoto | [オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | MIT |
| tri | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62)、[HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15)、[旅行代理店タスク対話コーパス](https://aclanthology.org/2022.lrec-1.619/)、[ヒューマンロボット対話コーパス](https://aclanthology.org/2025.naacl-long.367/)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | |
| tri_kyoto | [オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | MIT |

\* ノイズロバストモデル（`vap_mc`）では使用していません。

### 1チャネルモデル

[VAD モデル](vad_JP.md) と同一のデータで学習しています。

| lang | 学習データ |
| ---- | ---------- |
| jp | [旅行代理店タスク対話コーパス](https://aclanthology.org/2022.lrec-1.619/)、[ヒューマンロボット対話コーパス](https://aclanthology.org/2025.naacl-long.367/)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) |
| en | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62)、[Seamless Interaction](https://ai.meta.com/research/seamless-interaction/)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) |
| ch | [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) |

<br>

## サンプルスクリプト

### 2チャネル入力

- [マイク2本の入力](../example/vap/vap_2mic.py) 🎤
- [wav ファイル2本の入力](../example/vap/vap_2wav.py) 🎵

### 1チャネル入力

- [マイク1本の入力](../example/vap/vap_mic.py) 🎤
- [マイク1本の入力・Mimi エンコーダ (`model_type="normal-ver2"`)](../example/vap/vap_mic_ver2.py) 🎤
- [マイク1本の入力・ノイズロバストモデル](../example/vap_mc/vap_mc_mic.py) 🎤
- [マイク1本の入力・1チャネルモデル](../example/vap_mono/vap_mono_mic.py) 🎤
- [wav ファイル1本の入力・1チャネルモデル](../example/vap_mono/vap_mono_wav.py) 🎵

<br>

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

### 多言語モデル（`tri`）

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

### ノイズロバストモデル（`mc`）

ノイズロバストモデルを利用する場合は、以下も引用してください。

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
