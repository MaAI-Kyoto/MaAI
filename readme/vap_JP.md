<h1>
<p align="center">
ターンテイキング (VAP) モデル
</p>
</h1>
<p align="center">
README: <a href="vap.md">English </a> | <a href="vap_JP.md">Japanese (日本語) </a>
</p>

`Maai` クラスの `mode` パラメータに `vap` を指定してください。

このモデルは**2話者の近い未来の音声活動**を予測します。これが音声対話システムにおけるターンテイキング判断の基礎になります。

このモデルは2話者のチャネルを同時に受け取り、チャネル間のアテンションを用いて処理します。そのため、一方の話者に対する予測は常にもう一方の話者の状況を踏まえたものになります。

入力は 2 チャネル・16kHz の音声データです。

## 出力

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

## 対応言語・フレームレート

`Maai` クラスの `lang` パラメータで言語を指定してください。

| lang | model_type | frame_rate |
| ---- | ---------- | ---------- |
| jp | `normal` (CPC エンコーダ) | 5, 10, 20 |
| jp | `normal-ver2` (Mimi エンコーダ) | 12.5 |
| jp_kyoto | `normal` (CPC エンコーダ) | 5, 10, 20 |
| jp_kyoto | `normal-ver2` (Mimi エンコーダ) | 12.5 |
| en | `normal` (CPC エンコーダ) | 5, 10, 20 |
| en | `normal-ver2` (Mimi エンコーダ) | 12.5 |
| en_kyoto | `normal` (CPC エンコーダ) | 5, 10 |
| en_kyoto | `normal-ver2` (Mimi エンコーダ) | 12.5 |
| ch | `normal` (CPC エンコーダ) | 5, 10, 20 |
| ch | `normal-ver2` (Mimi エンコーダ) | 12.5 |
| ch_kyoto | `normal` (CPC エンコーダ) | 5, 10 |
| ch_kyoto | `normal-ver2` (Mimi エンコーダ) | 準備中 |
| tri | `normal` (CPC エンコーダ) | 5, 10 |
| tri | `normal-ver2` (Mimi エンコーダ) | 12.5 |
| tri_kyoto | `normal` (CPC エンコーダ) | 5, 10 |
| tri_kyoto | `normal-ver2` (Mimi エンコーダ) | 12.5 |

`model_type` はモデル種別を指定します。`"normal"` はこれまでのリリースで使っていた既存モデル、`"normal-ver2"` は Mimi をエンコーダとして使用する新しいモデルです。5/10/20 Hz のモデルが CPC ベース (`model_type="normal"`)、12.5 Hz のモデルが Mimi ベース (`model_type="normal-ver2"`) である点にご注意ください。

`frame_rate` は VAP モデルが1秒あたりに処理するサンプル数を指定します。ご利用の計算環境に合わせて調整してください。

## 学習データ

`tri` は3言語対応（日本語＋英語＋中国語）のモデルです。`*_kyoto` のモデルはオンライン会話データセットのみで学習されており、MIT ライセンスで公開されています。

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

## 使用例

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

サンプルスクリプト:
- [マイク2本の入力](../example/vap/vap_2mic.py) 🎤
- [wav ファイル2本の入力](../example/vap/vap_2wav.py) 🎵
- [マイク1本の入力（第2チャネルはゼロ信号）](../example/vap/vap_mic.py) 🎤
- [マイク1本の入力・Mimi エンコーダ (`model_type="normal-ver2"`)](../example/vap/vap_mic_ver2.py) 🎤

一方の話者の音声しか利用できない場合は、第2チャネルにゼロ信号を入力するのではなく、専用の[単一チャネルモデル (`vap_mono`)](vap_mono_JP.md)の利用を推奨します。

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
