<h1>
<p align="center">
ノイズロバストターンテイキング (VAP) モデル (MC-VAP)
</p>
</h1>
<p align="center">
README: <a href="vap_mc.md">English </a> | <a href="vap_mc_JP.md">Japanese (日本語) </a>
</p>

`Maai` クラスの `mode` パラメータに `vap_mc` を指定してください。

これは通常の [VAP モデル](vap_JP.md)のマルチコンディション版です。学習データに様々な環境雑音を重畳し、さらに発話音声のゲインもランダムに変更させています。そのため実環境で通常のモデルより頑健に動作することが期待されます。

学習条件以外は、モデル構造・入力・出力とも通常の [VAP モデル](vap_JP.md)と同じです。

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

`vad` は各入力チャネルの現フレームにおける音声活動の確率を表す、2要素の float のリストです。

`return_p_bins=True` を指定すると、`p_bins` として4つのビン (0〜200, 200〜600, 600〜1200, 1200〜2000 ミリ秒) ごと・話者ごとの活動確率が得られ、`p_bins_now` / `p_bins_future` はそれぞれ `p_now` / `p_future` の範囲での平均です。`p_now` や `p_future` とは異なり、これらは話者間で正規化されていません。

## 対応言語・フレームレート

`Maai` クラスの `lang` パラメータで言語を指定してください。
括弧内は、そのフレームレートで公開されている `context_len_sec`（モデルへ入力する音声文脈の長さ・秒）の値です。既定値は `context_len_sec=20` です。

| lang | model_type | frame_rate（公開されている `context_len_sec`） |
| ---- | ---------- | ---------------------------------------------- |
| jp | `normal` (CPC エンコーダ) | 5 (3, 5, 10, 20)、10 (3, 5, 10, 20)、20 (2.5, 10, 20) |
| jp | `normal-ver2` (Mimi エンコーダ) | 12.5 (20) |
| jp_kyoto | `normal` (CPC エンコーダ) | 5 (3, 5, 20)、10 (3, 5, 20)、20 (2.5, 20) |
| jp_kyoto | `normal-ver2` (Mimi エンコーダ) | 12.5 (20) |
| en | `normal` (CPC エンコーダ) | 5 (3, 5, 20)、10 (3, 5, 20)、20 (2.5, 20) |
| en | `normal-ver2` (Mimi エンコーダ) | 12.5 (20) |
| en_kyoto | `normal` (CPC エンコーダ) | 5 (20)、10 (20) |
| en_kyoto | `normal-ver2` (Mimi エンコーダ) | 12.5 (20) |
| ch | `normal` (CPC エンコーダ) | 5 (3, 5, 20)、10 (3, 5, 20)、20 (2.5, 20) |
| ch | `normal-ver2` (Mimi エンコーダ) | 12.5 (20) |
| ch_kyoto | `normal` (CPC エンコーダ) | 5 (20)、10 (20) |
| ch_kyoto | `normal-ver2` (Mimi エンコーダ) | 12.5 (20) |
| tri | `normal` (CPC エンコーダ) | 5 (3, 5, 20)、10 (3, 5, 20)、20 (2.5) |
| tri | `normal-ver2` (Mimi エンコーダ) | 12.5 (20) |
| tri_kyoto | `normal` (CPC エンコーダ) | 5 (20)、10 (20) |
| tri_kyoto | `normal-ver2` (Mimi エンコーダ) | 12.5 (20) |

`model_type` はモデル種別を指定します。`"normal"` はこれまでのリリースで使っていた既存モデル、`"normal-ver2"` は Mimi をエンコーダとして使用する新しいモデルです。5/10/20 Hz のモデルが CPC ベース (`model_type="normal"`)、12.5 Hz のモデルが Mimi ベース (`model_type="normal-ver2"`) である点にご注意ください。

`frame_rate` は VAP モデルが1秒あたりに処理するサンプル数を指定します。ご利用の計算環境に合わせて調整してください。

## 学習データ

通常の [VAP モデル](vap_JP.md)と同じデータに、環境雑音の重畳とゲインのランダム変更を適用しています。
`tri` は3言語対応（日本語＋英語＋中国語）のモデルです。`*_kyoto` のモデルはオンライン会話データセットのみで学習されており、MIT ライセンスで公開されています。

| lang | 学習データ | ライセンス |
| ---- | ---------- | ---------- |
| jp | [旅行代理店タスク対話コーパス](https://aclanthology.org/2022.lrec-1.619/)、[ヒューマンロボット対話コーパス](https://aclanthology.org/2025.naacl-long.367/)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | |
| jp_kyoto | [オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | MIT |
| en | [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62)、[オンライン会話データセット](https://www.arxiv.org/abs/2506.21191) | |
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
    mode="vap_mc",
    lang="jp",
    frame_rate=10,
    context_len_sec=5,
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
- [マイク1本の入力（第2チャネルはゼロ信号）](../example/vap_mc/vap_mc_mic.py) 🎤

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
