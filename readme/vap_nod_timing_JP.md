<h1>
<p align="center">
頷き予測モデル（タイミング専用）
</p>
</h1>
<p align="center">
README: <a href="vap_nod_timing.md">English </a> | <a href="vap_nod_timing_JP.md">Japanese (日本語) </a>
</p>

`Maai` クラスの `mode` パラメータは `nod_timing` に設定してください。

本モデルは2チャンネルの16kHz音声データを入力とし、ch1をユーザ、ch2をシステムの音声として想定しています。

- `p_nod`: 頷きの発生確率
- `p_bc`: 相槌の発生確率

</br>

## 対応言語

現時点では日本語のみ対応しています。
`Maai` クラスの `lang` パラメータで指定してください。

### 日本語（`lang=jp`）

本モデルは以下の日本語データセットで学習されています：
- [ヒューマンロボット対話コーパス]()

</br>

## 実装例

```python
from maai import Maai, MaaiInput

mic = MaaiInput.Mic(mic_device_index=0)
zero = MaaiInput.Zero()

maai = Maai(mode="nod_timing", lang="jp", frame_rate=10, audio_ch1=mic, audio_ch2=zero, device="cpu")
maai.start()

while True:
    result = maai.get_result()

    print(result['p_nod'])
    print(result['p_bc'])
```

</br>

## パラメータ

利用可能なパラメータを以下にまとめます。

`model_type` はモデルのバリアントを選択します：`"normal-ver2"` はMimiをエンコーダに使う新しいバリアント、`"normal"` は以前のリリースから使われているCPCエンコーダのバリアントです。

`frame_rate` はVAPモデルが1秒あたりに処理するサンプル数を指定します。ご利用の計算環境に合わせて、この値を調整してください。

### `model_type="normal-ver2"`（Mimiエンコーダ）

| lang | frame_rate |
| ---- | ---------- |
| jp | 12.5 |

### `model_type="normal"`（CPCエンコーダ）

| lang | frame_rate |
| ---- | ---------- |
| jp | 5, 10, 20 |

<br>

## 📚 論文・参考文献

このモデルを利用した成果を発表する際は、以下の論文を引用してください。🙏

Kazushi Kato, Koji Inoue, Divesh Lala, Keiko Ochi, Tatsuya Kawahara<br>
__Real-time Generation of Various Types of Nodding for Avatar Attentive Listening System__<br>
https://www.arxiv.org/abs/2507.23298<br>

```
@inproceedings{kato2025icmi,
    author = {Kazushi Kato and Koji Inoue and Divesh Lala and Keiko Ochi and Tatsuya Kawahara},
    title = {Real-time Generation of Various Types of Nodding for Avatar Attentive Listening System},
    booktitle = {International Conference on Multimodal Interaction (ICMI)},
    year = {2025},
}
```
