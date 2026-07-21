<h1>
<p align="center">
1チャネル音声用ターンテイキング (VAP) モデル (Mono-VAP)
</p>
</h1>
<p align="center">
README: <a href="vap_mono.md">English </a> | <a href="vap_mono_JP.md">Japanese (日本語) </a>
</p>

`Maai` クラスの `mode` パラメータに `vap_mono` を指定してください。

このモードは、標準の VAP モデル (`vap`) と同一のモデル・学習済み重みを内部で動かしますが、インターフェイスは1チャネルです。`audio_ch1` のみを指定すればよく、2チャネル目には内部で無音(ゼロ信号)が供給されます。
片方の話者の音声しか得られないユースケース(例: 音声対話システムでのマイク1本の入力)を想定しています。

入力は 1 チャネル・16kHz の音声データです。

## 出力

`p_now` と `p_future` は入力音声に対する単一の float 値です(2要素リストではありません):

- 内部では、入力チャネルと無音チャネルの間で話者正規化された通常の `p_now` / `p_future` を計算します。
- そのうえで、1チャネル目の [0.5, 1.0] の値を [0.0, 1.0] に線形に引き伸ばします。0.5 以下の値は 0.0 になります。

```
p_mono = max(0.0, (p[0] - 0.5) * 2.0)
```

`p_now` は入力話者の 0〜600 ミリ秒先の音声活動を、`p_future` は 600〜2000 ミリ秒先を表します。
`vad` も入力チャネルに対する単一の float 値です。

## 対応言語・フレームレート・コンテキスト長

`vap_mono` は標準 VAP モデルとチェックポイントを共有するため、指定可能な `lang`、`frame_rate`、`context_len_sec` は [VAP モデル](vap_JP.md) と同一です。

## 使用例

```python
from maai import Maai, MaaiInput, MaaiOutput

mic = MaaiInput.Mic()

maai = Maai(
    mode="vap_mono",
    lang="jp",
    frame_rate=10,
    audio_ch1=mic,   # audio_ch2 は不要
    device="cpu",
)
maai.start()

while True:
    result = maai.get_result()
    print(result["p_now"], result["p_future"], result["vad"])  # すべて単一の float
```

サンプルスクリプト:
- [マイク1本の入力](../example/vap_mono/vap_mono_mic.py) 🎤
- [wav ファイル1本の入力](../example/vap_mono/vap_mono_wav.py) 🎵
