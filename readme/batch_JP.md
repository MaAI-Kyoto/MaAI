<h1>
<p align="center">
MaaiBatch - 録音済み音声のバッチ推論
</p>
</h1>
<p align="center">
README: <a href="batch.md">English </a> | <a href="batch_JP.md">Japanese (日本語) </a>
</p>

`Maai` はリアルタイム入力向けに作られており、音声エンコーダとモデルを1フレーム（12.5 Hz なら 80 ms）ごとに1回ずつ実行します。
対話コーパスの分析や合成対話の評価のように録音全体が手元にある場合は、`MaaiBatch` を使うと同じ出力を録音全体についてまとめて計算できます。

- 音声エンコーダは長いチャンク（1回あたり `encoder_chunk_frames` フレーム）単位で、複数の録音を並列に（`batch_size`）処理します。ストリーミング状態はチャンク間で引き継ぐので、特徴量は1フレームずつ処理した場合と同じです。
  - Mimi：リサンプラ・畳み込みのパディング・Transformer の KV キャッシュを引き継ぎます。
  - CPC：リアルタイム処理では各フレームを直前 320 サンプルの重なりと一緒に符号化し、窓の端に掛かる畳み込み出力を捨てているため、複数フレーム分の長い窓で計算しても同じ出力になります。LSTM の状態を引き継ぎ、モデルごとのダウンサンプリング層はリアルタイムと同じく窓ごとに適用します。
- 各モデルの Transformer は系列全体を1回の順伝播で処理します。
  文脈長の帯マスクにより、直近 `context_len_sec` 秒を保持するリアルタイムの KV キャッシュと同じ計算になります。
  長い録音は重なりのあるウィンドウに分割して1つのバッチとして処理します。重なりは Transformer の受容野を覆うので、分割しても結果は変わりません。
- 複数モデル（`configs`）を指定すると、`MaaiMultiple` と同様に音声のエンコードを1回で共有します。

</br>

## 対応範囲

- Mimi のモデル（`model_type="normal-ver2"`、`MaaiBatch` の既定値）で Mimi のフレームレート（`frame_rate=12.5`、Mimi の既定値）のもの、
  または CPC のモデル（`model_type="normal"`）で 16000 / `frame_rate` が 160 サンプルの倍数になるフレームレート（5, 10, 20, 50 Hz など。既定値 10 Hz）のもの。
- モード：`vap`, `vad`, `bc_det`, `bc`, `bc_2type`, `nod`, `nod_timing`, `nod_para`（`vap_prompt` は非対応）。
  そのモード・言語・エンコーダ・フレームレート・文脈長の学習済みモデルが必要です（[HuggingFaceリポジトリ](https://huggingface.co/maai-kyoto)を参照）。`local_model` も指定できます。
- 入力：話者ごとの 16 kHz モノラル float32 波形（numpy 配列）。

</br>

## 使い方

```python
import soundfile as sf
from maai import MaaiBatch

ch1, _ = sf.read("speaker1_16k.wav", dtype="float32")
ch2, _ = sf.read("speaker2_16k.wav", dtype="float32")

batch = MaaiBatch(mode="vap", lang="jp", device="cuda")
result = batch.process(ch1, ch2)

result["time_sec"]   # (フレーム数,)    各予測の時刻 [秒]
result["p_now"]      # (フレーム数, 2)
result["p_future"]   # (フレーム数, 2)
result["vad"]        # (フレーム数, 2)
```

複数の録音は `batch_size` 件ずつまとめて処理します（長さの近いものを同じバッチにまとめます）。

```python
results = batch.process_many([(a1, a2), (b1, b2), (c1, c2)])
```

複数モデルはエンコーダ処理を共有します。結果はラベル（省略時はモード名）ごとに入れ子になります。

```python
batch = MaaiBatch(
    configs=[
        {"mode": "vap", "lang": "jp"},
        {"mode": "bc",  "lang": "jp"},
        {"mode": "nod", "lang": "jp"},
    ],
    device="cuda",
)
result = batch.process(ch1, ch2)
result["vap"]["p_now"], result["bc"]["p_bc"], result["nod"]["p_nod_short"]
```

`configs` の各要素には [MaaiMultiple](multiple_JP.md) と同じく `"mode"`, `"lang"`, `"mc"`, `"local_model"`, `"return_p_bins"`, `"label"` を指定できます。

CPC のモデルは `Maai` と同じく `model_type="normal"` で選びます。

```python
batch = MaaiBatch(mode="vap", lang="jp", model_type="normal", frame_rate=10, context_len_sec=20)
```

`mono=True` のときは `audio_ch1` だけを渡します。`vap`・`vad`・`bc_det` は1チャネルモデルを使い、それ以外のモデルには第2チャネルとして無音を与えます（`Maai` で `MaaiInput.Zero()` を渡した場合と同じ）。

```python
batch = MaaiBatch(mode="vap", lang="jp", mono=True)
result = batch.process(ch1)
result["p_now"]      # (フレーム数,)
```

`MaaiBatch.to_frames(result)` は結果を `Maai.get_result()` と同じ形式（`t`, `x1`, `x2` を除く）の1フレーム1辞書のリストに変換します。

スクリプト例は [example/batch/batch_2wav.py](../example/batch/batch_2wav.py) にあります。

</br>

## 出力

- `end_sample`, `time_sec`：各予測を行う入力位置（モデルが見た音声の終端）。リアルタイムの `Maai` と同じく1フレームに1つです。
  Mimi では、リアルタイムの既定（ONNX Mimi）と同じく最初のエンコーダ出力を捨てるので、最初の予測は 0.16 秒で、以降 80 ms ごとです。
  CPC では最初の予測は1フレーム後（10 Hz なら 0.1 秒）です。
  末尾の1フレームに満たない端数は推論しません。
- モデルの出力は `Maai.get_result()` と同じ名前・同じ値で、1フレーム1行の配列です（`p_now`: `(n, 2)`、`p_bc`: `(n,)`、`nod_repetitions`: `(n, 3)` など）。
  `p_bins`, `p_bins_now`, `p_bins_future` は `return_p_bins=True` のときだけ返します。

</br>

## 精度

リアルタイムの `Maai` との比較（`context_len_sec=20`、日本語サンプル対話の 40 秒、CPU）：

| リアルタイムの `Maai` | フレーム数 | 最大絶対誤差 |
| --- | --- | --- |
| Mimi 12.5 Hz、`use_mimi_onnx=False`（PyTorch Mimi）と `MaaiBatch(drop_first_frame=False)` | 500 / 500 | `vap` 2.8e-5、`bc` 1.5e-5 |
| Mimi 12.5 Hz、既定（ONNX Mimi）と `MaaiBatch()` | 499 / 499 | `vap` 5.4e-5、`bc` 2.5e-5 |
| CPC 10 Hz と `MaaiBatch(model_type="normal")` | 400 / 400 | `vap` 7.5e-6、`bc` 1.3e-6 |

他のモデルでも同様です（30 秒、フレーム数は一致、最大絶対誤差 4.3e-5 以下）。
Mimi では `vad`・`bc_det`・`nod`・`nod_timing`・`nod_para` と1チャネル版の `vap`・`vad`・`bc_det`（`nod_para` の `*_pred` は全フレームで一致）、
CPC では 5 / 10 / 20 Hz の `vap`、`vad`、`bc_2type`、`nod`、`nod_timing`、1チャネル版の `vap`（10 Hz）と `vad`（50 Hz）で確認しています。
残る差は浮動小数点の丸め誤差（Mimi の既定の場合は ONNX Runtime と PyTorch の差も）です。
テストは [tests/test_batch.py](../tests/test_batch.py) にあります（`MAAI_TEST_PRETRAINED=1` で学習済みモデルとの比較も実行します）。

</br>

## 速度

日本語サンプル対話（266 秒）、CPU（16 スレッド）での計測：

| | 処理時間 |
| --- | --- |
| リアルタイムの `Maai`（`vap`、ONNX Mimi、12.5 Hz）で1フレームずつ | 32 ms/フレーム、1モデルあたり約 106 秒 |
| `MaaiBatch` で `vap` + `bc`（Mimi、エンコーダ共有） | 2モデル合計で 12.4 秒 |
| リアルタイムの `Maai`（`vap`、CPC、10 Hz）で1フレームずつ | 12.5 ms/フレーム、1モデルあたり約 33 秒 |
| `MaaiBatch` で `vap` + `bc`（CPC、エンコーダ共有） | 2モデル合計で 9.9 秒 |

GPU では差がさらに大きくなる見込みです（ここでは未計測）。リアルタイム処理の1フレームごとの呼び出しでは GPU を使い切れないのに対し、`MaaiBatch` は大きなバッチ単位で計算するためです。

</br>

## パラメータ

| パラメータ | 既定値 | 説明 |
| --- | --- | --- |
| `frame_rate` | 12.5（Mimi）/ 10（CPC） | モデルのフレームレート。 |
| `cpc_model` | `~/.cache/cpc/60k_epoch4-d0f474de.pt` | CPC の学習済み重み（CPC のみ。`Maai` と同じく、無ければダウンロードします）。 |
| `batch_size` | `8` | 一緒にエンコードする録音の数、および Transformer の1回の順伝播で処理するウィンドウ数。メモリが足りない場合は小さくしてください。 |
| `encoder_chunk_frames` | `50` | エンコーダの1回の呼び出しで処理するフレーム数。大きいほどメモリを使います。結果には影響しません。 |
| `fold_context_frames` | 受容野 | 長い録音を分割するとき、各ウィンドウの前に置く文脈のフレーム数。既定値（20 秒・12.5 Hz で 996 フレーム）なら分割しても結果は変わりません。 |
| `fold_step_frames` | 文脈長の2倍 | 1ウィンドウあたりの新しいフレーム数（20 秒・12.5 Hz で 500 フレーム）。 |
| `drop_first_frame` | `True`（Mimi）/ `False`（CPC） | リアルタイムの ONNX Mimi エンコーダと同じく最初のエンコーダ出力を捨てます。Mimi で `False` にすると `Maai(..., use_mimi_onnx=False)` と一致します。リアルタイムの CPC は捨てません。 |

メモリ使用量は `batch_size` × 録音の長さに比例して増えます（1バッチ分のエンコーダ特徴量を、全モデルの処理が終わるまで保持するため）。
