<h1>
<p align="center">
MaaiMultiple - 音声エンコーダを共有して複数モデルを同時実行
</p>
</h1>
<p align="center">
README: <a href="multiple.md">English </a> | <a href="multiple_JP.md">Japanese (日本語) </a>
</p>

`MaaiMultiple` は、複数の Maai モデルを 1 つの音声エンコーダを共有しながら同時に動作させるためのクラスです。
Maai の各モデル（`vap`, `vap_mc`, `bc`, `bc_2type`, `nod`, `nod_para`, `vap_prompt`）は、いずれも同じ音声エンコーダ（`model_type="normal"` の場合は CPC、`model_type="normal-ver2"` の場合は Mimi）を入口として持っています。
モデルごとに別々の `Maai` を作る素朴な方法では、1 フレーム処理するたびに同じエンコーダを N 回走らせることになります。
`MaaiMultiple` ではエンコーダを 1 回だけ実行し、その出力をすべてのサブモデルに配るので、特に CPU 実行時の処理コストを大幅に下げられます。

</br>

## どのような場面で使うか

同じ音声入力に対して複数のふるまいを同時に推定したい場合に使用します。例：

- ターンテイキング ＋ 相槌 ＋ 頷きを同時に動かす
- 2 種類の相槌モデル（`bc` と `bc_2type`）を同時に動かす
- `vap` と `vap_mc` を並べて比較する

1 モデルだけで十分な場合は、これまで通り `Maai` を直接使ってください。

</br>

## 制約

エンコーダを共有するため、エンコーダ関連の設定はすべてのサブモデルで共通である必要があります。
以下のパラメータは `MaaiMultiple` に一度だけ渡し、すべてのサブモデルに適用されます。

- `model_type`（`"normal"` または `"normal-ver2"`）
- `frame_rate`
- `context_len_sec`
- `device`
- `cpc_model`
- `mimi_*` 各種パラメータ（`model_type="normal-ver2"` のとき有効）
- `cache_dir`, `force_download`
- `use_kv_cache`
- `inference_chunk_frames`（[複数フレームをまとめて推論して高速化](../README_JP.md#複数フレームをまとめて推論して高速化) を参照）
- `audio_ch1`, `audio_ch2`

サブモデルごとに変えられる設定は `configs` リストで指定します。
各要素は以下のキーを持つ `dict` です。

- `"mode"`（必須）：`"vap"`, `"vap_mc"`, `"vad"`, `"bc"`, `"bc_det"`,
  `"bc_2type"`, `"nod"`, `"nod_para"`, `"vap_prompt"` のいずれか
- `"lang"`（必須）：`Maai` と同じ指定
- `"label"`（任意）：結果辞書のキー名。省略時は `"mode"` を使用。
  同じ `mode` を複数登録するとき（例：複数言語の `vap` を比較するとき）に区別するために使います。
- `"local_model"`（任意）：ローカルに保存したチェックポイントのパス
- `"return_p_bins"`（任意）：`Maai` と同じフラグ（`vap` / `vap_mc` のみ意味があります）

</br>

## 使用例

```python
from maai import MaaiMultiple, MaaiInput

mic = MaaiInput.Mic()
zero = MaaiInput.Zero()

maai_multi = MaaiMultiple(
    configs=[
        {"mode": "vap", "lang": "jp"},
        {"mode": "bc",  "lang": "jp"},
        {"mode": "nod", "lang": "jp"},
    ],
    audio_ch1=mic,
    audio_ch2=zero,
    frame_rate=10,
    device="cpu",
    model_type="normal",
)

maai_multi.start()

while True:
    result = maai_multi.get_result()
    # フレーム共通フィールド：
    #   result["t"]   - タイムスタンプ
    #   result["x1"]  - チャネル1の音声チャンク
    #   result["x2"]  - チャネル2の音声チャンク
    # 各サブモデルの出力は label（省略時は mode）の下にネストされます。
    print(result["vap"]["p_now"])
    print(result["bc"]["p_bc"])
    print(result["nod"]["p_nod_short"])
```

そのまま実行できる例は [`example/multiple/multi_mic.py`](../example/multiple/multi_mic.py) にあります。

</br>

## 結果のフォーマット

`get_result()` は 1 フレームにつき以下のような辞書を返します。

```text
{
    "t":  <フレームのタイムスタンプ>,
    "x1": <チャネル1の音声チャンク, np.ndarray>,
    "x2": <チャネル2の音声チャンク, np.ndarray>,
    "<label-1>": { ...そのモードで Maai が返すフィールドと同じもの... },
    "<label-2>": { ... },
    ...
}
```

各 label 配下には、その mode を単体の `Maai` で動かした場合の `get_result()` と同じフィールドが入ります（`t`, `x1`, `x2` はトップレベルにまとめてあるため重複しません）。

</br>

## API

```python
MaaiMultiple(
    configs: list[dict],
    audio_ch1, audio_ch2,
    frame_rate: float = 10,
    context_len_sec: int = 20,
    device: str = "cpu",
    model_type: str = "normal",
    use_kv_cache: bool = True,
    inference_chunk_frames: int = 1,
    # ... mimi_* / cache_dir / cpc_model / force_download は Maai と同じ
)
```

メソッドは `Maai` と同様です。

- `start()` ：ワーカースレッドと音声入力を開始
- `stop(wait=True, timeout=2.0)` ：ワーカースレッドを停止
- `process(x1, x2)` ：音声チャンクを直接渡して処理（マイクワーカーを使わない場合用）
- `get_result()` ：次のフレームの結果辞書をブロッキングで取得
- `get_sub_maai(label) -> Maai` ：内部の `Maai` インスタンスを取得
- `set_prompt_ch1(prompt, label=None)` /
  `set_prompt_ch2(prompt, label=None)` ：すべての `vap_prompt` サブモデル（または指定 `label` のみ）にプロンプトを反映

</br>

## 出力: ConsoleBar

`MaaiOutput.ConsoleBar` は `MaaiMultiple` の結果フォーマットを自動で判別します。
フレームごとの共通フィールド（`t`, `x1`, `x2`）は上部に 1 度だけ表示し、
その下にサブモデルごとのセクションを見出し付きで描画するので、複数モデルを同時に動かしてもコンソール表示が見やすく保たれます。

```python
from maai import MaaiMultiple, MaaiInput, MaaiOutput

mic = MaaiInput.Mic()
zero = MaaiInput.Zero()

maai_multi = MaaiMultiple(
    configs=[
        {"mode": "vap", "lang": "jp"},
        {"mode": "bc",  "lang": "jp"},
        {"mode": "nod", "lang": "jp"},
    ],
    audio_ch1=mic, audio_ch2=zero, frame_rate=10, device="cpu",
)

output = MaaiOutput.ConsoleBar(bar_type="balance")
maai_multi.start()

while True:
    output.update(maai_multi.get_result())
```

同じ `ConsoleBar` インスタンスは単一の `Maai` でもそのまま使えます。
セクション分割表示は、結果にネストされたサブモデル辞書が含まれているときだけ自動で有効になります。

</br>

## 補足

- すべてのサブモデルのエンコーダモジュールは、最初のサブモデルのエンコーダへの参照に置き換えられます。
  そのため、エンコーダの重みおよび Mimi のストリーミング状態はメモリ上・デバイス上に 1 つしか存在しません。
- KV キャッシュ（`use_kv_cache=True`）はサブモデルごとに独立で、毎ステップ `context_len_sec` まで切り詰められます（`Maai` と同じ挙動）。
- `use_kv_cache=False` の場合は、エンコーダ出力のローリングコンテキスト（長さ `context_len_sec`）が全サブモデルで共有されます。
- `inference_chunk_frames` は全サブモデルが同じエンコーダを共有するため、共通の設定として1つだけ指定します。`N > 1` の場合、`N` フレームにつき1つの結果が出力されます。
  各サブモデルの `decrease_dimension` 投影はその上で個別に適用されます。
