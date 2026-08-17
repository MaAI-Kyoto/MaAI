<p align="center">
<img src="img/logo_ver2.png" width="300">
</p>

<h1>
<p align="center">
MaAI
</p>
<p align="center">
<img src="https://img.shields.io/pypi/v/maai" alt="PyPI - Version">
</p>
</h1>
<p align="center"><b>リアルタイム</b>かつ<b>軽量</b>な会話AI向け<b>非言語的ふるまい生成</b>ソフトウェア</p>
<p align="center">（Voice Activity Projectionのリアルタイム実装）</p>
<p align="center">
📄 README: <a href="README.md">English </a> | <a href="README_JP.md">Japanese (日本語) </a>
<br>
📚 API Documentation: <a href="https://maai-kyoto.github.io/MaAI/">https://maai-kyoto.github.io/MaAI/</a>
</p>

<b>MaAI</b>は（１）<b>ターンテイキング</b>、（２）<b>相槌</b>、（３）<b>頷き</b>をリアルタイムかつ連続的に予測するソフトウェアです。
日本語・英語・中国語に対応しており、今後も対応言語やふるまいの種類を拡大予定です。
会話AI（対話システムやロボット等）向けに設計されており、2チャネル（ユーザ・システム）または1チャネル（ユーザのみ）の音声入力に対応します🎙️
軽量設計のため、CPUのみでも高速に動作します⚡

プロジェクトの名称の<b>MaAI</b>は、日本語の<b>間（ま）</b>や<b>間合い</b>に由来し、会話におけるタイミングや間合いの調整をAIで実現することを目指しています。

現在サポートしているモデルは主にVoice Activity Projection（VAP）およびその拡張です。
VAPモデルの詳細は以下のリポジトリを参照してください：
https://github.com/ErikEkstedt/VoiceActivityProjection

<b>MaAIソフトウェアを使用したシステム開発・共同研究</b>などは京都大学の[井上昂治](https://inokoj.github.io/)までご相談いただけますと幸いです。

<br>

https://github.com/user-attachments/assets/e7d1b266-0613-4a46-8718-1808a3682bd3

<br>

## 🆕 新着情報

- MaAIの新バージョン（0.2.0）をリリースしました！大幅な性能改善を実現しています（2026年4月17日）
- [3言語対応相槌予測モデル（英語・中国語・日本語）を公開しました](readme/vap_bc_JP.md)（2025年11月19日）
- 計算効率の改善により長いコンテキスト長のモデルがより高速に動作するようになりました（2025年9月3日）
- MaAIプロジェクトおよびリポジトリを公開しました！🚀（2025年8月13日）

<br>

## 🚀 インストール

MaAIはpipで簡単にインストールできます：

```bash
pip install maai
```

> 💡 **注意:** デフォルトではPyTorchのCPU版がインストールされます。GPUで利用したい場合は、事前にCUDA環境に合ったPyTorchのGPU版をインストールしてください。

以下のように実行できます🏃‍♂️
タスク（mode）に応じたモデルやパラメータは自動でダウンロードされます。
下記は、１チャネル目はマイク入力（ユーザ）、２チャネル目は無音（システム）をターンテイキングモデル（VAP）に入力する例です。

```python
from maai import Maai, MaaiInput, MaaiOutput

mic = MaaiInput.Mic()
zero = MaaiInput.Zero() 

maai = Maai(mode="vap", lang="jp", frame_rate=10, audio_ch1=mic, audio_ch2=zero, device="cpu")
maai_output_bar = MaaiOutput.ConsoleBar()

maai.start()
while True:
    result = maai.get_result()
    maai_output_bar.update(result)
```

`Maai` のモデル種別指定には `model_type` を使います。

- `"normal"`: これまでのリリースで使っていた既存の標準モデル
- `"normal-ver2"`: Mimi encoder を利用する新しいモデル

<br>

## 🧩 モデル

対応モデル（ふるまい・言語・設定など）は今後も追加予定です。
現在利用可能なモデルは[HuggingFaceリポジトリ](https://huggingface.co/maai-kyoto)をご覧ください。

### ターンテイキング

ターンテイキングモデルはVAPを用い、次の瞬間にどちらが発話するかを予測します。

- [VAPモデル](readme/vap_JP.md)
- [ノイズロバストVAPモデル（<b>推奨</b>）](readme/vap_mc_JP.md)
- [1チャンネルVAPモデル](readme/vap_mono_JP.md)

### 音声区間検出

音声区間検出モデルは、各参加者が現時点で発話しているかどうかを検出します。2つのチャネルを同時に処理するため、一方の話者の声が他方のマイクに回り込んでいる場合でも、どちらが実際に話しているのかを判定できます。

- [VADモデル（1チャンネル版もあり）](readme/vad_JP.md)

### 相槌

相槌は「うん」「はい」などの短い聞き手反応で、ターンテイキングとも関連します。

- [VAPベースの相槌予測モデル - タイミング](readme/vap_bc_JP.md)
- [VAPベースの相槌予測モデル - 2種類のタイミング](readme/vap_bc_2type_JP.md)

上記のモデルが相槌を打つべきタイミングを*予測*するのに対し、相槌検出モデルは今まさに発話されている内容が相槌かどうかを*検出*します。

- [相槌検出モデル（1チャンネル版もあり）](readme/bc_det_JP.md)


### 頷き

頷きは頭の上下運動で、相槌と関連します。発声を伴わず非言語的に聞き手反応を示すことができます。

- [VAPベースの頷き予測モデル](readme/vap_nod_JP.md)
- [VAPベースの頷きパラメーター予測モデル](readme/vap_nod_para_JP.md)

### 音声エンコーダを共有して複数モデルを同時実行

同じ音声入力に複数の Maai モデルを動かしたい場合は、`MaaiMultiple` クラスを使うと音声エンコーダを 1 つだけ共有して動かすことができ、計算コストの大きいエンコーダ処理を 1 フレームに 1 回で済ませられます。

- [MaaiMultiple - 音声エンコーダを共有して複数モデルを同時実行](readme/multiple_JP.md)

<br>

## 🎚️ 入出力

MaAIモデルへの入力は、`Maai`クラスインスタンスの`process`メソッドを直接呼び出せます。
`MaaiInput`クラスにより、WAVファイル・マイク・TCP通信など柔軟な音声入力が可能です。

- WAVファイル入力: `Wav`クラス 📁
- マイク入力: `Mic`クラス 🎙️
- TCP通信: `TCPReceiver` / `TCPTransmitter`クラス 🌐
- チャンク入力: `Chunk`クラス 📦

これらのクラスを使うことで、用途に応じた音声入力方法を簡単に切り替えられます。

処理結果は、`Maai`クラスインスタンスの`get_result`メソッドで取得できます。
また、`MaaiOutput`クラスを使うことで、さまざまな可視化やTCP通信による出力も可能です。

- コンソール動的出力: `ConsoleBar`クラス 📊
- GUIバーグラフ表示: `GuiBar`クラス 🖼️
- GUIプロット表示: `GuiPlot`クラス 📈
- TCP通信: `TCPReceiver` / `TCPTransmitter`クラス 🌐

詳細は以下のREADMEもご参照ください：
- [入力について](readme/input_JP.md)
- [出力について](readme/output_JP.md)

<br>

## 💡 実装例

`example`ディレクトリにMaAIモデルの実装例があります。

- ターンテイキング（VAP）
    - [1マイク入力](example/vap/vap_mic.py) 🎤
    - [2マイク入力](example/vap/vap_2mic.p) 🎤🎤
    - [1マイク入力 と 1wavファイル入力](example/vap/vap_mic_wav.py) 🎤🎧
    - [2wavファイル入力](example/vap/vap_2wav.py) 🎧🎧
    - [1マイクチャンク入力](example/vap/vap_mic_chunk.py) 🎤
    - [1マイクTCP経由入力](example/vap/vap_mic_tcp.py) 🎤🌐
    - [2wavファイル入力 と TCP経由出力](example/vap/vap_2wav_output_tcp.py) 🎧🎧🌐
    - [1マイクチャンクTCP経由入力](example/vap/vap_mic_chunk_tcp.py) 🎤🌐

- ノイズロバストターンテイキング（VAP）
    - [1マイク入力](example/vap_mc/vap_mic.py) 🎤

- 相槌
    - [1マイク入力](example/bc/bc_mic.py) 🎤
    - [1マイク入力 (normal-ver2)](example/bc/bc_mic_ver2.py) 🎤
    - [1マイク入力](example/bc_2type/bc_2type_mic.py) 🎤

- 頷き
    - 3種類の頷き予測
        - [1マイク入力](example/nod/nod_mic.py) 🎤
    - 頷きの運動学的パラメーター予測
        - [1マイク入力](example/nod/nod_para_mic.py) 🎤

- 音声エンコーダを共有した複数モデル同時実行
    - [VAP + BC + Nod を 1 マイク入力で実行](example/multiple/multi_mic.py) 🎤

- 出力
    - [コンソール動的出力](example/output/vap_2wav_ConsoleBar.py) 📊
    - [GUIバーグラフ表示](example/output/vap_2wav_GuiBar.py) 🖼️
    - [GUIプロット表示](example/output/vap_2wav_GuiPlot.py) 📈
    - [TCP通信](example/output/vap_2wav_TCP.py) 🌐
<br>

## 📚 論文・参考文献

本リポジトリを利用した成果を発表する際は、以下の論文を引用してください🙏

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

ノイズロバストVAPモデルを利用する場合は、以下も引用してください。

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

相槌VAPモデルを利用する場合は、以下も引用してください。

Koji Inoue, Divesh Lala, Gabriel Skantze, Tatsuya Kawaharaa<br>
__Yeah, Un, Oh: Continuous and Real-time Backchannel Prediction with Fine-tuning of Voice Activity Projection__<br>
https://aclanthology.org/2025.naacl-long.367/<br>

```
@inproceedings{inoue2025vapbc,
    author = {Koji Inoue and Divesh Lala and Gabriel Skantze and Tatsuya Kawahara},
    title = {Yeah, Un, Oh: Continuous and Real-time Backchannel Prediction with Fine-tuning of Voice Activity Projection},
    booktitle = {Proceedings of the Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL)},
    pages = {7171--7181},
    year = {2025},
    url = {https://aclanthology.org/2025.naacl-long.367/},
}
```

<br>

## 🛠️ トラブルシューティング

### Windows + Anaconda 環境で `model_type="normal-ver2"` が `onnxruntime` の DLL ロードエラーになる

Windows の Anaconda 環境で `model_type="normal-ver2"` を使うと、Mimi encoder が依存している `onnxruntime` のインポート時に以下のようなエラーが出る場合があります。

```
ImportError: DLL load failed while importing onnxruntime_pybind11_state:
ダイナミック リンク ライブラリ (DLL) 初期化ルーチンの実行に失敗しました。
```

これは conda 環境に Visual C++ のランタイムが入っていないことが原因です。conda-forge からランタイムを入れることで解消します。

```bash
conda install -c conda-forge vs2015_runtime
```

これだけで `model_type="normal-ver2"`（および同設定の `MaaiMultiple`）がそのまま動くようになります。

<br>

## 📝 ライセンス

本リポジトリのソースコードはMITライセンスです。
学習済みモデルは各モデルのREADMEまたはHugging Faceに記載されたライセンスに従います。

`model_type="normal-ver2"` を利用する場合、MaAI は Mimi encoder を使用します。
Mimi は https://huggingface.co/kyutai/mimi で公開されており、ライセンスは CC BY 4.0 です。
MaAI では Mimi 自体は改変せず、そのままエンコーダとして利用し、別の下流モデルと結合して使用しています。

CPCモデルはオリジナルCPCプロジェクト由来です。
ライセンスの詳細は以下を参照してください：
https://github.com/facebookresearch/CPC_audio
