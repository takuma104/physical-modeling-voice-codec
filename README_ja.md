# physical-modeling-voice-codec

[English](README.md)

物理モデル音声合成をデコーダに使うニューラルコーデックの検討リポジトリです。
潜在表現を調音・音源の制御パラメータにし、デコーダを声道の物理シミュレーションにすれば、潜在表現が多少外れても人間の発声としてあり得る音になるのではないか、という仮説を確かめます。

- **デモ**(インストール不要): <https://huggingface.co/spaces/takuma104/pmvc-tts?ja>(English: [`?en`](https://huggingface.co/spaces/takuma104/pmvc-tts?en))
- **モデルの重み**(CC BY 4.0): <https://huggingface.co/takuma104/pmvc-tts>
- **テクニカルレポート**: [日本語](docs/report/physical_modeling_voice_codec_report_ja.pdf) · [English](docs/report/physical_modeling_voice_codec_report_en.pdf)
- **実験の手順**: [docs/experiments_ja.md](docs/experiments_ja.md)

最初の企画メモと先行研究の調査は [docs/plans/idea.md](docs/plans/idea.md) にあります。先行研究との比較は、レポートの 9 節にまとめています。

## すぐ試す: PMVC-TTS(テキスト音声合成のデモ)

テキストから制御列(21 チャネル × 100 Hz)を書くモデルと、それをブラウザの中で音にする画面です。日本語と英語を話します。
サーバが返すのは制御列だけで、音は CHIMERA のエンジン(WebAssembly)がブラウザの中で作ります。声の高さと明るさは、話している途中でも変えられます。

Rust(`wasm32-unknown-unknown` ターゲット)、[uv](https://docs.astral.sh/uv/)、ブラウザが必要です。

```sh
git clone --recursive https://github.com/takuma104/physical-modeling-voice-codec && cd physical-modeling-voice-codec
rustup target add wasm32-unknown-unknown
uv sync                                                                    # Python の環境。pmvc-engine(Rust)もここでビルドされる
cargo build -p pmvc-wasm --release --target wasm32-unknown-unknown         # ブラウザ用のエンジン(667 KB)
uvx --from huggingface_hub hf download takuma104/pmvc-tts --local-dir models
uv run python -m pmvc.serve --model models/ja.json models/en.json          # → http://127.0.0.1:7860/
```

画面の言語はブラウザの言語で決まります。アドレスに `?ja` か `?en` を付けると、その言語で開きます(`http://127.0.0.1:7860/?en`)。

画面は CHIMERA の Speak 画面に倣って作り直したもので、CHIMERA そのものではありません。声道のシミュレーションは Jameson Hodge さんの [CHIMERA](https://github.com/jamesondh/chimera-vocal-synth)(MIT)を改変せずに使っています。歌わせたり、舌や唇を直接動かしたりするには本家をどうぞ。

## 構成

| パス | 内容 |
|---|---|
| `third-party/chimera-vocal-synth` | 物理モデル音声合成 CHIMERA(サブモジュール、無改造で使う) |
| `third-party/arti-6` | 6 次元調音特徴の逆推定・合成モデル(サブモジュール) |
| `third-party/NISQA` | 音声だけから主観品質を予測するモデル(サブモジュール、無改造で使う。重みは CC BY-NC-SA 4.0) |
| `crates/pmvc-synth` | CHIMERA を JSON で駆動する CLI。音素長・F0 を外から与える `render`、制御列をそのまま再生する `drive` |
| `crates/pmvc-engine` | CHIMERA を Python から直接呼ぶ拡張モジュール(PyO3)。物理状態ごと分岐できる `Voice.fork()`、声門や狭めの閉じ方のパラメータを直接指定する `Voice.set_glottis()`・`Voice.set_airway()`、鼻腔の形(断面積、長さ、副鼻腔の開口と容積)を指定する `Voice.set_nose()`、体格を既定の 3 種の混合ではなく 4 つの値(大きさ、咽頭の割合、舌の半径、唇の幅)で直接指定する入力、内部の状態(声門や狭めの面積、圧力、流量)を読む `Voice.probe`、ある姿勢での声道の断面の輪郭を返す `outline()`(レポートの図 1 はこれで描く)を持つ |
| `crates/pmvc-voice` | 制御列で CHIMERA を鳴らす中身。`pmvc-engine` と `pmvc-wasm` が共有するので、同じ制御列は Python でもブラウザでも同じサンプルになる |
| `crates/pmvc-wasm` | ブラウザ用のエンジン(C ABI、import なし。AudioWorklet の中で動く)。断面図を描くための幾何も返す |
| `web/` | デモの画面(ビルド不要の静的ファイル)。`pmvc.serve` が配る |
| `space/` | Hugging Face Space(Gradio SDK、ZeroGPU 上で CPU 推論)用の `app.py` など。`tools/space.sh` で組み立てる |
| `voices/v1.json` | モデルが学習に使った 2 つの声(低い声・高い声)と、明るさのつまみの定義 |
| `pmvc/` | 実験スクリプト(Python)。コーデックのエンコーダは `pmvc.encoder`、テキストからの合成は `pmvc.tts`、ニューラルデコーダとの対照は `pmvc.control` と `pmvc.vocoder` |
| `docs/` | テクニカルレポート(`docs/report/`、英語と日本語)、実験の手順(`docs/experiments_ja.md`)、企画メモと関連論文のメモ |

## 実験

レポートの実験を流し直す手順(セットアップ、必要なデータ、コマンド)は [docs/experiments_ja.md](docs/experiments_ja.md) にあります。
