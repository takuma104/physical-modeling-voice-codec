# physical-modeling-voice-codec

物理モデル音声合成をデコーダに使うニューラルコーデックの検討リポジトリです。
潜在表現を調音・音源の制御パラメータにし、デコーダを声道の物理シミュレーションにすれば、潜在表現が多少外れても人間の発声としてあり得る音になるのではないか、という仮説を確かめます。
背景と先行研究は [docs/plans/idea.md](docs/plans/idea.md) にあります。

## 構成

| パス | 内容 |
|---|---|
| `third-party/chimera-vocal-synth` | 物理モデル音声合成 CHIMERA(サブモジュール、無改造で使う) |
| `third-party/arti-6` | 6 次元調音特徴の逆推定・合成モデル(サブモジュール) |
| `crates/pmvc-synth` | CHIMERA を JSON で駆動する CLI。音素長・F0 を外から与える `render`、制御列をそのまま再生する `drive` |
| `pmvc/` | 実験スクリプト(Python) |
| `docs/` | 企画メモと関連論文 |

## セットアップ

```sh
git submodule update --init
cargo build --release          # target/release/pmvc-synth
cargo build --release -p chimera-cli --manifest-path third-party/chimera-vocal-synth/Cargo.toml
uv sync
```

実験は TSP speech database(48 kHz、254 MB、Simplified BSD)を `~/datasets/tsp` に置いて使います。
サブモジュールの `tools/tsp/fetch.sh` は `sha256sum` を使うので、macOS では次のようにします。

```sh
mkdir -p ~/datasets/tsp && cd ~/datasets/tsp
curl -fL -C - -o 48k.zip https://www.mmsp.ece.mcgill.ca/Documents/Data/TSP-Speech-Database/48k.zip
shasum -a 256 48k.zip   # 9cfb3a3a13014c8ff90770a5d1923f376da73ac927b9100f09826f60cf06cf43
unzip -qo 48k.zip
```

別の場所に置いた場合は環境変数 `TSP_DIR` で指定します。
ffmpeg(試聴ページの音声変換)も必要です。

## 実験の流し方

出力はすべて `out/`(git 管理外)に出ます。

```sh
# 1. 韻律移植: 実録音の音素列・音素長・F0 で CHIMERA に同じ文を言わせる(200 文)
uv run python -m pmvc.transplant
uv run python -m pmvc.evaluate

# 2. 摂動テスト: 制御列(100 Hz × 21 チャネル)にノイズ・量子化・間引きを入れて再生する(40 文)
uv run python -m pmvc.perturb
uv run python -m pmvc.evaluate --against drive --name perturb_results \
  --variants "$(uv run python -c "from pmvc.perturb import conditions; print(','.join(c for c in conditions() if c != 'drive'))")"

# 3. 試聴ページ(音声埋め込みの HTML 1 枚)
uv run python -m pmvc.page --notes notes.html
```

各スクリプトの条件や指標の定義は、それぞれのファイル冒頭の docstring にあります。
