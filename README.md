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
| `crates/pmvc-engine` | CHIMERA を Python から直接呼ぶ拡張モジュール(PyO3)。物理状態ごと分岐できる `Voice.fork()` を持つ |
| `pmvc/` | 実験スクリプト(Python) |
| `docs/` | 企画メモと関連論文 |

## セットアップ

```sh
git submodule update --init
cargo build --release          # target/release/pmvc-synth
cargo build --release -p chimera-cli --manifest-path third-party/chimera-vocal-synth/Cargo.toml
uv sync                        # pmvc-engine もここでビルドされる(Rust が必要)
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

手順 5 と 6 は別の録音を使います。Hi-Fi-CAPTAIN(日本語と英語の女声・男声、CC BY-NC-SA 4.0)の `eval` の音声とテキストを `~/datasets/hfc/hi-fi-captain/` に、JSUT の `basic5000` を `~/datasets/jsut/jsut_ver1.1/` に展開しておきます。使うのは各 100 文です。
手順 6 は JVS(100 話者、再配布不可)の `parallel100` を `~/datasets/jvs/jvs_ver1/` に置いて使います。

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

# 3. コピー合成: 制御列を実録音に合わせて最適化する
uv run python -m pmvc.warp                      # 出発点: 規則の発音を録音の時間軸に伸縮
uv run python -m pmvc.copysynth                 # 局所探索(40 文、1 文 1〜2 分。書き起こしも使う)
uv run python -m pmvc.evaluate --variants rule,f0,warp,copy --aligned warp,copy --name results_copy
uv run python -m pmvc.second_judge --name results_copy   # 第 2 の判定役(Whisper large-v3-turbo)

# 4. エンコーダ(コーデックの試作): 録音 → 制御列 → 音声。文の書き起こしは使わない
uv run python -m pmvc.transplant --all --resume    # 全 1,372 文の下準備(録音、F0、規則の合成)
uv run python -m pmvc.warp --resume
#   A: CHIMERA 自身の合成音と制御列の対だけで学習し、全録音を符号化する
uv run python -m pmvc.encoder train --name sim --no-real --synthetic warp --epochs 30 --every 3 --stride 4
uv run python -m pmvc.encoder encode --name sim --out sim --split all
#   C′: 声の代役(制御列 → リスナーにどう聞こえるか)を合成音で学習し、それを通して録音で閉ループ学習する
uv run python -m pmvc.standin --name standin0 --variants warp,sim
uv run python -m pmvc.encoder train --name encn --target sim --synthetic sim --tune 6 --standin standin0
#   比較用。探索: A の制御列を録音に近づくよう直す(1 文 1〜2 分。--shard K/N で並走できる)
uv run python -m pmvc.copysynth --all --base sim --nll-weight 0 --iters 20 --out simcopy --resume
#   B: 直した制御列を教師に学習。C: さらに代役を通した閉ループの誤差を加える
uv run python -m pmvc.encoder train --name encb --tune 6
uv run python -m pmvc.standin
uv run python -m pmvc.encoder train --name enc --tune 6 --standin standin
#   テスト話者で評価する
for m in encb enc encn; do uv run python -m pmvc.encoder encode --name $m --out $m --split test; done
uv run python -m pmvc.evaluate --variants rule,sim,simcopy,encb,enc,encn --aligned sim,simcopy,encb,enc,encn --name results_codec
uv run python -m pmvc.second_judge --name results_codec
uv run python -m pmvc.heard --name results_codec    # リスナー距離(探索が下げる量)
#   エンコーダの制御列への摂動(テスト話者の 40 文)
uv run python -m pmvc.perturb --base encn --tag ec --ids "$(ls out/transplant/encn | grep tracks.json | sed 's/.tracks.json//' | awk 'NR%6==1' | paste -sd, -)" \
  --conds "$(uv run python -c "from pmvc.perturb import conditions; print(','.join(conditions() + ['coded_5bit_50hz', 'coded_4bit_50hz', 'coded_5bit_25hz', 'coded_4bit_25hz']))")"
uv run python -m pmvc.evaluate --against ec_drive --name perturb_ec_results \
  --variants "$(uv run python -c "import json; print(','.join(c for c in json.load(open('out/transplant/perturb_ec.json'))['conditions'] if c != 'ec_drive'))")"

# 5. 英語(TSP)で学習したエンコーダに、ほかの録音を通す(日本語 300 文と英語 200 文。出力は out/ja)
H=~/datasets/hfc/hi-fi-captain; J=~/datasets/jsut/jsut_ver1.1/basic5000
uv run python -m pmvc.recordings --set hjf --lang ja --sex F --wavs $H/ja-JP/female/wav/eval --text $H/ja-JP/female/text/eval.txt
uv run python -m pmvc.recordings --set hjm --lang ja --sex M --wavs $H/ja-JP/male/wav/eval --text $H/ja-JP/male/text/eval.txt
uv run python -m pmvc.recordings --set jsut --lang ja --sex F --wavs $J/wav --text $J/transcript_utf8.txt
uv run python -m pmvc.recordings --set hef --lang en --sex F --wavs $H/en-US/female/wav/eval --text $H/en-US/female/text/eval.txt
uv run python -m pmvc.recordings --set hem --lang en --sex M --wavs $H/en-US/male/wav/eval --text $H/en-US/male/text/eval.txt
for m in sim enc encn; do uv run python -m pmvc.encoder encode out/ja --models out/transplant/models --name $m --out $m --split all; done
uv run python -m pmvc.judge                                                   # Whisper 2 種、音声記号の認識器、リスナー距離
uv run python -m pmvc.judge out/transplant --split test --name results_judge  # 同じ物差しで TSP のテスト話者
#   診断: C′ の制御列を、書き起こしなしの探索で録音に近づける(日本語 24 文)
IDS=$(for s in hjf hjm jsut; do for k in 005 017 029 041 053 065 077 089; do printf "%s_%s," $s $k; done; done | sed 's/,$//')
uv run python -m pmvc.copysynth out/ja --ids $IDS --base encn --nll-weight 0 --iters 20 --out encncopy --scale out/transplant/warp
uv run python -m pmvc.judge --ids $IDS --variants encn,encncopy --name results_search

# 6. 日本語の録音を閉ループ学習に加える(JVS: 学習 86 話者 × 10 文、検証 6 話者、テスト 8 話者 × 25 文)
uv run python -m pmvc.jvs                                  # out/jvs に録音を並べる
uv run python -m pmvc.encoder encode out/jvs --models out/transplant/models --name sim --out sim --split all
uv run python -m pmvc.yardstick                            # 物差し(リスナー)の候補を、同じ文を読んだ 2 人の録音で比べる
M="--listener utter-project/mHuBERT-147 --layer 8"         # 選んだ物差し: 多言語の mHuBERT-147 の第 8 層
#   D: 日本語の録音を加える(物差しは英語の wav2vec2 のまま)
uv run python -m pmvc.standin --name standin_j --variants warp,sim --also out/jvs
uv run python -m pmvc.encoder train --name encd --target sim --synthetic sim --tune 6 --standin standin_j --also out/jvs
#   E: 録音は英語のまま、物差しを替える
uv run python -m pmvc.standin --name standin_m0 --variants warp,sim $M
uv run python -m pmvc.encoder train --name ence --target sim --synthetic sim --tune 6 --standin standin_m0
#   F: 両方
uv run python -m pmvc.standin --name standin_m --variants warp,sim --also out/jvs $M
uv run python -m pmvc.encoder train --name encf --target sim --synthetic sim --tune 6 --standin standin_m --also out/jvs
#   G: F に加えて、エンコーダが録音を聞く入口も mHuBERT に替える
uv run python -m pmvc.encoder train --name encg --target sim --synthetic sim --tune 6 --standin standin_m --also out/jvs --ssl utter-project/mHuBERT-147
#   3 つのテスト(JVS の未使用話者、手順 5 の録音、TSP のテスト話者)で評価する
for m in encn encd ence encf encg; do
  uv run python -m pmvc.encoder encode out/jvs --models out/transplant/models --name $m --out $m --split test
  uv run python -m pmvc.encoder encode out/ja --models out/transplant/models --name $m --out $m --split all
  uv run python -m pmvc.encoder encode --name $m --out $m --split test
done
uv run python -m pmvc.judge out/jvs --split test --variants encn,encd,ence,encf,encg
uv run python -m pmvc.judge out/ja --variants encn,encd,ence,encf,encg --name results_taught
uv run python -m pmvc.judge out/transplant --split test --variants encn,encd,ence,encf,encg --name results_taught

# 7. 試聴ページ(音声埋め込みの HTML 2 枚: 英語編 listening.html と日本語編 listening_ja.html)
uv run python -m pmvc.page --notes notes.html --notes-ja notes_ja.html
```

エンコーダの学習と評価は話者で分けています。TSP では同じ文字の女性・男性話者が同じ文リストを読むので、FK・MK・FL・ML をテスト、FJ・MJ を検証に取り置くと、その話者の文も学習に出てきません。
学習は GPU があると速く(RTX 5090 で A が 3 分、代役が 4〜5 分、C′ が 12 分ほど)、探索は CPU を使います(16 コアの Mac で 1 時間に約 270 文)。
日本語を加えた D・F・G は、同じ GPU で 1 本 25〜40 分でした。JVS も話者と文の両方で分けていて、テストの 8 話者と 25 文は学習にも検証にも出てきません。

`pmvc.calibrate` は雑音源のゲインと声門音源の設定を実録音に合わせて調整するものですが、未使用のデータでの改善はわずかだったため、コピー合成では既定で使いません(使うときは `--calibrated`)。

各スクリプトの条件や指標の定義は、それぞれのファイル冒頭の docstring にあります。
