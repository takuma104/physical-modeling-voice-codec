# physical-modeling-voice-codec

物理モデル音声合成をデコーダに使うニューラルコーデックの検討リポジトリです。
潜在表現を調音・音源の制御パラメータにし、デコーダを声道の物理シミュレーションにすれば、潜在表現が多少外れても人間の発声としてあり得る音になるのではないか、という仮説を確かめます。
背景と先行研究は [docs/plans/idea.md](docs/plans/idea.md) にあります。

## 構成

| パス | 内容 |
|---|---|
| `third-party/chimera-vocal-synth` | 物理モデル音声合成 CHIMERA(サブモジュール、無改造で使う) |
| `third-party/arti-6` | 6 次元調音特徴の逆推定・合成モデル(サブモジュール) |
| `third-party/NISQA` | 音声だけから主観品質を予測するモデル(サブモジュール、無改造で使う。重みは CC BY-NC-SA 4.0) |
| `crates/pmvc-synth` | CHIMERA を JSON で駆動する CLI。音素長・F0 を外から与える `render`、制御列をそのまま再生する `drive` |
| `crates/pmvc-engine` | CHIMERA を Python から直接呼ぶ拡張モジュール(PyO3)。物理状態ごと分岐できる `Voice.fork()`、声門や狭めの閉じ方のパラメータを直接指定する `Voice.set_glottis()`・`Voice.set_airway()`、鼻腔の形(断面積、長さ、副鼻腔の開口と容積)を指定する `Voice.set_nose()`、体格を既定の 3 種の混合ではなく 4 つの値(大きさ、咽頭の割合、舌の半径、唇の幅)で直接指定する入力、内部の状態(声門や狭めの面積、圧力、流量)を読む `Voice.probe`、ある姿勢での声道の断面の輪郭を返す `outline()`(レポートの図 1 はこれで描く)を持つ |
| `pmvc/` | 実験スクリプト(Python)。コーデックのエンコーダは `pmvc.encoder`、テキストからの合成は `pmvc.tts`、ニューラルデコーダとの対照は `pmvc.control` と `pmvc.vocoder` |
| `docs/` | 企画メモと関連論文。`docs/report/` にテクニカルレポートの下書き(英語・日本語) |

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

手順 5 以降は別の録音を使います。Hi-Fi-CAPTAIN(日本語と英語の女声・男声、CC BY-NC-SA 4.0)の `eval` の音声とテキストを `~/datasets/hfc/hi-fi-captain/` に、JSUT の `basic5000` を `~/datasets/jsut/jsut_ver1.1/` に展開しておきます。使うのは各 100 文です。手順 7 では、Hi-Fi-CAPTAIN の日本語女性の `dev` も使います。
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

# 7. 発話の外の余計な音を直し(エンコーダ H)、高い声の設定を NISQA に合わせて調整する
#   H: G と同じ作り方で、発話の外では「息なし」を手本にする
uv run python -m pmvc.encoder train --name ench --target sim --synthetic sim --tune 6 --standin standin_m --also out/jvs --ssl utter-project/mHuBERT-147 --silences
T=("out/jvs --split test" "out/ja --split all" "out/transplant --split test")
for t in "${T[@]}"; do uv run python -m pmvc.encoder encode $t --models out/transplant/models --name ench --out ench; done
#   声の設定: Hi-Fi-CAPTAIN の女性話者の dev 100 文のうち 16 文に合わせ、残り 84 文で確かめる
uv run python -m pmvc.recordings out/fit --set hjfd --lang ja --sex F --wavs $H/ja-JP/female/wav/dev --text $H/ja-JP/female/text/dev.txt
for m in encg ench; do uv run python -m pmvc.encoder encode out/fit --models out/transplant/models --name $m --out $m --split all; done
#   調整 1: NISQA だけに合わせる(--clicks 0)。声門は長く開く設定になるが、歯の雑音源も 2.2 倍になり、舌が閉じる瞬間などにもクリック音が出る
uv run python -m pmvc.calibrate out/fit --sex F --tracks encg --prefix hjfd --by nisqa --clicks 0 --n 16 --evals 220 --name voice-F1 \
  --only noiseGlottis,noiseConstriction,noiseTeeth,pressure,tension,breathiness,brightness,roughness,openQuotient,amplitudeScale \
  --start '{"tension": 0.0745, "brightness": 0.0264, "breathiness": 0.0465, "roughness": 0.0084, "noiseGlottis": 1.18, "noiseConstriction": 1.55, "noiseTeeth": 1.87, "openQuotient": 0.88}'
#   その設定がどの高さの話者に効くか(17 話者)。encode --voice はこの結果に従い、話者の声の高さで設定を切り替える
uv run python -m pmvc.pitchmix --voice out/fit/voice-F1.json $(for s in ML MK FL FK; do echo out/transplant:ench:$s; done) \
  $(for s in hem hef hjm hjf jsut; do echo out/ja:ench:$s; done) $(for n in 093 094 095 096 097 098 099 100; do echo out/jvs:ench:jvs$n; done)
#   対策 1: 無音区間(破裂音の閉鎖、ポーズ)の中の音(pmvc.clicks.in_silences が数える)
#   --no-wall-air は、声道の壁が動いて押しのける空気を音にしない(エンジンの公開スイッチ)。息を止めても出る数百 Hz 以下の低い音
#   --quiet-pauses は、録音のポーズ(発話の中の 0.12 秒以上の無音)で息を止める
#   対策 2: 舌が閉じる瞬間のクリック音(pmvc.clicks.extra が数える)。舌を鳴らすように聞こえるのはこちら
#   --closed-area は、狭めを「閉じた」とみなす面積。エンジンの既定(2e-4 cm²)では、まだ流れが残っているうちに 1 サンプルで止めるので 16 kHz まで届く衝撃になる
Q="--no-wall-air --quiet-pauses"; C="$Q --closed-area 1e-6"
uv run python -m pmvc.encoder encode out/fit --models out/transplant/models --name ench --out enchq --split all $Q
#   調整 2 と 3: 調整 1 の声門の開き方のまま、振動の振幅を 1.4 倍にし、雑音源を既定値(調整 2)、狭めと歯だけ 0.6 倍(調整 3)にする。
#   高い帯域のクリック音(pmvc.clicks.measure)を見ながら手で決めた値(--evals 0 は探索せず、--start の設定の点数を測って書き出す)
with() { python3 -c "
import json, sys; s = json.load(open('out/fit/voice-F1.json'))['settings']; s.update(noiseGlottis=1, amplitudeScale=1.4)
s.update(noiseConstriction=float(sys.argv[1]), noiseTeeth=float(sys.argv[1])); print(json.dumps(s))" $1; }
uv run python -m pmvc.calibrate out/fit --sex F --tracks enchq --prefix hjfd --by nisqa --n 16 --evals 0 --name voice-F2 --start "$(with 1.0)"
uv run python -m pmvc.calibrate out/fit --sex F --tracks enchq --prefix hjfd --by nisqa --n 16 --evals 0 --name voice-F3 --start "$(with 0.6)"
#   3 つのテストで評価する: H と 3 つの声を、対策なし・対策 1・対策 1+2 で
for t in "${T[@]}"; do
  enc() { uv run python -m pmvc.encoder encode $t --models out/transplant/models --name ench "$@"; }
  enc --out enchs --no-wall-air; enc --out enchq $Q; enc --out enchc $C
  enc --out enchv --voice out/fit/voice-F1.json; enc --out enchvq --voice out/fit/voice-F1.json $Q; enc --out enchvc --voice out/fit/voice-F1.json $C
  enc --out ench2q --voice out/fit/voice-F2.json $Q; enc --out ench2c --voice out/fit/voice-F2.json $C
  enc --out ench3q --voice out/fit/voice-F3.json $Q; enc --out ench3c --voice out/fit/voice-F3.json $C
  uv run python -m pmvc.judge $t --variants encn,encg,ench,enchs,enchq,enchc,enchv,enchvq,enchvc,ench2q,ench2c,ench3q,ench3c --name results_tuned
  uv run python -m pmvc.natural ${t%% *} --name results_tuned       # NISQA の点数を足す
done
#   対策ありの制御列で、調整 2 がどの高さの話者に効くかを測り直す(試聴ページの表)
uv run python -m pmvc.pitchmix --voice out/fit/voice-F2.json $(for s in ML MK FL FK; do echo out/transplant:enchc:$s; done) \
  $(for s in hem hef hjm hjf jsut; do echo out/ja:enchc:$s; done) $(for n in 093 094 095 096 097 098 099 100; do echo out/jvs:enchc:jvs$n; done)

# 8. 調整後の声でエンコーダを学習し直す(エンコーダ I)。調整後の声 = 声の高い話者に調整 2、対策 1+2
#   H は CHIMERA の既定の声を通して学習したので、まず学習用の制御列を調整後の声で鳴らし直し(制御列は同じ)、その声の代役を作る
V="--voice out/fit/voice-F2.json --no-wall-air --closed-area 1e-6"
uv run python -m pmvc.encoder revoice --of warp --out warp2c $V
uv run python -m pmvc.encoder revoice --of sim --out sim2c $V
uv run python -m pmvc.encoder revoice out/jvs --of sim --out sim2c $V
uv run python -m pmvc.standin --name standin_2c --variants warp2c,sim2c --also out/jvs $M --voiced   # --voiced: 話者ごとの「調整した声の割合」も読む
#   H と同じ作り方で、代役と、エポックを選ぶ聞き比べを調整後の声にする
uv run python -m pmvc.encoder train --name enci --target sim --synthetic sim2c --tune 6 --standin standin_2c --also out/jvs --ssl utter-project/mHuBERT-147 --silences $V --quiet-pauses
for t in "${T[@]}"; do
  uv run python -m pmvc.encoder encode $t --models out/transplant/models --name enci --out enci $V --quiet-pauses
  uv run python -m pmvc.judge $t --variants encn,encg,ench,enchs,enchq,enchc,enchv,enchvq,enchvc,ench2q,ench2c,ench3q,ench3c,enci --name results_tuned --add
  uv run python -m pmvc.natural ${t%% *} --name results_tuned
done

# 9. TTS: テキスト → 制御列。Hi-Fi-CAPTAIN の日本語男性話者 1 人(学習 18,655 文・約 24 時間、検証 100 文、テストは手順 5 の hjm の 100 文)
H=~/datasets/hfc/hi-fi-captain/ja-JP/male
uv run python -m pmvc.recordings out/hfcm --set hmt --speaker hjm --lang ja --sex M --wavs $H/wav/train_parallel --text $H/text/train_parallel.txt --n 0 --split train --link
uv run python -m pmvc.recordings out/hfcm --set hmv --speaker hjm --lang ja --sex M --wavs $H/wav/dev --text $H/text/dev.txt --n 0 --split valid --link
#   手本はエンコーダ I の制御列(鳴らさずに制御列だけ書く)
uv run python -m pmvc.encoder encode out/hfcm --models out/transplant/models --name enci --out enci --split all $V --quiet-pauses --no-render
#   各音素が録音のどのフレームにあたるかを、録音と読みだけから学習する。読点のないポーズにはポーズ記号(sp)を入れる
uv run python -m pmvc.align train out/hfcm
uv run python -m pmvc.align write out/hfcm
#   読みとアクセント(Open JTalk)→ 音素ごとの長さ → 制御列。誤差は、手本の制御列との差と、声の代役を通した録音との聞き比べ
#   学習の最後に、読点のない位置で話者がポーズするかどうかの判定器も作る(すでにあるモデルには pmvc.tts pauses)
uv run python -m pmvc.tts train out/hfcm --name tts $V
#   学習データを減らしたモデル(データが足りないとどう間違えるか)
uv run python -m pmvc.tts train out/hfcm --name tts2000 --limit 2000 --epochs 100 --every 10 $V
uv run python -m pmvc.tts train out/hfcm --name tts200 --limit 200 --epochs 300 --every 25 $V
uv run python -m pmvc.tts train out/hfcm --name tts50 --limit 50 --epochs 600 --every 50 $V
uv run python -m pmvc.tts train out/hfcm --name tts20 --limit 20 --epochs 1000 --every 100 $V
#   テストの 100 文: 音素の長さもモデルが決める tts と、長さだけ録音に合わせる ttsd(録音と同じ時間軸になるので、エンコーダと同じ物差しで比べられる)
IDS=$(ls out/ja/meta | grep hjm | sed 's/.json//' | paste -sd, -)
uv run python -m pmvc.align write out/ja --models out/hfcm/models --ids $IDS
uv run python -m pmvc.tts speak out/ja --models out/hfcm/models --name tts --ids $IDS --out ttsd --timing recording
for n in tts tts2000 tts200 tts50 tts20; do
  uv run python -m pmvc.tts speak out/ja --models out/hfcm/models --name $n --ids $IDS --out $n
done
uv run python -m pmvc.judge out/ja --variants enci,ttsd,tts,tts2000,tts200,tts50,tts20 --ids $IDS --name results_tts
uv run python -m pmvc.natural out/ja --name results_tts
#   録音のない文(docs/sentences_ja.txt)
uv run python -m pmvc.tts speak out/say --models out/hfcm/models --name tts --sentences docs/sentences_ja.txt --out tts
for n in tts2000 tts200 tts50 tts20; do
  uv run python -m pmvc.tts speak out/say --models out/hfcm/models --name $n --out $n
done
uv run python -m pmvc.judge out/say --variants tts,tts2000,tts200,tts50,tts20
uv run python -m pmvc.natural out/say
#   1 文だけ鳴らす
uv run python -m pmvc.tts say "こんにちは。今日はいい天気ですね。" --out say.wav

# 10. 複数話者の TTS: JVS の 100 話者(女性 51・男性 49)。話者ではなく文で分ける
#   全話者が読む 100 文のうち 1〜70 番と、話者ごとに違う 30 文(nonpara30)が学習(9,998 文)、71〜75 番が検証、76〜100 番がテスト
uv run python -m pmvc.jvs --all                       # → out/jvsall
uv run python -m pmvc.encoder encode out/jvsall --models out/transplant/models --name enci --out enci --split all $V --quiet-pauses --no-render
uv run python -m pmvc.align train out/jvsall
uv run python -m pmvc.align write out/jvsall
#   話者ごとにベクトルを 1 つ持ち、声の高さは話者ごとの平均と幅を基準に書く。ほかは手順 9 と同じ
uv run python -m pmvc.tts train out/jvsall --name ttsj --stride 5 $V
uv run python -m pmvc.tts train out/jvsall --name ttsj2000 --limit 2000 --epochs 100 --every 10 --stride 5 $V   # 1 話者あたり 20 文
uv run python -m pmvc.tts train out/jvsall --name ttsj500 --limit 500 --epochs 300 --every 25 --stride 5 $V     # 1 話者あたり 5 文
#   テスト: 手順 6 と同じ 8 話者 × 25 文(out/jvs)。話者の声は学習に出てくるが、文は出てこない
uv run python -m pmvc.align write out/jvs --models out/jvsall/models --split test
uv run python -m pmvc.tts speak out/jvs --models out/jvsall/models --name ttsj --split test --out ttsjd --timing recording
for n in ttsj ttsj2000 ttsj500; do
  uv run python -m pmvc.tts speak out/jvs --models out/jvsall/models --name $n --split test --out $n
done
uv run python -m pmvc.judge out/jvs --variants enci,ttsjd,ttsj,ttsj2000,ttsj500 --split test --name results_tts
uv run python -m pmvc.natural out/jvs --name results_tts
#   誰の声に聞こえるか(話者照合モデルによる判定)。8 話者と、全 100 話者のテスト文で
uv run python -m pmvc.whose out/jvs --variants enci,ttsjd,ttsj,ttsj500 --split test --name whose_tts
uv run python -m pmvc.encoder revoice out/jvsall --of enci --out encir --split test $V     # エンコーダ I の制御列を鳴らす
uv run python -m pmvc.tts speak out/jvsall --name ttsj --split test --out ttsj
uv run python -m pmvc.whose out/jvsall --variants encir,ttsj --split test --name whose_tts
#   録音のない文を 4 人の声で
for sp in jvs093 jvs095 jvs097 jvs100; do
  uv run python -m pmvc.tts speak out/say --models out/jvsall/models --name ttsj --speaker $sp --out ttsj_$sp
done
uv run python -m pmvc.judge out/say --variants tts,tts2000,tts200,tts50,tts20,tts2000n,ttsj_jvs093,ttsj_jvs095,ttsj_jvs097,ttsj_jvs100 --add
uv run python -m pmvc.natural out/say

# 11. 対照実験: 手順 9 と同じモデルに、ニューラルデコーダの潜在表現を書かせる(制御列のおかげか、モデルの形のおかげか)
#   mel: メルスペクトログラム + UnivNet ボコーダ、encodec: EnCodec の量子化前の潜在表現、mimi: Mimi の 32 コードブックが表すベクトル
#   3 つとも transformers の公開モデルをそのまま使う(24 kHz)。対照には FastSpeech 2 のピッチ予測器が付く
for L in mel encodec mimi; do
  uv run python -m pmvc.control copy out/ja --latent $L --ids $IDS       # デコーダの上限: 録音 → 潜在表現 → 音
  uv run python -m pmvc.control train out/hfcm --latent $L --name ctl$L
  uv run python -m pmvc.control train out/hfcm --latent $L --name ctl${L}2000 --limit 2000 --epochs 100 --every 10
  uv run python -m pmvc.control train out/hfcm --latent $L --name ctl${L}200 --limit 200 --epochs 300 --every 25
  uv run python -m pmvc.control train out/hfcm --latent $L --name ctl${L}50 --limit 50 --epochs 600 --every 50
  uv run python -m pmvc.control train out/hfcm --latent $L --name ctl${L}20 --limit 20 --epochs 1000 --every 100
done
#   物理モデルの側は、誤差を揃えるために代役なしで学習し直す(--closed 0)。2,000 文は手順 9 の tts2000n
uv run python -m pmvc.tts train out/hfcm --name tts200n --limit 200 --epochs 300 --every 25 --closed 0 $V
uv run python -m pmvc.tts train out/hfcm --name tts50n --limit 50 --epochs 600 --every 50 --closed 0 $V
uv run python -m pmvc.tts train out/hfcm --name tts20n --limit 20 --epochs 1000 --every 100 --closed 0 $V
#   テストの 100 文と、録音のない 12 文
C="ctlmel ctlmel2000 ctlmel200 ctlmel50 ctlmel20 ctlencodec ctlencodec2000 ctlencodec200 ctlencodec50 ctlencodec20 ctlmimi ctlmimi2000 ctlmimi200 ctlmimi50 ctlmimi20"
for n in $C; do
  uv run python -m pmvc.control speak out/ja --models out/hfcm/models --name $n --ids $IDS
  uv run python -m pmvc.control speak out/say --models out/hfcm/models --name $n
done
for n in tts200n tts50n tts20n; do
  uv run python -m pmvc.tts speak out/ja --models out/hfcm/models --name $n --ids $IDS --out $n
  uv run python -m pmvc.tts speak out/say --models out/hfcm/models --name $n --out $n
done
N="tts,tts2000n,tts200n,tts50n,tts20n,$(echo $C | tr ' ' ',')"
uv run python -m pmvc.judge out/ja --variants enci,copy_mel,copy_encodec,copy_mimi,$N --ids $IDS --name results_ctl
uv run python -m pmvc.natural out/ja --name results_ctl
uv run python -m pmvc.judge out/say --variants $N --name results_ctl
uv run python -m pmvc.natural out/say --name results_ctl
#   対照を強くする: メルのボコーダ(UnivNet)を、モデルが書いたスペクトログラムで調整する(この分野の通常の手順)
#   UnivNet の学習に使われた識別器が要る。transformers 版には入っていないので、作者が公開している学習チェックポイント
#   (github.com/maum-ai/univnet の UnivNet-c32、Google Drive)から読む。生成器は transformers 版と同じもの
mkdir -p out/univnet && uvx gdown 1QZFprpvYEhLWCDF90gSl6Dpn0gonS_Rv -O out/univnet/univ_c32.pt
tune() {  # tune モデル ステップ数 確認の間隔: 学習に使った録音のスペクトログラムを録音の音素長とピッチで書かせ、それでボコーダを調整する
  uv run python -m pmvc.control write out/hfcm --name $1 --folds 0 --out ${1}p
  uv run python -m pmvc.vocoder tune out/hfcm --model $1 --written ${1}p --name ${1}vp --steps $2 --every $3 --quality 30
  uv run python -m pmvc.control speak out/ja --models out/hfcm/models --name $1 --vocoder ${1}vp --out ${1}vp --ids $IDS
  uv run python -m pmvc.control speak out/say --models out/hfcm/models --name $1 --vocoder ${1}vp --out ${1}vp
}
tune ctlmel20 1500 250; tune ctlmel50 2000 250; tune ctlmel200 3000 250; tune ctlmel2000 6000 500; tune ctlmel 15000 1000
uv run python -m pmvc.control copy out/ja --latent mel --vocoder ctlmelvp --models out/hfcm/models --out copy_melvp --ids $IDS   # 調整したボコーダの上限
V2="ctlmelvp,ctlmel2000vp,ctlmel200vp,ctlmel50vp,ctlmel20vp"
uv run python -m pmvc.judge out/ja --variants copy_melvp,$V2 --ids $IDS --name results_ctl --add
uv run python -m pmvc.natural out/ja --name results_ctl
uv run python -m pmvc.judge out/say --variants $V2 --name results_ctl --add
uv run python -m pmvc.natural out/say --name results_ctl
#   各録音のスペクトログラムを、その録音を学習していないモデルに書かせる版も試した(良くならなかった):
#   pmvc.control train ... --name ctlmel200f$i --fold $i/5 を i = 0..4 で学習し、pmvc.control write --folds 5、pmvc.vocoder tune

# 12. 話者性: 話者ごとに体格(と声門の設定)を当てはめる。制御列は変えず、鳴らすときの設定だけを変える
#   学習用の文(話者ごとに 6 文の最初の 4 秒)を候補の設定で鳴らし、話者照合モデルが「その話者の録音に近い」と判定する設定を探す
uv run python -m pmvc.speaker fit out/jvsall --only female,child --evals 40 --name speakers_body    # 体格の 2 つの値だけ
uv run python -m pmvc.speaker fit out/jvsall --name speakers                                        # 体格 + 声門の開放率・明るさ・張り
#   さらに 11 個の値: 体格の 4 つの値を直接(成人男性より大きい声道も許す)、声門の 3 つ、構えの癖 4 つ
#   (喉頭の上の狭めと、喉頭の高さ・鼻への通路・舌の根のずれ。エンコーダの制御列に全フレームで足す)。前の当てはめの値から始める
uv run python -m pmvc.speaker fit out/jvsall --name speakers_more --start out/jvsall/speakers.json --evals 200 \
  --only scale,pharynx,tongueRadius,lipWidth,openQuotient,brightness,tension,epilarynx,larynxShift,velumShift,rootShift
#   テストの文(当てはめに使っていない)を、当てはめた設定で鳴らし直す。8 話者(out/jvs)と全 100 話者(out/jvsall)
for d in out/jvs out/jvsall; do
  e=enci; [ $d = out/jvsall ] && e=encir
  uv run python -m pmvc.encoder revoice $d --of $e --out encib --split test $V --speakers out/jvsall/speakers_body.json
  uv run python -m pmvc.encoder revoice $d --of $e --out encis --split test $V --speakers out/jvsall/speakers.json
  uv run python -m pmvc.encoder revoice $d --of ttsj --out ttsjb --split test $V --speakers out/jvsall/speakers_body.json
  uv run python -m pmvc.encoder revoice $d --of ttsj --out ttsjs --split test $V --speakers out/jvsall/speakers.json
  uv run python -m pmvc.encoder revoice $d --of $e --out encit --split test $V --speakers out/jvsall/speakers_more.json
  uv run python -m pmvc.encoder revoice $d --of ttsj --out ttsjt --split test $V --speakers out/jvsall/speakers_more.json
done
#   誰の声に聞こえるか。当てはめに使った照合モデル(WavLM)と、使っていない照合モデル(UniSpeech-SAT)で
for m in "whose_spk" "whose_spk2 --model microsoft/unispeech-sat-base-plus-sv"; do
  uv run python -m pmvc.whose out/jvs --variants enci,ttsj,encis,ttsjs,encib,ttsjb,encit,ttsjt --split test --name $m
done
for m in "whose_speakers" "whose_speakers2 --model microsoft/unispeech-sat-base-plus-sv"; do
  uv run python -m pmvc.whose out/jvsall --variants encir,encib,encis,encit,ttsj,ttsjb,ttsjs,ttsjt --name $m
done
#   聞き取りやすさと音の質が変わっていないか
uv run python -m pmvc.judge out/jvs --variants enci,ttsj,encis,ttsjs,encib,ttsjb,encit,ttsjt --split test --name results_spk
uv run python -m pmvc.natural out/jvs --name results_spk
#   女性の鼻音(試聴で「鼻が詰まったように聞こえる」との指摘): 鼻腔に付く副鼻腔の容積を CHIMERA の 14 cm³ から 24 cm³ にして鳴らし直す(--sinus)
for o in "encin" "encibn --speakers out/jvsall/speakers_body.json" "encisn --speakers out/jvsall/speakers.json" "encitn --speakers out/jvsall/speakers_more.json"; do
  uv run python -m pmvc.encoder revoice out/jvs --of enci --out $o --split test $V --sinus 24
done
for m in "whose_nose" "whose_nose2 --model microsoft/unispeech-sat-base-plus-sv"; do   # 結果には女性と男性に分けた値も入る
  uv run python -m pmvc.whose out/jvs --variants enci,encin,encib,encibn,encis,encisn,encit,encitn --split test --name $m
done
#   鼻音と母音の倍音を、録音と合成音で音素の種類ごとに比べる(鼻音の強さ、鼻音とその隣の母音・離れた母音での倍音の強さ)
uv run python -m pmvc.nasals out/jvs --variants enci,encin,encib,encibn,encis,encisn,encit,encitn --split test
uv run python -m pmvc.judge out/jvs --variants enci,ttsj,encis,ttsjs,encib,ttsjb,encit,ttsjt,encisn,encibn --split test --name results_spk --add
uv run python -m pmvc.natural out/jvs --name results_spk
#   女性の声の明るさ(試聴で「華やかさが足りない」「声門を当てはめるとブザーのよう」との指摘): 女性の母音は 1.6〜5.5 kHz が録音より 7〜12 dB 弱い
#   声門の閉じ方の鋭さを、話者ごとに録音のスペクトルから測る(照合モデルは使わない。学習用の 6 文を鋭さ 6 通りで鳴らし、有声区間の 1.6〜5.5 kHz が録音と同じ強さになる値)
uv run python -m pmvc.speaker closure out/jvsall --start out/jvsall/speakers_body.json --sex F --name speakers_closure               # 測った値
uv run python -m pmvc.speaker closure out/jvsall --start out/jvsall/speakers_body.json --sex F --share 0.5 --name speakers_closure_half  # その半分
uv run python -m pmvc.speaker closure out/jvsall --start out/jvsall/speakers_body.json --sex F --value 0.3 --name speakers_closure3     # 全員 0.3
for o in "encibh --speakers out/jvsall/speakers_closure_half.json" "encibf --speakers out/jvsall/speakers_closure.json" "encibk --speakers out/jvsall/speakers_closure3.json"; do
  uv run python -m pmvc.encoder revoice out/jvs --of enci --out $o --split test $V --sinus 24
done
#   母音と摩擦音の強さを帯域ごとに録音と比べる(倍音の釣り合い、各帯域の強さ、帯域ごとの倍音と雑音の比)
uv run python -m pmvc.bands out/jvs --variants enci,encib,encibn,encibh,encibf,encibk,encis,encit --split test
uv run python -m pmvc.nasals out/jvs --variants enci,encibn,encibh,encibf,encibk --split test --name nasals_closure
for m in "whose_closure" "whose_closure2 --model microsoft/unispeech-sat-base-plus-sv"; do
  uv run python -m pmvc.whose out/jvs --variants enci,encib,encibn,encibh,encibf,encibk,encis,encit --split test --name $m
done
uv run python -m pmvc.judge out/jvs --variants enci,ttsj,encis,ttsjs,encib,ttsjb,encit,ttsjt,encisn,encibn,encibk,encibf,encibh --split test --name results_spk --add
uv run python -m pmvc.natural out/jvs --name results_spk
#   全 100 話者でも同じ鳴らし直しと照合(女性と男性に分けた値が結果に入る)
for o in "encibn --speakers out/jvsall/speakers_body.json" "encibf --speakers out/jvsall/speakers_closure.json" "encibk --speakers out/jvsall/speakers_closure3.json"; do
  uv run python -m pmvc.encoder revoice out/jvsall --of encir --out $o --split test $V --sinus 24
done
for m in "whose_closure" "whose_closure2 --model microsoft/unispeech-sat-base-plus-sv"; do
  uv run python -m pmvc.whose out/jvsall --variants encir,encib,encibn,encibk,encibf,encis,encit --name $m
done

# 13. 公開(V1)に向けて: 公開できる材料だけでエンコーダを学習し直す(エンコーダ J)
#   エンコーダ I は公開できない: 入口と物差しの mHuBERT-147 が CC BY-NC-SA 4.0、録音の JVS が研究用で再配布不可
#   声: 話者ごとの当てはめの代わりに、2 つの基準の声(voices/v1.json)。男性は低い声、女性は高い声(声の高さでは選ばない)
#   低い声 = CHIMERA の成人男性 + 喉頭の上の狭め 0.78 + 舌の根を前へ 0.35 cm。高い声 = 調整 2 + 女性の平均の体格 + 副鼻腔 24 cm³ + 閉じ方の鋭さ 0.18
V="--voice voices/v1.json --no-wall-air --closed-area 1e-6"
uv run python -m pmvc.encoder revoice out/jvs --of enci --out enciv --split test $V        # I の制御列を基準の声で(舌の根の構えも足す)
uv run python -m pmvc.bands out/jvs --variants enci,encibh,encit,encim,enciv --name bands_v1
uv run python -m pmvc.whose out/jvs --variants enci,encibh,encit,encim,enciv --split test --name whose_v1
uv run python -m pmvc.judge out/jvs --variants enci,encibh,encit,enciv --split test --name results_v1 && uv run python -m pmvc.natural out/jvs --name results_v1
#   物差しの候補(重みのライセンスが Apache-2.0 か MIT のもの)を手順 6 と同じ比べ方で
for m in "xlsr:facebook/wav2vec2-xls-r-300m:6,8,10,12,16" "hubert:facebook/hubert-base-ls960:6,8,9,10,12" "wavlm:microsoft/wavlm-base-plus:6,8,9,10,12"; do
  uv run python -m pmvc.yardstick --name yardstick_${m%%:*} --models "${m#*:}"
done
#   録音: Emilia-YODAS(CC BY 4.0)の日本語と英語から 324 話者 × 6 文ずつ(学習 300、検証 12、テスト 12 話者)。性別の表示がないので推定する
E=/path/to/Emilia-Dataset/Emilia-YODAS
uv run python -m pmvc.emilia --tars $E/JA --lang ja --shards 6        # → out/emja
uv run python -m pmvc.emilia --tars $E/EN --lang en --shards 4        # → out/emen
uv run python -m pmvc.sexes out/emja && uv run python -m pmvc.sexes out/emen     # 声の高さで決まらない話者は、照合モデルのベクトルで男女どちらに近いか
uv run python -m pmvc.sexes out/jvs out/transplant --check                       # 性別のわかっている話者での当たり具合
#   規則だけで学習したエンコーダの制御列を手本にし、すべて基準の声で鳴らし直す
for d in out/emja out/emen; do uv run python -m pmvc.encoder encode $d --models out/transplant/models --name sim --out sim --split all --no-render; done
uv run python -m pmvc.encoder revoice --of warp --out warpv $V
for d in out/transplant out/jvs out/emja out/emen; do uv run python -m pmvc.encoder revoice $d --of sim --out simv $V; done
X="--listener facebook/wav2vec2-xls-r-300m --layer 16"
T="--target simv --synthetic simv --tune 6 --silences $V --quiet-pauses"
#   I′: I と同じ材料(TSP + JVS、mHuBERT)で、基準の声を通して学習し直す(声を替えた分だけを見る)
uv run python -m pmvc.standin --name standin_vm --variants warpv,simv --also out/jvs --voiced $M
uv run python -m pmvc.encoder train --name encvm --standin standin_vm --also out/jvs --ssl utter-project/mHuBERT-147 $T
#   J0: 物差しを XLS-R に、入口を wav2vec 2.0 base に替える(録音は TSP + JVS のまま)
uv run python -m pmvc.standin --name standin_vj --variants warpv,simv --also out/jvs --voiced $X
uv run python -m pmvc.encoder train --name encvj --standin standin_vj --also out/jvs $T
#   J: 録音も TSP + Emilia-YODAS に替える。公開できる材料だけ
uv run python -m pmvc.standin --name standin_v --variants warpv,simv --also out/emja,out/emen --voiced $X
uv run python -m pmvc.encoder train --name encv --standin standin_v --also out/emja,out/emen $T
#   これまでと同じ 3 つのテストで比べる
for t in "${T3[@]}"; do    # T3=("out/jvs --split test" "out/ja --split all" "out/transplant --split test")
  uv run python -m pmvc.encoder revoice ${t%% *} --of enci --out enciv --split ${t##* } $V
  for m in encvm encvj encv; do uv run python -m pmvc.encoder encode $t --models out/transplant/models --name $m --out $m $V --quiet-pauses; done
  uv run python -m pmvc.judge $t --variants enci,enciv,encvm,encvj,encv --name results_rel && uv run python -m pmvc.natural ${t%% *} --name results_rel
done
#   声のつまみ(基準の声の明るさ、声の高さ、もう一方の声): 鳴らすときの設定だけを変える
for o in "encivd --bright -1" "encivu --bright 1" "encivlo --pitch -3" "encivhi --pitch 3" "encivL --share 0 --pitch -10" "encivH --share 1 --pitch 9"; do
  uv run python -m pmvc.encoder revoice out/jvs --of enci --out ${o%% *} --split test $V ${o#* }
done
uv run python -m pmvc.bands out/jvs --variants enciv,encivd,encivu --name bands_dial
uv run python -m pmvc.judge out/jvs --variants enciv,encivd,encivu,encivlo,encivhi,encivL,encivH --split test --name results_dial && uv run python -m pmvc.natural out/jvs --name results_dial
#   公開できる録音だけの TTS(最初の試み): Emilia-YODAS の日本語から 800 話者 × 12 文。J の制御列、基準の声
uv run python -m pmvc.emilia --tars $E/JA --lang ja --out out/emjatts --shards 30 --speakers 800 --per 12 --held 0 --valid-clips 1 --draw 2
uv run python -m pmvc.sexes out/emjatts --also out/emja
uv run python -m pmvc.encoder encode out/emjatts --models out/transplant/models --name encv --out encv --split all $V --quiet-pauses --no-render
uv run python -m pmvc.align train out/emjatts --epochs 40 --hinted 12 --sharpen 16     # 話し言葉と自動書き起こしでは、既定の 12 エポックでは対応付けが見つからない
uv run python -m pmvc.align write out/emjatts
uv run python -m pmvc.align prune out/emjatts                                           # 書き起こしと音が合っていない録音を外す(9,600 → 5,356 文)
uv run python -m pmvc.tts train out/emjatts --name ttse --tracks encv --standin standin_v --stride 8 $V
#   JVS のテスト文を、学習した話者のうち男性 1 人・女性 1 人の声で読ませる
uv run python -m pmvc.tts speak out/jvs --models out/emjatts/models --name ttse --split test --out ttse_m --speaker JA_BrOouMZCkBA_SPEAKER_00
uv run python -m pmvc.tts speak out/jvs --models out/emjatts/models --name ttse --split test --out ttse_f --speaker JA_yD5DCCvSJSI_SPEAKER_00
uv run python -m pmvc.judge out/jvs --variants ttsj,ttse_m,ttse_f --split test --name results_ttse && uv run python -m pmvc.natural out/jvs --name results_ttse
#   話す速さ: Emilia-YODAS の話者は読み上げより速い(男性の声で 1 秒に 13.3 音素、JVS の録音は 10.2)。--rate で揃える
uv run python -m pmvc.tts speak out/jvs --models out/emjatts/models --name ttse --split test --out ttse_mr --speaker JA_BrOouMZCkBA_SPEAKER_00 --rate 10.2
uv run python -m pmvc.tts speak out/jvs --models out/emjatts/models --name ttse --split test --out ttse_fr --speaker JA_yD5DCCvSJSI_SPEAKER_00 --rate 10.2
uv run python -m pmvc.judge out/jvs --variants ttsj,ttse_m,ttse_f,ttse_mr,ttse_fr --split test --name results_ttse --add
#   J の損(物差しと入口の入れ替え)を取り返す候補: J0 と同じ録音(TSP + JVS)で 3 つ。JVS と TSP で比べる(results_listen.json)
uv run python -m pmvc.encoder train --name encvxf --standin standin_vj --also out/jvs --ssl facebook/wav2vec2-xls-r-300m $T                   # 入口を XLS-R に
uv run python -m pmvc.standin --name standin_vx12 --variants warpv,simv --also out/jvs --voiced --listener facebook/wav2vec2-xls-r-300m --layer 12
uv run python -m pmvc.encoder train --name encvx12 --standin standin_vx12 --also out/jvs $T                                                   # 物差しを XLS-R の第 12 層に
uv run python -m pmvc.standin --name standin_vt --variants warpv,simv --also out/jvs --voiced --listener sarulab-speech/hubert-base-jtube --layer 8
uv run python -m pmvc.encoder train --name encvt --standin standin_vt --also out/jvs $T                                                       # 物差しを日本語の HuBERT(MIT)に: Whisper の誤りが I' の水準に戻る
#   J2: その物差しで、録音は TSP + Emilia-YODAS
uv run python -m pmvc.standin --name standin_vte --variants warpv,simv --also out/emja,out/emen --voiced --listener sarulab-speech/hubert-base-jtube --layer 8
uv run python -m pmvc.encoder train --name encv2 --standin standin_vte --also out/emja,out/emen $T
#   読み上げの録音での TTS: Common Voice の日本語(CC0)。性別を申告している話者から女性 300 人・男性 300 人、16 文ずつ
uv run python -m pmvc.commonvoice --corpus /path/to/cv-corpus-27.0-2026-09-11/ja      # → out/cvja
uv run python -m pmvc.encoder encode out/cvja --models out/transplant/models --name encv --out encv --split all $V --quiet-pauses --no-render
uv run python -m pmvc.align train out/cvja --epochs 40 --hinted 12 --sharpen 16 && uv run python -m pmvc.align write out/cvja && uv run python -m pmvc.align prune out/cvja
uv run python -m pmvc.tts train out/cvja --name ttsc --tracks encv --standin standin_v --stride 8 $V

# 14. 試聴ページ(音声埋め込みの HTML 3 枚: 英語編 listening.html、日本語編 listening_ja.html、TTS 編 listening_tts.html)
uv run python -m pmvc.page --notes notes.html --notes-ja notes_ja.html --notes-tts notes_tts.html

# 15. テクニカルレポート(下書き。英語と日本語の PDF)。tectonic が必要
bash docs/report/build.sh
```

エンコーダの学習と評価は話者で分けています。TSP では同じ文字の女性・男性話者が同じ文リストを読むので、FK・MK・FL・ML をテスト、FJ・MJ を検証に取り置くと、その話者の文も学習に出てきません。
学習は GPU があると速く(RTX 5090 で A が 3 分、代役が 4〜5 分、C′ が 12 分ほど)、探索は CPU を使います(16 コアの Mac で 1 時間に約 270 文)。
日本語を加えた D・F・G は、同じ GPU で 1 本 25〜40 分でした。手順 8 は、鳴らし直しが 7 分(3,864 本)、代役が 3 分、I が 22 分でした。手順 9 は、録音の取り込みが 1 分、エンコーダ I の制御列が 7 分(18,755 文)、音素の対応付けが学習 4 分と書き出し 4 分、TTS の学習が 1 エポック約 1 分(60 エポック)でした。合成は、4 秒の文でテキストから制御列までが CPU で 17 ms、制御列から音声までが CHIMERA の 1 スレッドで 0.5 秒です。手順 10 は、取り込みが 1 分、制御列が 5 分(12,997 文)、音素の対応付けが学習 4 分と書き出し 3 分、TTS の学習が約 55 分(60 エポック)でした。JVS も話者と文の両方で分けていて、テストの 8 話者と 25 文は学習にも検証にも出てきません。
声の設定の調整は CPU で 15 分ほどです。手順 7 の最初の `--start` は、設定を 1 つずつ動かして見つけた柔らかい発声の値を、狭い範囲で一度当てはめた結果です。クリック音(`pmvc.clicks`)を減点に入れて当てはめ直すこともできますが(`--clicks`、既定 0.3)、開放率を下げて振幅を倍にする、息の雑音を増やして紛らわせる、といった別の逃げ道に行くので、クリック音に効く 4 つの設定は指標を見ながら手で決めています。

`pmvc.calibrate` は雑音源のゲインと声門音源の設定を実録音に合わせて調整するものですが、未使用のデータでの改善はわずかだったため、コピー合成では既定で使いません(使うときは `--calibrated`)。

各スクリプトの条件や指標の定義は、それぞれのファイル冒頭の docstring にあります。
