# Running the experiments

[日本語](experiments_ja.md)

How to run the experiments of the technical report ([English](report/physical_modeling_voice_codec_report_en.pdf), [日本語](report/physical_modeling_voice_codec_report_ja.pdf)) from the start.
To run the demo alone, the steps in the [README](../README.md) are enough. Every command is run from the root of the repository.

## Setup

```sh
git submodule update --init
cargo build --release          # target/release/pmvc-synth
cargo build --release -p chimera-cli --manifest-path third-party/chimera-vocal-synth/Cargo.toml
uv sync                        # this also builds pmvc-engine (needs Rust)
```

The experiments use the TSP speech database (48 kHz, 254 MB, Simplified BSD) in `~/datasets/tsp`.
The submodule's `tools/tsp/fetch.sh` calls `sha256sum`, so on macOS do this instead:

```sh
mkdir -p ~/datasets/tsp && cd ~/datasets/tsp
curl -fL -C - -o 48k.zip https://www.mmsp.ece.mcgill.ca/Documents/Data/TSP-Speech-Database/48k.zip
shasum -a 256 48k.zip   # 9cfb3a3a13014c8ff90770a5d1923f376da73ac927b9100f09826f60cf06cf43
unzip -qo 48k.zip
```

If it is somewhere else, name the place in the environment variable `TSP_DIR`.
ffmpeg is needed too (it converts the audio of the listening pages).

From step 5 on other recordings are used. Unpack the `eval` audio and text of Hi-Fi-CAPTAIN (a woman and a man in Japanese and in English, CC BY-NC-SA 4.0) into `~/datasets/hfc/hi-fi-captain/`, and JSUT's `basic5000` into `~/datasets/jsut/jsut_ver1.1/`. A hundred sentences of each are used. Step 7 also uses the `dev` set of Hi-Fi-CAPTAIN's Japanese woman.
Step 6 uses `parallel100` of JVS (100 speakers, not to be redistributed) in `~/datasets/jvs/jvs_ver1/`.

## The steps

Everything is written under `out/`, which git does not track.

```sh
# 1. Prosody transplant: CHIMERA says the same sentences with the recordings' phonemes, durations and F0 (200 sentences)
uv run python -m pmvc.transplant
uv run python -m pmvc.evaluate

# 2. Perturbation test: noise, quantization and thinning on the control track (100 Hz × 21 channels), which is then rendered (40 sentences)
uv run python -m pmvc.perturb
uv run python -m pmvc.evaluate --against drive --name perturb_results \
  --variants "$(uv run python -c "from pmvc.perturb import conditions; print(','.join(c for c in conditions() if c != 'drive'))")"

# 3. Copy synthesis: a control track is optimized to match a recording
uv run python -m pmvc.warp                      # where it starts: the rules' pronunciation stretched onto the recording's time axis
uv run python -m pmvc.copysynth                 # local search (40 sentences, 1 to 2 minutes each; this one uses the transcript too)
uv run python -m pmvc.evaluate --variants rule,f0,warp,copy --aligned warp,copy --name results_copy
uv run python -m pmvc.second_judge --name results_copy   # a second judge (Whisper large-v3-turbo)

# 4. Encoders (the codec's first form): recording → control track → sound. The sentences' transcripts are not used
uv run python -m pmvc.transplant --all --resume    # what all 1,372 sentences need first (recordings, F0, the rule synthesis)
uv run python -m pmvc.warp --resume
#   A: taught with nothing but pairs of CHIMERA's own renders and their control tracks; it then encodes every recording
uv run python -m pmvc.encoder train --name sim --no-real --synthetic warp --epochs 30 --every 3 --stride 4
uv run python -m pmvc.encoder encode --name sim --out sim --split all
#   C′: a stand-in for the voice (control track → what a listener hears of its render) is taught with renders, and through it the encoder is taught with recordings, in a closed loop
uv run python -m pmvc.standin --name standin0 --variants warp,sim
uv run python -m pmvc.encoder train --name encn --target sim --synthetic sim --tune 6 --standin standin0
#   For comparison. A search: A's tracks are moved towards the recordings (1 to 2 minutes a sentence; --shard K/N runs several side by side)
uv run python -m pmvc.copysynth --all --base sim --nll-weight 0 --iters 20 --out simcopy --resume
#   B: taught to copy the searched tracks. C: with the closed-loop error through the stand-in besides
uv run python -m pmvc.encoder train --name encb --tune 6
uv run python -m pmvc.standin
uv run python -m pmvc.encoder train --name enc --tune 6 --standin standin
#   Evaluate on the test speakers
for m in encb enc encn; do uv run python -m pmvc.encoder encode --name $m --out $m --split test; done
uv run python -m pmvc.evaluate --variants rule,sim,simcopy,encb,enc,encn --aligned sim,simcopy,encb,enc,encn --name results_codec
uv run python -m pmvc.second_judge --name results_codec
uv run python -m pmvc.heard --name results_codec    # listener distance (what the search lowers)
#   Perturbing the encoder's tracks (40 sentences of the test speakers)
uv run python -m pmvc.perturb --base encn --tag ec --ids "$(ls out/transplant/encn | grep tracks.json | sed 's/.tracks.json//' | awk 'NR%6==1' | paste -sd, -)" \
  --conds "$(uv run python -c "from pmvc.perturb import conditions; print(','.join(conditions() + ['coded_5bit_50hz', 'coded_4bit_50hz', 'coded_5bit_25hz', 'coded_4bit_25hz']))")"
uv run python -m pmvc.evaluate --against ec_drive --name perturb_ec_results \
  --variants "$(uv run python -c "import json; print(','.join(c for c in json.load(open('out/transplant/perturb_ec.json'))['conditions'] if c != 'ec_drive'))")"

# 5. Other recordings through the encoders taught with English (TSP): 300 Japanese sentences and 200 English ones. Output goes to out/ja
H=~/datasets/hfc/hi-fi-captain; J=~/datasets/jsut/jsut_ver1.1/basic5000
uv run python -m pmvc.recordings --set hjf --lang ja --sex F --wavs $H/ja-JP/female/wav/eval --text $H/ja-JP/female/text/eval.txt
uv run python -m pmvc.recordings --set hjm --lang ja --sex M --wavs $H/ja-JP/male/wav/eval --text $H/ja-JP/male/text/eval.txt
uv run python -m pmvc.recordings --set jsut --lang ja --sex F --wavs $J/wav --text $J/transcript_utf8.txt
uv run python -m pmvc.recordings --set hef --lang en --sex F --wavs $H/en-US/female/wav/eval --text $H/en-US/female/text/eval.txt
uv run python -m pmvc.recordings --set hem --lang en --sex M --wavs $H/en-US/male/wav/eval --text $H/en-US/male/text/eval.txt
for m in sim enc encn; do uv run python -m pmvc.encoder encode out/ja --models out/transplant/models --name $m --out $m --split all; done
uv run python -m pmvc.judge                                                   # two Whispers, the phone recognizer, listener distance
uv run python -m pmvc.judge out/transplant --split test --name results_judge  # TSP's test speakers by the same yardsticks
#   A diagnosis: C′'s tracks are moved towards the recordings by a search that uses no transcript (24 Japanese sentences)
IDS=$(for s in hjf hjm jsut; do for k in 005 017 029 041 053 065 077 089; do printf "%s_%s," $s $k; done; done | sed 's/,$//')
uv run python -m pmvc.copysynth out/ja --ids $IDS --base encn --nll-weight 0 --iters 20 --out encncopy --scale out/transplant/warp
uv run python -m pmvc.judge --ids $IDS --variants encn,encncopy --name results_search

# 6. Japanese recordings in the closed loop (JVS: 86 speakers × 10 sentences to teach with, 6 speakers for validation, 8 speakers × 25 sentences for the test)
uv run python -m pmvc.jvs                                  # lays the recordings out in out/jvs
uv run python -m pmvc.encoder encode out/jvs --models out/transplant/models --name sim --out sim --split all
uv run python -m pmvc.yardstick                            # candidates for the yardstick (the listener), compared on two people's recordings of the same sentence
M="--listener utter-project/mHuBERT-147 --layer 8"         # the yardstick chosen: layer 8 of the multilingual mHuBERT-147
#   D: Japanese recordings are added (the yardstick is still the English wav2vec2)
uv run python -m pmvc.standin --name standin_j --variants warp,sim --also out/jvs
uv run python -m pmvc.encoder train --name encd --target sim --synthetic sim --tune 6 --standin standin_j --also out/jvs
#   E: the recordings stay English, the yardstick is changed
uv run python -m pmvc.standin --name standin_m0 --variants warp,sim $M
uv run python -m pmvc.encoder train --name ence --target sim --synthetic sim --tune 6 --standin standin_m0
#   F: both
uv run python -m pmvc.standin --name standin_m --variants warp,sim --also out/jvs $M
uv run python -m pmvc.encoder train --name encf --target sim --synthetic sim --tune 6 --standin standin_m --also out/jvs
#   G: as F, and the front end through which the encoder hears the recording is mHuBERT too
uv run python -m pmvc.encoder train --name encg --target sim --synthetic sim --tune 6 --standin standin_m --also out/jvs --ssl utter-project/mHuBERT-147
#   Evaluate on three tests (JVS's unused speakers, the recordings of step 5, TSP's test speakers)
for m in encn encd ence encf encg; do
  uv run python -m pmvc.encoder encode out/jvs --models out/transplant/models --name $m --out $m --split test
  uv run python -m pmvc.encoder encode out/ja --models out/transplant/models --name $m --out $m --split all
  uv run python -m pmvc.encoder encode --name $m --out $m --split test
done
uv run python -m pmvc.judge out/jvs --split test --variants encn,encd,ence,encf,encg
uv run python -m pmvc.judge out/ja --variants encn,encd,ence,encf,encg --name results_taught
uv run python -m pmvc.judge out/transplant --split test --variants encn,encd,ence,encf,encg --name results_taught

# 7. The stray sounds outside the speech are removed (encoder H), and the settings of the high voice are tuned against NISQA
#   H: made as G is, but outside the speech the reference is "no breath"
uv run python -m pmvc.encoder train --name ench --target sim --synthetic sim --tune 6 --standin standin_m --also out/jvs --ssl utter-project/mHuBERT-147 --silences
T=("out/jvs --split test" "out/ja --split all" "out/transplant --split test")
for t in "${T[@]}"; do uv run python -m pmvc.encoder encode $t --models out/transplant/models --name ench --out ench; done
#   The voice's settings: fitted to 16 of the 100 dev sentences of Hi-Fi-CAPTAIN's woman, and checked on the other 84
uv run python -m pmvc.recordings out/fit --set hjfd --lang ja --sex F --wavs $H/ja-JP/female/wav/dev --text $H/ja-JP/female/text/dev.txt
for m in encg ench; do uv run python -m pmvc.encoder encode out/fit --models out/transplant/models --name $m --out $m --split all; done
#   Adjustment 1: fitted to NISQA alone (--clicks 0). The glottis comes out open for longer, but the noise source at the teeth is 2.2 times stronger too, and clicks appear, at the moment the tongue closes among other places
uv run python -m pmvc.calibrate out/fit --sex F --tracks encg --prefix hjfd --by nisqa --clicks 0 --n 16 --evals 220 --name voice-F1 \
  --only noiseGlottis,noiseConstriction,noiseTeeth,pressure,tension,breathiness,brightness,roughness,openQuotient,amplitudeScale \
  --start '{"tension": 0.0745, "brightness": 0.0264, "breathiness": 0.0465, "roughness": 0.0084, "noiseGlottis": 1.18, "noiseConstriction": 1.55, "noiseTeeth": 1.87, "openQuotient": 0.88}'
#   Which speakers' pitches those settings suit (17 speakers). encode --voice goes by this result and switches the settings by the pitch of the speaker's voice
uv run python -m pmvc.pitchmix --voice out/fit/voice-F1.json $(for s in ML MK FL FK; do echo out/transplant:ench:$s; done) \
  $(for s in hem hef hjm hjf jsut; do echo out/ja:ench:$s; done) $(for n in 093 094 095 096 097 098 099 100; do echo out/jvs:ench:jvs$n; done)
#   Remedy 1: the sounds inside silences (a stop's closure, a pause), which pmvc.clicks.in_silences counts
#   --no-wall-air: the air that the tract's moving walls push aside makes no sound (a public switch of the engine). This is the low sound, below a few hundred Hz, that comes out even with the breath held
#   --quiet-pauses: the breath is held in the recording's pauses (silences of 0.12 s or more inside the speech)
#   Remedy 2: the click at the moment the tongue closes, which pmvc.clicks.extra counts. This is the one that sounds like the tongue being clicked
#   --closed-area: the area under which a constriction counts as closed. At the engine's default (2e-4 cm²) the flow is stopped in one sample while some of it is left, which makes an impulse that reaches 16 kHz
Q="--no-wall-air --quiet-pauses"; C="$Q --closed-area 1e-6"
uv run python -m pmvc.encoder encode out/fit --models out/transplant/models --name ench --out enchq --split all $Q
#   Adjustments 2 and 3: with the glottis opening as in adjustment 1, the vibration's amplitude is 1.4 times larger, and the noise sources are at their defaults (adjustment 2) or, the constriction's and the teeth's only, at 0.6 of them (adjustment 3).
#   The values were set by hand, watching the clicks in the high band (pmvc.clicks.measure); --evals 0 searches nothing, and scores and writes out the settings given by --start
with() { python3 -c "
import json, sys; s = json.load(open('out/fit/voice-F1.json'))['settings']; s.update(noiseGlottis=1, amplitudeScale=1.4)
s.update(noiseConstriction=float(sys.argv[1]), noiseTeeth=float(sys.argv[1])); print(json.dumps(s))" $1; }
uv run python -m pmvc.calibrate out/fit --sex F --tracks enchq --prefix hjfd --by nisqa --n 16 --evals 0 --name voice-F2 --start "$(with 1.0)"
uv run python -m pmvc.calibrate out/fit --sex F --tracks enchq --prefix hjfd --by nisqa --n 16 --evals 0 --name voice-F3 --start "$(with 0.6)"
#   Evaluate on the three tests: H and the three voices, with no remedy, with remedy 1, and with remedies 1 and 2
for t in "${T[@]}"; do
  enc() { uv run python -m pmvc.encoder encode $t --models out/transplant/models --name ench "$@"; }
  enc --out enchs --no-wall-air; enc --out enchq $Q; enc --out enchc $C
  enc --out enchv --voice out/fit/voice-F1.json; enc --out enchvq --voice out/fit/voice-F1.json $Q; enc --out enchvc --voice out/fit/voice-F1.json $C
  enc --out ench2q --voice out/fit/voice-F2.json $Q; enc --out ench2c --voice out/fit/voice-F2.json $C
  enc --out ench3q --voice out/fit/voice-F3.json $Q; enc --out ench3c --voice out/fit/voice-F3.json $C
  uv run python -m pmvc.judge $t --variants encn,encg,ench,enchs,enchq,enchc,enchv,enchvq,enchvc,ench2q,ench2c,ench3q,ench3c --name results_tuned
  uv run python -m pmvc.natural ${t%% *} --name results_tuned       # adds NISQA's scores
done
#   With the tracks made under the remedies, measure again which speakers' pitches adjustment 2 suits (the table on the listening page)
uv run python -m pmvc.pitchmix --voice out/fit/voice-F2.json $(for s in ML MK FL FK; do echo out/transplant:enchc:$s; done) \
  $(for s in hem hef hjm hjf jsut; do echo out/ja:enchc:$s; done) $(for n in 093 094 095 096 097 098 099 100; do echo out/jvs:enchc:jvs$n; done)

# 8. The encoder is taught again through the adjusted voice (encoder I). The adjusted voice: adjustment 2 for speakers with a high voice, and remedies 1 and 2
#   H was taught through CHIMERA's default voice. So the tracks it is taught with are first rendered again by the adjusted voice (the tracks stay as they are), and a stand-in is made for that voice
V="--voice out/fit/voice-F2.json --no-wall-air --closed-area 1e-6"
uv run python -m pmvc.encoder revoice --of warp --out warp2c $V
uv run python -m pmvc.encoder revoice --of sim --out sim2c $V
uv run python -m pmvc.encoder revoice out/jvs --of sim --out sim2c $V
uv run python -m pmvc.standin --name standin_2c --variants warp2c,sim2c --also out/jvs $M --voiced   # --voiced: it also reads each speaker's share of the adjusted voice
#   Made as H is, with the adjusted voice in the stand-in and in the listening that picks the epoch
uv run python -m pmvc.encoder train --name enci --target sim --synthetic sim2c --tune 6 --standin standin_2c --also out/jvs --ssl utter-project/mHuBERT-147 --silences $V --quiet-pauses
for t in "${T[@]}"; do
  uv run python -m pmvc.encoder encode $t --models out/transplant/models --name enci --out enci $V --quiet-pauses
  uv run python -m pmvc.judge $t --variants encn,encg,ench,enchs,enchq,enchc,enchv,enchvq,enchvc,ench2q,ench2c,ench3q,ench3c,enci --name results_tuned --add
  uv run python -m pmvc.natural ${t%% *} --name results_tuned
done

# 9. TTS: text → control track. One speaker, Hi-Fi-CAPTAIN's Japanese man (18,655 sentences, about 24 hours, to teach with; 100 for validation; the test is the 100 sentences of hjm from step 5)
H=~/datasets/hfc/hi-fi-captain/ja-JP/male
uv run python -m pmvc.recordings out/hfcm --set hmt --speaker hjm --lang ja --sex M --wavs $H/wav/train_parallel --text $H/text/train_parallel.txt --n 0 --split train --link
uv run python -m pmvc.recordings out/hfcm --set hmv --speaker hjm --lang ja --sex M --wavs $H/wav/dev --text $H/text/dev.txt --n 0 --split valid --link
#   The targets are encoder I's control tracks (only the tracks are written; nothing is rendered)
uv run python -m pmvc.encoder encode out/hfcm --models out/transplant/models --name enci --out enci --split all $V --quiet-pauses --no-render
#   Which frames of a recording each phoneme takes is learned from the recordings and their readings alone. A pause where the text has no comma gets a pause mark (sp)
uv run python -m pmvc.align train out/hfcm
uv run python -m pmvc.align write out/hfcm
#   Reading and accent (Open JTalk) → each phoneme's duration → control track. The loss is the error against the target tracks, and the distance to the recording as heard through the voice's stand-in
#   At the end of training a classifier is made too, of whether the speaker pauses where the text has no comma (for a model that exists already: pmvc.tts pauses)
uv run python -m pmvc.tts train out/hfcm --name tts $V
#   Models taught with fewer recordings (how does it err when the data is not enough?)
uv run python -m pmvc.tts train out/hfcm --name tts2000 --limit 2000 --epochs 100 --every 10 $V
uv run python -m pmvc.tts train out/hfcm --name tts200 --limit 200 --epochs 300 --every 25 $V
uv run python -m pmvc.tts train out/hfcm --name tts50 --limit 50 --epochs 600 --every 50 $V
uv run python -m pmvc.tts train out/hfcm --name tts20 --limit 20 --epochs 1000 --every 100 $V
#   The 100 test sentences: tts, where the model sets the durations too, and ttsd, where only the durations are the recording's (it then shares the recording's time axis and can be measured by the encoder's yardsticks)
IDS=$(ls out/ja/meta | grep hjm | sed 's/.json//' | paste -sd, -)
uv run python -m pmvc.align write out/ja --models out/hfcm/models --ids $IDS
uv run python -m pmvc.tts speak out/ja --models out/hfcm/models --name tts --ids $IDS --out ttsd --timing recording
for n in tts tts2000 tts200 tts50 tts20; do
  uv run python -m pmvc.tts speak out/ja --models out/hfcm/models --name $n --ids $IDS --out $n
done
uv run python -m pmvc.judge out/ja --variants enci,ttsd,tts,tts2000,tts200,tts50,tts20 --ids $IDS --name results_tts
uv run python -m pmvc.natural out/ja --name results_tts
#   Sentences of which there is no recording (docs/sentences_ja.txt)
uv run python -m pmvc.tts speak out/say --models out/hfcm/models --name tts --sentences docs/sentences_ja.txt --out tts
for n in tts2000 tts200 tts50 tts20; do
  uv run python -m pmvc.tts speak out/say --models out/hfcm/models --name $n --out $n
done
uv run python -m pmvc.judge out/say --variants tts,tts2000,tts200,tts50,tts20
uv run python -m pmvc.natural out/say
#   Say one sentence
uv run python -m pmvc.tts say "こんにちは。今日はいい天気ですね。" --out say.wav

# 10. TTS for many speakers: JVS's 100 speakers (51 women, 49 men). The split is by sentence, not by speaker
#   Of the 100 sentences that everyone reads, numbers 1 to 70 and each speaker's own 30 sentences (nonpara30) are taught with (9,998 sentences), 71 to 75 are for validation and 76 to 100 for the test
uv run python -m pmvc.jvs --all                       # → out/jvsall
uv run python -m pmvc.encoder encode out/jvsall --models out/transplant/models --name enci --out enci --split all $V --quiet-pauses --no-render
uv run python -m pmvc.align train out/jvsall
uv run python -m pmvc.align write out/jvsall
#   The model has one vector for each speaker and writes the pitch against each speaker's own mean and spread. The rest is as in step 9
uv run python -m pmvc.tts train out/jvsall --name ttsj --stride 5 $V
uv run python -m pmvc.tts train out/jvsall --name ttsj2000 --limit 2000 --epochs 100 --every 10 --stride 5 $V   # 20 sentences a speaker
uv run python -m pmvc.tts train out/jvsall --name ttsj500 --limit 500 --epochs 300 --every 25 --stride 5 $V     # 5 sentences a speaker
#   The test: the 8 speakers × 25 sentences of step 6 (out/jvs). The speakers' voices are in the training, the sentences are not
uv run python -m pmvc.align write out/jvs --models out/jvsall/models --split test
uv run python -m pmvc.tts speak out/jvs --models out/jvsall/models --name ttsj --split test --out ttsjd --timing recording
for n in ttsj ttsj2000 ttsj500; do
  uv run python -m pmvc.tts speak out/jvs --models out/jvsall/models --name $n --split test --out $n
done
uv run python -m pmvc.judge out/jvs --variants enci,ttsjd,ttsj,ttsj2000,ttsj500 --split test --name results_tts
uv run python -m pmvc.natural out/jvs --name results_tts
#   Whose voice it sounds like (as a speaker-verification model judges it). On the 8 speakers, and on the test sentences of all 100
uv run python -m pmvc.whose out/jvs --variants enci,ttsjd,ttsj,ttsj500 --split test --name whose_tts
uv run python -m pmvc.encoder revoice out/jvsall --of enci --out encir --split test $V     # renders encoder I's tracks
uv run python -m pmvc.tts speak out/jvsall --name ttsj --split test --out ttsj
uv run python -m pmvc.whose out/jvsall --variants encir,ttsj --split test --name whose_tts
#   Sentences of which there is no recording, in four people's voices
for sp in jvs093 jvs095 jvs097 jvs100; do
  uv run python -m pmvc.tts speak out/say --models out/jvsall/models --name ttsj --speaker $sp --out ttsj_$sp
done
uv run python -m pmvc.judge out/say --variants tts,tts2000,tts200,tts50,tts20,tts2000n,ttsj_jvs093,ttsj_jvs095,ttsj_jvs097,ttsj_jvs100 --add
uv run python -m pmvc.natural out/say

# 11. The control: the model of step 9 writes the latents of neural decoders (is it the control track's doing, or the model's shape?)
#   mel: a mel spectrogram and the UnivNet vocoder; encodec: EnCodec's latent before quantization; mimi: the vectors that Mimi's 32 codebooks stand for
#   All three are public models from transformers, used as they are (24 kHz). The control has FastSpeech 2's pitch predictor
for L in mel encodec mimi; do
  uv run python -m pmvc.control copy out/ja --latent $L --ids $IDS       # the decoder's best: recording → latent → sound
  uv run python -m pmvc.control train out/hfcm --latent $L --name ctl$L
  uv run python -m pmvc.control train out/hfcm --latent $L --name ctl${L}2000 --limit 2000 --epochs 100 --every 10
  uv run python -m pmvc.control train out/hfcm --latent $L --name ctl${L}200 --limit 200 --epochs 300 --every 25
  uv run python -m pmvc.control train out/hfcm --latent $L --name ctl${L}50 --limit 50 --epochs 600 --every 50
  uv run python -m pmvc.control train out/hfcm --latent $L --name ctl${L}20 --limit 20 --epochs 1000 --every 100
done
#   On the physical model's side the models are taught again without the stand-in, so that the losses are alike (--closed 0). For 2,000 sentences that is tts2000n of step 9
uv run python -m pmvc.tts train out/hfcm --name tts200n --limit 200 --epochs 300 --every 25 --closed 0 $V
uv run python -m pmvc.tts train out/hfcm --name tts50n --limit 50 --epochs 600 --every 50 --closed 0 $V
uv run python -m pmvc.tts train out/hfcm --name tts20n --limit 20 --epochs 1000 --every 100 --closed 0 $V
#   The 100 test sentences, and the 12 of which there is no recording
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
#   A stronger control: the spectrogram's vocoder (UnivNet) is tuned on the spectrograms the model wrote (the usual practice in the field)
#   This needs the discriminators UnivNet was trained with. The transformers version does not have them, so they are read from the training checkpoint its authors publish
#   (UnivNet-c32 of github.com/maum-ai/univnet, on Google Drive). The generator is the same as the transformers version's
mkdir -p out/univnet && uvx gdown 1QZFprpvYEhLWCDF90gSl6Dpn0gonS_Rv -O out/univnet/univ_c32.pt
tune() {  # tune model steps interval-between-checks: the model writes the spectrograms of the recordings it was taught with, at the recordings' durations and pitch, and the vocoder is tuned on those
  uv run python -m pmvc.control write out/hfcm --name $1 --folds 0 --out ${1}p
  uv run python -m pmvc.vocoder tune out/hfcm --model $1 --written ${1}p --name ${1}vp --steps $2 --every $3 --quality 30
  uv run python -m pmvc.control speak out/ja --models out/hfcm/models --name $1 --vocoder ${1}vp --out ${1}vp --ids $IDS
  uv run python -m pmvc.control speak out/say --models out/hfcm/models --name $1 --vocoder ${1}vp --out ${1}vp
}
tune ctlmel20 1500 250; tune ctlmel50 2000 250; tune ctlmel200 3000 250; tune ctlmel2000 6000 500; tune ctlmel 15000 1000
uv run python -m pmvc.control copy out/ja --latent mel --vocoder ctlmelvp --models out/hfcm/models --out copy_melvp --ids $IDS   # the tuned vocoder's best
V2="ctlmelvp,ctlmel2000vp,ctlmel200vp,ctlmel50vp,ctlmel20vp"
uv run python -m pmvc.judge out/ja --variants copy_melvp,$V2 --ids $IDS --name results_ctl --add
uv run python -m pmvc.natural out/ja --name results_ctl
uv run python -m pmvc.judge out/say --variants $V2 --name results_ctl --add
uv run python -m pmvc.natural out/say --name results_ctl
#   Also tried, and no better: each recording's spectrogram written by a model that was not taught with that recording:
#   pmvc.control train ... --name ctlmel200f$i --fold $i/5 for i = 0..4, then pmvc.control write --folds 5 and pmvc.vocoder tune

# 12. Who is speaking: a body (and settings of the glottis) is fitted to each speaker. The control tracks are not changed, only the settings they are rendered with
#   Training sentences (the first 4 seconds of 6 sentences a speaker) are rendered with candidate settings, and the search is for the settings a speaker-verification model judges nearest to that speaker's recordings
uv run python -m pmvc.speaker fit out/jvsall --only female,child --evals 40 --name speakers_body    # the body's two numbers only
uv run python -m pmvc.speaker fit out/jvsall --name speakers                                        # the body, and the glottis's open quotient, brightness and tension
#   Eleven numbers more: the body's four numbers directly (a tract larger than the adult male's is allowed), three of the glottis, and four habits of posture
#   (the narrowing above the larynx, and shifts of the larynx's height, of the passage to the nose and of the tongue root, added to every frame of the encoder's track). It starts from the earlier fit's values
uv run python -m pmvc.speaker fit out/jvsall --name speakers_more --start out/jvsall/speakers.json --evals 200 \
  --only scale,pharynx,tongueRadius,lipWidth,openQuotient,brightness,tension,epilarynx,larynxShift,velumShift,rootShift
#   The test sentences (not used in the fit) are rendered again with the fitted settings. The 8 speakers (out/jvs) and all 100 (out/jvsall)
for d in out/jvs out/jvsall; do
  e=enci; [ $d = out/jvsall ] && e=encir
  uv run python -m pmvc.encoder revoice $d --of $e --out encib --split test $V --speakers out/jvsall/speakers_body.json
  uv run python -m pmvc.encoder revoice $d --of $e --out encis --split test $V --speakers out/jvsall/speakers.json
  uv run python -m pmvc.encoder revoice $d --of ttsj --out ttsjb --split test $V --speakers out/jvsall/speakers_body.json
  uv run python -m pmvc.encoder revoice $d --of ttsj --out ttsjs --split test $V --speakers out/jvsall/speakers.json
  uv run python -m pmvc.encoder revoice $d --of $e --out encit --split test $V --speakers out/jvsall/speakers_more.json
  uv run python -m pmvc.encoder revoice $d --of ttsj --out ttsjt --split test $V --speakers out/jvsall/speakers_more.json
done
#   Whose voice it sounds like. By the verification model used in the fit (WavLM) and by one that was not (UniSpeech-SAT)
for m in "whose_spk" "whose_spk2 --model microsoft/unispeech-sat-base-plus-sv"; do
  uv run python -m pmvc.whose out/jvs --variants enci,ttsj,encis,ttsjs,encib,ttsjb,encit,ttsjt --split test --name $m
done
for m in "whose_speakers" "whose_speakers2 --model microsoft/unispeech-sat-base-plus-sv"; do
  uv run python -m pmvc.whose out/jvsall --variants encir,encib,encis,encit,ttsj,ttsjb,ttsjs,ttsjt --name $m
done
#   Whether intelligibility and the quality of the sound are unchanged
uv run python -m pmvc.judge out/jvs --variants enci,ttsj,encis,ttsjs,encib,ttsjb,encit,ttsjt --split test --name results_spk
uv run python -m pmvc.natural out/jvs --name results_spk
#   The women's nasals (the listener: "they sound as if the nose were blocked"): rendered again with the sinus on the nasal cavity at 24 cm³ instead of CHIMERA's 14 cm³ (--sinus)
for o in "encin" "encibn --speakers out/jvsall/speakers_body.json" "encisn --speakers out/jvsall/speakers.json" "encitn --speakers out/jvsall/speakers_more.json"; do
  uv run python -m pmvc.encoder revoice out/jvs --of enci --out $o --split test $V --sinus 24
done
for m in "whose_nose" "whose_nose2 --model microsoft/unispeech-sat-base-plus-sv"; do   # the results also give the values for women and for men
  uv run python -m pmvc.whose out/jvs --variants enci,encin,encib,encibn,encis,encisn,encit,encitn --split test --name $m
done
#   The harmonics of nasals and vowels, recordings against renders, by kind of phoneme (how strong the nasals are; how strong the harmonics are in nasals, in the vowels beside them and in vowels away from them)
uv run python -m pmvc.nasals out/jvs --variants enci,encin,encib,encibn,encis,encisn,encit,encitn --split test
uv run python -m pmvc.judge out/jvs --variants enci,ttsj,encis,ttsjs,encib,ttsjb,encit,ttsjt,encisn,encibn --split test --name results_spk --add
uv run python -m pmvc.natural out/jvs --name results_spk
#   The brightness of the women's voices (the listener: "not brilliant enough", and "like a buzzer once the glottis is fitted"): the women's vowels are 7 to 12 dB weaker than the recordings between 1.6 and 5.5 kHz
#   How sharply the glottis closes is measured for each speaker from the recordings' spectra (no verification model: 6 training sentences are rendered at 6 sharpnesses, and the value is the one at which the voiced stretches are as strong between 1.6 and 5.5 kHz as in the recordings)
uv run python -m pmvc.speaker closure out/jvsall --start out/jvsall/speakers_body.json --sex F --name speakers_closure               # the measured values
uv run python -m pmvc.speaker closure out/jvsall --start out/jvsall/speakers_body.json --sex F --share 0.5 --name speakers_closure_half  # half of them
uv run python -m pmvc.speaker closure out/jvsall --start out/jvsall/speakers_body.json --sex F --value 0.3 --name speakers_closure3     # 0.3 for everyone
for o in "encibh --speakers out/jvsall/speakers_closure_half.json" "encibf --speakers out/jvsall/speakers_closure.json" "encibk --speakers out/jvsall/speakers_closure3.json"; do
  uv run python -m pmvc.encoder revoice out/jvs --of enci --out $o --split test $V --sinus 24
done
#   The strength of vowels and fricatives by band, against the recordings (the balance of the harmonics, each band's strength, the ratio of harmonics to noise in each band)
uv run python -m pmvc.bands out/jvs --variants enci,encib,encibn,encibh,encibf,encibk,encis,encit --split test
uv run python -m pmvc.nasals out/jvs --variants enci,encibn,encibh,encibf,encibk --split test --name nasals_closure
for m in "whose_closure" "whose_closure2 --model microsoft/unispeech-sat-base-plus-sv"; do
  uv run python -m pmvc.whose out/jvs --variants enci,encib,encibn,encibh,encibf,encibk,encis,encit --split test --name $m
done
uv run python -m pmvc.judge out/jvs --variants enci,ttsj,encis,ttsjs,encib,ttsjb,encit,ttsjt,encisn,encibn,encibk,encibf,encibh --split test --name results_spk --add
uv run python -m pmvc.natural out/jvs --name results_spk
#   The same re-rendering and verification for all 100 speakers (the results give the values for women and for men)
for o in "encibn --speakers out/jvsall/speakers_body.json" "encibf --speakers out/jvsall/speakers_closure.json" "encibk --speakers out/jvsall/speakers_closure3.json"; do
  uv run python -m pmvc.encoder revoice out/jvsall --of encir --out $o --split test $V --sinus 24
done
for m in "whose_closure" "whose_closure2 --model microsoft/unispeech-sat-base-plus-sv"; do
  uv run python -m pmvc.whose out/jvsall --variants encir,encib,encibn,encibk,encibf,encis,encit --name $m
done

# 13. Towards the release (V1): the encoder is taught again with nothing but what may be passed on (encoder J)
#   Encoder I may not be: mHuBERT-147, its front end and its yardstick, is CC BY-NC-SA 4.0, and JVS's recordings are for research and not to be redistributed
#   The voice: in place of settings fitted to each speaker, a pair of reference voices (voices/v1.json). Men get the low voice and women the high one (the pitch does not decide)
#   The low voice: CHIMERA's adult male, the narrowing above the larynx at 0.78, the tongue root 0.35 cm forward. The high voice: adjustment 2, the women's average body, a sinus of 24 cm³, a closure sharpness of 0.18
V="--voice voices/v1.json --no-wall-air --closed-area 1e-6"
uv run python -m pmvc.encoder revoice out/jvs --of enci --out enciv --split test $V        # I's tracks by the reference voices (with the tongue root's posture added)
uv run python -m pmvc.bands out/jvs --variants enci,encibh,encit,encim,enciv --name bands_v1
uv run python -m pmvc.whose out/jvs --variants enci,encibh,encit,encim,enciv --split test --name whose_v1
uv run python -m pmvc.judge out/jvs --variants enci,encibh,encit,enciv --split test --name results_v1 && uv run python -m pmvc.natural out/jvs --name results_v1
#   Candidates for the yardstick (weights under Apache-2.0 or MIT), compared as in step 6
for m in "xlsr:facebook/wav2vec2-xls-r-300m:6,8,10,12,16" "hubert:facebook/hubert-base-ls960:6,8,9,10,12" "wavlm:microsoft/wavlm-base-plus:6,8,9,10,12"; do
  uv run python -m pmvc.yardstick --name yardstick_${m%%:*} --models "${m#*:}"
done
#   Recordings: from the Japanese and the English of Emilia-YODAS (CC BY 4.0), 324 speakers × 6 clips each (300 speakers to teach with, 12 for validation, 12 for the test). The corpus does not say a speaker's sex, so it is estimated
E=/path/to/Emilia-Dataset/Emilia-YODAS
uv run python -m pmvc.emilia --tars $E/JA --lang ja --shards 6        # → out/emja
uv run python -m pmvc.emilia --tars $E/EN --lang en --shards 4        # → out/emen
uv run python -m pmvc.sexes out/emja && uv run python -m pmvc.sexes out/emen     # a speaker whose pitch does not decide it: whether the verification model's vector is nearer the men's or the women's
uv run python -m pmvc.sexes out/jvs out/transplant --check                       # how often that is right for speakers whose sex is known
#   The targets are the tracks of the encoder taught with the rules alone, and everything is rendered again by the reference voices
for d in out/emja out/emen; do uv run python -m pmvc.encoder encode $d --models out/transplant/models --name sim --out sim --split all --no-render; done
uv run python -m pmvc.encoder revoice --of warp --out warpv $V
for d in out/transplant out/jvs out/emja out/emen; do uv run python -m pmvc.encoder revoice $d --of sim --out simv $V; done
X="--listener facebook/wav2vec2-xls-r-300m --layer 16"
T="--target simv --synthetic simv --tune 6 --silences $V --quiet-pauses"
#   I′: taught again with I's materials (TSP and JVS, mHuBERT) through the reference voices (to see what the change of voice alone does)
uv run python -m pmvc.standin --name standin_vm --variants warpv,simv --also out/jvs --voiced $M
uv run python -m pmvc.encoder train --name encvm --standin standin_vm --also out/jvs --ssl utter-project/mHuBERT-147 $T
#   J0: the yardstick becomes XLS-R and the front end wav2vec 2.0 base (the recordings are still TSP and JVS)
uv run python -m pmvc.standin --name standin_vj --variants warpv,simv --also out/jvs --voiced $X
uv run python -m pmvc.encoder train --name encvj --standin standin_vj --also out/jvs $T
#   J: the recordings become TSP and Emilia-YODAS too. Nothing but what may be passed on
uv run python -m pmvc.standin --name standin_v --variants warpv,simv --also out/emja,out/emen --voiced $X
uv run python -m pmvc.encoder train --name encv --standin standin_v --also out/emja,out/emen $T
#   Compared on the same three tests as before
for t in "${T3[@]}"; do    # T3=("out/jvs --split test" "out/ja --split all" "out/transplant --split test")
  uv run python -m pmvc.encoder revoice ${t%% *} --of enci --out enciv --split ${t##* } $V
  for m in encvm encvj encv; do uv run python -m pmvc.encoder encode $t --models out/transplant/models --name $m --out $m $V --quiet-pauses; done
  uv run python -m pmvc.judge $t --variants enci,enciv,encvm,encvj,encv --name results_rel && uv run python -m pmvc.natural ${t%% *} --name results_rel
done
#   The voice's dials (the reference voices' brightness, the pitch, the other voice): only the settings the tracks are rendered with change
for o in "encivd --bright -1" "encivu --bright 1" "encivlo --pitch -3" "encivhi --pitch 3" "encivL --share 0 --pitch -10" "encivH --share 1 --pitch 9"; do
  uv run python -m pmvc.encoder revoice out/jvs --of enci --out ${o%% *} --split test $V ${o#* }
done
uv run python -m pmvc.bands out/jvs --variants enciv,encivd,encivu --name bands_dial
uv run python -m pmvc.judge out/jvs --variants enciv,encivd,encivu,encivlo,encivhi,encivL,encivH --split test --name results_dial && uv run python -m pmvc.natural out/jvs --name results_dial
#   A TTS model from recordings that may be passed on (the first attempt): the Japanese of Emilia-YODAS, 800 speakers × 12 clips. J's tracks, the reference voices
uv run python -m pmvc.emilia --tars $E/JA --lang ja --out out/emjatts --shards 30 --speakers 800 --per 12 --held 0 --valid-clips 1 --draw 2
uv run python -m pmvc.sexes out/emjatts --also out/emja
uv run python -m pmvc.encoder encode out/emjatts --models out/transplant/models --name encv --out encv --split all $V --quiet-pauses --no-render
uv run python -m pmvc.align train out/emjatts --epochs 40 --hinted 12 --sharpen 16     # with people talking and automatic transcripts, the default 12 epochs do not find the alignment
uv run python -m pmvc.align write out/emjatts
uv run python -m pmvc.align prune out/emjatts                                           # drops the recordings whose transcript does not match the sound (9,600 → 5,356 clips)
uv run python -m pmvc.tts train out/emjatts --name ttse --tracks encv --standin standin_v --stride 8 $V
#   JVS's test sentences read in the voices of two of the speakers it was taught with, a man and a woman
uv run python -m pmvc.tts speak out/jvs --models out/emjatts/models --name ttse --split test --out ttse_m --speaker JA_BrOouMZCkBA_SPEAKER_00
uv run python -m pmvc.tts speak out/jvs --models out/emjatts/models --name ttse --split test --out ttse_f --speaker JA_yD5DCCvSJSI_SPEAKER_00
uv run python -m pmvc.judge out/jvs --variants ttsj,ttse_m,ttse_f --split test --name results_ttse && uv run python -m pmvc.natural out/jvs --name results_ttse
#   The pace: Emilia-YODAS's speakers are faster than people reading aloud (13.3 phonemes a second in the man's voice; JVS's recordings have 10.2). --rate makes them level
uv run python -m pmvc.tts speak out/jvs --models out/emjatts/models --name ttse --split test --out ttse_mr --speaker JA_BrOouMZCkBA_SPEAKER_00 --rate 10.2
uv run python -m pmvc.tts speak out/jvs --models out/emjatts/models --name ttse --split test --out ttse_fr --speaker JA_yD5DCCvSJSI_SPEAKER_00 --rate 10.2
uv run python -m pmvc.judge out/jvs --variants ttsj,ttse_m,ttse_f,ttse_mr,ttse_fr --split test --name results_ttse --add
#   Candidates for winning back what J lost (by the change of yardstick and front end): three on J0's recordings (TSP and JVS), compared on JVS and TSP (results_listen.json)
uv run python -m pmvc.encoder train --name encvxf --standin standin_vj --also out/jvs --ssl facebook/wav2vec2-xls-r-300m $T                   # XLS-R as the front end
uv run python -m pmvc.standin --name standin_vx12 --variants warpv,simv --also out/jvs --voiced --listener facebook/wav2vec2-xls-r-300m --layer 12
uv run python -m pmvc.encoder train --name encvx12 --standin standin_vx12 --also out/jvs $T                                                   # layer 12 of XLS-R as the yardstick
uv run python -m pmvc.standin --name standin_vt --variants warpv,simv --also out/jvs --voiced --listener sarulab-speech/hubert-base-jtube --layer 8
uv run python -m pmvc.encoder train --name encvt --standin standin_vt --also out/jvs $T                                                       # a HuBERT trained on Japanese (MIT) as the yardstick: Whisper's errors are back at I′'s level
#   J2: with that yardstick, and TSP and Emilia-YODAS as the recordings
uv run python -m pmvc.standin --name standin_vte --variants warpv,simv --also out/emja,out/emen --voiced --listener sarulab-speech/hubert-base-jtube --layer 8
uv run python -m pmvc.encoder train --name encv2 --standin standin_vte --also out/emja,out/emen $T
#   A TTS model from recordings of people reading: the Japanese of Common Voice (CC0). Of the speakers who state their sex, 300 women and 300 men, 16 sentences each
uv run python -m pmvc.commonvoice --corpus /path/to/cv-corpus-27.0-2026-09-11/ja      # → out/cvja
uv run python -m pmvc.encoder encode out/cvja --models out/transplant/models --name encv --out encv --split all $V --quiet-pauses --no-render
uv run python -m pmvc.align train out/cvja --epochs 40 --hinted 12 --sharpen 16 && uv run python -m pmvc.align write out/cvja && uv run python -m pmvc.align prune out/cvja
uv run python -m pmvc.tts train out/cvja --name ttsc --tracks encv --standin standin_v --stride 8 $V

#   A TTS model without speakers' vectors (it is given the share of the high voice and the pace; a dial can then set the voice)
uv run python -m pmvc.tts train out/cvja --name ttscv --by voice --tracks encv --standin standin_v --stride 8 $V
#   The release's configuration: J2's tracks (and J2's stand-in for the yardstick). Common Voice, Emilia-YODAS, and both (out/jatts is the two joined by links)
for d in out/cvja out/emjatts; do uv run python -m pmvc.encoder encode $d --models out/transplant/models --name encv2 --out encv2 --split all $V --quiet-pauses --no-render; done
uv run python -m pmvc.tts train out/cvja --name ttscv2 --by voice --tracks encv2 --standin standin_vte --stride 8 $V
uv run python -m pmvc.tts train out/emjatts --name ttsev2 --by voice --tracks encv2 --standin standin_vte --stride 8 $V
uv run python -m pmvc.tts train out/jatts --name ttsja --by voice --tracks encv2 --standin standin_vte --stride 16 $V
#   JVS's test sentences in the low voice and in the high one (at the recordings' pace)
uv run python -m pmvc.tts speak out/jvs --models out/cvja/models --name ttscv2 --split test --out ttscv2_m --share 0 --pitch 125 --rate 10.2
uv run python -m pmvc.tts speak out/jvs --models out/cvja/models --name ttscv2 --split test --out ttscv2_f --share 1 --pitch 240 --rate 10.2
#   English: LibriTTS-R (CC BY 4.0). The text is read by CHIMERA's own English front end (pmvc.english)
uv run python -m pmvc.libritts --corpus /path/to/LibriTTS_R              # → out/lten
uv run python -m pmvc.encoder encode out/lten --models out/transplant/models --name encv2 --out encv2 --split all $V --quiet-pauses --no-render
uv run python -m pmvc.align train out/lten --lang en --epochs 40 --hinted 12 --sharpen 16 && uv run python -m pmvc.align write out/lten --lang en && uv run python -m pmvc.align prune out/lten
uv run python -m pmvc.tts train out/lten --name ttsen --by voice --tracks encv2 --standin standin_vte --stride 8 $V
uv run python -m pmvc.tts say "こんにちは。" --models out/cvja/models --name ttscv --share 1 --pitch 240 --rate 10 --out say.wav
#   Sound in the browser (the base of V1's WebUI): the engine becomes WebAssembly and turns the control track into sound inside the browser
#   crates/pmvc-voice is what pmvc_engine (for Python) and pmvc-wasm (for the browser) share. The same control track gives the same samples
PATH="$(brew --prefix rustup)/bin:$PATH" cargo build -p pmvc-wasm --release --target wasm32-unknown-unknown    # needs rustup's wasm32-unknown-unknown
node tools/play_wasm.mjs target/wasm32-unknown-unknown/release/pmvc_wasm.wasm out/jvs/enciv/jvs093_079.tracks.json /tmp/a.f32    # render it in Node, to check
uv run python -m pmvc.serve --model out/jatts/models/ttsja.pt out/lten/models/ttsen.pt    # → http://127.0.0.1:7860/ (text → control track on the server, the sound in the browser; one model a language)
node tools/test_page.mjs http://127.0.0.1:7860 target/wasm32-unknown-unknown/release/pmvc_wasm.wasm /tmp/b.f32 1 240 -1          # the page's own path, without a browser
node tools/test_browser.mjs http://127.0.0.1:7860/ /tmp/shots       # opens the page in a real Chrome (headless), makes it speak, turns a dial while it speaks, and measures the pitch of what comes out
node tools/test_browser.mjs "http://127.0.0.1:7860/?en" /tmp/shots --lang ja    # that ?ja / ?en in the address sets the language, and that switching on the page changes the address too
#   The models in the form they are published in: the weights (safetensors) and whatever else is needed (JSON). The names of the speakers they were taught with are left out
uv run python -m pmvc.tts export out/jatts/models/ttsja.pt ../pmvc-tts/ja
uv run python -m pmvc.tts export out/lten/models/ttsen.pt ../pmvc-tts/en
uv run python -m pmvc.serve --model ../pmvc-tts/ja.json ../pmvc-tts/en.json
#   The Hugging Face Space (Gradio SDK. ZeroGPU hardware runs only Gradio applications, so app.py serves the page and /api/* as they are
#   through Gradio's Server mode (gr.Server). The models run on the CPU): on a Linux x86_64 machine, build the engine's wheel and assemble everything in a clone of the Space's repository
bash tools/wheel.sh                                                  # → dist/pmvc_engine-…-cp312-cp312-manylinux_2_28_x86_64.whl (built inside Docker)
MODELS=../pmvc-tts bash tools/space.sh ../pmvc-tts-space             # space/'s app.py, README and requirements.txt, with the page, the server, the engine and the models
PORT=7861 uv run --with gradio==6.30.0 python ../pmvc-tts-space/app.py    # runs the Space's application here → http://127.0.0.1:7861/

# 14. The listening pages (three HTML pages with the audio inside: English in listening.html, Japanese in listening_ja.html, TTS in listening_tts.html)
uv run python -m pmvc.page --notes notes.html --notes-ja notes_ja.html --notes-tts notes_tts.html

# 15. The technical report (PDFs in English and Japanese). Needs tectonic
bash docs/report/build.sh

# 16. (V2) Yardsticks for work on the voice itself. Uses third-party/UTMOSv2 (git submodule update --init)
#   Parts swapped between a recording and its render (the spectral envelope and the rest, bands, voiced and unvoiced stretches and
#   silences), to see what a render lacks. Two speaker-verification models (100 speakers enrolled: the training recordings in
#   out/jvsall) and NISQA's two models judge the hybrids
uv run python -m pmvc.gap out/jvs --variant encv2 --enrol out/jvsall --out /tmp/gap --name gap_encv2
uv run python -m pmvc.gap out/jvs --variant encv2 --out /tmp/gap_low --name gap_low_encv2 --per 13 --bands 0-250,250-500,500-1000 --gates --no-speaker
#   The scoreboard: recordings and renders are brought to 24 kHz and to one speech level before any judge hears them. Naturalness and
#   quality are given for the clip as it is and through a fixed recording condition (pmvc.room: a high-pass, a short tail, a noise
#   floor). --opinions also asks SQUIM and UTMOSv2
uv run python -m pmvc.bench out/jvs --variants encv2,encit --enrol out/jvsall --judged results_rel --opinions --name bench_v1
uv run python -m pmvc.bench out/transplant --variants encv2 --judged results_rel --opinions --name bench_v1      # English (TSP): no speaker identification and no envelope terms
#   Does the engine come nearer a recording's envelope when asked (phase 3-1 of the plan)? Each speaker's settings fitted to the
#   distance between envelopes alone, with no verification model
S8=jvs093,jvs094,jvs095,jvs096,jvs097,jvs098,jvs099,jvs100
uv run python -m pmvc.speaker fit out/jvsall --name speakers_env --start out/jvsall/speakers.json --evals 200 --by envelope --speakers $S8 \
  --only scale,pharynx,tongueRadius,lipWidth,openQuotient,brightness,tension,closureSharpness,epilarynx,larynxShift,velumShift,rootShift
uv run python -m pmvc.encoder revoice out/jvs --of enci --out encie --split test --voice out/fit/voice-F2.json --no-wall-air --closed-area 1e-6 --speakers out/jvsall/speakers_env.json
uv run python -m pmvc.bench out/jvs --variants enci,encit,encie --enrol out/jvsall --name bench_static
#   The tracks searched one utterance at a time (the five shortest test sentences of the 8 speakers). The objective is the listener's
#   distance + a weight x the envelope distance. Weight 3.0 was run on the first two sentences only (16 recordings)
IDS=$(for k in 091 099 098 088 080; do for s in ${S8//,/ }; do printf "%s_%s," $s $k; done; done | sed 's/,$//')
uv run python -m pmvc.copysynth out/jvs --ids $IDS --base encit --out reach_t --nll-weight 0 --shape-weight 0.3 --iters 50 \
  --listener utter-project/mHuBERT-147 --layer 8 --scale out/transplant/warp --also epilarynx=0.15
uv run python -m pmvc.bench out/jvs --variants encit,reach_t,reach_s --enrol out/jvsall --name bench_reach --per 5
#   Where the tract's resonances are (the yardstick of the plan's phase 2-1): the transfer function in the vowels' frames, by the
#   engine's own frequency-domain model. --set holds channels of the tracks at a value in every frame (what a setting does to the
#   resonances, before anything is rendered)
uv run python -m pmvc.tract out/jvs --variant encv2
uv run python -m pmvc.tract out/jvs --variant encv2 --set epilarynx=1.0
#   The fourth listening page (listening_v2.html): the hybrids, and renders through the recording condition
uv run python -m pmvc.page2 out/jvs --notes out/transplant/notes_v2.html
```

The encoders are trained and evaluated on separate speakers. In TSP a woman and a man with the same letter read the same sentence list, so with FK, MK, FL and ML held out for the test and FJ and MJ for validation, those speakers' sentences are not in the training either.
Training is quick with a GPU (on an RTX 5090: 3 minutes for A, 4 to 5 for a stand-in, about 12 for C′), and the search uses the CPU (about 270 sentences an hour on a 16-core Mac).
D, F and G, which add Japanese, took 25 to 40 minutes each on the same GPU. In step 8, rendering again took 7 minutes (3,864 tracks), the stand-in 3 minutes and I 22 minutes. In step 9, taking in the recordings took 1 minute, encoder I's tracks 7 minutes (18,755 sentences), the alignment 4 minutes to train and 4 to write out, and the TTS model about 1 minute an epoch (60 epochs). To synthesize a 4-second sentence, text to control track takes 17 ms on a CPU, and control track to sound 0.5 s on one thread of CHIMERA. In step 10, taking in the recordings took 1 minute, the tracks 5 minutes (12,997 sentences), the alignment 4 minutes to train and 3 to write out, and the TTS model about 55 minutes (60 epochs). JVS is split both by speaker and by sentence: the 8 test speakers and the 25 test sentences are in neither the training nor the validation.
Tuning the voice's settings takes about 15 minutes on a CPU. The first `--start` of step 7 is the result of one fit, within a narrow range, around the values of a soft phonation that were found by moving one setting at a time. The fit can be run again with the clicks (`pmvc.clicks`) counted against a setting (`--clicks`, 0.3 by default), but it then finds other ways out, lowering the open quotient and doubling the amplitude, or raising the breath noise to mask the clicks, so the four settings that matter for the clicks are set by hand, watching the measures.

`pmvc.calibrate` adjusts the noise sources' gains and the glottal source's settings to recordings; on unused data the gain was slight, so copy synthesis does not use it by default (`--calibrated` turns it on).

The conditions and the definitions of the measures are in the docstring at the head of each script.
