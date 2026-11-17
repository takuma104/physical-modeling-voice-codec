"""A listening page for the transplant and perturbation renders: one self-contained HTML file
with the audio embedded, the scores in tables and what Whisper heard under each clip.

usage: python -m pmvc.page [dir] [--abroad out/ja] [--notes notes.html] [--notes-ja notes_ja.html]
                           [--notes-tts notes_tts.html] [--say out/say]
                           [--sentences 12] [--perturbed 4] [--en-url URL] [--ja-url URL] [--tts-url URL]

Three pages, since one would be too large to publish: <dir>/listening.html, the experiments on
TSP's recordings; <dir>/listening_ja.html, the ones on recordings from outside it (the
Japanese ones above all); and <dir>/listening_tts.html, text to speech (`pmvc.tts`). The first
reads <dir>/results.json and <dir>/perturb_results.json (`pmvc.evaluate`), <dir>/perturb.json
(`pmvc.perturb`) and the WAVs, and its later sections appear once their results are there; the
second reads `pmvc.judge`'s results in `--abroad`; the third reads results_tts.json there and
`--say`'s results.json. Given the pages' published addresses, the first two point at each
other, the second at the third, and the third at both. `--notes` is an HTML fragment shown at the top (the findings,
written by whoever ran the experiment). The page is written without <html>/<head>/<body>, as
an Artifact wants it; browsers open it as it is.
"""

import argparse
import base64
import html
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .audio import load
from .paths import ROOT

VARIANT_LABELS: dict[str, str] = {
    "real": "実録音",
    "rule": "ルールのみ",
    "f0": "規則の音素長 + 話者の F0",
    "floor": "話者の音素長（下限つき）+ F0",
    "durf0": "話者の音素長 + F0",
    "fit": "話者の音素長 + F0 + 境界補正",
}
VARIANT_NOTES: dict[str, str] = {
    "real": "TSP の話者本人",
    "rule": "CHIMERA 標準（G2P・Klatt 規則の音素長・規則の抑揚）",
    "f0": "話者の音素列と F0 曲線。音素長は CHIMERA の規則",
    "floor": "話者の音素長。ただし母音 70 ms・子音 50 ms を下限にする",
    "durf0": "話者の音素長をそのまま",
    "fit": "有声の開始・終了が録音と合うよう音素境界を補正",
}
COPY_LABELS: dict[str, str] = {
    "real": "実録音",
    "rule": "ルールのみ",
    "f0": "規則の音素長 + 話者の F0",
    "warp": "時間伸縮",
    "copy": "コピー合成",
}
COPY_NOTES: dict[str, str] = {
    "real": "TSP の話者本人",
    "rule": "CHIMERA 標準",
    "f0": "話者の音素列と F0 曲線。音素長は CHIMERA の規則",
    "warp": "規則どおりに発音した制御列を、録音の時間軸へ音素ごとに伸縮（コピー合成の出発点）",
    "copy": "制御列を録音に近づくよう局所探索で最適化",
}
CODEC_LABELS: dict[str, str] = {
    "real": "実録音",
    "rule": "ルールのみ",
    "sim": "エンコーダ A",
    "simcopy": "A + 探索",
    "encb": "エンコーダ B",
    "enc": "エンコーダ C",
    "encn": "エンコーダ C′",
}
CODEC_NOTES: dict[str, str] = {
    "real": "TSP の話者本人（学習に使っていない話者）",
    "rule": "CHIMERA 標準。文のテキストから合成（録音は使わない）",
    "sim": "CHIMERA 自身の合成音と制御列の対だけで学習。録音から 1 回の推論で制御列を出す",
    "simcopy": "A の制御列を、合成音が録音に近づくよう探索で直す（1 文 1〜2 分）",
    "encb": "探索で直した制御列を教師にして学習。1 回の推論",
    "enc": "B に加えて、声の代役モデルを通した閉ループの誤差でも学習。1 回の推論",
    "encn": "C と同じ閉ループ学習を、探索の結果を一切使わずに行う。1 回の推論",
}
ABROAD_LABELS: dict[str, str] = {
    "real": "実録音",
    "sim": "エンコーダ A",
    "enc": "エンコーダ C",
    "encn": "エンコーダ C′",
    "encncopy": "C′ + 探索",
}
ABROAD_NOTES: dict[str, str] = {
    "real": "話者本人",
    "sim": "CHIMERA 自身の合成音だけで学習（実録音は一度も聞いていない）",
    "enc": "英語の録音で閉ループ学習（探索の結果も使う）",
    "encn": "英語の録音で閉ループ学習（探索なし）",
    "encncopy": "C′ の制御列を、合成音が録音に近づくよう探索で直す（1 文 数分）",
}
# the encoders taught with Japanese recordings, a multilingual listener, or both, beside C′
TAUGHT_LABELS: dict[str, str] = {
    "real": "実録音",
    "encn": "エンコーダ C′",
    "encd": "エンコーダ D",
    "ence": "エンコーダ E",
    "encf": "エンコーダ F",
    "encg": "エンコーダ G",
}
TAUGHT_NOTES: dict[str, str] = {
    "real": "話者本人",
    "encn": "英語の録音で学習。物差しは英語の wav2vec2",
    "encd": "英語と日本語の録音で学習。物差しは英語の wav2vec2",
    "ence": "英語の録音で学習。物差しは多言語の mHuBERT",
    "encf": "英語と日本語の録音で学習。物差しは多言語の mHuBERT",
    "encg": "F に加えて、エンコーダの入力も mHuBERT に替える",
}
# encoder H, and the female voice refitted
TUNED_LABELS: dict[str, str] = {
    "real": "実録音",
    "encn": "エンコーダ C′",
    "encg": "エンコーダ G",
    "ench": "エンコーダ H",
    "enchq": "H（対策 1）",
    "enchc": "H（対策 1+2）",
    "ench2q": "H + 調整 2（対策 1）",
    "ench2c": "H + 調整 2（対策 1+2）",
    "enchvc": "H + 調整 1（対策 1+2）",
    "ench3c": "H + 調整 3（対策 1+2）",
}
TUNED_NOTES: dict[str, str] = {
    "real": "話者本人",
    "encn": "英語の録音だけで学習",
    "encg": "日本語の録音、多言語の物差しと入口（第 2 節）",
    "ench": "G と同じ。ただし発話の外では「息なし」を手本にする",
    "enchq": "対策 1 は無音区間の音への対策: 動きの音を切り、録音のポーズでは息を止める",
    "enchc": "対策 2 は閉鎖のクリック音への対策: 狭めを「閉じた」とみなす面積を小さくする",
    "ench2q": "調整 2 は声の高い話者だけに使う声の設定（振幅 1.4 倍、雑音源は既定値）。前回お聴きいただいたもの",
    "ench2c": "前回の調整 2 に対策 2 を加えたもの",
    "enchvc": "調整 1 は NISQA だけに合わせた設定（歯の雑音源 2.2 倍）",
    "ench3c": "調整 3 は、調整 2 の狭めと歯の雑音源を 0.6 倍にしたもの",
}
RETAUGHT_LABELS: dict[str, str] = {
    "real": "実録音",
    "ench": "エンコーダ H（既定の声）",
    "ench2c": "エンコーダ H（調整後の声）",
    "enci": "エンコーダ I（調整後の声）",
}
RETAUGHT_NOTES: dict[str, str] = {
    "real": "話者本人",
    "ench": "CHIMERA の既定の声で学習し、既定の声で鳴らす（第 3 節の H）",
    "ench2c": "既定の声で学習した H を、調整後の声で鳴らす（第 3 節の「H + 調整 2（対策 1+2）」）",
    "enci": "H と同じ作り方で、調整後の声を通して学習し直し、調整後の声で鳴らす",
}
# towards the release: the pair of voices and the encoders taught with what may be passed on
RELEASE_KEPT: dict[str, str] = {"real": "実録音（値）", "enci": "これまで", "decided": "話者ごとの設定（決定済み）", "enciv": "2 つの基準の声"}
RELEASE_LABELS: dict[str, str] = {
    "real": "実録音",
    "enci": "エンコーダ I",
    "enciv": "I + 基準の声",
    "encvm": "エンコーダ I′",
    "encvj": "エンコーダ J0",
    "encv": "エンコーダ J",
    "encv2": "エンコーダ J2",
}
RELEASE_NOTES: dict[str, str] = {
    "real": "話者本人",
    "enci": "これまでのもの。調整後の声を通して学習し、その声で鳴らす（第 4 節）",
    "enciv": "I の制御列を、2 つの基準の声で鳴らす（学習し直さない）",
    "encvm": "I と同じ材料（TSP + JVS、mHuBERT）で、基準の声を通して学習し直す",
    "encvj": "I′ の物差しを XLS-R に、入口を wav2vec 2.0 base に替える（録音は TSP + JVS のまま）",
    "encv": "J0 の JVS を Emilia-YODAS（日本語と英語）に替える。公開できる材料だけで学習",
    "encv2": "J の物差しを、日本語の HuBERT（hubert-base-jtube、MIT）に替える。公開できる材料だけで学習",
}
RELEASE_MODELS: dict[str, str] = {
    "utter-project/mHuBERT-147": "mHuBERT-147（147 言語。これまでの物差し）｜CC BY-NC-SA 4.0",
    "facebook/wav2vec2-xls-r-300m": "XLS-R 300M（128 言語）｜Apache-2.0",
    "sarulab-speech/hubert-base-jtube": "hubert-base-jtube（日本語）｜MIT",
    "facebook/wav2vec2-large-xlsr-53": "XLSR-53（53 言語）｜Apache-2.0",
    "microsoft/wavlm-base-plus": "WavLM Base+（英語）｜CC BY-SA 3.0",
    "facebook/wav2vec2-base-960h": "wav2vec 2.0 base 960h（英語）｜Apache-2.0",
    "facebook/hubert-base-ls960": "HuBERT base（英語）｜Apache-2.0",
}
# the settings `pmvc.calibrate` fits, as the page names them
TTS_LABELS: dict[str, str] = {
    "real": "実録音",
    "enci": "エンコーダ I（録音から）",
    "ttsd": "TTS（音素の長さは録音）",
    "tts": "TTS（テキストだけから）",
}
TTS_NOTES: dict[str, str] = {
    "real": "Hi-Fi-CAPTAIN の男性話者",
    "enci": "録音を聞いて書いた制御列。TTS の手本で、ここが上限",
    "ttsd": "テキストから書いた制御列。各音素の長さだけ録音に合わせた",
    "tts": "音素の長さもモデルが決める。録音は使わない",
}
# the same model taught with fewer of the recordings: the variant, how many, and how long they are together
TTS_SIZES: dict[str, tuple[str, str]] = {
    "tts": ("18,655 文", "約 24 時間。上の「TTS（テキストだけから）」"),
    "tts2000": ("2,000 文", "約 2.6 時間"),
    "tts200": ("200 文", "約 15 分"),
    "tts50": ("50 文", "約 4 分"),
    "tts20": ("20 文", "約 1.5 分"),
}
# text to speech for JVS's hundred speakers: the variants in `--jvs`'s results_tts.json
MANY_LABELS: dict[str, str] = {
    "real": "実録音",
    "enci": "エンコーダ I（録音から）",
    "ttsjd": "TTS（音素の長さは録音）",
    "ttsj": "TTS（テキストだけから）",
    "ttsj2000": "TTS（2,000 文で学習）",
    "ttsj500": "TTS（500 文で学習）",
}
MANY_NOTES: dict[str, str] = {
    "real": "JVS のテストの 8 話者",
    "enci": "録音を聞いて書いた制御列。TTS の手本",
    "ttsjd": "テキストと話者から書いた制御列。各音素の長さだけ録音に合わせた",
    "ttsj": "100 話者の 9,998 文（約 20 時間）で学習。録音は使わない",
    "ttsj2000": "1 話者あたり 20 文（約 2 分）",
    "ttsj500": "1 話者あたり 5 文（約 35 秒）",
}
# the control: the same model on the latents of neural decoders (`pmvc.control`). For each, the
# label, a note, the variant of the model taught with so many recordings ("" for all), and the
# variant that is the decoder's best (a recording through it and back; the encoder's render; ""
# where that was not made)
CONTROL_LATENTS: dict[str, tuple[str, str, str, str]] = {
    "track": ("CHIMERA の制御列", "物理モデル。対照と同じ誤差で学習（代役なし）", "tts{}n", "enci"),
    "mel": ("メルスペクトログラム", "UnivNet ボコーダで音にする。この形のモデルの標準の出力先", "ctlmel{}", "copy_mel"),
    "melv": ("メル + 調整したボコーダ", "上と同じモデル。UnivNet を、モデルの学習と同じ録音で調整", "ctlmel{}vp", "copy_melvp"),
    "melf": ("同、交差予測で調整", "各録音のメルを、その録音を学習していないモデルに書かせて調整", "ctlmel{}v", ""),
    "encodec": ("EnCodec の潜在表現", "量子化前の連続値。128 次元、75 Hz", "ctlencodec{}", "copy_encodec"),
    "mimi": ("Mimi の潜在表現", "32 コードブックが表すベクトル。512 次元、12.5 Hz", "ctlmimi{}", "copy_mimi"),
}
CONTROL_SIZES: dict[str, str] = {"20": "20 文", "50": "50 文", "200": "200 文", "2000": "2,000 文", "": "18,655 文"}
# the latents that are in the tables only: the page has room for so many clips
CONTROL_UNHEARD: set[str] = {"melf"}
# each speaker's own body and glottis (`pmvc.speaker`): the encoder's tracks and the
# text-to-speech model's, as they were, with the body fitted, with the body and the glottis
SPEAKER_LABELS: dict[str, str] = {
    "enci": "これまで（エンコーダ I）",
    "encib": "体格を当てはめ",
    "encis": "体格 + 声門を当てはめ",
    "encit": "体格 4 値 + 声門 + 構えを当てはめ",
    "encisn": "体格 + 声門、副鼻腔を大きく",
    "encibn": "体格だけ、副鼻腔を大きく",
    "encibh": "体格 + 副鼻腔、声門の閉じ方を鋭く（半分）",
    "encibf": "体格 + 副鼻腔、声門の閉じ方を鋭く（測った値）",
    "encibk": "体格 + 副鼻腔、声門の閉じ方を鋭く（全員 0.3）",
    "ttsj": "これまで（TTS）",
    "ttsjb": "TTS、体格を当てはめ",
    "ttsjs": "TTS、体格 + 声門を当てはめ",
    "ttsjt": "TTS、体格 4 値 + 声門 + 構えを当てはめ",
}
SPEAKER_NOTES: dict[str, str] = {
    "enci": "声に伝えるのは性別と声の高さの中央値だけ",
    "encib": "話者ごとに体格の 2 つの値。声門は調整後の声のまま",
    "encis": "話者ごとに体格と声門の 5 つの値",
    "encit": "話者ごとに 11 個の値: 体格の 4 つ、声門の 3 つ、構えの 4 つ",
    "encisn": "「体格 + 声門」のまま、鼻腔につながる副鼻腔の容積を 14 → 24 cm³ に",
    "encibn": "体格の 2 つの値だけを当てはめ、声門は調整後の声のまま。副鼻腔は 24 cm³",
    "encibh": "「体格だけ、副鼻腔を大きく」から、女性の声門の閉じ方の鋭さを、録音から測った値の半分に",
    "encibf": "同じく、録音から測った値に（話者ごと。男性は変えない）",
    "encibk": "同じく、女性の全員を 0.3 に",
    "ttsj": "100 話者の TTS。声に伝えるのは性別と声の高さだけ",
    "ttsjb": "同じ制御列を、話者の体格で鳴らす",
    "ttsjs": "同じ制御列を、話者の体格と声門で鳴らす",
    "ttsjt": "同じ制御列を、話者の 11 個の値で鳴らす",
}
# what the cards of the speakers' section play beside the recording and the render as it was: the
# women's with the body fitted and the nose's sinus enlarged (the listener's "a good direction"
# for one of them) and from there with the glottis closing more sharply, half as much as measured
# and as measured (what the listener missed in the women's voices was brilliance; the glottis
# fitted to a verification model was a buzz to that ear); the men's with the eleven settings
# (better than as they were, to the same ear)
SPEAKER_HEARD: dict[str, tuple[tuple[str, str], ...]] = {
    "F": (("encibh", "体格 + 副鼻腔 + 閉じ方を鋭く（半分）"),),
    "M": (("encit", "体格 4 値 + 声門 + 構えを当てはめ"),),
}
# the speakers whose voices the page's cards are in
MANY_VOICES: dict[str, str] = {"jvs093": "女性 jvs093", "jvs095": "女性 jvs095", "jvs097": "男性 jvs097", "jvs100": "男性 jvs100"}
VOICE_LABELS: dict[str, str] = {
    "noiseGlottis": "雑音源のゲイン（声門）",
    "noiseConstriction": "雑音源のゲイン（狭め）",
    "noiseTeeth": "雑音源のゲイン（歯）",
    "pressure": "肺圧 cmH₂O",
    "tension": "声門の張り（0 ゆるい 〜 1 強い）",
    "breathiness": "息漏れ（0〜1）",
    "brightness": "明るさ（閉じ方の急さ、0〜1）",
    "roughness": "ゆらぎ（周期と振幅の乱れ、0〜1）",
    "openQuotient": "開放率（1 周期のうち声門が開いている割合）",
    "amplitudeScale": "振動の振幅（倍率）",
    "sealRate": "狭めが閉じきる速さ cm²/s",
}
# the speakers `pmvc.pitchmix` was run on, as the page names them
PITCH_SPEAKERS: dict[str, str] = {
    "ML": "TSP 男性 ML", "MK": "TSP 男性 MK", "FL": "TSP 女性 FL", "FK": "TSP 女性 FK",
    "hem": "HFC 英語 男性", "hef": "HFC 英語 女性", "hjm": "HFC 日本語 男性", "hjf": "HFC 日本語 女性", "jsut": "JSUT 女性",
    "jvs093": "JVS 女性 jvs093", "jvs094": "JVS 女性 jvs094", "jvs095": "JVS 女性 jvs095", "jvs096": "JVS 女性 jvs096",
    "jvs097": "JVS 男性 jvs097", "jvs098": "JVS 男性 jvs098", "jvs099": "JVS 男性 jvs099", "jvs100": "JVS 男性 jvs100",
}  # fmt: skip
# the listener candidates `pmvc.yardstick` compared, as the page names them
YARDSTICK_LABELS: dict[str, str] = {
    "facebook/wav2vec2-base-960h": "wav2vec2-base-960h（英語。これまでの物差し）",
    "utter-project/mHuBERT-147": "mHuBERT-147（147 言語）",
    "sarulab-speech/hubert-base-jtube": "hubert-base-jtube（日本語）",
    "facebook/wav2vec2-large-xlsr-53": "wav2vec2-large-xlsr-53（53 言語）",
}
# the sets of recordings `pmvc.recordings` brought in, as the page names them
ABROAD_SETS: dict[str, str] = {
    "hjf": "日本語・女性（Hi-Fi-CAPTAIN）",
    "hjm": "日本語・男性（Hi-Fi-CAPTAIN）",
    "jsut": "日本語・女性（JSUT）",
    "hef": "英語・女性（Hi-Fi-CAPTAIN）",
    "hem": "英語・男性（Hi-Fi-CAPTAIN）",
}
# Open JTalk's phoneme symbols that need a word of explanation
SOUND_NOTES: dict[str, str] = {
    "N": "撥音「ん」", "cl": "促音「っ」", "ts": "「つ」の子音", "sh": "「し」「しゃ」の子音", "ch": "「ち」「ちゃ」の子音", "j": "「じ」「じゃ」の子音",
    "z": "ざ行の子音", "f": "「ふ」の子音", "r": "ら行の子音", "h": "は行の子音", "g": "が行の子音", "d": "だ行の子音", "t": "た行の子音",
    "w": "「わ」の子音", "b": "ば行の子音", "k": "か行の子音", "y": "や行の子音",
}  # fmt: skip
KIND_LABELS: dict[str, str] = {
    "white": "白色ノイズ（フレームごとに独立）",
    "smooth": "ゆっくり変動するノイズ（4 Hz 以下）",
    "quant": "量子化（チャネルごと）",
    "rate": "フレームレート削減",
}


# what a plain code of the 17 channels that move would cost, per bit and frame a second
CODED_CHANNELS: int = 17
CODED_LABEL: dict[str, str] = {"coded": "間引きと量子化を同時に（素朴な符号化）"}


def level_label(cond: str) -> str:
    kind, arg = cond.split("_", 1)
    if kind in ("white", "smooth"):
        return f"SNR {arg[3:]} dB"
    if kind == "quant":
        return f"{arg[:-3]} bit"
    if kind == "coded":
        bits, hz = arg.split("_")
        return f"{bits[:-3]} bit・{hz[:-2]} Hz（{CODED_CHANNELS * int(bits[:-3]) * int(hz[:-2]) / 1000:.1f} kbps）"
    return f"{arg[:-2].replace('p', '.')} Hz"


def clip_uri(wav: Path) -> str:
    """The clip as an MP3 data URI, its speech brought to a common level."""
    x, sr = load(wav)
    frame = sr // 100
    rms = np.sqrt((x[: len(x) // frame * frame].reshape(-1, frame) ** 2).mean(axis=1))
    active = rms[rms > rms.max() * 0.1]
    gain = 0.07 / max(1e-9, float(np.sqrt((active**2).mean())))
    x = x * min(gain, 0.95 / max(1e-9, float(np.abs(x).max())))
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "a.wav", Path(tmp) / "a.mp3"
        sf.write(str(src), x, sr, subtype="PCM_16")
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-c:a", "libmp3lame", "-b:a", "96k", str(dst)],
            check=True,
        )
        return "data:audio/mpeg;base64," + base64.b64encode(dst.read_bytes()).decode()


def esc(s: Any) -> str:
    return html.escape(str(s))


def button(clips: dict[str, str], key: str, wav: Path, label: str, heard: str, kind: str = "", judge: str = "small.en") -> str:
    if key not in clips:
        clips[key] = clip_uri(wav)
    cls = f"clip {kind}".strip()
    return f'<button type="button" class="{cls}" data-clip="{key}" data-heard="{esc(heard)}" data-judge="{judge}" aria-pressed="false">{esc(label)}</button>'


def rate_cell(s: dict[str, Any], key: str) -> str:
    """A rate with its interval, as `pmvc.judge` writes them."""
    if key not in s:
        return '<td class="num">–</td>'
    w = s[key]
    return f'<td class="num">{w[0]:.1f}<span class="ci"> [{w[1]:.1f}, {w[2]:.1f}]</span></td>'


def wer_cell(s: dict[str, Any]) -> str:
    w = s["werCapped"]
    return f'<td class="num">{w[0]:.1f}<span class="ci"> [{w[1]:.1f}, {w[2]:.1f}]</span></td>'


def num_cell(s: dict[str, Any], key: str, fmt: str, scale: float = 1.0) -> str:
    return f'<td class="num">{format(s[key] * scale, fmt)}</td>' if key in s else '<td class="num">–</td>'


def transplant_table(summary: dict[str, Any]) -> str:
    rows = []
    for v, label in VARIANT_LABELS.items():
        a = summary["all"][v]
        rows.append(
            f'<tr><th scope="row">{esc(label)}<span class="sub">{esc(VARIANT_NOTES[v])}</span></th>'
            + wer_cell(a)
            + wer_cell(summary["female"][v])
            + wer_cell(summary["male"][v])
            + num_cell(a, "voicing", ".0f", 100.0)
            + num_cell(a, "timing", ".0f")
            + num_cell(a, "f0Corr", ".2f")
            + num_cell(a, "mcd", ".2f")
            + "</tr>"
        )
    other = summary["all"].get("other")
    if other:
        rows.append(
            '<tr><th scope="row">別の実話者<span class="sub">同じ文を読んだ他の話者（参考）</span></th>'
            + '<td class="num">–</td>' * 6
            + f'<td class="num">{other["mcd"]:.2f}</td></tr>'
        )
    n = summary["all"]["n"]
    return f"""<div class="scroll"><table>
<caption>韻律移植の評価（{n} 文、女性 {summary["female"]["n"]}・男性 {summary["male"]["n"]}）。単語誤り率は Whisper small.en、角括弧は 95% 区間。1 文の誤りは語数を上限に数えています。</caption>
<thead><tr><th scope="col">条件</th><th scope="col">単語誤り率 %</th><th scope="col">女声 %</th><th scope="col">男声 %</th>
<th scope="col">有声一致 %</th><th scope="col">時間ずれ ms</th><th scope="col">F0 相関</th><th scope="col">ケプストラム距離 dB</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>"""


def copy_table(summary: dict[str, Any]) -> str:
    a = summary["all"]
    rows = []
    for v, label in COPY_LABELS.items():
        w2 = a[v]["wer2"]
        rows.append(
            f'<tr><th scope="row">{esc(label)}<span class="sub">{esc(COPY_NOTES[v])}</span></th>'
            + wer_cell(a[v])
            + f'<td class="num">{w2[0]:.1f}<span class="ci"> [{w2[1]:.1f}, {w2[2]:.1f}]</span></td>'
            + num_cell(a[v], "voicing", ".0f", 100.0)
            + num_cell(a[v], "timing", ".0f")
            + "</tr>"
        )
    return f"""<div class="scroll"><table>
<caption>コピー合成の評価（{a["n"]} 文）。単語誤り率は、最適化に使っていない 2 つの認識器によるもの。角括弧は 95% 区間。</caption>
<thead><tr><th scope="col">条件</th><th scope="col">Whisper small.en %</th><th scope="col">Whisper large-v3-turbo %</th><th scope="col">有声一致 %</th><th scope="col">時間ずれ ms</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>"""


def copy_section(d: Path, metas: dict[str, Any], clips: dict[str, str], n: int) -> str:
    """The copy-synthesis section, or nothing before `pmvc.copysynth` has been run and judged."""
    path = d / "results_copy.json"
    if not path.exists():
        return ""
    results = json.loads(path.read_text())
    cards = []
    for r in results["utterances"][:n]:
        uid, m = r["id"], metas[r["id"]]
        buttons = "".join(
            button(
                clips, f"{uid}/{v}", d / v / f"{uid}.wav", label,
                f"{r['variants'][v].get('heard', '')}」／large-v3-turbo「{r['variants'][v].get('heard2', '')}", "ref" if v == "real" else "",
            )
            for v, label in COPY_LABELS.items()
        )  # fmt: skip
        cards.append(
            f"""<article class="sentence">
<p class="meta">{esc(uid)} · {"女性" if m["sex"] == "F" else "男性"}話者 {esc(m["speaker"])}</p>
<p class="text">{esc(m["text"])}</p>
<div class="clips">{buttons}</div>
<p class="heard" aria-live="polite">クリップを選ぶと Whisper の聞き取りを表示します。</p>
</article>"""
        )
    return f"""<section aria-labelledby="h-c"><h2 id="h-c">3. コピー合成</h2>
<p>制御列そのものを実録音に合わせて最適化します。出発点は「時間伸縮」で、規則どおりに発音した制御列を録音の時間軸へ伸縮したものです。そこから制御列を少しずつ変えては合成し、音声認識モデル wav2vec2 の中間特徴が録音に近づき、かつ文として読める変更だけを残します。エンジンは微分できないので、物理状態のスナップショットから数百ミリ秒だけ再合成する局所探索です。</p>
{copy_table(results["summary"])}
<p class="small">探索は 1 文あたり 40 反復（1 反復で 96 個の変更を試す）。Whisper は探索に使っていません。</p>
<div class="stack">{"".join(cards)}</div>
</section>"""


def codec_section(d: Path, metas: dict[str, Any], clips: dict[str, str], n: int) -> str:
    """The encoder section, or nothing before `pmvc.encoder`'s renders have been judged."""
    path = d / "results_codec.json"
    if not path.exists():
        return ""
    results = json.loads(path.read_text())
    a = results["summary"]["all"]
    labels = {v: label for v, label in CODEC_LABELS.items() if v in a}
    rows = []
    for v, label in labels.items():
        w2 = a[v]["wer2"]
        rows.append(
            f'<tr><th scope="row">{esc(label)}<span class="sub">{esc(CODEC_NOTES[v])}</span></th>'
            + wer_cell(a[v])
            + f'<td class="num">{w2[0]:.1f}<span class="ci"> [{w2[1]:.1f}, {w2[2]:.1f}]</span></td>'
            + num_cell(a[v], "distance", ".2f")
            + num_cell(a[v], "timing", ".0f")
            + "</tr>"
        )
    table = f"""<div class="scroll"><table>
<caption>エンコーダの評価（学習に使っていない話者 4 人の {a["n"]} 文。文も学習に出てこない）。角括弧は 95% 区間。</caption>
<thead><tr><th scope="col">条件</th><th scope="col">Whisper small.en %</th><th scope="col">Whisper large-v3-turbo %</th><th scope="col">リスナー距離</th><th scope="col">時間ずれ ms</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>"""
    # the same number of sentences from every speaker
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for r in results["utterances"]:
        by_speaker.setdefault(metas[r["id"]]["speaker"], []).append(r)
    per = max(1, n // max(1, len(by_speaker)))
    cards = []
    # each speaker's sentences from a different place in the lists, since pairs of them share lists
    for r in [r for j, rs in enumerate(by_speaker.values()) for r in rs[j * per : (j + 1) * per]]:
        uid, m = r["id"], metas[r["id"]]
        buttons = "".join(
            button(
                clips, f"{uid}/{v}", d / v / f"{uid}.wav", label,
                f"{r['variants'][v].get('heard', '')}」／large-v3-turbo「{r['variants'][v].get('heard2', '')}", "ref" if v == "real" else "",
            )
            for v, label in labels.items()
        )  # fmt: skip
        cards.append(
            f"""<article class="sentence">
<p class="meta">{esc(uid)} · {"女性" if m["sex"] == "F" else "男性"}話者 {esc(m["speaker"])}</p>
<p class="text">{esc(m["text"])}</p>
<div class="clips">{buttons}</div>
<p class="heard" aria-live="polite">クリップを選ぶと Whisper の聞き取りを表示します。</p>
</article>"""
        )
    return f"""<section aria-labelledby="h-e"><h2 id="h-e">4. エンコーダ（コーデックの試作）</h2>
{{{{codec_text}}}}
{table}
<p class="small">リスナー距離は、探索が下げようとする量そのもの（wav2vec2 の中間特徴の 1 − コサイン類似度を、録音と合成音のフレームどうしで比べた平均）。同じ文を読んだ別の実話者どうしで約 0.16 です。時間ずれは第 1 節と同じ定義。Whisper は学習にも探索にも使っていません。</p>
<div class="stack">{"".join(cards)}</div>
{codec_perturbed(d, metas, clips, 2)}
</section>"""


def abroad_section(d: Path, ja: Path, clips: dict[str, str]) -> str:
    """Recordings from outside TSP through the same encoders: `pmvc.judge`'s results in `ja`,
    beside its results for TSP's test speakers in `d`. Nothing before they exist."""
    if not (ja / "results.json").exists():
        return ""
    results = json.loads((ja / "results.json").read_text())
    summary = results["summary"]
    metas = {p.stem: json.loads(p.read_text()) for p in (ja / "meta").glob("*.json")}
    all_ja = summary["ja"]

    def row(label: str, note: str, cells: str) -> str:
        return f'<tr><th scope="row">{esc(label)}<span class="sub">{esc(note)}</span></th>{cells}</tr>'

    # every encoder on the Japanese recordings
    rows = [
        row(ABROAD_LABELS[v], ABROAD_NOTES[v], rate_cell(all_ja[v], "rate") + rate_cell(all_ja[v], "rate2") + rate_cell(all_ja[v], "phoneRate") + num_cell(all_ja[v], "distance", ".2f"))
        for v in ("real", "sim", "enc", "encn")
    ]
    if "other" in all_ja:
        rows.append(row("別の実話者", f"同じ文を読んだもう 1 人の録音（{all_ja['other']['n']} 文）", '<td class="num">–</td>' * 2 + rate_cell(all_ja["other"], "phoneRate") + '<td class="num">–</td>'))
    by_encoder = f"""<div class="scroll"><table>
<caption>日本語の録音 {all_ja["n"]} 文（3 話者 × 100 文）。Whisper の 2 列は音素誤り率で、文と聞き取りをどちらも読みの音素列に直して比べています。角括弧は 95% 区間。</caption>
<thead><tr><th scope="col">条件</th><th scope="col">Whisper small %</th><th scope="col">Whisper large-v3-turbo %</th><th scope="col">音声記号の不一致 %</th><th scope="col">リスナー距離</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>"""

    # encoder C′ by set of recordings, English beside Japanese
    sets: list[tuple[str, str, dict[str, Any]]] = []
    home = d / "results_judge.json"
    if home.exists():
        sets.append(("英語・TSP のテスト話者", "学習と同じコーパスの、学習に使っていない 4 人", json.loads(home.read_text())["summary"]["en"]))
    sets += [(label, "", summary[name]) for name, label in ABROAD_SETS.items() if name in summary]
    sets.sort(key=lambda x: x[2]["lang"] == "ja")
    rows = []
    for label, note, g in sets:
        fine = "rate2" if g["lang"] == "ja" else "letterRate2"
        rows.append(
            row(label, note, f'<td class="num">{g["n"]}</td>'
                + (rate_cell(g["encn"], "rate2") if g["lang"] != "ja" else '<td class="num">–</td>')
                + rate_cell(g["real"], fine) + rate_cell(g["encn"], fine)
                + rate_cell(g["encn"], "phoneRate") + rate_cell(g.get("other", {}), "phoneRate")
                + num_cell(g["encn"], "distance", ".2f") + num_cell(g["encn"], "voicing", ".0f", 100.0) + num_cell(g["encn"], "f0Corr", ".2f"))
        )  # fmt: skip
    by_set = f"""<div class="scroll"><table>
<caption>エンコーダ C′ を、録音の種類ごとに同じ物差しで測ったもの。誤り率は Whisper large-v3-turbo。「細かい単位」は、英語は文字、日本語は音素で数えた誤り率です。</caption>
<thead><tr><th scope="col">録音</th><th scope="col">文数</th><th scope="col">単語誤り率 C′ %</th><th scope="col">細かい単位 実録音 %</th><th scope="col">細かい単位 C′ %</th>
<th scope="col">音声記号の不一致 C′ %</th><th scope="col">同 別の実話者 %</th><th scope="col">リスナー距離</th><th scope="col">有声一致 %</th><th scope="col">F0 相関</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>"""

    # which phonemes of the sentences were lost
    table = results["sounds"].get("ja", {})
    lost = sorted(
        ((p, table["real"][p], table["sim"][p], table["encn"][p]) for p in table.get("encn", {}) if table["encn"][p][1] >= 60),
        key=lambda x: x[1][0] / x[1][1] - x[3][0] / x[3][1],
    )
    rows = [
        f'<tr><th scope="row">{esc(p)}<span class="sub">{esc(SOUND_NOTES.get(p, ""))}</span></th><td class="num">{real[1]}</td>'
        + "".join(f'<td class="num">{100.0 * a / n:.1f}</td>' for a, n in (real, sim, encn))
        + "</tr>"
        for p, real, sim, encn in lost[:10]
    ]
    total = [sum(table[v][p][0] for p in table[v]) / sum(table[v][p][1] for p in table[v]) for v in ("real", "sim", "encn")] if table else []
    by_sound = f"""<div class="scroll"><table>
<caption>文の音素のうち、Whisper large-v3-turbo の聞き取りで別の音になったか消えた割合。C′ と実録音の差が大きい順に 10 個（60 回以上出てくる音素から）。全音素の平均は実録音 {100 * total[0]:.1f}%、A {100 * total[1]:.1f}%、C′ {100 * total[2]:.1f}% です。</caption>
<thead><tr><th scope="col">音素</th><th scope="col">出現数</th><th scope="col">実録音 %</th><th scope="col">エンコーダ A %</th><th scope="col">エンコーダ C′ %</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>""" if rows else ""  # fmt: skip

    # the search from C′'s tracks, where it was run
    searched: dict[str, Any] = {}
    by_search = ""
    if (ja / "results_search.json").exists():
        search = json.loads((ja / "results_search.json").read_text())
        searched = {r["id"]: r for r in search["utterances"]}
        g = search["summary"]["ja"]
        rows = [
            row(ABROAD_LABELS[v], ABROAD_NOTES[v], rate_cell(g[v], "rate") + rate_cell(g[v], "rate2") + rate_cell(g[v], "phoneRate") + num_cell(g[v], "distance", ".2f"))
            for v in ("real", "encn", "encncopy")
        ]
        by_search = f"""<div class="scroll"><table>
<caption>C′ の制御列を探索で録音に近づけた場合（日本語 {g["n"]} 文、3 話者 × 8 文）。列は最初の表と同じです。</caption>
<thead><tr><th scope="col">条件</th><th scope="col">Whisper small %</th><th scope="col">Whisper large-v3-turbo %</th><th scope="col">音声記号の不一致 %</th><th scope="col">リスナー距離</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>"""

    # two sentences of each Japanese speaker (searched ones where there are any), one of each English
    by_id = {r["id"]: r for r in results["utterances"]}
    cards = []
    for name, label in ABROAD_SETS.items():
        mine = sorted(i for i in by_id if metas[i]["speaker"] == name)
        ja_set = bool(mine) and by_id[mine[0]]["lang"] == "ja"
        chosen = ([i for i in mine if i in searched] or mine)[1:3] if ja_set else mine[:1]
        for uid in chosen:
            r, m = by_id[uid], metas[uid]
            variants = {**r["variants"], **{v: x for v, x in searched.get(uid, {}).get("variants", {}).items() if v == "encncopy"}}
            buttons = "".join(
                button(
                    clips, f"{uid}/{v}", ja / v / f"{uid}.wav", ABROAD_LABELS[v],
                    f"{variants[v]['heard']}」／large-v3-turbo「{variants[v]['heard2']}", "ref" if v == "real" else "", "small" if ja_set else "small.en",
                )
                for v in ABROAD_LABELS if v in variants and (ja_set or v in ("real", "encn"))
            )  # fmt: skip
            cards.append(
                f"""<article class="sentence">
<p class="meta">{esc(uid)} · {esc(label)}</p>
<p class="text">{esc(m["text"])}</p>
<div class="clips">{buttons}</div>
<p class="heard" aria-live="polite">クリップを選ぶと Whisper の聞き取りを表示します。</p>
</article>"""
            )
    return f"""<section aria-labelledby="h-j"><h2 id="h-j">1. 英語で学習したエンコーダに日本語の録音を通す</h2>
{{{{abroad_text}}}}
{by_encoder}
{{{{abroad_measures}}}}
{by_set}
{by_sound}
{by_search}
<div class="stack">{"".join(cards)}</div>
</section>"""


def yardstick_table(ja: Path) -> str:
    """`pmvc.yardstick`'s comparison: each model at the layer that tells a Japanese render from
    another speaker best, and the listener in use."""
    path = ja / "yardstick.json"
    if not path.exists():
        return ""
    best: dict[str, dict[str, Any]] = {}
    for r in json.loads(path.read_text()):
        if r["model"] not in best or (r["ja"]["dprime"] > best[r["model"]]["ja"]["dprime"] and r["model"] != "facebook/wav2vec2-base-960h") or (r["model"], r["layer"]) == ("facebook/wav2vec2-base-960h", 9):
            best[r["model"]] = r
    rows = []
    for name, label in YARDSTICK_LABELS.items():
        if name in best:
            r = best[name]
            cells = "".join(f'<td class="num">{r[lang]["other"]:.2f}</td><td class="num">{r[lang]["render"]:.2f}</td><td class="num">{r[lang]["ratio"]:.2f}</td>' for lang in ("en", "ja"))
            search = f'<td class="num">{r["search"][0]:.2f} → {r["search"][1]:.2f}</td>' if "search" in r else '<td class="num">–</td>'
            rows.append(f'<tr><th scope="row">{esc(label)}<span class="sub">第 {r["layer"]} 層</span></th>{cells}{search}</tr>')
    return f"""<div class="scroll"><table>
<caption>物差しの候補の比較。同じ文を読んだ 2 人の録音どうしの距離（別の話者）と、録音とエンコーダ C′ の合成音の距離（合成音）を、どちらも時間伸縮を許して測ったもの。比が 1 なら、合成音は別人の読み上げと同じ近さです。右端は、英語の物差しに向けた探索の前後の距離です。</caption>
<thead><tr><th scope="col">モデル</th><th scope="col">英語 別の話者</th><th scope="col">英語 合成音</th><th scope="col">英語 比</th>
<th scope="col">日本語 別の話者</th><th scope="col">日本語 合成音</th><th scope="col">日本語 比</th><th scope="col">探索の前 → 後</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>"""


Tests = tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]


def three_tests(d: Path, ja: Path, jvs: Path, name_jvs: str, name: str) -> Tests:
    """`pmvc.judge`'s results on JVS's held-out speakers (`jvs`), on the recordings of `ja` and on
    TSP's test speakers (`d`), and their summaries as a table's columns: (heading, summary)."""
    files = {"jvs": jvs / f"{name_jvs}.json", "ja": ja / f"{name}.json", "tsp": d / f"{name}.json"}
    results = {k: json.loads(p.read_text()) for k, p in files.items() if p.exists()}
    sets: list[tuple[str, dict[str, Any]]] = [("JVS 日本語（8 話者）", results["jvs"]["summary"]["ja"])]
    if "ja" in results:
        # "日本語・女性（Hi-Fi-CAPTAIN）" as a column's heading: the corpus first, shortened
        sets += [(("JSUT " if "JSUT" in label else "HFC ") + label.split("（")[0], results["ja"]["summary"][name]) for name, label in ABROAD_SETS.items() if results["ja"]["summary"][name]["lang"] == "ja"]
    if "tsp" in results:
        sets.append(("TSP 英語（4 話者）", results["tsp"]["summary"]["en"]))
    if "ja" in results:
        sets.append(("HFC 英語（2 話者）", results["ja"]["summary"]["en"]))
    return results, sets


def grid(sets: list[tuple[str, dict[str, Any]]], variants: list[str], labels: dict[str, str], notes: dict[str, str], caption: str, key: str, other: bool = False) -> str:
    """A table of one measure: a row for each variant that has it, a column for each test."""
    head = "".join(f'<th scope="col">{esc(label)}<span class="sub">{g["n"]} 文</span></th>' for label, g in sets)

    def cell(m: dict[str, Any]) -> str:
        formats = {"quality": ".2f", "natural": ".2f", "oddClicks": ".2f", "releases": ".2f", "releaseLevel": ".1f", "silenceSounds": ".2f", "silenceLevel": ".1f", "closureClicks": ".2f", "extra4k": ".2f"}
        return rate_cell(m, key) if isinstance(m.get(key), list) else num_cell(m, key, formats.get(key, ".0f"))

    rows = [
        f'<tr><th scope="row">{esc(labels[v])}<span class="sub">{esc(notes[v])}</span></th>' + "".join(cell(g[v]) for _, g in sets) + "</tr>"
        for v in variants if key in sets[0][1][v]
    ]  # fmt: skip
    if other:
        rows.append('<tr><th scope="row">別の実話者<span class="sub">同じ文を読んだもう 1 人の録音</span></th>' + "".join(rate_cell(g.get("other", {}), key) for _, g in sets) + "</tr>")
    return f"""<div class="scroll"><table>
<caption>{caption}</caption>
<thead><tr><th scope="col">条件</th>{head}</tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>"""


def sentence_cards(clips: dict[str, str], ja: Path, jvs: Path, results: dict[str, Any], variants: list[str], labels: dict[str, str], voiced: tuple[str, ...] = (), plain: tuple[str, ...] = ()) -> str:
    """The cards of the three tests: one sentence each of four JVS speakers, then the sentences
    of the first section's cards, then English. `voiced`: variants rendered with a fitted
    voice, left out where the speaker's pitch leaves the voice as it was. `plain`: variants
    shown only there, in their place."""

    def unchanged(where: Path, v: str, uid: str) -> bool:
        glottis = json.loads((where / v / f"{uid}.tracks.json").read_text()).get("glottis")
        return glottis is None or abs(glottis["openQuotient"] - 0.6) < 0.03

    shown: list[tuple[Path, dict[str, Any], dict[str, Any], str]] = []
    if "ja" in results:
        metas = {p.stem: json.loads(p.read_text()) for p in (ja / "meta").glob("*.json")}
        by_id = {r["id"]: r for r in results["ja"]["utterances"]}
        searched = sorted(json.loads((ja / "results_search.json").read_text())["utterances"], key=lambda r: r["id"]) if (ja / "results_search.json").exists() else []
        for name, label in ABROAD_SETS.items():
            mine = [r["id"] for r in searched if metas[r["id"]]["speaker"] == name][1:3] or sorted(i for i in by_id if metas[i]["speaker"] == name)[:1]
            shown += [(ja, by_id[uid], metas[uid], label) for uid in mine]
    rows = sorted(results["jvs"]["utterances"], key=lambda r: r["id"])
    speakers = sorted({r["speaker"] for r in rows})
    # two women and two men, each on a different sentence
    for k, name in enumerate(speakers[:2] + speakers[-2:]):
        r = [x for x in rows if x["speaker"] == name][3 + 5 * k]
        shown.insert(k, (jvs, r, json.loads((jvs / "meta" / f"{r['id']}.json").read_text()), f"日本語・{'女性' if r['sex'] == 'F' else '男性'}（JVS {name}）"))
    cards = []
    for where, r, m, label in shown:
        buttons = "".join(
            button(
                clips, f"{r['id']}/{v}", where / v / f"{r['id']}.wav", labels[v],
                f"{r['variants'][v]['heard']}」／large-v3-turbo「{r['variants'][v]['heard2']}", "ref" if v == "real" else "", "small" if r["lang"] == "ja" else "small.en",
            )
            for v in variants if v in r["variants"] and not (v in voiced and unchanged(where, v, r["id"])) and not (v in plain and not all(unchanged(where, w, r["id"]) for w in voiced))
        )  # fmt: skip
        cards.append(
            f"""<article class="sentence">
<p class="meta">{esc(r["id"])} · {esc(label)}</p>
<p class="text">{esc(m["text"])}</p>
<div class="clips">{buttons}</div>
<p class="heard" aria-live="polite">クリップを選ぶと Whisper の聞き取りを表示します。</p>
</article>"""
        )
    return "".join(cards)


def taught_section(d: Path, ja: Path, jvs: Path, clips: dict[str, str]) -> str:
    """Encoders taught with Japanese recordings, a multilingual listener, or both. Nothing
    before `pmvc.judge`'s results for them on JVS's held-out speakers exist."""
    if not (jvs / "results.json").exists():
        return ""
    results, sets = three_tests(d, ja, jvs, "results", "results_taught")
    variants = [v for v in TAUGHT_LABELS if all(v in g for _, g in sets)]
    tables = (
        grid(sets, variants, TAUGHT_LABELS, TAUGHT_NOTES, "Whisper large-v3-turbo の誤り率 %。日本語の列は音素、英語の列は単語で数えています。角括弧は 95% 区間。", "rate2")
        + grid(sets, variants, TAUGHT_LABELS, TAUGHT_NOTES, "Whisper small の誤り率 %（英語の列は small.en）。", "rate")
        + grid(sets, variants, TAUGHT_LABELS, TAUGHT_NOTES, "音声記号の不一致 %。録音と合成音を音声記号認識器に書き取らせて、食い違った記号の割合です。", "phoneRate", other=True)
    )
    return f"""<section aria-labelledby="h-k"><h2 id="h-k">2. 日本語の録音を学習に加える</h2>
{{{{taught_text}}}}
{yardstick_table(ja)}
{{{{taught_between}}}}
{tables}
<div class="stack">{sentence_cards(clips, ja, jvs, results, [v for v in variants if v in ("real", "encg")], TAUGHT_LABELS)}</div>
</section>"""


def tuned_section(d: Path, ja: Path, jvs: Path, fit: Path, clips: dict[str, str]) -> str:
    """The sounds before the speech taken out (encoder H), and the female voice refitted
    (`pmvc.calibrate --by nisqa`'s settings in `fit`). Nothing before `pmvc.judge`'s and
    `pmvc.natural`'s results for them on JVS's held-out speakers exist."""
    if not (jvs / "results_tuned.json").exists():
        return ""
    results, sets = three_tests(d, ja, jvs, "results_tuned", "results_tuned")
    variants = [v for v in TUNED_LABELS if all(v in g for _, g in sets)]
    stray = grid(sets, [v for v in variants if v in ("encn", "encg", "ench", "enchc")], TUNED_LABELS, TUNED_NOTES,
                 "発話の外（録音が無音のところ）で、合成音が 30 ms 以上有声になった録音の割合 %。発話の前後 60 ms は除いています。TSP と JSUT の録音は、発話の前にほとんど無音がありません。", "stray")  # fmt: skip
    # NISQA by sex: the same files' summaries by language and sex
    by_sex: list[tuple[str, dict[str, Any]]] = []
    for key, heading in (("jvs", "JVS"), ("ja", "HFC・JSUT"), ("tsp", "TSP 英語"), ("ja", "HFC 英語")):
        if key in results:
            lang = "en" if "英語" in heading else "ja"
            by_sex += [(f"{heading} {'女性' if sex == 'F' else '男性'}", results[key]["summary"][f"{lang}-{sex}"]) for sex in "FM" if f"{lang}-{sex}" in results[key]["summary"]]
    quality = grid(by_sex, variants, TUNED_LABELS, TUNED_NOTES, "NISQA が予測する品質（1〜5 の平均オピニオン評点）。女性の列と男性の列を分けています。JVS の男性 4 人のうち 3 人は声が高く（中央値 172〜200 Hz）、調整の対象です。", "quality")
    by_pitch = ""
    if (fit / "voice-F2.pitchmix.json").exists():
        mix = json.loads((fit / "voice-F2.pitchmix.json").read_text())
        head = "".join(f'<th scope="col">{w:g}</th>' for w in mix["shares"])
        rows = "".join(
            f'<tr><th scope="row">{esc(PITCH_SPEAKERS.get(r["speaker"], r["speaker"]))}</th><td class="num">{r["pitch"]:.0f}</td>' + "".join(f'<td class="num">{r["quality"][str(w)]:.2f}</td>' for w in mix["shares"]) + "</tr>"
            for r in mix["speakers"]
        )
        by_pitch = f"""<div class="scroll"><table>
<caption>調整した設定（調整 2、対策 1+2）をどの程度使うと NISQA の点数がどうなるか（話者ごとに {mix["n"]} 文、エンコーダ H の制御列）。0 は CHIMERA の既定、1 は調整後、その間は設定を混ぜたもの。話者は声の高さの順です。</caption>
<thead><tr><th scope="col">話者</th><th scope="col">声の高さ Hz</th>{head}</tr></thead>
<tbody>{rows}</tbody></table></div>"""
    tables = (
        grid(sets, variants, TUNED_LABELS, TUNED_NOTES, "Whisper large-v3-turbo の誤り率 %。日本語の列は音素、英語の列は単語で数えています。角括弧は 95% 区間。", "rate2")
        + grid(sets, variants, TUNED_LABELS, TUNED_NOTES, "Whisper small の誤り率 %（英語の列は small.en）。", "rate")
        + grid(sets, variants, TUNED_LABELS, TUNED_NOTES, "音声記号の不一致 %。", "phoneRate", other=True)
    )
    settings = ""
    if (fit / "voice-F2.json").exists():
        from .calibrate import PARAMS

        files = {"調整 1": fit / "voice-F1.json", "調整 2": fit / "voice-F2.json", "調整 3": fit / "voice-F3.json"}
        fits = [json.loads(p.read_text())["settings"] for p in files.values() if p.exists()]
        rows = "".join(
            f'<tr><th scope="row">{esc(VOICE_LABELS[name])}</th><td class="num">{default:.3g}</td>' + "".join(f'<td class="num">{f.get(name, default):.3g}</td>' for f in fits) + "</tr>"
            for name, default, _, _ in PARAMS
        )
        heads = "".join(f'<th scope="col">{esc(name)}</th>' for name, p in files.items() if p.exists())
        settings = f"""<div class="scroll"><table>
<caption>声の設定。CHIMERA の既定値と、Hi-Fi-CAPTAIN の女性話者の録音 16 文に合わせて調整した値。雑音源のゲインは既定値に対する倍率です。</caption>
<thead><tr><th scope="col">設定</th><th scope="col">既定</th>{heads}</tr></thead>
<tbody>{rows}</tbody></table></div>"""
    clicking = [v for v in variants if v not in ("encn", "encg")]
    silences = grid(sets, clicking, TUNED_LABELS, TUNED_NOTES, "録音が無音の区間（破裂音の閉鎖とポーズ。発話のレベルより 45 dB 以上小さい区間で、両端の 15 ms は除く）の中で、合成音が発話のレベルの −40 dB を超えた回数（発話 1 秒あたり）。", "silenceSounds") + grid(
        sets, clicking, TUNED_LABELS, TUNED_NOTES, "同じ区間の音量（発話のレベルに対する dB）。", "silenceLevel"
    )
    closure = grid(sets, clicking, TUNED_LABELS, TUNED_NOTES, "閉鎖のクリック音の回数（発話 1 秒あたり）。4〜16 kHz の帯域で、録音が前後 15 ms にしていることより 12 dB 以上大きい 20 ms 以下の音のうち、録音が無音になった直後か、無音の中にあるもの。", "closureClicks") + grid(
        sets, clicking, TUNED_LABELS, TUNED_NOTES, "4〜16 kHz の帯域で、録音にない音の回数（発話 1 秒あたり）。場所を問わず数えたもの。JVS の録音は 24 kHz なので 4〜11.5 kHz で数えています。", "extra4k"
    )
    clicks = (
        grid(sets, clicking, TUNED_LABELS, TUNED_NOTES, "実録音にはほとんどない種類のクリック音の回数（1 秒あたり）: 舌が閉じる瞬間（録音が音から無音に変わるところ）と、音が続いている途中のもの。クリック音は、2〜9 kHz の帯域で前後から 8〜12 dB 以上突き出た短い音として数えています。", "oddClicks")
        + grid(sets, clicking, TUNED_LABELS, TUNED_NOTES, "破裂（録音が無音から音に変わるところ）のクリック音の回数（1 秒あたり）。", "releases")
        + grid(sets, clicking, TUNED_LABELS, TUNED_NOTES, "破裂のクリック音の大きさ（発話のレベルに対する dB、平均）。", "releaseLevel")
    )
    return f"""<section aria-labelledby="h-u"><h2 id="h-u">3. 発話の前の余計な音と、女声の声色</h2>
{{{{tuned_text}}}}
{stray}
{{{{tuned_voice}}}}
{by_pitch}
{settings}
{{{{tuned_rule}}}}
{quality}
{{{{tuned_walls}}}}
{silences}
{{{{tuned_closure}}}}
{closure}
{{{{tuned_clicks}}}}
{clicks}
{{{{tuned_check}}}}
{tables}
<p class="small">下のカードには、H、対策 1+2、決まった「調整 2（対策 1+2）」を残しています（「対策 1」だけの音は、後の節の音の場所を空けるために外しました）。</p>
<div class="stack">{sentence_cards(clips, ja, jvs, results, ["real", "ench", "enchc", "ench2c"], TUNED_LABELS, ("ench2c",), ("enchc",))}</div>
</section>"""


def retaught_section(d: Path, ja: Path, jvs: Path, clips: dict[str, str]) -> str:
    """Encoder I: the recipe of H, taught through a stand-in for the voice the listening
    settled on. Nothing before `pmvc.judge` has its results on JVS's held-out speakers."""
    if not (jvs / "results_tuned.json").exists():
        return ""
    results, sets = three_tests(d, ja, jvs, "results_tuned", "results_tuned")
    if not all("enci" in g for _, g in sets):
        return ""
    variants = list(RETAUGHT_LABELS)
    by_sex: list[tuple[str, dict[str, Any]]] = []
    for key, heading in (("jvs", "JVS"), ("ja", "HFC・JSUT"), ("tsp", "TSP 英語"), ("ja", "HFC 英語")):
        lang = "en" if "英語" in heading else "ja"
        by_sex += [(f"{heading} {'女性' if sex == 'F' else '男性'}", results[key]["summary"][f"{lang}-{sex}"]) for sex in "FM" if f"{lang}-{sex}" in results[key]["summary"]]
    tables = (
        grid(sets, variants, RETAUGHT_LABELS, RETAUGHT_NOTES, "音声記号の不一致 %。角括弧は 95% 区間。", "phoneRate", other=True)
        + grid(sets, variants, RETAUGHT_LABELS, RETAUGHT_NOTES, "Whisper large-v3-turbo の誤り率 %。日本語の列は音素、英語の列は単語で数えています。", "rate2")
        + grid(sets, variants, RETAUGHT_LABELS, RETAUGHT_NOTES, "Whisper small の誤り率 %（英語の列は small.en）。", "rate")
        + grid(by_sex, variants, RETAUGHT_LABELS, RETAUGHT_NOTES, "NISQA が予測する品質（1〜5）。", "quality")
    )
    sounds = (
        grid(sets, variants, RETAUGHT_LABELS, RETAUGHT_NOTES, "閉鎖のクリック音の回数（発話 1 秒あたり。数え方は第 3 節と同じ）。", "closureClicks")
        + grid(sets, variants, RETAUGHT_LABELS, RETAUGHT_NOTES, "録音が無音の区間の中で、合成音が鳴った回数（発話 1 秒あたり）。", "silenceSounds")
        + grid(sets, variants, RETAUGHT_LABELS, RETAUGHT_NOTES, "破裂のクリック音の大きさ（発話のレベルに対する dB、平均）。", "releaseLevel")
    )
    return f"""<section aria-labelledby="h-i"><h2 id="h-i">4. 調整後の声でエンコーダを学習し直す</h2>
{{{{retaught_text}}}}
{tables}
{{{{retaught_sounds}}}}
{sounds}
<div class="stack">{sentence_cards(clips, ja, jvs, results, ["real", "ench2c", "enci"], RETAUGHT_LABELS)}</div>
</section>"""


def speaker_section(jvs: Path, everyone: Path, clips: dict[str, str]) -> str:
    """Each speaker's own body and glottis (`pmvc.speaker`): whose voice the renders of JVS's
    eight test speakers then are (`jvs`/whose_spk.json and whose_spk2.json, a second model's
    say), and of all hundred (`everyone`/whose_speakers.json, whose_speakers2.json), what it
    costs (`jvs`/results_spk.json) and the settings (`jvs`/speakers.json, speakers_body.json).
    Nothing before the first exists."""
    if not (jvs / "whose_spk.json").exists() or not (jvs / "results_spk.json").exists():
        return ""
    results = json.loads((jvs / "results_spk.json").read_text())
    summary = results["summary"]

    def whose_table(names: tuple[str, str], where: Path, rows: dict[str, str], what: str) -> str:
        """Two models' say on a set of clips; nothing if the first was not asked."""
        if not (where / f"{names[0]}.json").exists():
            return ""
        first = json.loads((where / f"{names[0]}.json").read_text())
        second = json.loads((where / f"{names[1]}.json").read_text())["variants"] if (where / f"{names[1]}.json").exists() else {}

        def cells(m: dict[str, float] | None) -> str:
            return '<td class="num">–</td>' * 3 if m is None else f'<td class="num">{100 * m["top1"]:.1f}</td><td class="num">{m["own"]:.3f}</td><td class="num">{m["other"]:.3f}</td>'

        lines = "".join(f'<tr><th scope="row">{esc(label)}</th>{cells(first["variants"][v])}{cells(second.get(v))}</tr>' for v, label in rows.items() if v in first["variants"])
        return f"""<div class="scroll"><table>
<caption>{what}（{first["speakers"]} 話者、{first["clips"]:,} 文）。「本人」は、最も近い話者が本人だった割合 %。類似度はコサイン類似度で、「他人」は同性の他の話者との平均。左の 3 列は当てはめに使った照合モデル（WavLM）、右の 3 列は当てはめに使っていない照合モデル（UniSpeech-SAT）。</caption>
<thead><tr><th scope="col">条件</th><th scope="col">本人</th><th scope="col">本人との類似度</th><th scope="col">他人との類似度</th><th scope="col">本人（別モデル）</th><th scope="col">本人との類似度</th><th scope="col">他人との類似度</th></tr></thead>
<tbody>{lines}</tbody></table></div>"""

    whose = (
        whose_table(("whose_spk", "whose_spk2"), jvs, {"real": "実録音", **{v: SPEAKER_LABELS[v] for v in ("enci", "encib", "encis", "encit")}}, "エンコーダ I の制御列。テストの 8 話者")
        + whose_table(("whose_speakers", "whose_speakers2"), everyone, {"real": "実録音", "encir": SPEAKER_LABELS["enci"], **{v: SPEAKER_LABELS[v] for v in ("encib", "encis", "encit")}}, "同じく、全 100 話者のテスト文")
        + whose_table(("whose_spk", "whose_spk2"), jvs, {v: SPEAKER_LABELS[v] for v in ("ttsj", "ttsjb", "ttsjs", "ttsjt")}, "TTS が書いた制御列。テストの 8 話者")
        + whose_table(("whose_speakers", "whose_speakers2"), everyone, {v: SPEAKER_LABELS[v] for v in ("ttsj", "ttsjb", "ttsjs", "ttsjt")}, "同じく、全 100 話者のテスト文")
    )
    def apart_table(names: tuple[str, str], rows: dict[str, str], where: Path = jvs) -> str:
        """The same for the women's clips and the men's apart (`pmvc.whose`'s `sexes`), of the eight
        test speakers or (`where`) of all hundred; nothing before it is there."""
        if not (where / f"{names[0]}.json").exists():
            return ""
        whole = json.loads((where / f"{names[0]}.json").read_text())
        first = whole["variants"]
        second = json.loads((where / f"{names[1]}.json").read_text())["variants"] if (where / f"{names[1]}.json").exists() else {}
        few = whole["speakers"] <= 8
        digits = ".0f" if few else ".1f"

        def cells(v: str) -> str:
            out = "".join(f'<td class="num">{100 * first[v]["sexes"][x]["top1"]:{digits}}</td><td class="num">{first[v]["sexes"][x]["own"]:.3f}</td><td class="num">{first[v]["sexes"][x]["other"]:.3f}</td>' for x in "FM")
            return out + "".join(f'<td class="num">{100 * second[v]["sexes"][x]["top1"]:{digits}}</td>' if v in second else '<td class="num">–</td>' for x in "FM")

        lines = "".join(f'<tr><th scope="row">{esc(label)}</th>{cells(v)}</tr>' for v, label in rows.items() if v in first)
        what = "テストの 8 話者の照合を、女性の 100 文と男性の 100 文に分けたもの。「本人」は、8 話者のうち" if few else f"全 {whole['speakers']} 話者のテスト文（{whole['clips']:,} 文）の照合を、女性と男性に分けたもの。「本人」は、{whole['speakers']} 話者のうち"
        return f"""<div class="scroll"><table>
<caption>{what}最も近い話者が本人だった割合 %。類似度は WavLM のもので、「他人」は同性の他の話者との平均。右の 2 列は、当てはめに使っていない照合モデル（UniSpeech-SAT）での「本人」。</caption>
<thead><tr><th scope="col">条件</th><th scope="col">女性: 本人</th><th scope="col">本人との類似度</th><th scope="col">他人との類似度</th><th scope="col">男性: 本人</th><th scope="col">本人との類似度</th><th scope="col">他人との類似度</th><th scope="col">女性: 本人（別モデル）</th><th scope="col">男性: 本人（別モデル）</th></tr></thead>
<tbody>{lines}</tbody></table></div>"""

    def nasal_table(rows: dict[str, str]) -> str:
        """The renders' nasals and vowels against the recordings' (`pmvc.nasals`); nothing before it is there."""
        if not (jvs / "nasals.json").exists():
            return ""
        got = json.loads((jvs / "nasals.json").read_text())["sexes"]
        keys = ("level", "nasal", "beside", "h2", "h3")

        def cells(v: str) -> str:
            return "".join(f'<td class="num">{got[x][v][k] - (got[x]["real"][k] if v != "real" else 0.0):+.1f}</td>' for x in "FM" for k in keys)

        lines = "".join(f'<tr><th scope="row">{esc(label)}</th>{cells(v)}</tr>' for v, label in rows.items() if v in got["F"])
        heads = "".join(f'<th scope="col">{x}{esc(h)}</th>' for x in ("女性: ", "男性: ") for h in ("鼻音の強さ", "鼻音の第 2 倍音", "鼻音の隣の母音の第 2 倍音", "母音の第 2 倍音", "母音の第 3 倍音"))
        return f"""<div class="scroll"><table>
<caption>録音と合成音の比較（テストの女性 {got["F"]["n"]} 文、男性 {got["M"]["n"]} 文）。単位は dB。1 行目は録音の値、2 行目からは録音との差で、0 に近いほど録音どおりです。「鼻音の強さ」は、同じ文の鼻音から離れた母音に対する鼻音（m、n、ん）の強さ。倍音はどれも基本波に対する強さで、「母音」は鼻音から離れた母音です。</caption>
<thead><tr><th scope="col">条件</th>{heads}</tr></thead>
<tbody>{lines}</tbody></table></div>"""

    def bright_table(rows: dict[str, str]) -> str:
        """The renders' vowels against the recordings', band by band (`pmvc.bands`); nothing before it is there."""
        if not (jvs / "bands.json").exists():
            return ""
        got = json.loads((jvs / "bands.json").read_text())
        edges, by = got["edges"], got["sexes"]
        picked = [k for k, lo in enumerate(edges[:-1]) if 1000 <= lo < 7500]

        def cells(v: str) -> str:
            out = ""
            for x in "FM":
                real = by[x]["real"]
                out += f'<td class="num">{by[x][v]["h2"] - (real["h2"] if v != "real" else 0.0):+.1f}</td>'
                out += "".join(f'<td class="num">{by[x][v]["vowels"][k] - (real["vowels"][k] if v != "real" else 0.0):+.1f}</td>' for k in picked)
            return out

        lines = "".join(f'<tr><th scope="row">{esc(label)}</th>{cells(v)}</tr>' for v, label in rows.items() if v in by["F"])
        heads = "".join(f'<th scope="col">{x}第 2 倍音</th>' + "".join(f'<th scope="col">{edges[k] / 1000:g}〜{edges[k + 1] / 1000:g} kHz</th>' for k in picked) for x in ("女性: ", "男性: "))
        return f"""<div class="scroll"><table>
<caption>母音の中ほどのフレームでの、録音と合成音の比較（テストの女性 {by["F"]["n"]} 文、男性 {by["M"]["n"]} 文）。単位は dB。1 行目は録音の値、2 行目からは録音との差で、0 に近いほど録音どおりです。「第 2 倍音」は基本波に対する強さ、帯域は母音全体の強さに対するその帯域の強さです。閉じ方の鋭さは女性だけに当てているので、その 3 行の男性の値は「体格だけ、副鼻腔を大きく」と同じです。</caption>
<thead><tr><th scope="col">条件</th>{heads}</tr></thead>
<tbody>{lines}</tbody></table></div>"""

    def closure_table() -> str:
        """The sharpness of the closure measured for the test's women (`pmvc.speaker closure`); nothing before it is there."""
        if not (jvs / "speakers_closure.json").exists():
            return ""
        got = json.loads((jvs / "speakers_closure.json").read_text())["speakers"]
        lines = "".join(
            f'<tr><th scope="row">{esc(n)}</th><td class="num">{got[n]["closure"]["level"][0]:+.1f}</td><td class="num">{got[n]["settings"]["closureSharpness"]:.2f}</td><td class="num">{got[n]["settings"]["closureSharpness"] / 2:.2f}</td></tr>'
            for n in speakers if "closure" in got.get(n, {})
        )  # fmt: skip
        return f"""<div class="scroll"><table>
<caption>テストの女性 4 人の、声門の閉じ方の鋭さ（0 がいちばん柔らかく、1 がいちばん鋭い）。学習用の 6 文で測りました。「鋭さ 0 での差」は、これまでの声（鋭さ 0）で鳴らしたときの 1.6〜5.5 kHz の強さの、録音との差（dB）。</caption>
<thead><tr><th scope="col">話者</th><th scope="col">鋭さ 0 での差</th><th scope="col">測った値</th><th scope="col">半分</th></tr></thead>
<tbody>{lines}</tbody></table></div>"""

    with_closure = {
        "real": "実録音", "enci": "これまで", "encibn": "体格だけ、副鼻腔を大きく", "encibh": "＋ 閉じ方を鋭く（半分）", "encibf": "＋ 閉じ方を鋭く（測った値）", "encibk": "＋ 閉じ方を鋭く（全員 0.3）",
        "encis": "体格 + 声門を当てはめ", "encit": "体格 4 値 + 声門 + 構えを当てはめ",
    }  # fmt: skip
    with_sinus = {
        "real": "実録音", "enci": "これまで", "encin": "これまで、副鼻腔を大きく", "encib": "体格を当てはめ", "encibn": "体格だけ、副鼻腔を大きく", "encis": "体格 + 声門を当てはめ",
        "encisn": "体格 + 声門、副鼻腔を大きく", "encit": "体格 4 値 + 声門 + 構えを当てはめ", "encitn": "体格 4 値 + 声門 + 構え、副鼻腔を大きく",
    }  # fmt: skip
    variants = [v for v in SPEAKER_LABELS if v in summary["ja"]]
    sets = [("JVS 女性（4 話者）", summary["ja-F"]), ("JVS 男性（4 話者）", summary["ja-M"])]
    costs = (
        grid(sets, variants, SPEAKER_LABELS, SPEAKER_NOTES, "Whisper large-v3-turbo の音素誤り率 %。角括弧は 95% 区間。", "rate2")
        + grid(sets, variants, SPEAKER_LABELS, SPEAKER_NOTES, "NISQA が予測する品質（1〜5）。", "quality")
        + grid(sets, variants, SPEAKER_LABELS, SPEAKER_NOTES, "NISQA のもう 1 つのモデルが予測する、合成音声の自然さ（1〜5）。", "natural")
    )
    fitted = {name: json.loads((jvs / f"{name}.json").read_text())["speakers"] for name in ("speakers_body", "speakers", "speakers_more") if (jvs / f"{name}.json").exists()}
    speakers = sorted({r["speaker"] for r in results["utterances"]})
    sexes = {r["speaker"]: r["sex"] for r in results["utterances"]}

    def settings_row(name: str) -> str:
        cells = ""
        for fit, keys in (("speakers_body", ("female", "child")), ("speakers", ("female", "child", "openQuotient", "brightness", "tension"))):
            got = fitted.get(fit, {}).get(name, {}).get("settings")
            cells += "".join(f'<td class="num">{got[k]:.2f}</td>' if got else '<td class="num">–</td>' for k in keys)
        return f'<tr><th scope="row">{esc(name)}<span class="sub">{"女性" if sexes[name] == "F" else "男性"}</span></th>{cells}</tr>'

    settings = f"""<div class="scroll"><table>
<caption>テストの 8 話者に当てはまった設定。体格は 0 が成人男性、1 が成人女性（これまでは性別どおりの 0 か 1）。子どもの軸は 0 が大人。声門は、開放率（1 周期のうち開いている割合）、明るさ（閉じ方の鋭さ）、張り。</caption>
<thead><tr><th scope="col">話者</th><th scope="col">体格だけ: 女性</th><th scope="col">子ども</th><th scope="col">体格 + 声門: 女性</th><th scope="col">子ども</th><th scope="col">開放率</th><th scope="col">明るさ</th><th scope="col">張り</th></tr></thead>
<tbody>{"".join(settings_row(n) for n in speakers)}</tbody></table></div>""" if fitted else ""
    more = (
        ("scale", "大きさ", ".2f"), ("pharynx", "咽頭", ".2f"), ("tongueRadius", "舌の半径", ".2f"), ("lipWidth", "唇の幅", ".1f"), ("openQuotient", "開放率", ".2f"), ("brightness", "明るさ", ".2f"),
        ("tension", "張り", ".2f"), ("epilarynx", "喉頭の上の狭め", ".2f"), ("larynxShift", "喉頭の高さ", "+.2f"), ("velumShift", "鼻への通路", "+.2f"), ("rootShift", "舌の根", "+.2f"),
    )  # fmt: skip
    if "speakers_more" in fitted:
        lines = "".join(
            f'<tr><th scope="row">{esc(n)}<span class="sub">{"女性" if sexes[n] == "F" else "男性"}</span></th>' + "".join(f'<td class="num">{fitted["speakers_more"][n]["settings"][k]:{fmt}}</td>' for k, _, fmt in more) + "</tr>"
            for n in speakers if n in fitted["speakers_more"]
        )  # fmt: skip
        settings += f"""<div class="scroll"><table>
<caption>同じ 8 話者の、11 個の値の当てはめ。体格は 4 つの値をそのまま当てはめます（大きさと咽頭の割合は成人男性が 1、舌の半径と唇の幅は cm）。構えは、喉頭の上の狭め（0〜1。これまでは 0）と、エンコーダが書いた値に全フレームで足すずれ（喉頭の高さと舌の根は cm、鼻への通路は cm²）。</caption>
<thead><tr><th scope="col">話者</th>{"".join(f'<th scope="col">{esc(label)}</th>' for _, label, _ in more)}</tr></thead>
<tbody>{lines}</tbody></table></div>"""

    def card(r: dict[str, Any]) -> str:
        buttons = "".join(
            button(clips, f"{r['id']}/{v}", jvs / v / f"{r['id']}.wav", label, f"{r['variants'][v]['heard']}」／large-v3-turbo「{r['variants'][v]['heard2']}", "ref" if v == "real" else "", "small")
            for v, label in (("real", "実録音"), ("enci", "これまで"), *SPEAKER_HEARD[r["sex"]]) if v in r["variants"]
        )  # fmt: skip
        return f"""<article class="sentence">
<p class="meta">{esc(r["id"])} · {"女性" if r["sex"] == "F" else "男性"}（JVS {esc(r["speaker"])}）</p>
<p class="text">{esc(r["text"])}</p>
<div class="clips">{buttons}</div>
<p class="heard" aria-live="polite">クリップを選ぶと Whisper の聞き取りを表示します。</p>
</article>"""

    rows = sorted(results["utterances"], key=lambda r: r["id"])
    # the four speakers and sentences of the cards above: their recordings are on the page already
    picked = [[x for x in rows if x["speaker"] == name][3 + 5 * k] for k, name in enumerate(speakers[:2] + speakers[-2:])]
    return f"""<section aria-labelledby="h-speaker"><h2 id="h-speaker">5. 話者ごとに体格・声門・構えを当てはめる</h2>
{{{{speaker_text}}}}
{whose}
{{{{speaker_costs}}}}
{costs}
{settings}
{SPEAKER_NOSE.replace("{{nasal_table}}", nasal_table({**with_sinus, "real": "実録音（値）"})).replace("{{apart_table}}", apart_table(("whose_nose", "whose_nose2"), with_sinus)) if (jvs / "whose_nose.json").exists() else ""}
{SPEAKER_BRIGHT.replace("{{bright_table}}", bright_table({**with_closure, "real": "実録音（値）"})).replace("{{closure_table}}", closure_table()).replace("{{closure_apart}}", apart_table(("whose_closure", "whose_closure2"), with_closure)).replace("{{closure_everyone}}", apart_table(("whose_closure", "whose_closure2"), {"real": "実録音", "encir": "これまで", "encib": "体格を当てはめ", **{v: label for v, label in with_closure.items() if v not in ("real", "enci")}}, everyone)) if (jvs / "whose_closure.json").exists() else ""}
{{{{speaker_sounds}}}}
<div class="stack">{"".join(card(r) for r in picked)}</div>
</section>"""


def release_section(d: Path, ja: Path, jvs: Path, clips: dict[str, str], number: int) -> str:
    """Towards a release: what the models so far depend on, the pair of voices that stands in
    for the speakers' own settings (`voices/v1.json`; `jvs`/bands_v1.json, whose_v1.json and
    results_v1.json say what it keeps of them), the listeners that may be used, and the encoders
    taught with them (`results_rel.json` of the three tests). Nothing before the pair of voices
    was measured."""
    if not (jvs / "bands_v1.json").exists() or not (ROOT / "voices" / "v1.json").exists():
        return ""
    voice = json.loads((ROOT / "voices" / "v1.json").read_text())
    names = {
        "body.female": "体格: 女性の体格の割合", "body.child": "体格: 子どもの体格の割合", "source.pressure": "肺圧 cmH₂O", "source.tension": "声門の張り", "source.breathiness": "息漏れ",
        "source.brightness": "明るさ", "source.roughness": "ゆらぎ", "glottis.openQuotient": "開放率", "glottis.amplitudeScale": "振動の振幅（倍率）", "glottis.closureSharpness": "閉じ方の鋭さ",
        "nose.sinusVolume": "副鼻腔の容積 cm³", "set.epilarynx": "喉頭の上の狭め（0〜1、全フレームで固定）", "shift.root": "舌の根のずれ cm（前が正）",
    }  # fmt: skip
    rows = "".join(
        f'<tr><th scope="row">{esc(label)}</th><td class="num">{voice["low"][part][key]:.3g}</td><td class="num">{voice["high"][part][key]:.3g}</td></tr>'
        for (part, key), label in ((tuple(k.split(".")), v) for k, v in names.items()) if key in voice["low"].get(part, {})
    )  # fmt: skip
    settings = f"""<div class="scroll"><table>
<caption>2 つの基準の声（voices/v1.json）。低い声は CHIMERA の成人男性に 2 つの構えを足したもの、高い声は第 3 節の調整 2 に、第 5 節で決まった設定の女性の平均を足したものです。</caption>
<thead><tr><th scope="col">設定</th><th scope="col">低い声（男性）</th><th scope="col">高い声（女性）</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""

    # what the pair keeps of the speakers' own settings: bands, the verifier, the judges
    bands = json.loads((jvs / "bands_v1.json").read_text())
    edges, by = bands["edges"], bands["sexes"]
    picked = [k for k, lo in enumerate(edges[:-1]) if 1000 <= lo < 5500]
    whose = json.loads((jvs / "whose_v1.json").read_text())["variants"] if (jvs / "whose_v1.json").exists() else {}
    judged = json.loads((jvs / "results_v1.json").read_text())["summary"] if (jvs / "results_v1.json").exists() else {}

    def kept(x: str, v: str) -> str:
        of = {"decided": "encibh" if x == "F" else "encit"}.get(v, v)
        real = by[x]["real"]
        out = f'<td class="num">{by[x][of]["h2"] - (real["h2"] if of != "real" else 0.0):+.1f}</td>'
        out += "".join(f'<td class="num">{by[x][of]["vowels"][k] - (real["vowels"][k] if of != "real" else 0.0):+.1f}</td>' for k in picked)
        out += f'<td class="num">{whose[of]["sexes"][x]["own"]:.3f}</td>' if of in whose else '<td class="num">–</td>'
        m = judged.get(f"ja-{x}", {}).get(of, {})
        out += f'<td class="num">{m["rate2"][0]:.1f}</td>' if "rate2" in m else '<td class="num">–</td>'
        out += f'<td class="num">{m["phoneRate"][0]:.1f}</td>' if "phoneRate" in m else '<td class="num">–</td>'
        return out + (f'<td class="num">{m["quality"]:.2f}</td>' if "quality" in m else '<td class="num">–</td>')

    heads = '<th scope="col">第 2 倍音</th>' + "".join(f'<th scope="col">{edges[k] / 1000:g}〜{edges[k + 1] / 1000:g} kHz</th>' for k in picked) + '<th scope="col">本人との類似度</th><th scope="col">large-v3-turbo %</th><th scope="col">音声記号の不一致 %</th><th scope="col">NISQA 品質</th>'
    keeps = "".join(
        f"""<div class="scroll"><table>
<caption>{sex}のテスト {by[x]["n"]} 文。エンコーダ I の制御列を 3 通りの設定で鳴らしたものです。左の 5 列は母音の録音との差（dB、0 に近いほど録音どおり。1 行目は録音の値）、「本人との類似度」は話者照合モデル（WavLM）によるその話者の録音との近さ（実録音どうしは 0.98）。「話者ごとの設定」は、{note}</caption>
<thead><tr><th scope="col">条件</th>{heads}</tr></thead>
<tbody>{"".join(f'<tr><th scope="row">{esc(label)}</th>{kept(x, v)}</tr>' for v, label in RELEASE_KEPT.items())}</tbody></table></div>"""
        for x, sex, note in (("F", "女性", "体格 + 副鼻腔 + 閉じ方の鋭さ（測った値の半分）。"), ("M", "男性", "体格 4 値 + 声門 + 構えの 11 個の値。"))
    )

    # the dial: brightness, pitch, and the other body
    dial = ""
    if (jvs / "results_dial.json").exists() and (jvs / "bands_dial.json").exists():
        turned = json.loads((jvs / "results_dial.json").read_text())["summary"]
        lit = json.loads((jvs / "bands_dial.json").read_text())["sexes"]
        band = next(k for k, lo in enumerate(edges[:-1]) if lo == 2500)

        def turn(x: str, v: str) -> str:
            m = turned[f"ja-{x}"][v]
            level = f'{lit[x][v]["vowels"][band] - lit[x]["real"]["vowels"][band]:+.1f}' if v in lit[x] else "–"
            return f'<td class="num">{level}</td><td class="num">{m["rate2"][0]:.1f}</td><td class="num">{m["phoneRate"][0]:.1f}</td><td class="num">{m["quality"]:.2f}</td>'

        hands = (
            ("enciv", "enciv", "基準の声のまま"), ("encivd", "encivd", "明るさ −1"), ("encivu", "encivu", "明るさ +1"),
            ("encivlo", "encivlo", "声の高さ −3 半音"), ("encivhi", "encivhi", "声の高さ +3 半音"),
            ("encivL", "encivH", "もう一方の声で鳴らす（女性の制御列を低い声で 10 半音下、男性の制御列を高い声で 9 半音上）"),
        )  # fmt: skip
        dial = f"""<div class="scroll"><table>
<caption>つまみを動かしたときの、テストの女性 100 文と男性 100 文。制御列はエンコーダ I のもので、鳴らすときの設定だけを変えています。「2.5〜4 kHz」は母音のその帯域の、録音との差（dB）。</caption>
<thead><tr><th scope="col">条件</th>{"".join(f'<th scope="col">{x}2.5〜4 kHz</th><th scope="col">large-v3-turbo %</th><th scope="col">音声記号の不一致 %</th><th scope="col">NISQA 品質</th>' for x in ("女性: ", "男性: "))}</tr></thead>
<tbody>{"".join(f'<tr><th scope="row">{esc(label)}</th>{turn("F", f)}{turn("M", m)}</tr>' for f, m, label in hands)}</tbody></table></div>"""

    # the listeners that may be used
    listeners = ""
    found = [r for p in sorted(ja.glob("yardstick*.json")) for r in json.loads(p.read_text())]
    if found:
        best: dict[str, dict[str, Any]] = {}
        for r in found:
            if r["model"] not in best or r["ja"]["dprime"] > best[r["model"]]["ja"]["dprime"]:
                best[r["model"]] = r
        lines = "".join(
            f'<tr><th scope="row">{esc(label.split("｜")[0])}<span class="sub">第 {best[name]["layer"]} 層</span></th><td>{esc(label.split("｜")[1])}</td>'
            + "".join(f'<td class="num">{best[name][lang]["ratio"]:.2f}</td><td class="num">{best[name][lang]["dprime"]:+.2f}</td>' for lang in ("ja", "en"))
            + f'<td class="num">{best[name]["judges"]["phones"]:.2f}</td></tr>'
            for name, label in RELEASE_MODELS.items() if name in best
        )  # fmt: skip
        listeners = f"""<div class="scroll"><table>
<caption>物差し（聞き手のモデル）の候補。第 2 節と同じ比べ方で、各モデルの、日本語の合成音と別の話者をいちばんよく区別する層を載せています。「比」は、録音と合成音の距離を、同じ文を読んだ別の話者との距離で割ったもの（1 なら合成音は別人の読み上げと同じ近さ）。「差」はその 2 つの距離の差を、ばらつきで割ったもの（大きいほど、合成音の足りなさを聞き取れる）。右端は、音声記号認識器の判定との順位相関です。</caption>
<thead><tr><th scope="col">モデル</th><th scope="col">重みのライセンス</th><th scope="col">日本語 比</th><th scope="col">日本語 差</th><th scope="col">英語 比</th><th scope="col">英語 差</th><th scope="col">判定との相関</th></tr></thead>
<tbody>{lines}</tbody></table></div>"""

    # the encoders
    encoders, cards = "", ""
    if (jvs / "results_rel.json").exists():
        results, sets = three_tests(d, ja, jvs, "results_rel", "results_rel")
        variants = [v for v in RELEASE_LABELS if all(v in g for _, g in sets)]
        by_sex: list[tuple[str, dict[str, Any]]] = []
        for key, heading in (("jvs", "JVS"), ("ja", "HFC・JSUT"), ("tsp", "TSP 英語"), ("ja", "HFC 英語")):
            if key in results:
                lang = "en" if "英語" in heading else "ja"
                by_sex += [(f"{heading} {'女性' if sex == 'F' else '男性'}", results[key]["summary"][f"{lang}-{sex}"]) for sex in "FM" if f"{lang}-{sex}" in results[key]["summary"]]
        encoders = (
            grid(sets, variants, RELEASE_LABELS, RELEASE_NOTES, "音声記号の不一致 %。角括弧は 95% 区間。", "phoneRate", other=True)
            + grid(sets, variants, RELEASE_LABELS, RELEASE_NOTES, "Whisper large-v3-turbo の誤り率 %。日本語の列は音素、英語の列は単語で数えています。", "rate2")
            + grid(sets, variants, RELEASE_LABELS, RELEASE_NOTES, "Whisper small の誤り率 %（英語の列は small.en）。", "rate")
            + grid(by_sex, variants, RELEASE_LABELS, RELEASE_NOTES, "NISQA が予測する品質（1〜5）。", "quality")
        )
        rows_ = sorted(results["jvs"]["utterances"], key=lambda r: r["id"])
        speakers = sorted({r["speaker"] for r in rows_})
        own = json.loads((jvs / "results_spk.json").read_text())["utterances"] if (jvs / "results_spk.json").exists() else []
        decided = {r["id"]: r for r in own}

        dialled = {r["id"]: r["variants"] for r in json.loads((jvs / "results_dial.json").read_text())["utterances"]} if (jvs / "results_dial.json").exists() else {}
        spoken = {r["id"]: r["variants"] for r in json.loads((jvs / "results_ttse.json").read_text())["utterances"]} if (jvs / "results_ttse.json").exists() else {}

        def card(r: dict[str, Any]) -> str:
            mine = "encibh" if r["sex"] == "F" else "encit"
            shown = [("real", "実録音", r["variants"].get("real"))]
            if r["id"] in decided and mine in decided[r["id"]]["variants"]:
                shown.append((mine, "話者ごとの設定（決定済み）", decided[r["id"]]["variants"][mine]))
            shown.append(("enciv", "基準の声", r["variants"].get("enciv")))
            shown += [(v, label, dialled.get(r["id"], {}).get(v)) for v, label in (("encivd", "基準の声、明るさ −1"), ("encivu", "基準の声、明るさ +1"))]
            shown.append(("encv2", "エンコーダ J2（公開用）", r["variants"].get("encv2")))
            # the text-to-speech models, each in the voice of the sentence's speaker's sex: the two taught with one corpus, and the one taught with both
            voice = "f" if r["sex"] == "F" else "m"
            shown += [(f"{v}_{voice}", label, spoken.get(r["id"], {}).get(f"{v}_{voice}")) for v, label in (("ttsev2", "TTS: Emilia-YODAS だけ"), ("ttscv2", "TTS: Common Voice だけ"), ("ttsja", "TTS: 両方（V1 の日本語モデル）"))]
            buttons = "".join(button(clips, f"{r['id']}/{v}", jvs / v / f"{r['id']}.wav", label, f"{m['heard']}」／large-v3-turbo「{m['heard2']}", "ref" if v == "real" else "", "small") for v, label, m in shown if m)
            return f"""<article class="sentence">
<p class="meta">{esc(r["id"])} · {"女性" if r["sex"] == "F" else "男性"}（JVS {esc(r["speaker"])}）</p>
<p class="text">{esc(r["text"])}</p>
<div class="clips">{buttons}</div>
<p class="heard" aria-live="polite">クリップを選ぶと Whisper の聞き取りを表示します。</p>
</article>"""

        picked_rows = [[x for x in rows_ if x["speaker"] == name][3 + 5 * k] for k, name in enumerate(speakers[:2] + speakers[-2:])]
        cards = f'<div class="stack">{"".join(card(r) for r in picked_rows)}</div>'
    return f"""<section aria-labelledby="h-release"><h2 id="h-release">{number}. 公開に向けて: 2 つの基準の声と、公開できる材料で学習し直したエンコーダ</h2>
{{{{release_text}}}}
{{{{release_voices}}}}
{settings}
{keeps}
{{{{release_dial}}}}
{dial}
{{{{release_listeners}}}}
{listeners}
{{{{release_encoders}}}}
{encoders}
{{{{release_more}}}}
{{{{release_tts}}}}
{{{{release_web}}}}
{{{{release_sounds}}}}
{cards}
</section>"""


def release_web(d: Path, jvs: Path) -> str:
    """The subsection on the page in the browser, with a picture of it if there is one
    (<dir>/web.jpg) and the command for the model that the cards play."""
    shot = d / "web.jpg"
    picture = f'<figure><img src="data:image/jpeg;base64,{base64.b64encode(shot.read_bytes()).decode()}" alt="PMVC-TTS の画面: 左に声道の断面図、右に文の入力、声のつまみ、制御列"><figcaption>話している途中の画面。断面図の橙色の文字と、制御列の縦線が、いま鳴っている音です。</figcaption></figure>' if shot.exists() else ""
    both = (jvs / "ttsja_m").exists()
    return RELEASE_WEB.replace("{{web_shot}}", picture).replace("{{web_model}}", "../pmvc-tts/ja.json ../pmvc-tts/en.json" if both else "out/cvja/models/ttscv2.pt")


def release_tts_table(jvs: Path) -> str:
    """The text-to-speech models taught with recordings that may be passed on, on JVS's test
    sentences, the low voice and the high one side by side: beside the recordings, encoder J2
    and the model taught with JVS (their men's and women's sentences)."""
    if not (jvs / "results_ttse.json").exists():
        return ""
    spoken = json.loads((jvs / "results_ttse.json").read_text())["summary"]
    heard = json.loads((jvs / "results_rel.json").read_text())["summary"] if (jvs / "results_rel.json").exists() else {}

    def both(of: dict[str, Any], v: str) -> tuple[Any, Any]:
        return of.get("ja-M", {}).get(v), of.get("ja-F", {}).get(v)

    def voices(name: str) -> tuple[Any, Any]:
        return spoken["ja"].get(f"{name}_m"), spoken["ja"].get(f"{name}_f")

    rows = (
        ("実録音", "話者本人。男性 4 人と女性 4 人の各 100 文", both(spoken, "real")),
        ("エンコーダ J2", "録音から書いた制御列", both(heard, "encv2")),
        ("TTS: JVS、エンコーダ I、話者ベクトル", "これまでのモデル（公開できない）。各文をその話者の声で", both(spoken, "ttsj")),
        ("TTS: Emilia-YODAS、J、話者ベクトル", "最初の試み。学習した話者 1 人ずつの声で", (spoken["ja"].get("ttse_mr"), spoken["ja"].get("ttse_fr"))),
        ("TTS: Common Voice、J、話者ベクトル", "同じ作りで録音を替える", voices("ttsc")),
        ("TTS: Common Voice、J、つまみ用", "話者ベクトルをやめ、声の割合と速さを入力にする", voices("ttscv")),
        ("TTS: Common Voice、J2、つまみ用", "制御列を J2 のものに", voices("ttscv2")),
        ("TTS: Emilia-YODAS、J2、つまみ用", "同じ作りで録音を替える", voices("ttsev2")),
        ("TTS: 両方、J2、つまみ用", "Common Voice と Emilia-YODAS を合わせて学習", voices("ttsja")),
    )  # fmt: skip

    def cells(m: Any) -> str:
        if not m:
            return '<td class="num">–</td>' * 3
        return f'<td class="num">{m["rate2"][0]:.1f}</td><td class="num">{m["rate"][0]:.1f}</td><td class="num">{m["quality"]:.2f}</td>' if "quality" in m else f'<td class="num">{m["rate2"][0]:.1f}</td><td class="num">{m["rate"][0]:.1f}</td><td class="num">–</td>'

    lines = "".join(f'<tr><th scope="row">{esc(label)}<span class="sub">{esc(note)}</span></th>{cells(m)}{cells(f)}</tr>' for label, note, (m, f) in rows if m or f)
    heads = "".join(f'<th scope="col">{x} large-v3-turbo %</th><th scope="col">small %</th><th scope="col">NISQA 品質</th>' for x in ("低い声（男性）:", "高い声（女性）:"))
    return f"""<div class="scroll"><table>
<caption>JVS のテスト文（8 話者 × 25 文。どのモデルも学習していない文）。Whisper の誤りは音素の割合 %。TTS は 200 文すべてを、低い声（125 Hz）と高い声（240 Hz）でそれぞれ読ませたもので、話す速さは録音と同じ 1 秒 10.2 音素に揃えています。話者ベクトルのモデルは、学習した話者のうち男性 1 人と女性 1 人の声です。</caption>
<thead><tr><th scope="col">条件</th>{heads}</tr></thead>
<tbody>{lines}</tbody></table></div>"""


def perturb_table(summary: dict[str, Any], conds: list[str], tag: str = "", kinds: dict[str, str] = KIND_LABELS, drive: str = "100 Hz × 21 チャネルで再駆動（以下の基準）") -> str:
    """The table of a perturbation test; its conditions are named <tag>_<condition> if tagged."""
    a = summary["all"]
    pre = f"{tag}_" if tag else ""

    def cells(c: str) -> str:
        return wer_cell(a[c]) + num_cell(a[c], "mcdLinear", ".2f") + num_cell(a[c], "voicing", ".0f", 100.0) + num_cell(a[c], "f0Rmse", ".2f")

    rows = [
        f'<tr><th scope="row">実録音</th>{cells("real")}</tr>',
        f'<tr><th scope="row">制御列そのまま<span class="sub">{esc(drive)}</span></th>{cells(pre + "drive")}</tr>',
    ]
    for kind, label in kinds.items():
        rows.append(f'<tr class="group"><th scope="rowgroup" colspan="5">{esc(label)}</th></tr>')
        rows.extend(
            f'<tr><th scope="row" class="level">{esc(level_label(c[len(pre) :]))}</th>{cells(c)}</tr>' for c in conds if c.startswith(pre + kind + "_")
        )
    return f"""<div class="scroll"><table>
<caption>摂動テストの評価（{a["n"]} 文）。右 3 列は「制御列そのまま」の音に対する変化です。</caption>
<thead><tr><th scope="col">条件</th><th scope="col">単語誤り率 %</th><th scope="col">ケプストラム距離 dB</th><th scope="col">有声一致 %</th><th scope="col">F0 ずれ 半音</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>"""


def perturbed_cards(d: Path, metas: dict[str, Any], clips: dict[str, str], ids: list[str], rows: dict[str, Any], conds: list[str], tag: str = "", kinds: dict[str, str] = KIND_LABELS) -> list[str]:
    """One card per recording of a perturbation test: the recording, the track as it is, and
    every condition by kind."""
    pre = f"{tag}_" if tag else ""
    cards = []
    for uid in ids:
        m, r = metas[uid], rows[uid]

        def heard(c: str, r: dict[str, Any] = r) -> str:
            return r["variants"][c].get("heard", "")

        head = button(clips, f"{uid}/real", d / "real" / f"{uid}.wav", "実録音", heard("real"), "ref") + button(
            clips, f"{uid}/{pre}drive", d / f"{pre}drive" / f"{uid}.wav", "制御列そのまま", heard(pre + "drive")
        )
        grid = []
        for kind, label in kinds.items():
            cs = [c for c in conds if c.startswith(pre + kind + "_")]
            cells = "".join(button(clips, f"{uid}/{c}", d / c / f"{uid}.wav", level_label(c[len(pre) :]), heard(c)) for c in cs)
            grid.append(f'<div class="kind"><p class="kind-label">{esc(label)}</p><div class="clips">{cells}</div></div>')
        cards.append(
            f"""<article class="sentence">
<p class="meta">{esc(uid)} · {"女性" if m["sex"] == "F" else "男性"}話者 {esc(m["speaker"])}</p>
<p class="text">{esc(m["text"])}</p>
<div class="clips">{head}</div>
{"".join(grid)}
<p class="heard" aria-live="polite">クリップを選ぶと Whisper の聞き取りを表示します。</p>
</article>"""
        )
    return cards


def codec_perturbed(d: Path, metas: dict[str, Any], clips: dict[str, str], n: int) -> str:
    """The perturbation test of the encoder's own tracks (`pmvc.perturb --tag ec`), if run."""
    if not (d / "perturb_ec_results.json").exists():
        return ""
    results = json.loads((d / "perturb_ec_results.json").read_text())
    listing = json.loads((d / "perturb_ec.json").read_text())
    rows = {r["id"]: r for r in results["utterances"]}
    kinds = {**KIND_LABELS, **CODED_LABEL}
    # a woman and a man, with different sentences
    women, men = ([i for i in listing["ids"] if i[0] == sex] for sex in "FM")
    ids = [women[0], men[len(men) // 2]][:n]
    cards = perturbed_cards(d, metas, clips, ids, rows, listing["conditions"], "ec", kinds)
    table = perturb_table(results["summary"], listing["conditions"], "ec", kinds, "エンコーダ C′ が録音から書いた制御列（以下の基準）")
    return f"""<h3>エンコーダの制御列への摂動</h3>
<p>第 2 節と同じ摂動を、エンコーダ C′ が実録音から書いた制御列（つまりこのコーデックの潜在表現そのもの）にかけます。最後の群は、間引いた値を一様に丸めるだけの素朴な符号化で、括弧内は動く 17 チャネルぶんのビットレートです。</p>
{table}
<div class="stack">{"".join(cards)}</div>"""


def tts_section(ja: Path, say: Path, clips: dict[str, str]) -> str:
    """Text to speech on the latent: `pmvc.judge`'s results for the test sentences of the
    speaker the model was taught with (`ja`/results_tts.json: the recording, the encoder's
    render and the model's, and those of models taught with fewer recordings) and for
    sentences nobody recorded (`say`/results.json)."""
    results = json.loads((ja / "results_tts.json").read_text())
    speaker = results["utterances"][0]["speaker"]
    summary = results["summary"][speaker]
    variants = [v for v in TTS_LABELS if v in summary]
    sizes = [v for v in TTS_SIZES if v in summary]

    def row(label: str, note: str, cells: str) -> str:
        return f'<tr><th scope="row">{esc(label)}<span class="sub">{esc(note)}</span></th>{cells}</tr>'

    said = "".join(
        row(TTS_LABELS[v], TTS_NOTES[v], rate_cell(summary[v], "rate") + rate_cell(summary[v], "rate2") + rate_cell(summary[v], "phoneRate") + num_cell(summary[v], "quality", ".2f"))
        for v in variants
    )
    along = "".join(
        row(TTS_LABELS[v], TTS_NOTES[v], num_cell(summary[v], "distance", ".3f") + num_cell(summary[v], "f0Corr", ".2f") + num_cell(summary[v], "voicing", ".1f", 100.0) + num_cell(summary[v], "ownReleases", ".2f") + num_cell(summary[v], "ownReleaseLevel", ".1f"))
        for v in variants
    )
    tables = f"""<div class="scroll"><table>
<caption>テストの {summary["n"]} 文（学習にも検証にも使っていない文）。Whisper の 2 列は音素誤り率、角括弧は 95% 区間。NISQA は予測品質（1〜5）。</caption>
<thead><tr><th scope="col">条件</th><th scope="col">Whisper small %</th><th scope="col">Whisper large-v3-turbo %</th><th scope="col">音声記号の不一致 %</th><th scope="col">NISQA</th></tr></thead>
<tbody>{said}</tbody></table></div>
{{{{tts_between}}}}
<div class="scroll"><table>
<caption>録音との近さと、破裂のクリック音。リスナー距離・ピッチの相関・有声無声の一致は、録音と同じ時間軸の合成音でだけ測れます。クリック音は、どの音もその音自身の発話区間で数えています。</caption>
<thead><tr><th scope="col">条件</th><th scope="col">リスナー距離</th><th scope="col">ピッチの相関</th><th scope="col">有声・無声の一致 %</th><th scope="col">破裂のクリック音 回/秒</th><th scope="col">その大きさ dB</th></tr></thead>
<tbody>{along}</tbody></table></div>"""

    def cards(where: Path, rows: list[dict[str, Any]], shown: dict[str, str]) -> str:
        out = []
        for r in rows:
            buttons = "".join(
                button(clips, f"{r['id']}/{v}", where / v / f"{r['id']}.wav", label, f"{r['variants'][v]['heard']}」／large-v3-turbo「{r['variants'][v]['heard2']}", "ref" if v == "real" else "", "small")
                for v, label in shown.items() if v in r["variants"]
            )  # fmt: skip
            out.append(
                f"""<article class="sentence">
<p class="meta">{esc(r["id"])}</p>
<p class="text">{esc(r["text"])}</p>
<div class="clips">{buttons}</div>
<p class="heard" aria-live="polite">クリップを選ぶと Whisper の聞き取りを表示します。</p>
</article>"""
            )
        return "".join(out)

    tested = sorted(results["utterances"], key=lambda r: r["id"])
    body = f"""<section aria-labelledby="h-test"><h2 id="h-test">1. テストの文</h2>
{{{{tts_text}}}}
{tables}
<div class="stack">{cards(ja, tested[4::10], {v: TTS_LABELS[v] for v in variants})}</div>
</section>"""
    free = json.loads((say / "results.json").read_text()) if (say / "results.json").exists() else None
    if free is not None:
        group = free["summary"]["ja"]
        body += f"""<section aria-labelledby="h-say"><h2 id="h-say">2. コーパスにない文</h2>
{{{{say_text}}}}
<div class="stack">{cards(say, sorted(free["utterances"], key=lambda r: r["id"]), {"tts": "TTS"})}</div>
</section>"""
    if len(sizes) > 1:
        by_size = "".join(
            row(TTS_SIZES[v][0], TTS_SIZES[v][1], rate_cell(summary[v], "rate") + rate_cell(summary[v], "rate2") + rate_cell(summary[v], "phoneRate") + num_cell(summary[v], "quality", ".2f")
                + (rate_cell(group[v], "rate") + rate_cell(group[v], "rate2") if free is not None and v in group else ""))
            for v in sizes
        )  # fmt: skip
        outside = '<th scope="col">録音のない文 small %</th><th scope="col">同 large-v3-turbo %</th>' if free is not None else ""
        labels = {v: TTS_SIZES[v][0] for v in sizes}
        more = cards(say, sorted(free["utterances"], key=lambda r: r["id"])[:4], labels) if free is not None else ""
        body += f"""<section aria-labelledby="h-few"><h2 id="h-few">3. 学習データを減らす</h2>
{{{{few_text}}}}
<div class="scroll"><table>
<caption>学習に使う録音の数を変えた TTS。左の 4 列はテストの {summary["n"]} 文、右の 2 列は第 2 節の録音のない文。Whisper の列は音素誤り率、角括弧は 95% 区間。</caption>
<thead><tr><th scope="col">学習データ</th><th scope="col">Whisper small %</th><th scope="col">Whisper large-v3-turbo %</th><th scope="col">音声記号の不一致 %</th><th scope="col">NISQA</th>{outside}</tr></thead>
<tbody>{by_size}</tbody></table></div>
<div class="stack">{cards(ja, tested[9::30], {"real": TTS_LABELS["real"], **labels})}{more}</div>
</section>"""
    return body


def many_section(jvs: Path, everyone: Path, say: Path, clips: dict[str, str]) -> str:
    """Text to speech for JVS's hundred speakers: `pmvc.judge`'s results for the eight test
    speakers' test sentences (`jvs`/results_tts.json), `pmvc.whose`'s say on whose voice the
    renders are (`jvs`/whose_tts.json for those eight, `everyone`/whose_tts.json for all
    hundred), and sentences nobody recorded in four of the voices (`say`/results.json,
    variants ttsj_<speaker>). Nothing before the first exists."""
    if not (jvs / "results_tts.json").exists():
        return ""
    results = json.loads((jvs / "results_tts.json").read_text())
    summary = results["summary"]
    variants = [v for v in MANY_LABELS if v in summary["ja"]]
    sets = [("JVS 女性（4 話者）", summary["ja-F"]), ("JVS 男性（4 話者）", summary["ja-M"])]
    tables = (
        grid(sets, variants, MANY_LABELS, MANY_NOTES, "Whisper large-v3-turbo の音素誤り率 %。角括弧は 95% 区間。", "rate2")
        + grid(sets, variants, MANY_LABELS, MANY_NOTES, "Whisper small の音素誤り率 %。", "rate")
        + grid(sets, variants, MANY_LABELS, MANY_NOTES, "音声記号の不一致 %。", "phoneRate", other=True)
        + grid(sets, variants, MANY_LABELS, MANY_NOTES, "NISQA が予測する品質（1〜5）。", "quality")
    )
    def whose_table(path: Path, what: str) -> str:
        """`pmvc.whose`'s say on a set of clips, as a table; nothing if it was not run."""
        if not path.exists():
            return ""
        w = json.loads(path.read_text())
        # the encoder's renders are `encir` where they were rendered for this alone
        lines = "".join(
            f'<tr><th scope="row">{esc(MANY_LABELS[v.replace("encir", "enci")])}</th><td class="num">{m["own"]:.3f}</td><td class="num">{m["other"]:.3f}</td><td class="num">{100 * m["top1"]:.1f}</td><td class="num">{100 * m["top1sex"]:.1f}</td></tr>'
            for v, m in w["variants"].items() if v.replace("encir", "enci") in MANY_LABELS
        )  # fmt: skip
        return f"""<div class="scroll"><table>
<caption>{what}（{w["speakers"]} 話者、{w["clips"]:,} 文）。類似度はコサイン類似度、右の 2 列は「最も近い話者が本人だった」割合 %。</caption>
<thead><tr><th scope="col">条件</th><th scope="col">本人との類似度</th><th scope="col">同性の他人との類似度</th><th scope="col">{w["speakers"]} 話者の中で本人</th><th scope="col">同性の中で本人</th></tr></thead>
<tbody>{lines}</tbody></table></div>"""

    tables_whose = whose_table(jvs / "whose_tts.json", "テストの 8 話者") + whose_table(everyone / "whose_tts.json", "全 100 話者のテスト文")
    whose = f"{{{{many_whose}}}}{tables_whose}" if tables_whose else ""

    def card(where: Path, r: dict[str, Any], label: str, shown: dict[str, str]) -> str:
        buttons = "".join(
            button(clips, f"{r['id']}/{v}", where / v / f"{r['id']}.wav", name, f"{r['variants'][v]['heard']}」／large-v3-turbo「{r['variants'][v]['heard2']}", "ref" if v == "real" else "", "small")
            for v, name in shown.items() if v in r["variants"]
        )  # fmt: skip
        return f"""<article class="sentence">
<p class="meta">{esc(label)}</p>
<p class="text">{esc(r["text"])}</p>
<div class="clips">{buttons}</div>
<p class="heard" aria-live="polite">クリップを選ぶと Whisper の聞き取りを表示します。</p>
</article>"""

    tested: list[dict[str, Any]] = sorted(results["utterances"], key=lambda r: r["id"])
    shown: dict[str, str] = {v: label for v, label in (("real", "実録音"), ("enci", "エンコーダ I"), ("ttsj", "TTS"), ("ttsj500", "TTS（500 文で学習）")) if v in variants}
    # each of the four voices on a different sentence
    picked = [[r for r in tested if r["speaker"] == name][3 + 5 * k] for k, name in enumerate(MANY_VOICES)]
    cards = "".join(card(jvs, r, f"{r['id']} · {MANY_VOICES[r['speaker']]}", shown) for r in picked)
    free = json.loads((say / "results.json").read_text())["utterances"] if (say / "results.json").exists() else []
    voices = {"tts": "男性（Hi-Fi-CAPTAIN、第 1 節の TTS）", **{f"ttsj_{name}": label for name, label in MANY_VOICES.items()}}
    spoken = "".join(card(say, r, r["id"], voices) for r in sorted(free, key=lambda r: r["id"])[:5] if any(v in r["variants"] for v in voices if v != "tts"))
    return f"""<section aria-labelledby="h-many"><h2 id="h-many">4. JVS の 100 話者</h2>
{{{{many_text}}}}
{tables}
{whose}
<div class="stack">{cards}</div>
{{{{many_say}}}}
<div class="stack">{spoken}</div>
</section>"""


def control_section(ja: Path, say: Path, clips: dict[str, str]) -> str:
    """The control: `pmvc.judge`'s results for the same model writing neural decoders' latents
    (`ja`/results_ctl.json, `say`/results_ctl.json). Nothing before the first exists."""
    if not (ja / "results_ctl.json").exists():
        return ""
    results = json.loads((ja / "results_ctl.json").read_text())
    summary = results["summary"][results["utterances"][0]["speaker"]]
    free = json.loads((say / "results_ctl.json").read_text()) if (say / "results_ctl.json").exists() else None
    latents = {k: v for k, v in CONTROL_LATENTS.items() if any(v[2].format(n) in summary for n in CONTROL_SIZES)}

    def named(name: str, n: str) -> str:
        """The variant of a latent's model taught with `n` recordings; of the track with all of
        them there is only the model with the stand-in."""
        v = name.format(n)
        return "tts" if v == "ttsn" and v not in summary else v

    def table(group: dict[str, Any], key: str, caption: str, best: bool = True) -> str:
        def cell(v: str) -> str:
            m = group.get(v, {})
            if key not in m:
                return '<td class="num">–</td>'
            return f'<td class="num">{m[key][0]:.1f}</td>' if isinstance(m[key], list) else f'<td class="num">{m[key]:.2f}</td>'

        rows = "".join(
            f'<tr><th scope="row">{esc(label)}<span class="sub">{esc(sub)}</span></th>' + "".join(cell(named(name, n)) for n in CONTROL_SIZES) + (cell(top) if best else "") + "</tr>"
            for label, sub, name, top in latents.values()
        )
        head = "".join(f'<th scope="col">{esc(label)}</th>' for label in CONTROL_SIZES.values()) + ('<th scope="col">デコーダの上限</th>' if best else "")
        return f"""<div class="scroll"><table>
<caption>{caption}</caption>
<thead><tr><th scope="col">モデルが書くもの</th>{head}</tr></thead>
<tbody>{rows}</tbody></table></div>"""

    tables = (
        table(summary, "rate2", f"テストの {summary['n']} 文。Whisper large-v3-turbo の音素誤り率 %。列は学習に使った録音の数。「デコーダの上限」は、録音をその潜在表現にして戻した音（制御列ではエンコーダ I の合成音。調整したボコーダでは、全データで調整したもの）。")
        + table(summary, "rate", "同じく Whisper small の音素誤り率 %。")
        + "{{control_between}}"
        + table(summary, "quality", "NISQA が予測する品質（1〜5）。")
        + table(summary, "natural", "参考: NISQA のもう 1 つのモデルが予測する「合成音声の自然さ」（1〜5）。品質とは順位が逆になる。実録音は 4.52。")
    )
    if free is not None:
        tables += table(free["summary"]["ja"], "rate2", f"録音のない {free['summary']['ja']['n']} 文（第 2 節の文）。Whisper large-v3-turbo の音素誤り率 %。", best=False)
        tables += table(free["summary"]["ja"], "quality", "同じ文での NISQA。", best=False)

    def card(where: Path, r: dict[str, Any], real: bool) -> str:
        def clip(v: str, label: str, kind: str = "") -> str:
            c = r["variants"][v]
            return button(clips, f"{r['id']}/{v}", where / v / f"{r['id']}.wav", label, f"{c['heard']}」／large-v3-turbo「{c['heard2']}", kind, "small")

        groups = "".join(
            f'<div class="kind"><span class="kind-label">{esc(size)}で学習</span><div class="clips">'
            + "".join(clip(named(name, n), label) for key, (label, _, name, _) in latents.items() if key not in CONTROL_UNHEARD and named(name, n) in r["variants"])
            + "</div></div>"
            for n, size in CONTROL_SIZES.items() if n in ("20", "200", "")
        )
        first = f'<div class="clips">{clip("real", "実録音", "ref")}</div>' if real else ""
        return f"""<article class="sentence">
<p class="meta">{esc(r["id"])}</p>
<p class="text">{esc(r["text"])}</p>
{first}{groups}
<p class="heard" aria-live="polite">クリップを選ぶと Whisper の聞き取りを表示します。</p>
</article>"""

    tested = sorted(results["utterances"], key=lambda r: r["id"])
    cards = "".join(card(ja, r, True) for r in tested[19::40])
    if free is not None:
        cards += "".join(card(say, r, False) for r in sorted(free["utterances"], key=lambda r: r["id"])[1:2])
    return f"""<section aria-labelledby="h-control"><h2 id="h-control">5. 対照: 同じモデルにニューラルデコーダの潜在表現を書かせる</h2>
{{{{control_text}}}}
{tables}
{{{{control_sounds}}}}
<div class="stack">{cards}</div>
</section>"""


def shell(title: str, lede: str, notes: str, body: str, credits: str, clips: dict[str, str]) -> str:
    return (
        PAGE.replace("{{title}}", esc(title)).replace("{{lede}}", esc(lede)).replace("{{notes}}", notes).replace("{{body}}", body)
        .replace("{{credits}}", esc(credits)).replace("{{clips}}", json.dumps(clips))
    )  # fmt: skip


def pointer(number: int, heading: str, text: str, title: str, url: str, name: str = "more") -> str:
    """A section that only points at another page, or nothing without its address."""
    if not url:
        return ""
    return f'''<section aria-labelledby="h-{name}"><h2 id="h-{name}">{number}. {esc(heading)}</h2>
<p>{esc(text)}<a href="{esc(url)}">{esc(title)}</a></p></section>'''


def build(d: Path, notes: str, n_sentences: int, n_perturbed: int, ja_url: str = "") -> str:
    """The first page: the experiments on TSP's English recordings."""
    results = json.loads((d / "results.json").read_text())
    presults = json.loads((d / "perturb_results.json").read_text())
    perturb = json.loads((d / "perturb.json").read_text())
    rows = {r["id"]: r for r in results["utterances"]}
    prows = {r["id"]: r for r in presults["utterances"]}
    metas = {p.stem: json.loads(p.read_text()) for p in (d / "meta").glob("*.json")}
    ids: list[str] = perturb["ids"]
    clips: dict[str, str] = {}

    sentences = []
    for uid in ids[:n_sentences]:
        m, r = metas[uid], rows[uid]
        buttons = "".join(
            button(clips, f"{uid}/{v}", d / v / f"{uid}.wav", label, r["variants"][v].get("heard", ""), "ref" if v == "real" else "")
            for v, label in VARIANT_LABELS.items()
        )
        who = f'{"女性" if m["sex"] == "F" else "男性"}話者 {esc(m["speaker"])}'
        sentences.append(
            f"""<article class="sentence">
<p class="meta">{esc(uid)} · {who}</p>
<p class="text">{esc(m["text"])}</p>
<p class="ipa" lang="en-fonipa">{esc(m["ipa"])}</p>
<div class="clips">{buttons}</div>
<p class="heard" aria-live="polite">クリップを選ぶと Whisper の聞き取りを表示します。</p>
</article>"""
        )

    perturbed = perturbed_cards(d, metas, clips, ids[:n_perturbed], prows, perturb["conditions"])

    channels = "、".join(perturb["keys"])
    body = (
        BODY_EN.replace("{{transplant_table}}", transplant_table(results["summary"]))
        .replace("{{perturb_table}}", perturb_table(presults["summary"], perturb["conditions"]))
        .replace("{{sentences}}", "\n".join(sentences))
        .replace("{{perturbed}}", "\n".join(perturbed))
        .replace("{{copy_section}}", copy_section(d, metas, clips, n_sentences))
        .replace("{{codec_section}}", codec_section(d, metas, clips, 8).replace("{{codec_text}}", CODEC_TEXT))
        .replace("{{channels}}", esc(channels))
        .replace("{{more}}", pointer(5, "日本語の録音", "日本語の録音を使った実験は別のページにあります: ", TITLE_JA, ja_url))
    )
    return shell(TITLE_EN, LEDE_EN, notes, body, CREDITS_EN, clips)


def build_ja(d: Path, ja: Path, jvs: Path, fit: Path, notes: str, en_url: str = "", tts_url: str = "", everyone: Path | None = None) -> str:
    """The second page: recordings from outside TSP, the Japanese ones above all."""
    clips: dict[str, str] = {}
    body = abroad_section(d, ja, clips).replace("{{abroad_text}}", ABROAD_TEXT).replace("{{abroad_measures}}", ABROAD_MEASURES)
    body += taught_section(d, ja, jvs, clips).replace("{{taught_text}}", TAUGHT_TEXT).replace("{{taught_between}}", TAUGHT_BETWEEN)
    body += tuned_section(d, ja, jvs, fit, clips).replace("{{tuned_text}}", TUNED_TEXT).replace("{{tuned_voice}}", TUNED_VOICE).replace("{{tuned_rule}}", TUNED_RULE).replace("{{tuned_walls}}", TUNED_WALLS).replace("{{tuned_closure}}", TUNED_CLOSURE).replace("{{tuned_clicks}}", TUNED_CLICKS).replace("{{tuned_check}}", TUNED_CHECK)
    retaught = retaught_section(d, ja, jvs, clips).replace("{{retaught_text}}", RETAUGHT_TEXT).replace("{{retaught_sounds}}", RETAUGHT_SOUNDS)
    body += retaught
    fitted = speaker_section(jvs, everyone, clips).replace("{{speaker_text}}", SPEAKER_TEXT).replace("{{speaker_costs}}", SPEAKER_COSTS).replace("{{speaker_sounds}}", SPEAKER_SOUNDS) if everyone is not None else ""
    body += fitted
    last = (5 if retaught else 4) + (1 if fitted else 0)
    released = release_section(d, ja, jvs, clips, last)
    for name, text in (("release_text", RELEASE_TEXT), ("release_voices", RELEASE_VOICES), ("release_dial", RELEASE_DIAL), ("release_listeners", RELEASE_LISTENERS), ("release_encoders", RELEASE_ENCODERS),
                       ("release_more", RELEASE_MORE.replace("{{more}}", RELEASE_MORE_RESULT)), ("release_tts", RELEASE_TTS if (jvs / "results_ttse.json").exists() else ""), ("release_web", release_web(d, jvs)), ("release_sounds", RELEASE_SOUNDS), ("tts_table", release_tts_table(jvs))):  # fmt: skip
        released = released.replace("{{" + name + "}}", text)
    body += released
    last += 1 if released else 0
    said = pointer(last, "テキストからの合成（TTS）", "エンコーダ I の制御列を手本に、テキストから制御列を書くモデルを作った結果は別のページにあります: ", TITLE_TTS, tts_url, "tts")
    body += said + pointer(last + 1 if said else last, "英語の録音での実験", "エンコーダ A・C・C′ の作り方と、英語（TSP）での結果は別のページにあります: ", TITLE_EN, en_url)
    return shell(TITLE_JA, LEDE_JA, notes, body, CREDITS_JA, clips)


def build_tts(ja: Path, say: Path, jvs: Path, everyone: Path, notes: str, en_url: str = "", ja_url: str = "") -> str:
    """The third page: text to speech on the latent."""
    clips: dict[str, str] = {}
    body = tts_section(ja, say, clips).replace("{{tts_text}}", TTS_TEXT).replace("{{tts_between}}", TTS_BETWEEN).replace("{{say_text}}", SAY_TEXT).replace("{{few_text}}", FEW_TEXT)
    body += many_section(jvs, everyone, say, clips).replace("{{many_text}}", MANY_TEXT).replace("{{many_whose}}", MANY_WHOSE).replace("{{many_say}}", MANY_SAY)
    body += control_section(ja, say, clips).replace("{{control_text}}", CONTROL_TEXT).replace("{{control_between}}", CONTROL_BETWEEN).replace("{{control_sounds}}", CONTROL_SOUNDS)
    links = "".join(f'<li><a href="{esc(url)}">{esc(title)}</a>：{esc(what)}</li>' for title, url, what in ((TITLE_JA, ja_url, "エンコーダと声の調整（日本語の録音）"), (TITLE_EN, en_url, "韻律移植、摂動、最初のエンコーダ（英語の録音）")) if url)
    if links:
        body += f'<section aria-labelledby="h-more"><h2 id="h-more">これまでの実験</h2><ul>{links}</ul></section>'
    return shell(TITLE_TTS, LEDE_TTS, notes, body, CREDITS_TTS, clips)


CODEC_TEXT = """<p>録音を聞いて制御列を書くエンコーダです。デコーダは CHIMERA そのものなので、これで「音声 → 制御列（100 Hz × 21 チャネル）→ 音声」のコーデックになります。文の書き起こしは、学習にも推論にも探索にも使いません。</p>
<p>エンコーダは、音声認識モデル wav2vec2 の特徴とピッチ抽出器の出力を入力にする再帰型ネットワークです。F0 だけは学習せず、ピッチ抽出器の曲線をそのまま使います。声の体格（女声・男声のプリセット）は話者の性別で選んでいます。</p>
<ul>
<li><b>A</b>: CHIMERA 自身が合成した音とその制御列の対だけで学習します。</li>
<li><b>A + 探索</b>: A が実録音に対して書いた制御列を、合成音が録音に近づくよう局所探索で直します。</li>
<li><b>B</b>: 探索で直した制御列を教師にして学習し直します。</li>
<li><b>C</b>: さらに「声の代役」を通した誤差を加えます。CHIMERA は微分できないので、制御列から「リスナー（wav2vec2）にどう聞こえるか」を予測するネットワークを合成音で学習し、それを通して「自分の書いた制御列の音が録音と同じに聞こえるか」をエンコーダに直接教えます。</li>
<li><b>C′</b>: C から探索の結果を除いたものです。代役は規則の制御列と A の制御列の合成音だけで学習し、エンコーダの教師には A 自身の出力を使います。</li>
</ul>"""


ABROAD_TEXT = """<p>英語の録音で作ったエンコーダ A・C・C′ を、そのまま日本語の録音に使います。エンコーダも、その入力に使う wav2vec2 も、学習に使ったのは英語（TSP）だけです。録音は Hi-Fi-CAPTAIN の日本語話者 2 人（女性と男性、同じ 100 文）と、JSUT の女性話者（別の 100 文）です。</p>
<p>比較のため、Hi-Fi-CAPTAIN の英語話者 2 人（各 100 文）にも同じことをしています。収録条件が日本語と同じなので、コーパスの違いと言語の違いを分けて見られます。</p>"""
ABROAD_MEASURES = """<p class="small">音声記号の不一致は、テキストも言語も仮定しない物差しです。多言語の音声記号認識器（wav2vec2 XLSR-53 を国際音声記号の書き取りに調整したもの）に録音と合成音を書き取らせ、録音側の記号のうち合成音側で違っていた割合を数えます。この認識器は日本語では書き取りが安定しないので、同じ文を読んだ別の実話者の録音を同じように比べた値を基準に置きます。リスナー距離は、学習と探索が下げようとする量そのものです（英語の音声認識モデル wav2vec2 の中間特徴の 1 − コサイン類似度を、録音と合成音のフレームどうしで比べた平均）。Whisper と音声記号認識器は、学習にも探索にも使っていません。</p>"""


PAGE = """<title>{{title}}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Noto+Sans:wght@400&family=Space+Grotesk:wght@500;600&display=swap">
<style>
/* Layout: a lab notebook. One reading column; findings first, then each table followed by the clips it summarises. */
:root {
  --bg: #f4f6f4; --surface: #ffffff; --ink: #1b2422; --muted: #5a6764; --line: #d6dcd9;
  --accent: #0a6b6b; --accent-ink: #ffffff; --ref: #8a5600; --playing: #dff1ee;
  --font-display: "Space Grotesk", "Hiragino Sans", "Yu Gothic", "Noto Sans JP", sans-serif;
  --font-body: "Inter", "Hiragino Sans", "Yu Gothic", "Noto Sans JP", system-ui, sans-serif;
  --font-ipa: "Noto Sans", "Inter", "Hiragino Sans", sans-serif;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111615; --surface: #19201e; --ink: #e4eae8; --muted: #98a5a1; --line: #2a3431;
  --accent: #55c2bd; --accent-ink: #0c1413; --ref: #e2b35a; --playing: #1d3a37; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111615; --surface: #19201e; --ink: #e4eae8; --muted: #98a5a1; --line: #2a3431;
  --accent: #55c2bd; --accent-ink: #0c1413; --ref: #e2b35a; --playing: #1d3a37; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 15px; line-height: 1.75;
  padding-block: 2.5rem 4rem; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main { max-width: 52rem; margin: 0 auto; display: flex; flex-direction: column; gap: 2.5rem; }
section { display: flex; flex-direction: column; gap: 1rem; min-width: 0; }
h1, h2 { font-family: var(--font-display); font-weight: 600; line-height: 1.3; margin: 0; text-wrap: balance; }
h1 { font-size: 1.75rem; } h2 { font-size: 1.25rem; padding-top: 1.25rem; border-top: 1px solid var(--line); }
h3 { font-family: var(--font-display); font-weight: 600; font-size: 1.0625rem; line-height: 1.3; margin: 0; padding-top: 0.75rem; text-wrap: balance; }
p, ul { margin: 0; } ul { padding-left: 1.2rem; } li + li { margin-top: 0.35rem; }
.lede { color: var(--muted); max-width: 44rem; }
.notes { display: flex; flex-direction: column; gap: 0.75rem; max-width: 44rem; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.875rem; font-variant-numeric: tabular-nums; }
caption { caption-side: top; text-align: left; color: var(--muted); padding-bottom: 0.5rem; }
figure { margin: 1rem 0; max-width: 46rem; } figure img { display: block; max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; } figcaption { color: var(--muted); font-size: 0.875rem; padding-top: 0.4rem; }
pre { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 0.75rem 1rem; font-size: 0.8125rem; }
th, td { padding: 0.45rem 0.75rem 0.45rem 0; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
thead th { font-weight: 500; color: var(--muted); white-space: nowrap; }
tbody th { font-weight: 500; min-width: 15rem; }
td.num, thead th:not(:first-child) { text-align: right; white-space: nowrap; }
.sub { display: block; font-weight: 400; color: var(--muted); font-size: 0.8125rem; }
.ci { color: var(--muted); }
tr.group th { color: var(--muted); font-weight: 500; padding-top: 0.9rem; }
tbody th.level { padding-left: 1rem; font-weight: 400; }
.sentence { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 1rem 1.1rem; display: flex; flex-direction: column; gap: 0.5rem; min-width: 0; }
.meta { color: var(--muted); font-size: 0.8125rem; letter-spacing: 0.02em; }
.text { font-weight: 500; font-size: 1.0625rem; line-height: 1.5; }
.ipa { font-family: var(--font-ipa); color: var(--muted); overflow-wrap: anywhere; line-height: 1.5; }
.clips { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.kind { display: flex; flex-direction: column; gap: 0.35rem; padding-top: 0.35rem; }
.kind-label { font-size: 0.8125rem; color: var(--muted); }
button.clip { font: inherit; font-size: 0.875rem; line-height: 1.3; color: var(--accent); background: transparent; border: 1px solid var(--accent);
  border-radius: 999px; padding: 0.35rem 0.85rem; cursor: pointer; }
button.clip.ref { color: var(--ref); border-color: var(--ref); }
button.clip:hover { background: var(--playing); }
button.clip:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
button.clip[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); }
button.clip.ref[aria-pressed="true"] { background: var(--ref); color: var(--accent-ink); }
.heard { font-size: 0.875rem; color: var(--muted); min-height: 1.6em; overflow-wrap: anywhere; }
.heard b { color: var(--ink); font-weight: 500; }
.small { font-size: 0.8125rem; color: var(--muted); }
.stack { display: flex; flex-direction: column; gap: 0.75rem; }
code { font-size: 0.875em; }
a { color: var(--accent); }
</style>
<main>
<header class="stack">
<h1>{{title}}</h1>
<p class="lede">{{lede}}</p>
</header>
<section aria-labelledby="h-notes"><h2 id="h-notes">わかったこと</h2><div class="notes">{{notes}}</div></section>
{{body}}
<footer class="small stack">
<p>{{credits}}</p>
</footer>
</main>
<script>
const CLIPS = {{clips}};
const audio = new Audio();
let current = null;
function stop() { if (current) current.setAttribute("aria-pressed", "false"); current = null; audio.pause(); }
audio.addEventListener("ended", stop);
document.addEventListener("click", (e) => {
  const b = e.target.closest("button.clip");
  if (!b) return;
  if (b === current) { stop(); return; }
  stop();
  current = b;
  b.setAttribute("aria-pressed", "true");
  const line = b.closest(".sentence").querySelector(".heard");
  line.textContent = "";
  const label = document.createElement("b");
  label.textContent = b.textContent + "：";
  line.append(label, b.dataset.heard ? b.dataset.judge + "「" + b.dataset.heard + "」" : "（聞き取りなし）");
  audio.src = CLIPS[b.dataset.clip];
  audio.play().catch(stop);
});
</script>
"""


TITLE_EN = "CHIMERA 韻律移植と摂動の試聴"
LEDE_EN = "物理モデル音声合成 CHIMERA を、実際の話者の録音に合わせて動かした結果です。合成エンジンは無改造で、変えたのは制御だけです。ボタンを押すと再生し、下に Whisper の聞き取りが出ます。"
CREDITS_EN = "録音は TSP speech database（Copyright © 2018 Peter Kabal、Simplified BSD License）。文は Harvard sentences。合成は CHIMERA（MIT License）。再生用に MP3 96 kbps へ変換し、音量をそろえています。"
BODY_EN = """<section aria-labelledby="h-t"><h2 id="h-t">1. 韻律移植</h2>
<p>同じ文について、CHIMERA 標準の読み上げと、話者の F0 や音素長を移した読み上げを比べます。調音（舌や唇の動き）はどれも規則のままです。</p>
{{transplant_table}}
<p class="small">有声一致は、発話区間の 10 ms フレームのうち有声／無声の判定が実録音と同じだった割合。時間ずれは、スペクトルの時間伸縮対応が対角線から外れた量の平均。この 3 列は録音と同じ時間軸の条件だけに出ます。ケプストラム距離は別の実話者どうしでも同程度に大きく、条件の優劣を分ける力は弱い指標です。</p>
<div class="stack">{{sentences}}</div>
</section>
<section aria-labelledby="h-p"><h2 id="h-p">2. 制御列への摂動</h2>
<p>「規則の音素長 + 話者の F0」の発話を、100 Hz × 21 チャネルの制御列（{{channels}}）に書き出し、プランナなしで再駆動します。そこにノイズや量子化を加えて、音がどう崩れるかを聴きます。ノイズの大きさは、各チャネルが 1 発話の中で動く幅（標準偏差）に対する比です。</p>
{{perturb_table}}
<div class="stack">{{perturbed}}</div>
</section>
{{copy_section}}
{{codec_section}}
{{more}}
"""

TAUGHT_TEXT = """<p>日本語の録音を学習に加えます。JVS の 86 話者から 10 文ずつ（860 文、約 1.9 時間）を、英語の TSP（1,012 文）と合わせて閉ループ学習に入れました。テストは 3 種類です。JVS のうち学習にも検証にも使っていない 8 話者（女性 4・男性 4）が読む、学習に出てこない 25 文。第 1 節の録音（Hi-Fi-CAPTAIN と JSUT。学習には使っていない別のコーパス）。そして英語です。</p>
<p>もう 1 つ替えたのが「物差し」です。閉ループ学習は、合成音が録音と同じに聞こえるかを音声モデルの特徴で測ります。これまでの物差しは英語専用の wav2vec2 でした。下の表のとおり、この物差しは日本語の合成音を「別人の読み上げより録音に近い」と聞いてしまいます（比 0.89）。147 言語で学習した mHuBERT は「別人より遠い」と聞き分け（比 1.27）、第 1 節の音声記号認識器と同じ向きの判定になります。英語では、どちらも合成音を別人と同程度かそれより近いと聞きます。</p>
<ul>
<li><b>D</b>: 日本語の録音を加える（物差しは英語の wav2vec2 のまま）</li>
<li><b>E</b>: 物差しを mHuBERT に替える（録音は英語のまま）</li>
<li><b>F</b>: その両方</li>
<li><b>G</b>: F に加えて、エンコーダが録音を聞く入口も wav2vec2 から mHuBERT に替える</li>
</ul>
<p>作り方はどれも C′ と同じで、探索も文の書き起こしも使いません。</p>"""
TAUGHT_BETWEEN = """<p class="small">表の層は、モデルごとに日本語の合成音と別の話者を最もよく分けた層です（英語の wav2vec2 だけは、これまで使ってきた第 9 層）。物差しには mHuBERT の第 8 層を選びました。以下の Whisper と音声記号認識器は、どのエンコーダの学習にも使っていません。下のカードの音は、G だけです（C′ と D〜F の音は、後の節の音の場所を空けるために外しました。数値は表に残っています）。</p>"""

TUNED_TEXT = """<p>第 2 節のエンコーダ G には、発話の前に余計な音が出る癖がありました。原因は学習の手本です。JVS の録音は発話の前に 0.5 秒ほど無音がありますが、学習で録音と聞き比べているのは発話の区間だけで、その外では「合成音だけで学習したエンコーダ A の出力」を手本にしていました。A は録音の部屋ノイズが始まった時点で息を出し始めるので、それを真似ていました。</p>
<p>エンコーダ H は、G と同じ作り方で、発話の外では「息なし」を手本にします。あわせて、エンコーダ自身が足している先頭 0.2 秒の無音では、制御列を落ち着いた値から始めます。</p>"""
TUNED_VOICE = """<p>女声の声色には、NISQA（音声だけから主観品質の評点を予測するモデル）を物差しに使いました。男性の合成音は実録音に近い点数になるのに、女性の合成音は大きく下がります。調べると、点数は性別ではなく声の高さに従っていました。話者ごとの声の高さ（中央値）と点数の相関は −0.9 で、声の高い男性も下がります。原因はエンコーダではなく、CHIMERA のエンジンが高い声を鳴らすときの声門の動かし方にありました。エンジンの声門は、1 周期のうち開いている割合（開放率）が声の高さによらず約 0.6 で、設定で動かせるのも 0.78 までです。実際の声は、高いほど、また女声のほうが、開放率が大きいことが知られています。</p>
<p>そこで、Hi-Fi-CAPTAIN の女性話者の録音 16 文（テストとは別の文）を手本に、声門と雑音源の設定を調整しました。NISQA の点数が上がり、かつ多言語の物差しで録音から離れないように合わせています。エンジンのコードは変えておらず、声門のパラメータを外から直接指定しています。</p>"""
TUNED_RULE = """<p>上の 2 つの表から、調整した設定は声の高さが 175 Hz 以上の話者に使い、150 Hz 以下の話者は既定のまま、その間は混ぜることにしました。「H + 調整 1」などは、エンコーダ H の制御列をこの規則で鳴らしたものです。声の高さは、その話者の録音全体の中央値で決めています。調整には 3 つの版があります（下で説明します。前回の報告では「前回」「中間」「今回」と呼んでいました）。</p>"""
TUNED_WALLS = """<p>声の設定とは別に、合成音には録音にない音が 2 種類ありました。録音が無音の区間（破裂音で舌や唇が閉じている間と、文の途中のポーズ）で鳴る低い音と、舌が閉じる瞬間の鋭いクリック音です。順に対策を入れました。どちらも制御列は H のままで、男性の録音にも使っています。</p>
<p>対策 1 は、無音区間の音への対策です。実録音は無音区間で発話より 55 dB ほど小さくなります。合成音は既定の声の H でも 10 dB ほど大きく、短い音が何度も突き出ます。原因は 2 つありました。1 つ目は声道の壁が動く音です。CHIMERA は、舌・唇・あごが動いて押しのけた空気を、声道の中の流れとして計算します。そのため、息を完全に止めた制御列でも無音にはならず、300 Hz 以下を中心にした低い音が出ます。大きさは発話より 40 dB ほど小さく（男性の体では 30 dB ほど）、舌が速く動く瞬間は 20 dB 差まで上がります。男性の声はこの帯域に声の成分があるので隠れますが、女性の声にはなく、無音区間では何にも隠れません。CHIMERA 自身の規則で作った制御列では、さらに 3〜4 dB 大きく出ます。エンコーダの癖ではなくエンジンの性質です。エンジンにはこの流れを切るスイッチがあり（コードは変えていません）、それを切ります。2 つ目はポーズの中の息です。エンコーダ H は発話の前後では息を止めますが、文の途中のポーズでは弱い息を出し続けていて、かすかな声と、息が戻る瞬間の音になっていました。録音が 0.12 秒以上無音になる区間で息を止めます（区間が終わる 30 ms 前に戻します）。</p>"""
TUNED_CLOSURE = """<p>対策 2 は、舌が閉じる瞬間のクリック音への対策です。対策 1 で長い無音区間は静かになりましたが、舌を鳴らすような湿った音は残りました。有声区間の直前と直後の 35 ms ほどを無音にすると消える、というご指摘の区間を調べると、破裂音で舌が閉じきる瞬間に、16 kHz まで広がる 1〜2 ms の衝撃が出ていました。発話より 25〜35 dB 小さい音ですが、実録音はそこで無音です。雑音源をすべて切っても変わらないので、乱流の雑音ではありません。</p>
<p>原因はエンジンの扱いでした。CHIMERA は、狭めの面積が 0.0002 cm² を下回ると「閉じた」とみなして、そこを通る流れを 1 サンプルで 0 にします。息が十分にあると、その時点でまだ 1 cm³/s ほど流れていて、それが急に止まる段差が衝撃になります。このしきい値は外から設定できるので（コードは変えていません）、0.000001 cm² にしました。流れは切られる前に自然に消えます。</p>
<p>下の表の「閉鎖のクリック音」は、この種類の音を数えたものです。破裂の瞬間（舌が離れるとき）の音は、この対策では変わりません。Hi-Fi-CAPTAIN の女性話者の破裂 653 か所で比べると、合成音の破裂は 4 kHz 以上では実録音と同程度の強さですが（中央値で +1〜3 dB）、4 kHz 以下の成分が 5〜7 dB 弱く、実録音より鋭い音になっています。</p>"""
TUNED_CLICKS = """<p>声の設定によって変わるクリック音もあります。2〜9 kHz で前後から突き出た短い音を数えると、実録音では破裂音で舌や唇が離れる瞬間のものがほとんどで、既定の声の H も同じです。調整 1 では、舌が閉じる瞬間や、音が続いている途中にも出ます。NISQA だけに合わせると歯の雑音源のゲインが既定の 2.2 倍になること、声門を長く開く設定が息の流量を増やすことが原因です。</p>
<p>調整 2 と 3 は、声門の開き方（開放率・張り・明るさ）を調整 1 のままにして、クリック音に効く設定を変えたものです。どちらも、声門の振動の振幅を 1.4 倍にして、声を破裂に対して大きくしています。雑音源のゲインは、調整 2 が既定値、調整 3 が狭めと歯を既定の 0.6 倍です。ゲインを下げるほどクリック音は減り、NISQA の点数も下がります。数値を当てはめに任せると別の抜け道に行ってしまったので、指標を見ながら手で決めました。</p>"""
TUNED_CHECK = """<p class="small">声の設定や対策を変えても言葉が保たれているかを、第 2 節と同じ物差しで確かめたのが以下の表です。Whisper と音声記号認識器は、学習にも調整にも使っていません。NISQA は女声の設定の調整にだけ使いました（表の点数は、調整に使っていない録音のものです）。対策 1 で動きの音を切ると、音声記号の不一致が最大 2 ポイントほど増えます。増えるのは、破裂音の有声・無声の取り違え（t を d と聞く、など）です。エンコーダは動きの音がある声で学習したままです。対策 2 では変わりません。</p>"""

RETAUGHT_TEXT = """<p>第 3 節で決めた鳴らし方（声の高い話者には調整 2、対策 1+2。以下「調整後の声」）は、エンコーダ H が学習したときの声ではありません。H は、CHIMERA の既定の声を真似る代役モデルを通して学習しています。調整後の声で鳴らすと、音声記号の不一致が最大 5 ポイント増えていました。主に破裂音の有声・無声の取り違えです。</p>
<p>エンコーダ I は、H と同じ作り方で、調整後の声を通して学習し直したものです。学習用の制御列 3,864 本を調整後の声で鳴らし直し（制御列は同じです）、その音から代役モデルを作り直しました。代役には、話者ごとに「調整した声をどの割合で使うか」も入力しています。エポックを選ぶ聞き比べも、調整後の声で鳴らして行います。学習データと物差しは H と同じです。</p>
<p>音声記号の不一致は、増えた分の半分ほどから全部が戻りました（JVS は 31.7% から 29.3% に。H を既定の声で鳴らすと 28.3%。Hi-Fi-CAPTAIN の女性は 27.5% から 25.2% に。同 26.0%）。日本語 500 文で数えた有声・無声の取り違えは、H を既定の声で鳴らすと 141 件、調整後の声で鳴らすと 278 件、I では 200 件です。Whisper の誤り率は H と同程度で（差は 95% 区間の内側）、JVS と TSP ではわずかに下がりました。NISQA の点数は変わりません。</p>"""
RETAUGHT_SOUNDS = """<p class="small">録音にない音は、H を調整後の声で鳴らしたときとほぼ同じです。閉鎖のクリック音と破裂の大きさは変わらず、無音区間の中の音は同じか少なくなりました。</p>"""

TITLE_JA = "CHIMERA 日本語エンコーダの試聴"
LEDE_JA = "物理モデル音声合成 CHIMERA をデコーダにしたコーデックを、日本語の録音で試した結果です。合成エンジンは無改造で、変えたのはエンコーダの学習だけです。ボタンを押すと再生し、下に Whisper の聞き取りが出ます。"
CREDITS_JA = "録音は Hi-Fi-CAPTAIN（NICT、CC BY-NC-SA 4.0）、JSUT と JVS（東京大学 猿渡研究室。JVS は一部の音声だけを載せています）、TSP speech database（Copyright © 2018 Peter Kabal、Simplified BSD License）。合成は CHIMERA（MIT License）。再生用に MP3 96 kbps へ変換し、音量をそろえています。"


TITLE_TTS = "CHIMERA 日本語 TTS の試聴"
LEDE_TTS = "テキストから制御列（100 Hz × 21 チャネル）を書くモデルを作り、物理モデル音声合成 CHIMERA に鳴らさせた結果です。合成エンジンは無改造です。ボタンを押すと再生し、下に Whisper の聞き取りが出ます。"
CREDITS_TTS = "録音は Hi-Fi-CAPTAIN（NICT、CC BY-NC-SA 4.0）の日本語男性話者と、JVS（東京大学 猿渡研究室。一部の音声だけを載せています）。読みとアクセントの推定は Open JTalk。合成は CHIMERA（MIT License）。再生用に MP3 96 kbps へ変換し、音量をそろえています。"
SPEAKER_TEXT = """<p>これまで、声に伝える話者の情報は性別（男性か女性かの体格）と声の高さの中央値だけでした。そのため、話者照合モデルは同性の話者の合成音を聞き分けられませんでした。CHIMERA の体格は連続に変えられます（成人男性から成人女性へ、大人から子どもへ。声道の大きさと、そのうちの喉の割合が変わります）。声門にも、人によって違う設定があります。これを話者ごとに当てはめました。制御列は同じものを使い、鳴らすときの設定だけを変えます。</p>
<ul>
<li><b>体格を当てはめ</b>: 体格の 2 つの値だけを話者ごとに決めます。声門は、日本語編で決めた「調整後の声」のままです。</li>
<li><b>体格 + 声門を当てはめ</b>: 体格に加えて、声門の開放率・明るさ・張りも話者ごとに決めます。</li>
<li><b>体格 4 値 + 声門 + 構えを当てはめ</b>: 体格を 2 つの軸ではなく 4 つの値（大きさ、咽頭の割合、舌の半径、唇の幅）で直接決め、成人男性より大きい声道も許します。さらに、エンコーダが書かない「構えの癖」を 4 つ決めます。喉頭の上の狭め（声の響き）と、喉頭の高さ・鼻への通路・舌の根のずれです。話者あたり 11 個の値です。</li>
</ul>
<p>当てはめには、話者ごとに学習用の 6 文（それぞれ最初の 4 秒）を使いました。候補の設定で鳴らし、話者照合モデル（WavLM）が「その話者の録音 20 文に近い」と判定する設定を探します（試行は順に 40 回、90 回、200 回）。言葉が変わらないように、多言語のリスナーから見た録音との距離が増えるぶんは減点します。下の表は、当てはめに使っていないテストの文での結果です。</p>"""
SPEAKER_COSTS = """<p>当てはめで変わるのは声色だけではありません。聞き取りやすさと音の質への影響を、同じテストの文で測りました。</p>"""
SPEAKER_NOSE = """<h3>女性の声が鼻詰まりのように聞こえる点</h3>
<p>試聴で、女性 2 人の合成音は「鼻にかかる感じが強く、鼻が詰まった状態の発話のようだ」という指摘がありました。そこで、録音と合成音を音素の種類ごとに比べました（下の表）。</p>
<ul>
<li><b>鼻音（m、n、ん）が弱い。</b>女性の録音では、鼻音は母音より 1.6 dB 強く出ています。合成音では逆に 3 dB ほど弱く、録音との差は 4.8 dB です（「体格 + 声門」では 6.2 dB）。男性での差は 2.3 dB です。</li>
<li><b>鼻音の音色が違う。</b>女性の録音の鼻音はほとんど基本波（約 260 Hz）だけで、第 2 倍音はそれより 20 dB 弱くなっています。合成音では 10 dB しか弱くありません（録音との差は 10 dB。「体格 + 声門」では 15 dB）。男性での差は 2 dB です。</li>
<li><b>鼻音の隣の母音でも、低い響きが足りない。</b>録音では、鼻音に隣り合う母音でも基本波が 4 dB ほど強まります。合成音では、ほとんど強まりません。</li>
<li><b>「体格 + 声門」の当てはめは、女性の母音の倍音を強めていた。</b>第 2・第 3 倍音が録音より 6.9 dB と 9.1 dB 強くなります（当てはめる前は 1.6 dB と 4.5 dB）。照合モデルに合わせて、声門の開放率が下がったためです。</li>
</ul>
{{nasal_table}}
<p>どれも「鼻を通る低い響きが足りない」向きのずれで、鼻が詰まった声の特徴と合います。ただし、指摘された聞こえ方の原因がこれだとは言えません。次の試聴では、副鼻腔だけを変えた音（「体格 + 声門」と、その副鼻腔を大きくした版）は聞き分けられず、鼻声かどうかも「自信がなくなってきた」とのことでした。</p>
<p>次に、制御列や設定を 1 つずつ変えて鳴らし、どれが効くかを調べました。鼻への通路（軟口蓋）を閉じ切っても、母音は変わりません。通路を 2 倍に開いても、鼻孔を狭めても、鼻音はほとんど変わりません。効いたのは副鼻腔の容積でした。CHIMERA の鼻腔には副鼻腔が 1 つ（14 cm³）付いていて、鼻音の 700 Hz 付近を削ります。人の上顎洞は左右を合わせて 30 cm³ ほどです。容積を 24 cm³ にすると、削る位置が 500 Hz 付近に下がり、女性の鼻音の第 2 倍音のずれは 10 dB から 5 dB に縮みます（鼻音から離れた母音は変わりません）。エンジンは無改造で、ラッパーから鼻腔の設定を渡しています。</p>
<p>鼻音の強さの不足（4.6 dB）は残りました。CHIMERA の鼻腔は壁の損失が大きく、低い共鳴が鈍いためで、ラッパーから変えられる設定では直せませんでした。</p>
<p>下の表は、話者の照合を女性と男性に分けたものです。副鼻腔を大きくすると、女性では本人の録音との類似度が 0.01 ほど上がり、男性では下がります（11 個の値を当てはめた声では変わりません）。聞き取りやすさと NISQA は変わりません（上の表の「副鼻腔を大きく」の行）。この表からは、体格の当てはめで本人と判定されるようになったのは主に女性で、男性は 11 個の値を当てはめてようやく 40% になることも分かります（8 話者の中での話です。100 話者での内訳は、次の小節の最後の表にあります）。</p>
{{apart_table}}"""
SPEAKER_BRIGHT = """<h3>女性の声に足りない「華やかさ」: 声門の閉じ方</h3>
<p>次の試聴での指摘は、こうでした。男性は「体格 4 値 + 声門 + 構え」のほうが良い。女性は、リアルさに何かが大きく足りない（鼻声かと問われると自信がない。華やかさのようなもの）。jvs093 は、声門を当てはめると有声音がブザーのように人工的になる。jvs093 の「これまで」から「体格だけ、副鼻腔を大きく」への変化は良い方向。jvs094 は 4 つとも聞き分けられない。</p>
<p>そこで、母音の強さを帯域ごとに録音と比べました（下の表）。</p>
<ul>
<li><b>女性の母音は、1.6〜5.5 kHz が録音より 7〜12 dB 弱い。</b>声の明るさにあたる帯域です。</li>
<li><b>「体格 + 声門」の当てはめは、この帯域を 3〜5 dB 戻す代わりに、低い倍音を強めていた。</b>第 2 倍音が録音より 8.0 dB 強くなります（当てはめる前は 2.8 dB）。照合モデルに合わせて、声門の開放率が下がったためです（jvs093 は 0.85 から 0.59、jvs094 は 0.75）。ブザーのような聞こえ方の原因はこれだと見ていましたが、次の試聴では、開放率を変えずに閉じ方を鋭くしたときにも同じ話者でブザー感が出ました（この小節の最後）。第 2 倍音だけでは説明できません。</li>
<li><b>男性では、「体格 4 値 + 声門 + 構え」が 1〜4 kHz を 5〜8 dB 戻していた。</b>試聴で「良い」と判定された音です。当てはめる前の不足は 9〜15 dB で、第 2 倍音は録音より 5.6 dB 弱かったのが、1.5 dB 強い程度になっています。</li>
</ul>
{{bright_table}}
<p>設定を 1 つずつ変えて鳴らすと、低い倍音を変えずにこの帯域を戻せるのは、声門の閉じ方の鋭さでした。日本語編で決めた高い声は、閉じ方がいちばん柔らかい設定（0）になっています。開放率を下げる、あるいは閉じる側を速くすると、この帯域と一緒に第 2・第 3 倍音が 4〜8 dB 強まります。喉頭の上の狭めは、女性ではほとんど効きません（2 dB 以内）。</p>
<p>鋭さは、照合モデルを使わずに話者ごとに測れます。学習用の 6 文を鋭さ 0〜0.8 の 6 通りで鳴らし、録音が有声の区間で 1.6〜5.5 kHz の強さが録音と同じになる値を取ります。JVS の女性 51 人では 0〜0.62、平均 0.35 でした。</p>
{{closure_table}}
<p>測った値で鳴らすと、女性の 1.6〜5.5 kHz の不足は 3 dB 以内になり、第 2 倍音のずれも 2.5 dB から 1.6 dB に縮みます。聞き取りやすさは変わりません（上の表の「声門の閉じ方を鋭く」の行）。代わりに悪くなる点が 2 つあります。</p>
<ul>
<li><b>5.5〜7.5 kHz が録音より 7 dB 強くなる。</b>録音の母音は、この帯域でその下より 10 dB 落ち込みます。合成音にはその落ち込みがなく、下の帯域を合わせるとここが出すぎます。</li>
<li><b>NISQA の品質が 3.90 から 3.55 に下がる。</b>自然さは 3.63 から 3.71 に上がります。多言語のリスナーから見た録音との距離は、0.217 から 0.227 に少し増えます。</li>
</ul>
<p>測った値の半分にすると、得るものも失うものも半分ほどです（1.6〜5.5 kHz の不足は 4〜6 dB、5.5〜7.5 kHz は +2.7 dB、品質は 3.68）。</p>
<p>話者の聞き分けには効いていません（下の 2 つの表）。8 話者の中では、モデルによって向きが違います。体格の当てはめに使っていないモデル（UniSpeech-SAT）では、女性を本人と判定する割合が 58% から 85% に上がり、体格の当てはめに使ったモデル（WavLM）では 79% から 68% に下がります。100 話者の中では、どちらのモデルでも変わりません（女性で 7.8% から 7.4% と、7.8% から 9.0%）。別のモデルでの本人との類似度は 0.958 から 0.967 に上がりますが、同性の他人との類似度も 0.942 から 0.949 に上がります（表は WavLM の類似度です）。閉じ方の鋭さは、女性の合成音を全体として録音に近づけますが、誰の声かは変えていません。</p>
{{closure_apart}}
{{closure_everyone}}
<p>100 話者の表からは、もう 1 つ分かります。前の小節で「体格の当てはめで本人と判定されるようになったのは主に女性」と書いたのは、8 話者の中だけの話でした。100 話者の中では、体格の当てはめは女性も男性も同じくらい押し上げます（2.4% から 5.4%、3.3% から 5.6%）。11 個の値で伸びるのは男性です（20.5%。女性は 8.2%）。</p>
<p>摩擦音も調べました。女性の無声摩擦音（s、sh、h、f）は、母音に対して録音より 3.8 dB 弱く、5.5 kHz より上では 8〜10 dB 弱くなっています（男性は 2.2 dB と 4〜6 dB）。歯の雑音を 2 倍にすると 5.5 kHz より上の不足は 2〜5 dB に縮みますが、破裂音の立ち上がりも 2.5〜5.5 kHz で 5 dB ほど強まります。破裂音は今でも録音より鋭いので、この変更は入れていません。</p>
<p><b>試聴の結果と決定。</b>閉じ方を鋭くすると高域が出て、良くなります。jvs094 は測った値（0.33）でも問題ありません。jvs093 は測った値（0.51）だとブザー感が出てしまい、半分（0.25）ならほとんど出ません。そこで、女性の声は「体格 + 副鼻腔 + 閉じ方の鋭さ（測った値の半分）」に決まりました。男性は前回のとおり「体格 4 値 + 声門 + 構え」です。</p>
<p>ブザー感は、これで 2 つの条件で出たことになります。開放率を下げたとき（第 2・第 3 倍音が録音より 8〜10 dB 強い）と、閉じ方を 0.51 まで鋭くしたとき（第 2 倍音は録音どおりで、5.5 kHz より上が録音より強い）です。両方に共通する原因は、まだ特定していません。</p>"""
RELEASE_TEXT = """<p>ここからは公開（V1）に向けた作業です。公開するモデルの重みは、再配布できる材料だけから作られている必要があります。調べると、これまでのモデルはその条件を満たしていませんでした。</p>
<div class="scroll"><table>
<caption>これまでのモデルが依存している材料と、その利用条件（2026-11-16 に各配布元の表示で確認）。</caption>
<thead><tr><th scope="col">材料</th><th scope="col">使っているところ</th><th scope="col">利用条件</th><th scope="col">公開版では</th></tr></thead>
<tbody>
<tr><th scope="row">mHuBERT-147</th><td>エンコーダ I が録音を聞く入口（調整した層が重みに入っている）と、学習の物差し</td><td>CC BY-NC-SA 4.0（非営利・継承）</td><td>入口は wav2vec 2.0 base、物差しは XLS-R 300M（どちらも Apache-2.0）</td></tr>
<tr><th scope="row">JVS</th><td>エンコーダの学習の録音、100 話者の TTS</td><td>研究と個人利用のみ。再配布不可</td><td>Emilia-YODAS（CC BY 4.0）。JVS は評価にだけ使う</td></tr>
<tr><th scope="row">Hi-Fi-CAPTAIN</th><td>1 話者の TTS、高い声の設定の調整</td><td>CC BY-NC-SA 4.0</td><td>TTS の録音は選び直す（未定）</td></tr>
<tr><th scope="row">TSP</th><td>英語の録音、規則で作る最初の制御列</td><td>BSD 2-Clause（著作権表示のみ）</td><td>そのまま使う</td></tr>
<tr><th scope="row">wav2vec 2.0 base 960h</th><td>規則だけで学習する最初のエンコーダの入口</td><td>Apache-2.0（学習は LibriSpeech、CC BY 4.0）</td><td>そのまま使う</td></tr>
<tr><th scope="row">NISQA、Whisper、音声記号認識器、話者照合モデル</th><td>評価。NISQA は高い声の設定の調整にも使った</td><td>NISQA の重みは CC BY-NC-SA 4.0</td><td>評価にだけ使う</td></tr>
</tbody></table></div>
<p>エンコーダ I は、入口の mHuBERT の層を重みの中に持っているので、公開できません。その制御列で学習した 2 つの TTS も同じです。そこで、エンコーダから作り直しました。替えたのは、声、物差し、録音の 3 つです。</p>"""
RELEASE_VOICES = """<h3>2 つの基準の声</h3>
<p>第 5 節の設定は、話者ごとに録音から当てはめたものです。V1 では利用者が声の高さと明るさを選ぶだけなので、当てはめは使えません。そこで、決まった設定を男女それぞれでまとめ、男性用の低い声と女性用の高い声の 2 つにしました（リポジトリの voices/v1.json）。</p>
<ul>
<li><b>高い声</b>は、第 3 節の調整 2 に、女性の体格の平均（声道は成人男性の 0.87 倍）、副鼻腔 24 cm³、閉じ方の鋭さ 0.18（測った値の半分の平均）を足したものです。</li>
<li><b>低い声</b>は、CHIMERA の成人男性に、喉頭の上の狭め 0.78 と、舌の根を 0.35 cm 前に出す構えを足したものです。11 個の値を 1 つずつ試すと、男性の合成音を録音に近づけるのはこの 2 つでした。話者照合モデルによる本人との類似度は 0.870 から 0.903 に上がります（11 個すべての平均を当てると 0.899）。開放率、息漏れ、ゆらぎの設定は効きませんでした。</li>
<li><b>どちらの声を使うかは、声の高さではなく性別で決めます。</b>JVS の男性には声の高さの中央値が 196 Hz の話者がいて、TSP の女性は 173 Hz から始まります。声の高さで体格を混ぜると、高めの男性に女性寄りの体格が当たり、男性の本人との類似度は 0.850 から 0.812 に下がりました。</li>
</ul>
<p>これまでの鳴らし方では、声の高さが 150 Hz を超える男性に、女性用の柔らかい声源が混ざっていました（テストの男性 4 人のうち 3 人。jvs099 は 9 割）。そのため第 2 倍音が録音より 5.6 dB 弱くなっていました。試聴で男性の「体格 4 値 + 声門 + 構え」が良く聞こえた理由の 1 つは、当てはめがこの声源を男性側に戻していたことだと見ています（耳では確かめていません）。</p>
<p>下の 2 つの表は、エンコーダ I の制御列をこの 2 つの声で鳴らした結果です。女性は、話者ごとの設定とほぼ同じ釣り合いになります。男性は、1〜4 kHz の不足が 9〜15 dB から 5〜8 dB に縮み、第 2 倍音は録音どおりになり、音声記号の不一致は 27.4% から 25.8% に下がります。話者ごとの設定に及ばないのは本人との類似度で、これは全員に同じ設定を当てているためです。</p>"""
RELEASE_DIAL = """<h3>声のつまみ: 明るさ、高さ、体格</h3>
<p><b>声の明るさのつまみ。</b>V1 の「こもる ←→ 明るい」にあたるつまみも、この 2 つの声に付けました。1 目盛りで、声門の閉じ方の鋭さを 0.18 動かし、低い声では喉頭の上の狭めも 0.22 動かします。母音の 2.5〜4 kHz は、1 目盛り下げると女性で 3.8 dB、男性で 5.0 dB 弱まり、1 目盛り上げると女性で 4.8 dB、男性で 4.5 dB 強まります。第 2 倍音の動きは 1.2 dB 以内です。鳴っている途中でも変えられる設定です。聞き取りやすさは変わらず、NISQA の品質は、女性の声では明るくするほど下がります（3.95、3.64、3.44）。</p>
<p class="small">下げ側は、試聴で「上げ側より変化が少し小さい」との指摘を受けて調整したあとの値です。閉じ方の鋭さは 0 より下がらず、高い声は 0.18 から 1 目盛りで底に着きます。そこで下げるときは、声門の波形の傾きも 0.05 下げるようにしました。調整前の下げ側は、女性で 3.0 dB、男性で 3.8 dB でした。</p>
<p><b>声の高さ</b>は、制御列の高さの列を上下するだけです。3 半音上げ下げしても、聞き取りやすさはほとんど変わりません。</p>
<p><b>体格を入れ替えても読めます。</b>女性の録音から書いた制御列を低い声（男性の体格）で、男性の録音から書いた制御列を高い声（女性の体格）で鳴らしても、Whisper の誤りは 3.1〜3.8% のままです。制御列は体格にあまり依存していないので、1 つの TTS で、つまみのどの位置の声も鳴らせる見通しです。この実験からはもう 1 つ分かります。男性の制御列を高い声で鳴らすと、音声記号の不一致は 25.8% から 31.7% に、NISQA は 4.34 から 3.59 になり、女性の制御列を低い声で鳴らすと 31.7% から 29.3% に、3.64 から 4.74 になります。女性の合成音の点が低いのは、女性の制御列のせいではなく、高い声そのもののせいです。</p>"""
RELEASE_LISTENERS = """<h3>公開できる物差し</h3>
<p>学習の物差し（聞き手のモデル）は、合成音と録音の距離を測る役です。重みのライセンスが Apache-2.0 か MIT のモデルを、第 2 節と同じ方法で比べ直しました。XLS-R 300M の第 16 層は、日本語でも英語でも合成音の足りなさを聞き取れます（差が +0.97 と +0.33。mHuBERT は +1.32 と −0.05）。これを物差しにし、エンコーダの入口は、最初のエンコーダと同じ wav2vec 2.0 base にしました。</p>
<p class="small">XLS-R の重みは Apache-2.0 ですが、その事前学習には VoxPopuli、MLS、Common Voice、VoxLingua107 のほかに、配布に制限のある BABEL が使われています。事前学習の材料まで公開できるものに限るなら、物差しは英語の wav2vec 2.0 base に戻ることになり、日本語の音声記号の不一致は 5 ポイントほど悪くなります（第 2 節のエンコーダ D と F の差）。</p>"""
RELEASE_ENCODERS = """<h3>エンコーダを学習し直す</h3>
<p>エンコーダは 3 つ学習しました。違いを 1 つずつ足して、何がどれだけ効くかを分けています。どれも基準の声を通して学習し、基準の声で鳴らします。</p>
<ul>
<li><b>I′</b>: I と同じ材料（TSP と JVS の録音、mHuBERT）のまま、声だけを基準の声に替えて学習し直したもの。</li>
<li><b>J0</b>: I′ の物差しを XLS-R に、入口を wav2vec 2.0 base に替えたもの。録音は TSP と JVS のままです。</li>
<li><b>J</b>: J0 の JVS を Emilia-YODAS に替えたもの（日本語と英語、各 300 話者 × 6 文、計 6 時間ほど）。公開できる材料だけで学習しています。</li>
</ul>
<p>Emilia-YODAS には話者の性別の表示がありません。声の高さで決まる話者（中央値が 135 Hz 未満か 235 Hz 超）はそのまま決め、その間の話者は、話者照合モデルのベクトルが男女どちらの集団に近いかで決めました。性別のわかっている JVS と TSP で試すと、その間の話者の 90〜95% が合います（声の高さだけで、いちばん良い境目を選んだ場合と同じ程度です）。</p>
<p>結果は下の表のとおりです。</p>
<ul>
<li><b>声を替えた分の学習し直しは、少ししか効きません。</b>JVS の音声記号の不一致は、I の制御列を基準の声で鳴らすだけで 29.3% から 28.8% になり、学習し直した I′ で 28.0% です。Whisper の誤りは区間の中で変わりません。英語（TSP）は 9.4% から 10.1% で、わずかに悪くなります。</li>
<li><b>物差しと入口を公開できるモデルに替えると、はっきり悪くなります。</b>J0 は I′ に比べて、JVS の音声記号の不一致が 28.0% から 32.8%、Whisper large-v3-turbo の誤りが 3.6% から 5.0% です。TSP でも 10.1% から 12.7%、4.1% から 7.0% です。</li>
<li><b>録音を Emilia-YODAS に替えるのは、損ではなく得でした。</b>J は J0 より、どのテストでも良くなります（JVS で 32.8% から 31.5%、TSP で 12.7% から 10.6%）。JVS の録音を一度も聞いていないのに、JVS のテストでも J0 を上回ります。話者が 86 人から 600 人に増えた効果だと見ています。</li>
<li><b>差し引きで、J は I′ より日本語で 3.5 ポイント悪い。</b>JVS で 28.0% から 31.5%（Whisper の誤りは 3.6% から 4.9%）、Hi-Fi-CAPTAIN と JSUT で 27.2% から 30.7%（0.5% から 1.2%）。英語は 0.5 ポイント以内です。NISQA の品質は変わりません。</li>
<li><b>物差しを日本語の HuBERT に替えた J2 で、その損がなくなりました。</b>J2 は、公開できる材料だけで学習して、JVS の音声記号の不一致が 28.1%、Whisper の誤りが 3.6% です（I′ は 28.0% と 3.6%）。Hi-Fi-CAPTAIN と JSUT は 27.5% と 0.5%（27.2% と 0.5%）、TSP は 10.5% と 4.6%（10.1% と 4.1%）で、区間の中で I′ と並びます。公開用のエンコーダは J2 にします。物差しの選び方は下の「物差しをさらに選び直す」のとおりです。</li>
</ul>"""
RELEASE_MORE = """<p><b>物差しをさらに選び直す。</b>J の損は物差しと入口から来ていたので、公開できる範囲で別の組み合わせを、J0 と同じ録音（TSP と JVS）で試しました。{{more}}</p>"""
RELEASE_MORE_RESULT = "入口を XLS-R にしたもの、物差しを XLS-R の第 12 層にしたもの、物差しを日本語の hubert-base-jtube（重みは MIT）にしたものの 3 つです。JVS の音声記号の不一致は 3 つとも 31.0〜31.1% でした（J0 は 32.8%、I′ は 28.0%）。Whisper large-v3-turbo の誤りは、JVS で順に 4.3%、4.3%、3.3%（J0 は 5.0%、I′ は 3.6%）、TSP で 5.5%、6.1%、5.3%（J0 は 7.0%、I′ は 4.1%）です。日本語の HuBERT を物差しにすると、Whisper の誤りは日本語でこれまでの水準に戻り、英語でも J0 より良くなります。この物差しと Emilia-YODAS の録音を組み合わせたのが、上の表の J2 です。録音を替えた分の得が加わって、音声記号の不一致も I′ と並びました。hubert-base-jtube の事前学習は YouTube の音声（JTubeSpeech）で、XLS-R と同じく、重みのライセンスで使えると判断しています。"
RELEASE_TTS = """<h3>公開できる録音だけで学習した TTS</h3>
<p>エンコーダの制御列を手本に、公開できる録音だけで日本語の TTS を学習しました。録音は 2 種類試しました。Emilia-YODAS（動画の中の話し言葉を自動で書き起こしたもの、CC BY 4.0）と、Common Voice（ボランティアが文を読み上げたもの、CC0）です。モデルも 2 種類あります。話者ごとにベクトルを持つこれまでの作りと、話者のベクトルをやめて「高い声の割合」と「話す速さ」の 2 つの数だけを入力にした、つまみ用の作りです。</p>
{{tts_table}}
<ul>
<li><b>エンコーダが録音をどれだけ写せるかで、TTS の出来が決まります。</b>エンコーダが書いた制御列を鳴らしたときの Whisper の誤りは、J では Emilia-YODAS の録音で 7.7%、Common Voice の録音で 13.9% でした（録音そのものは 4.0% と 3.2%）。J2 では 5.0% と 7.0% です。Common Voice は録音した部屋もマイクもまちまちで、エンコーダには難しい録音でした。J の制御列で学習した Common Voice の TTS が悪かった（話者を 6 人替えても 6.6〜15.7%）のは、このためです。</li>
<li><b>話者のベクトルをやめたほうが良く読めます。</b>同じ Common Voice と J の制御列で、話者ベクトルのモデルは 11.7% と 5.7%、つまみ用のモデルは 5.4% と 4.1% でした。1 人あたり十数文しかない話者のベクトルは、その話者の録音の悪さも覚えてしまいます。つまみ用のモデルは、同じ声の話者を全員まとめて学習します。</li>
<li><b>J2 の制御列で、実録音の水準に近づきました。</b>低い声は Emilia-YODAS で学習したものが 2.4%、高い声は Common Voice で学習したものが 2.8% です（実録音は 2.5%、JVS で学習したこれまでのモデルは 2.3%）。良いコーパスが声で分かれるのは、Emilia-YODAS に女性が少なく（学習した約 800 人のうち、女性と推定されたのは 100 人ほど）、Common Voice は録音の条件が悪いためだと見ています。</li>
<li><b>2 つの録音を合わせると、1 つのモデルで両方の声が読めます。</b>つまみ用のモデルには話者のベクトルがないので、2 つを合わせて学習できます（15,279 文、22 時間、約 1,400 人）。合わせたモデルは低い声 3.0%、高い声 2.9% でした。それぞれの声でいちばん良かった単独のモデル（2.4% と 2.8%）にはわずかに届きませんが、単独のモデルの苦手な側（3.6% と 4.8%）よりずっと良くなります。体格のつまみを 2 つの声の途中に置けるのも、1 つのモデルで両方を学習した場合だけです。V1 の日本語モデルはこれにします。</li>
</ul>
<p><b>英語のモデル。</b>読みには、CHIMERA 自身の英語フロントエンド（CMUdict と、辞書にない語のための綴りからの規則）を、エンジンを変えずに呼び出して使います。録音は LibriTTS-R の train-clean-100（朗読、CC BY 4.0。使ったのは 7,534 文、12.9 時間、240 人）で、モデルの作りは日本語のつまみ用モデルと同じです。TSP のテスト文（4 話者 × 60 文）を読ませると、Whisper の誤りは低い声で 1.9%、高い声で 3.1% でした（実録音は 1.6%、エンコーダ J2 の制御列は 4.6%）。このページに英語の音はありません。下の「ブラウザの画面」で聴けます。</p>
<p><b>話す速さ。</b>最初の版の TTS は、jvs100_094 が早口すぎるとの指摘でした。測ると、その声は全部の文で速く（1 秒に 13.3 音素。JVS の録音は 10.2）、いちばん長い 094 が最も速くなっていました。話し言葉は読み上げより速く、話者ごとの速さは話者のベクトルに入っていました。いまは速さを指定でき（<code>--rate</code>）、つまみ用のモデルでは入力の 1 つです。表とカードの TTS は、すべて録音と同じ 10.2 に揃えています。</p>
<p class="small"><b>前の版からの訂正。</b>前の版では「Emilia-YODAS は、書き起こしと音が合わない録音を 4 割強外した」と書きましたが、外す基準のほうが誤っていました。「1 フレームしかない母音が 1 割を超える録音」を外していたため、「おいしい」のように同じ母音が続く文まで落ちていました（続く 2 つの母音は、片方が 1 フレームになります）。「1 フレームの音素が 4 つ以上続く録音」（文にあって音にない語の跡）に直すと、外れるのは Emilia-YODAS で 27%、Common Voice で 14% です。また、Common Voice の録音は発話の前後に長い無音があり、切り詰めてから使っています。</p>"""
RELEASE_WEB = """<h3>ブラウザの画面</h3>
<p>V1 の画面を作りました。CHIMERA の Speak 画面に倣った構成で、配色は明るくして見分けがつくようにしています。左に声道の断面図と管の太さ、右に文の入力、声のつまみ、制御列の全 21 チャネルです。断面図は、いま鳴っているフレームの姿勢を、エンジンが管を計算するのに使う幾何そのもので描いています。日本語と英語を切り替えられ、画面の言葉もそれに合わせて替わります。アドレスに <code>?ja</code> か <code>?en</code> を付けると、その言語で開きます（付けなければブラウザの言語です）。画面で切り替えると、アドレスもそれに合わせて変わります。CHIMERA 本家へのリンクは冒頭と末尾に置きました。</p>
{{web_shot}}
<p>音はブラウザの中で作ります。CHIMERA のエンジンを、私たちの制御列を再生する形で WebAssembly にしたもので（667 KB）、Python で鳴らした音と 1 サンプルも違わず、1 コアで実時間の 5.5〜5.9 倍の速さで動きます。サーバが返すのは制御列だけです（1 文で数十 kB。2 コアの CPU で 0.15 秒）。声の高さと明るさは話している途中でも変えられ、体格、話す速さ、抑揚は次に話すときから変わります。</p>
<p><b>本物のブラウザで確かめました。</b>ヘッドレスの Chrome でページを開き、「話す」を押し、話している途中につまみを回して、出てきた音を測るテストです（<code>tools/test_browser.mjs</code>）。音が出ること、声の高さのつまみを 120 Hz から 240 Hz に回すと測った高さが 143 Hz から 235 Hz に移ること、「止める」で止まること、エラーが出ないこと、スマートフォンの幅で横にはみ出さないこと、アドレスの <code>?ja</code> と <code>?en</code> でその言語になることを確かめています。手元で試すには、リポジトリで次の 2 行を実行して http://127.0.0.1:7860/ を開きます。</p>
<pre><code>PATH="$(brew --prefix rustup)/bin:$PATH" cargo build -p pmvc-wasm --release --target wasm32-unknown-unknown
uv run python -m pmvc.serve --model {{web_model}}</code></pre>
<p><b>公開の準備。</b>モデルは公開する形に書き出しました。重みは safetensors、それ以外に要るものは JSON で、学習に使った話者の名前は入りません（チェックポイントと同じ制御列を書くことを確かめています）。日本語と英語の TTS、エンコーダ J2、2 つの声の定義を、モデルカードと一緒に Hugging Face のモデル用リポジトリに置きました（非公開のまま）。新しく clone したリポジトリで、README の手順どおりにモデルを取得して画面が動くところまで確かめています。</p>
<p><b>Space。</b>Hugging Face の Space で動いています（非公開のまま）。この Space のハードウェアは ZeroGPU で、ZeroGPU が動かせるのは Gradio のアプリだけです。そこで Gradio の Server モード（<code>gr.Server</code>。Gradio の画面を持たず、自前のページと応答をそのまま出せる形）を使いました。ページとその応答は手元のサーバと同じコードで、同じ文に同じ制御列を返します。モデルは Space の CPU で動き、1 文にかかるのは通信を除いて 0.05〜0.12 秒です。同じ呼び出しを Gradio の API としても置いてあり、そちらでは GPU も選べますが、待ち時間は変わりません（手元からの往復で CPU 1.5 秒、GPU 1.7 秒）。手元と同じブラウザのテストを、Hugging Face 上の Space に対しても通しました。</p>
<p>最初は FastAPI に Gradio を載せる形で作り、手元では動きましたが、Space は起動の直後に止められました。Hugging Face が入れる <code>spaces</code> パッケージ（0.51.3）は、Gradio の <code>launch()</code> が呼ばれたときにだけ ZeroGPU に起動を報告するためです。Server モードは <code>launch()</code> を通ります。</p>"""
RELEASE_SOUNDS = """<h3>試聴の結果と、いまのカード</h3>
<p>このページへの試聴は 3 回ありました。</p>
<ul>
<li><b>1 回目。</b>基準の声は問題ない。明るさのつまみは良いが、下げ側の変化が上げ側より少し小さい。エンコーダ J は I と比べると少し巻き舌に聞こえる（比べないと分からない程度）。TTS は、jvs100_094 が早口すぎる以外は出発点として良い。</li>
<li><b>2 回目</b>（つまみの下げ側の調整、エンコーダ J2、話す速さを揃えた TTS のあと）。明るさは良い。巻き舌は非常に微妙な差で、日本語の HuBERT を物差しにした J2 で当面は良い。速さは良い。</li>
<li><b>3 回目</b>（2 つの録音を合わせて学習した TTS と、ブラウザの画面のあと）。画面はほぼ思い描いていたとおり。V1 の音声はこれで完成でよい。</li>
</ul>
<p>これで、声（2 つの声と明るさのつまみ）、公開用のエンコーダ（J2）、話す速さ（1 秒 10.2 音素から始める）、V1 の日本語モデル（2 つの録音を合わせて学習したもの）が決まりました。巻き舌については、J の制御列は舌の側面を開く動きが I より大きい（ラ行で 0.154 と 0.127）ことまでは測りましたが、それが原因かどうかは確かめていません。J2 でもこの値は同じです。</p>
<p>下のカードは、これまでと同じ 4 話者・4 文です。決まったものだけを残し、TTS を 3 つ並べてあります（Emilia-YODAS だけ、Common Voice だけ、両方）。男性の文は低い声、女性の文は高い声で読ませています。数字の上では、男性の文は Emilia-YODAS だけ、女性の文は Common Voice だけのモデルがわずかに良く、両方のモデルはその中間です。</p>"""
SPEAKER_SOUNDS = """<p>下のカードは、第 2〜4 節と同じ 4 話者・4 文です。決まった設定だけを残しました。女性 2 人は、実録音、これまでの鳴らし方、「体格 + 副鼻腔 + 閉じ方を鋭く（半分）」です（鋭さは jvs093 が 0.25、jvs094 が 0.17）。男性 2 人は、実録音、これまでの鳴らし方、「体格 4 値 + 声門 + 構え」です。聞き比べに使った他の音（「体格だけ、副鼻腔を大きく」と、測った値の鋭さ）は、次の節の音の場所を空けるために外しました。</p>"""
CONTROL_TEXT = """<p>少ない録音で学習できるのは制御列だからなのか、それとも、このモデルの形ならどの潜在表現でもそうなのかを確かめる対照実験です。第 1 節と同じモデル（同じ音素とアクセントの入力、同じ音素の長さ、同じ層）に、制御列の代わりに、ニューラルネットが音に戻す潜在表現を書かせました。</p>
<ul>
<li><b>メルスペクトログラム</b>（100 帯域、93.75 Hz）。UnivNet ボコーダで音にします。この形のモデル（FastSpeech 2）の本来の出力先です。</li>
<li><b>EnCodec の潜在表現</b>（量子化前の連続値、128 次元、75 Hz）。</li>
<li><b>Mimi の潜在表現</b>（32 コードブックが表すベクトル、512 次元、12.5 Hz）。Pocket TTS が使うコーデックの公開版です。</li>
</ul>
<p><b>調整したボコーダ</b>の行は、メルの対照を強くしたものです。ボコーダは実録音のメルで学習されていて、モデルが書くぼやけたメルを見たことがありません。そこでこの分野の通常の手順どおり、モデルの学習に使ったのと同じ録音だけで、ボコーダを調整しました。入力は、モデルがその録音について書いたメル（録音の音素長とピッチで書かせたもの）、目標は録音の音です。UnivNet の学習に使われた識別器（作者が公開しているチェックポイントに入っています）と、同じ誤差を使いました。残す重みは、検証の文で NISQA が最も高いものです。NISQA で選んでいるのはこの行だけで、対照に有利な選び方です。</p>
<p>ほかはどれも公開されているモデルをそのまま使い、24 kHz で動かしています。対照のモデルには、FastSpeech 2 のピッチ予測器を付けました。制御列にはピッチのチャネルがありますが、これらの潜在表現にはないためです。物理モデルの側は、条件を揃えるために、代役を通した誤差を外して学習し直しました（18,655 文の列だけは、第 1 節のモデルです）。残す重みの選び方は、どのモデルも同じです。</p>"""
CONTROL_BETWEEN = """<p class="small">NISQA は、実録音を 4.56 と付けます。物理モデルの合成音がそれより高いのは、CHIMERA の音が雑音のないきれいな音だからで、話者に似ているからではありません。ニューラルデコーダの上限（録音を潜在表現にして戻した音）は、話者の声そのものです。ここまでの NISQA は「音声品質」のモデルの値です。「合成音声の自然さ」のモデルは CHIMERA の声を低く付けるので、その表も下に載せます。</p>"""
CONTROL_SOUNDS = """<p>下のカードで、20 文、200 文、18,655 文で学習したモデルを、出力先ごとに聞き比べられます。最後のカードは、第 2 節の録音のない文です。</p>"""
MANY_TEXT = """<p>同じモデルを、JVS の 100 話者（女性 51 人、男性 49 人）で学習しました。話者ごとにベクトルを 1 つ持たせ、すべての音素とフレームに足します。声の高さは、話者ごとの平均と幅を基準に書きます。ほかは第 1 節のモデルと同じです。</p>
<p>学習データは、全話者が読む 100 文のうち 1〜70 番と、話者ごとに異なる 30 文で、合わせて 9,998 文（約 20 時間。1 話者あたり約 100 文、12 分）です。テストは、エンコーダのテストと同じ 8 話者（女性 4 人、男性 4 人）が読む、どの話者の学習にも出てこない 25 文です。話者の声は学習に出てきますが、文は出てきません。鳴らし方は日本語編で決めた「調整後の声」で、声の高い話者には高い声の設定が入ります。</p>"""
MANY_WHOSE = """<p>声に伝えている話者の情報は、性別と声の高さの中央値だけです。それ以外の話者の違いは、制御列に書かれていなければ失われます。どれだけ残っているかを、話者照合モデルに判定させました。このモデルは日本語もこのコーパスも学習していません。</p>"""
MANY_SAY = """<p>第 2 節の録音のない文を、4 人の声で読ませたものです。</p>"""
TTS_TEXT = """<p>テキストから制御列を書くモデルです。日本語編のエンコーダは録音を聞いて制御列を書きましたが、こちらは文だけから書きます。鳴らすのは同じ CHIMERA（無改造）で、鳴らし方も日本語編で決めた「調整後の声」です。</p>
<p>入力は、Open JTalk が読んだ音素の列です。各音素に、アクセント句の中での位置（アクセント核から何モーラ離れているか、句の頭と末から何モーラ目か）と、疑問文かどうかを添えます。モデルはまず音素ごとの長さを決め、その長さぶんのフレームに、エンコーダが書く 16 チャネルとピッチを 100 Hz で書きます。FastSpeech 2 と同じ形の、自己回帰でないモデルです（約 1,900 万パラメータ）。</p>
<p>学習に使ったのは、Hi-Fi-CAPTAIN の男性話者の録音 18,655 文（約 24 時間）です。手本は、エンコーダ I がその録音から書いた制御列です。誤差は 2 つあります。手本の制御列との差と、書いた制御列が録音と同じに聞こえるかどうかです（声の代役モデルを通した聞き比べ。エンコーダ I の学習と同じ仕組みです）。各音素が録音のどこにあたるかは、録音と読みだけから学習して決めました。</p>
<p>テストは、日本語編と同じ 100 文です（Hi-Fi-CAPTAIN の評価用の文で、学習にも検証にも使っていません）。「TTS（音素の長さは録音）」は、音素の長さだけを録音から借りた合成音です。録音と同じ時間軸になるので、エンコーダと同じ物差しで比べられます。</p>"""
TTS_BETWEEN = """<p class="small">「TTS（テキストだけから）」は録音と時間軸が違うので、リスナー距離などは測れません。破裂のクリック音は、日本語編では録音と突き合わせて数えましたが、ここではどの音も、その音自身の発話区間で同じ数え方をしています（実録音と同じ数え方）。</p>"""
SAY_TEXT = """<p>コーパスにない文を読ませたものです。録音がないので、Whisper の聞き取りだけを添えます（誤り率は第 3 節の表の右 2 列）。早口言葉、外来語、アクセントだけが違う語（箸・橋・端）も入れました。読みとアクセントは Open JTalk の推定をそのまま使っているので、推定が外れた語はその読み、その高さで話します。</p>"""
FEW_TEXT = """<p>同じモデルを、学習に使う録音を減らして作り直しました。録音はコーパス全体から等間隔に選んでいます。モデルの大きさは同じです。</p>
<p>このコーデックの仮説は「潜在表現が外れても、人間の発声としてあり得る音になる」でした。データが足りない TTS は制御列を書き損じるので、書き損じがどう聞こえるかをここで確かめられます。</p>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "transplant")
    ap.add_argument("--abroad", type=Path, default=ROOT / "out" / "ja", help="recordings from outside TSP, judged by pmvc.judge")
    ap.add_argument("--jvs", type=Path, default=ROOT / "out" / "jvs", help="JVS's recordings, laid out by pmvc.jvs and judged")
    ap.add_argument("--jvsall", type=Path, default=ROOT / "out" / "jvsall", help="all of JVS, laid out by pmvc.jvs --all: pmvc.whose's results for all its speakers")
    ap.add_argument("--fit", type=Path, default=ROOT / "out" / "fit", help="where pmvc.calibrate wrote the female voice's settings")
    ap.add_argument("--notes", type=Path, default=None)
    ap.add_argument("--notes-ja", type=Path, default=None)
    ap.add_argument("--sentences", type=int, default=12)
    ap.add_argument("--perturbed", type=int, default=4)
    ap.add_argument("--out", type=str, default="listening.html")
    ap.add_argument("--out-ja", type=str, default="listening_ja.html")
    ap.add_argument("--notes-tts", type=Path, default=None)
    ap.add_argument("--say", type=Path, default=ROOT / "out" / "say", help="sentences nobody recorded, spoken by pmvc.tts and judged")
    ap.add_argument("--out-tts", type=str, default="listening_tts.html")
    ap.add_argument("--en-url", type=str, default="", help="where the first page is published, for the second to point at")
    ap.add_argument("--ja-url", type=str, default="", help="where the second page is published")
    ap.add_argument("--tts-url", type=str, default="", help="where the third page is published, for the second to point at")
    a = ap.parse_args()

    def notes(path: Path | None) -> str:
        return path.read_text() if path else "<p>（所見は未記入）</p>"

    pages = [(a.dir / a.out, build(a.dir, notes(a.notes), a.sentences, a.perturbed, a.ja_url))]
    if (a.abroad / "results.json").exists():
        pages.append((a.dir / a.out_ja, build_ja(a.dir, a.abroad, a.jvs, a.fit, notes(a.notes_ja), a.en_url, a.tts_url, a.jvsall)))
    if (a.abroad / "results_tts.json").exists():
        pages.append((a.dir / a.out_tts, build_tts(a.abroad, a.say, a.jvs, a.jvsall, notes(a.notes_tts), a.en_url, a.ja_url)))
    for out, page in pages:
        out.write_text(page)
        print(f"{out}: {len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
