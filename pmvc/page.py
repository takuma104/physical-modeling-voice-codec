"""A listening page for the transplant and perturbation renders: one self-contained HTML file
with the audio embedded, the scores in tables and what Whisper heard under each clip.

usage: python -m pmvc.page [dir] [--abroad out/ja] [--notes notes.html] [--notes-ja notes_ja.html]
                           [--sentences 12] [--perturbed 4] [--en-url URL] [--ja-url URL]

Two pages, since one would be too large to publish: <dir>/listening.html, the experiments on
TSP's recordings, and <dir>/listening_ja.html, the ones on recordings from outside it (the
Japanese ones above all). The first reads <dir>/results.json and <dir>/perturb_results.json
(`pmvc.evaluate`), <dir>/perturb.json (`pmvc.perturb`) and the WAVs, and its later sections
appear once their results are there; the second reads `pmvc.judge`'s results in `--abroad`.
Given the pages' published addresses, each points at the other. `--notes` is an HTML fragment shown at the top (the findings,
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
    "enchs": "H（動きの音なし）",
    "enchq": "H（無音区間の対策あり）",
    "enchv": "H + 調整 1",
    "enchvq": "H + 調整 1（対策あり）",
    "ench2q": "H + 調整 2（対策あり）",
    "ench3q": "H + 調整 3（対策あり）",
}
TUNED_NOTES: dict[str, str] = {
    "real": "話者本人",
    "encn": "英語の録音だけで学習",
    "encg": "日本語の録音、多言語の物差しと入口（第 2 節）",
    "ench": "G と同じ。ただし発話の外では「息なし」を手本にする",
    "enchs": "H の制御列を、声道の壁が押しのける空気を音にしない設定で鳴らす",
    "enchq": "動きの音なしに加えて、録音のポーズ（0.12 秒以上の無音）では息を止める",
    "enchv": "声の高い話者だけ声の設定を変える。NISQA だけに合わせた設定（前回の報告の「前回」）。対策なし",
    "enchvq": "調整 1 に、無音区間の対策（動きの音なし、ポーズで息を止める）を加える",
    "ench2q": "調整 2 は、調整 1 の声門のまま、振幅を 1.4 倍にし、雑音源を既定値に戻したもの",
    "ench3q": "調整 3 は、調整 2 の狭めと歯の雑音源を 0.6 倍にしたもの",
}
# the settings `pmvc.calibrate` fits, as the page names them
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
        formats = {"quality": ".2f", "oddClicks": ".2f", "releases": ".2f", "releaseLevel": ".1f", "silenceSounds": ".2f", "silenceLevel": ".1f"}
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


def sentence_cards(clips: dict[str, str], ja: Path, jvs: Path, results: dict[str, Any], variants: list[str], labels: dict[str, str], voiced: tuple[str, ...] = ()) -> str:
    """The cards of the three tests: one sentence each of four JVS speakers, then the sentences
    of the first section's cards, then English. `voiced`: variants rendered with a fitted
    voice, left out where the speaker's pitch leaves the voice as it was."""

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
            for v in variants if v in r["variants"] and not (v in voiced and unchanged(where, v, r["id"]))
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
<div class="stack">{sentence_cards(clips, ja, jvs, results, variants, TAUGHT_LABELS)}</div>
</section>"""


def tuned_section(d: Path, ja: Path, jvs: Path, fit: Path, clips: dict[str, str]) -> str:
    """The sounds before the speech taken out (encoder H), and the female voice refitted
    (`pmvc.calibrate --by nisqa`'s settings in `fit`). Nothing before `pmvc.judge`'s and
    `pmvc.natural`'s results for them on JVS's held-out speakers exist."""
    if not (jvs / "results_tuned.json").exists():
        return ""
    results, sets = three_tests(d, ja, jvs, "results_tuned", "results_tuned")
    variants = [v for v in TUNED_LABELS if all(v in g for _, g in sets)]
    stray = grid(sets, [v for v in variants if v in ("encn", "encg", "ench", "enchq")], TUNED_LABELS, TUNED_NOTES,
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
<caption>調整した設定（調整 2、無音区間の対策あり）をどの程度使うと NISQA の点数がどうなるか（話者ごとに {mix["n"]} 文、エンコーダ H の制御列）。0 は CHIMERA の既定、1 は調整後、その間は設定を混ぜたもの。話者は声の高さの順です。</caption>
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
{{{{tuned_clicks}}}}
{clicks}
{{{{tuned_check}}}}
{tables}
<div class="stack">{sentence_cards(clips, ja, jvs, results, [v for v in clicking if v != "enchs"], TUNED_LABELS, ("enchv", "enchvq", "ench2q", "ench3q"))}</div>
</section>"""


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


def shell(title: str, lede: str, notes: str, body: str, credits: str, clips: dict[str, str]) -> str:
    return (
        PAGE.replace("{{title}}", esc(title)).replace("{{lede}}", esc(lede)).replace("{{notes}}", notes).replace("{{body}}", body)
        .replace("{{credits}}", esc(credits)).replace("{{clips}}", json.dumps(clips))
    )  # fmt: skip


def pointer(number: int, heading: str, text: str, title: str, url: str) -> str:
    """A section that only points at the other page, or nothing without its address."""
    if not url:
        return ""
    return f'''<section aria-labelledby="h-more"><h2 id="h-more">{number}. {esc(heading)}</h2>
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


def build_ja(d: Path, ja: Path, jvs: Path, fit: Path, notes: str, en_url: str = "") -> str:
    """The second page: recordings from outside TSP, the Japanese ones above all."""
    clips: dict[str, str] = {}
    body = abroad_section(d, ja, clips).replace("{{abroad_text}}", ABROAD_TEXT).replace("{{abroad_measures}}", ABROAD_MEASURES)
    body += taught_section(d, ja, jvs, clips).replace("{{taught_text}}", TAUGHT_TEXT).replace("{{taught_between}}", TAUGHT_BETWEEN)
    body += tuned_section(d, ja, jvs, fit, clips).replace("{{tuned_text}}", TUNED_TEXT).replace("{{tuned_voice}}", TUNED_VOICE).replace("{{tuned_rule}}", TUNED_RULE).replace("{{tuned_walls}}", TUNED_WALLS).replace("{{tuned_clicks}}", TUNED_CLICKS).replace("{{tuned_check}}", TUNED_CHECK)
    body += pointer(4, "英語の録音での実験", "エンコーダ A・C・C′ の作り方と、英語（TSP）での結果は別のページにあります: ", TITLE_EN, en_url)
    return shell(TITLE_JA, LEDE_JA, notes, body, CREDITS_JA, clips)


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
TAUGHT_BETWEEN = """<p class="small">表の層は、モデルごとに日本語の合成音と別の話者を最もよく分けた層です（英語の wav2vec2 だけは、これまで使ってきた第 9 層）。物差しには mHuBERT の第 8 層を選びました。以下の Whisper と音声記号認識器は、どのエンコーダの学習にも使っていません。</p>"""

TUNED_TEXT = """<p>第 2 節のエンコーダ G には、発話の前に余計な音が出る癖がありました。原因は学習の手本です。JVS の録音は発話の前に 0.5 秒ほど無音がありますが、学習で録音と聞き比べているのは発話の区間だけで、その外では「合成音だけで学習したエンコーダ A の出力」を手本にしていました。A は録音の部屋ノイズが始まった時点で息を出し始めるので、それを真似ていました。</p>
<p>エンコーダ H は、G と同じ作り方で、発話の外では「息なし」を手本にします。あわせて、エンコーダ自身が足している先頭 0.2 秒の無音では、制御列を落ち着いた値から始めます。</p>"""
TUNED_VOICE = """<p>女声の声色には、NISQA（音声だけから主観品質の評点を予測するモデル）を物差しに使いました。男性の合成音は実録音に近い点数になるのに、女性の合成音は大きく下がります。調べると、点数は性別ではなく声の高さに従っていました。話者ごとの声の高さ（中央値）と点数の相関は −0.9 で、声の高い男性も下がります。原因はエンコーダではなく、CHIMERA のエンジンが高い声を鳴らすときの声門の動かし方にありました。エンジンの声門は、1 周期のうち開いている割合（開放率）が声の高さによらず約 0.6 で、設定で動かせるのも 0.78 までです。実際の声は、高いほど、また女声のほうが、開放率が大きいことが知られています。</p>
<p>そこで、Hi-Fi-CAPTAIN の女性話者の録音 16 文（テストとは別の文）を手本に、声門と雑音源の設定を調整しました。NISQA の点数が上がり、かつ多言語の物差しで録音から離れないように合わせています。エンジンのコードは変えておらず、声門のパラメータを外から直接指定しています。</p>"""
TUNED_RULE = """<p>上の 2 つの表から、調整した設定は声の高さが 175 Hz 以上の話者に使い、150 Hz 以下の話者は既定のまま、その間は混ぜることにしました。「H + 調整 1」などは、エンコーダ H の制御列をこの規則で鳴らしたものです。声の高さは、その話者の録音全体の中央値で決めています。調整には 3 つの版があります（下で説明します。前回の報告では「前回」「中間」「今回」と呼んでいました）。</p>"""
TUNED_WALLS = """<p>調整 1 で聴くと、舌を鳴らすような湿った音が混じります。鳴っているのは、録音が無音の区間でした。破裂音で舌や唇が閉じている間と、文の途中のポーズです。実録音はここで発話より 55 dB ほど小さくなります。合成音は既定の声の H でも 10 dB ほど大きく、短い音が何度も突き出ます。原因は 2 つありました。</p>
<p>1 つ目は、声道の壁が動く音です。CHIMERA は、舌・唇・あごが動いて押しのけた空気を、声道の中の流れとして計算します。そのため、息を完全に止めた制御列でも無音にはならず、300 Hz 以下を中心にした低い音が出ます。大きさは発話より 40 dB ほど小さく（男性の体では 30 dB ほど）、舌が速く動く瞬間は 20 dB 差まで上がります。男性の声はこの帯域に声の成分があるので隠れますが、女性の声にはなく、無音区間では何にも隠れません。CHIMERA 自身の規則で作った制御列でも同じ大きさで出るので、エンコーダの癖ではなくエンジンの性質です。エンジンにはこの流れを切るスイッチがあり（コードは変えていません）、「動きの音なし」はそれを切って鳴らしたものです。</p>
<p>2 つ目は、ポーズの中の息です。エンコーダ H は発話の前後では息を止めますが、文の途中のポーズでは弱い息を出し続けていて、かすかな声と、息が戻る瞬間の音になっていました。「無音区間の対策あり」は、動きの音を切ったうえで、録音が 0.12 秒以上無音になる区間で息を止めます（区間が終わる 30 ms 前に戻します）。制御列のほかの部分は H と同じです。男性の録音にも同じ対策を使っています。</p>"""
TUNED_CLICKS = """<p>無音区間の音とは別に、高い帯域（2〜9 kHz）の短いクリック音があります。実録音のクリック音は、破裂音で舌や唇が離れる瞬間のものがほとんどで、既定の声の H も同じです。調整 1 では、舌が閉じる瞬間や、音が続いている途中にもクリック音が出ます。NISQA だけに合わせると歯の雑音源のゲインが既定の 2.2 倍になること、声門を長く開く設定が息の流量を増やすことが原因で、無音区間の対策では減りません。</p>
<p>調整 2 と 3 は、声門の開き方（開放率・張り・明るさ）を調整 1 のままにして、クリック音に効く設定を変えたものです。どちらも、声門の振動の振幅を 1.4 倍にして、声を破裂に対して大きくしています。雑音源のゲインは、調整 2 が既定値、調整 3 が狭めと歯を既定の 0.6 倍です。ゲインを下げるほどクリック音は減り、NISQA の点数も下がります。数値を当てはめに任せると別の抜け道に行ってしまったので、指標を見ながら手で決めました。前回の報告の「中間」「今回」では狭めが閉じきる速さも半分にしていましたが、動きの音を切ると不要になったので、既定に戻しています。</p>"""
TUNED_CHECK = """<p class="small">声の設定や無音区間の対策を変えても言葉が保たれているかを、第 2 節と同じ物差しで確かめたのが以下の表です。Whisper と音声記号認識器は、学習にも調整にも使っていません。NISQA は女声の設定の調整にだけ使いました（表の点数は、調整に使っていない録音のものです）。動きの音を切ると、音声記号の不一致が最大 2 ポイントほど増えます。増えるのは、破裂音の有声・無声の取り違え（t を d と聞く、など）です。エンコーダは動きの音がある声で学習したままです。</p>"""

TITLE_JA = "CHIMERA 日本語エンコーダの試聴"
LEDE_JA = "物理モデル音声合成 CHIMERA をデコーダにしたコーデックを、日本語の録音で試した結果です。合成エンジンは無改造で、変えたのはエンコーダの学習だけです。ボタンを押すと再生し、下に Whisper の聞き取りが出ます。"
CREDITS_JA = "録音は Hi-Fi-CAPTAIN（NICT、CC BY-NC-SA 4.0）、JSUT と JVS（東京大学 猿渡研究室。JVS は一部の音声だけを載せています）、TSP speech database（Copyright © 2018 Peter Kabal、Simplified BSD License）。合成は CHIMERA（MIT License）。再生用に MP3 96 kbps へ変換し、音量をそろえています。"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "transplant")
    ap.add_argument("--abroad", type=Path, default=ROOT / "out" / "ja", help="recordings from outside TSP, judged by pmvc.judge")
    ap.add_argument("--jvs", type=Path, default=ROOT / "out" / "jvs", help="JVS's recordings, laid out by pmvc.jvs and judged")
    ap.add_argument("--fit", type=Path, default=ROOT / "out" / "fit", help="where pmvc.calibrate wrote the female voice's settings")
    ap.add_argument("--notes", type=Path, default=None)
    ap.add_argument("--notes-ja", type=Path, default=None)
    ap.add_argument("--sentences", type=int, default=12)
    ap.add_argument("--perturbed", type=int, default=4)
    ap.add_argument("--out", type=str, default="listening.html")
    ap.add_argument("--out-ja", type=str, default="listening_ja.html")
    ap.add_argument("--en-url", type=str, default="", help="where the first page is published, for the second to point at")
    ap.add_argument("--ja-url", type=str, default="", help="where the second page is published")
    a = ap.parse_args()

    def notes(path: Path | None) -> str:
        return path.read_text() if path else "<p>（所見は未記入）</p>"

    pages = [(a.dir / a.out, build(a.dir, notes(a.notes), a.sentences, a.perturbed, a.ja_url))]
    if (a.abroad / "results.json").exists():
        pages.append((a.dir / a.out_ja, build_ja(a.dir, a.abroad, a.jvs, a.fit, notes(a.notes_ja), a.en_url)))
    for out, page in pages:
        out.write_text(page)
        print(f"{out}: {len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
