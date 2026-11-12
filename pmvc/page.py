"""A listening page for the transplant and perturbation renders: one self-contained HTML file
with the audio embedded, the scores in tables and what Whisper heard under each clip.

usage: python -m pmvc.page [dir] [--abroad out/ja] [--notes notes.html] [--sentences 12] [--perturbed 4]
                           [--out listening.html]

Reads <dir>/results.json and <dir>/perturb_results.json (`pmvc.evaluate`), <dir>/perturb.json
(`pmvc.perturb`) and the WAVs; the later sections appear once their results are there, the last
from `--abroad` (`pmvc.judge`'s results for recordings from outside TSP). `--notes` is an HTML fragment shown at the top (the findings,
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
        sets.append(("英語・TSP のテスト話者", "学習と同じコーパス。第 4 節の 4 人", json.loads(home.read_text())["summary"]["en"]))
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
    return f"""<section aria-labelledby="h-j"><h2 id="h-j">5. 日本語の録音を通す</h2>
{{{{abroad_text}}}}
{by_encoder}
{{{{abroad_measures}}}}
{by_set}
{by_sound}
{by_search}
<div class="stack">{"".join(cards)}</div>
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


def build(d: Path, ja: Path, notes: str, n_sentences: int, n_perturbed: int) -> str:
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
    return (
        PAGE.replace("{{notes}}", notes)
        .replace("{{transplant_table}}", transplant_table(results["summary"]))
        .replace("{{perturb_table}}", perturb_table(presults["summary"], perturb["conditions"]))
        .replace("{{sentences}}", "\n".join(sentences))
        .replace("{{perturbed}}", "\n".join(perturbed))
        .replace("{{copy_section}}", copy_section(d, metas, clips, n_sentences))
        .replace("{{codec_section}}", codec_section(d, metas, clips, 8).replace("{{codec_text}}", CODEC_TEXT))
        .replace("{{abroad_section}}", abroad_section(d, ja, clips).replace("{{abroad_text}}", ABROAD_TEXT).replace("{{abroad_measures}}", ABROAD_MEASURES))
        .replace("{{channels}}", esc(channels))
        .replace("{{clips}}", json.dumps(clips))
    )


CODEC_TEXT = """<p>録音を聞いて制御列を書くエンコーダです。デコーダは CHIMERA そのものなので、これで「音声 → 制御列（100 Hz × 21 チャネル）→ 音声」のコーデックになります。文の書き起こしは、学習にも推論にも探索にも使いません。</p>
<p>エンコーダは、音声認識モデル wav2vec2 の特徴とピッチ抽出器の出力を入力にする再帰型ネットワークです。F0 だけは学習せず、ピッチ抽出器の曲線をそのまま使います。声の体格（女声・男声のプリセット）は話者の性別で選んでいます。</p>
<ul>
<li><b>A</b>: CHIMERA 自身が合成した音とその制御列の対だけで学習します。</li>
<li><b>A + 探索</b>: A が実録音に対して書いた制御列を、合成音が録音に近づくよう局所探索で直します。</li>
<li><b>B</b>: 探索で直した制御列を教師にして学習し直します。</li>
<li><b>C</b>: さらに「声の代役」を通した誤差を加えます。CHIMERA は微分できないので、制御列から「リスナー（wav2vec2）にどう聞こえるか」を予測するネットワークを合成音で学習し、それを通して「自分の書いた制御列の音が録音と同じに聞こえるか」をエンコーダに直接教えます。</li>
<li><b>C′</b>: C から探索の結果を除いたものです。代役は規則の制御列と A の制御列の合成音だけで学習し、エンコーダの教師には A 自身の出力を使います。</li>
</ul>"""


ABROAD_TEXT = """<p>第 4 節のエンコーダを、そのまま日本語の録音に使います。エンコーダも、その入力に使う wav2vec2 も、学習に使ったのは英語（TSP）だけです。録音は Hi-Fi-CAPTAIN の日本語話者 2 人（女性と男性、同じ 100 文）と、JSUT の女性話者（別の 100 文）です。</p>
<p>比較のため、Hi-Fi-CAPTAIN の英語話者 2 人（各 100 文）にも同じことをしています。収録条件が日本語と同じなので、コーパスの違いと言語の違いを分けて見られます。</p>"""
ABROAD_MEASURES = """<p class="small">音声記号の不一致は、テキストも言語も仮定しない物差しです。多言語の音声記号認識器（wav2vec2 XLSR-53 を国際音声記号の書き取りに調整したもの）に録音と合成音を書き取らせ、録音側の記号のうち合成音側で違っていた割合を数えます。この認識器は日本語では書き取りが安定しないので、同じ文を読んだ別の実話者の録音を同じように比べた値を基準に置きます。リスナー距離は第 4 節と同じです。Whisper と音声記号認識器は、学習にも探索にも使っていません。</p>"""


PAGE = """<title>CHIMERA 韻律移植と摂動の試聴</title>
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
</style>
<main>
<header class="stack">
<h1>CHIMERA 韻律移植と摂動の試聴</h1>
<p class="lede">物理モデル音声合成 CHIMERA を、実際の話者の録音に合わせて動かした結果です。合成エンジンは無改造で、変えたのは制御だけです。ボタンを押すと再生し、下に Whisper の聞き取りが出ます。</p>
</header>
<section aria-labelledby="h-notes"><h2 id="h-notes">わかったこと</h2><div class="notes">{{notes}}</div></section>
<section aria-labelledby="h-t"><h2 id="h-t">1. 韻律移植</h2>
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
{{abroad_section}}
<footer class="small stack">
<p>録音は TSP speech database（Copyright © 2018 Peter Kabal、Simplified BSD License）。文は Harvard sentences。第 5 節の録音は Hi-Fi-CAPTAIN（NICT、CC BY-NC-SA 4.0）と JSUT（東京大学 猿渡研究室）。合成は CHIMERA（MIT License）。再生用に MP3 96 kbps へ変換し、音量をそろえています。</p>
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "transplant")
    ap.add_argument("--abroad", type=Path, default=ROOT / "out" / "ja", help="recordings from outside TSP, judged by pmvc.judge")
    ap.add_argument("--notes", type=Path, default=None)
    ap.add_argument("--sentences", type=int, default=12)
    ap.add_argument("--perturbed", type=int, default=4)
    ap.add_argument("--out", type=str, default="listening.html")
    a = ap.parse_args()
    notes = a.notes.read_text() if a.notes else "<p>（所見は未記入）</p>"
    page = build(a.dir, a.abroad, notes, a.sentences, a.perturbed)
    out = a.dir / a.out
    out.write_text(page)
    print(f"{out}: {len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
