"""The listening page of the work on the voice itself: what a render lacks of its recording, for the ear to check the judges against.

usage: python -m pmvc.page2 [dir] [--variant encv2] [--notes notes_v2.html] [--out listening_v2.html]
                            [--pages out/transplant]

A fourth page beside `pmvc.page`'s three, in their form (its shell, its buttons). From <dir>
(JVS's test recordings and a variant's renders of them) it makes, for one sentence each of
SPEAKERS, the hybrids of `pmvc.gap` that say where a speaker's identity is (the render with the
recording's envelope, and for two of the speakers the recording with the render's) and the
render through `pmvc.room`'s recording condition, part by part; under each clip, what NISQA's
two models and UTMOSv2 make of it. The tables are read from what `pmvc.gap` and `pmvc.bench`
wrote in <dir>: gap_<variant>.json, gap_low_<variant>.json, bench_v1.json (each left out if it
is not there). JVS's terms allow only some of its audio on a page: the page holds ten clips made
of its recordings. `--notes` is an HTML fragment shown at the top. Writes <pages>/<out>.
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import gap
from .audio import Array, at_level
from .bench import CURVE, LEVEL, SEXES
from .encoder import ids_in
from .page import button, esc, shell
from .paths import ROOT
from .room import recorded

TITLE: str = "CHIMERA 声の改良の試聴"
LEDE: str = "物理モデル音声合成 CHIMERA の声そのものを良くしていく作業（V2）の試聴ページです。まず、いまの合成音が録音に比べて何を欠いているかを、録音と合成音の成分を入れ替えた音で確かめます。ボタンを押すと再生し、下に予測モデルの点数が出ます。"
CREDITS: str = "録音は JVS（東京大学 猿渡研究室。一部の音声だけを載せています。録音から作った音は 10 クリップです）。合成は CHIMERA（MIT License）。再生用に MP3 96 kbps へ変換し、音量をそろえています。"
# the speakers on the cards, and those whose recording is also heard with the render's envelope
SPEAKERS: dict[str, str] = {"jvs093": "女性 jvs093", "jvs095": "女性 jvs095", "jvs097": "男性 jvs097", "jvs100": "男性 jvs100"}
BOTH_WAYS: tuple[str, ...] = ("jvs093", "jvs100")
# a card's sentence is the speaker's test sentence nearest this long (seconds of speech)
LENGTH: float = 5.0
LABELS: dict[str, str] = {
    "R": "実録音",
    "S": "合成音（V1）",
    "S+envR": "合成音 + 録音の包絡",
    "R+envS": "録音 + 合成音の包絡",
    "S+floor": "合成音 + ノイズフロア",
    "S+tail": "合成音 + 短い残響",
    "S+room": "合成音 + 両方（採点の録音条件）",
}

SWAP_TEXT = """<p>録音とその合成音は、同じ時間軸・同じ高さで鳴っています。そこで両方を、フレームごとに「スペクトル包絡」（声道の形と声源の傾きが作る、なめらかな山と谷）と「それ以外」（倍音と雑音の比、パルスの位相、ゆらぎ）に分け、互い違いに組み直しました。ボコーダは使っていません。片方の短時間スペクトルに、包絡の比を掛けているだけです。</p>
<p>聞いていただきたいのは次の点です。</p>
<ul>
<li>「合成音 + 録音の包絡」は、「合成音（V1）」より本人に近く聞こえますか。照合モデルは、これで 100 話者の中から本人を選べるようになります（下の表）。エンジンの声道が録音の包絡を完全に出せた場合の、上限の音にあたります。</li>
<li>「録音 + 合成音の包絡」は、声源が本物でも、誰の声か分からなくなりますか。</li>
<li>予測モデルの点数（NISQA の自然さと品質、UTMOSv2）のうち、聞こえ方に合うのはどれですか。</li>
</ul>"""
ROOM_TEXT = """<p>合成音は、壁も雑音もない場所で口元の音をそのまま聞いたものです。NISQA の自然さのモデルは、これに声と無関係なノイズフロアや短い残響を足すだけで点を上げます（下の表）。各カードの 2 段めで、足す前と足した後を聞き比べられます。自然に聞こえるようになりますか。それとも、汚れただけに聞こえますか。</p>"""


def chosen(d: Path, speaker: str, k: int) -> dict[str, Any]:
    """Of the speaker's test recordings, the one whose speech is `k`-th nearest LENGTH (so that
    the cards do not all say one sentence)."""
    metas = [json.loads((d / "meta" / f"{i}.json").read_text()) for i in ids_in(d, "test")]
    return sorted((m for m in metas if m["speaker"] == speaker), key=lambda m: abs(m["span"][1] - m["span"][0] - LENGTH))[k]


def versions(d: Path, variant: str, m: dict[str, Any], pitch: float) -> dict[str, Array]:
    """A card's clips, at one speech level: the recording, the render, the two hybrids, and the render through the parts of the recording condition."""
    x, y, _ = gap.pair(d, variant, m["id"])
    made = gap.hybrids(x, y, pitch, bands={})
    dry = at_level(y, gap.SR, LEVEL)
    out = {k: at_level(made[k], gap.SR, LEVEL) for k in ("R", "S", "S+envR", "R+envS")}
    out["S+floor"] = recorded(dry, gap.SR, seconds=None, highpass=None)
    out["S+tail"] = recorded(dry, gap.SR, floor_db=None, highpass=None)
    out["S+room"] = recorded(dry, gap.SR)
    return out


def cards(d: Path, variant: str, clips: dict[str, str]) -> str:
    from .natural import score
    from .opinions import utmos

    metas = [json.loads((d / "meta" / f"{i}.json").read_text()) for i in ids_in(d, "test")]
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        made: list[tuple[dict[str, Any], str, Path]] = []
        for k, speaker in enumerate(SPEAKERS):
            m = chosen(d, speaker, k)
            pitch = float(np.median([x["f0Median"] for x in metas if x["speaker"] == speaker]))
            for k, x in versions(d, variant, m, pitch).items():
                if k == "R+envS" and speaker not in BOTH_WAYS:
                    continue
                p = Path(tmp) / f"{m['id']}_{len(made)}.wav"
                sf.write(str(p), x * min(1.0, 0.98 / max(float(np.abs(x).max()), 1e-9)), gap.SR, subtype="PCM_16")
                made.append((m, k, p))
        nisqa, opinion = score([p for _, _, p in made]), utmos([p for _, _, p in made])
        said = {(m["id"], k): f"NISQA 自然さ {n['natural']:.2f}、品質 {n['quality']:.2f}、UTMOSv2 {u:.2f}" for (m, k, _), n, u in zip(made, nisqa, opinion)}
        for speaker, label in SPEAKERS.items():
            mine = [(m, k, p) for m, k, p in made if m["speaker"] == speaker]
            m = mine[0][0]
            rows = [
                "".join(button(clips, f"{m['id']}/{k}", p, LABELS[k], said[m["id"], k], "ref" if k == "R" else "", "点数") for _, k, p in mine if (k in ("R", "S", "S+envR", "R+envS")) == first)
                for first in (True, False)
            ]
            out.append(
                f"""<article class="sentence">
<p class="meta">{esc(m["id"])} · {esc(label)}</p>
<p class="text">{esc(m["text"])}</p>
<div class="kind"><span class="kind-label">成分の入れ替え</span><div class="clips">{rows[0]}</div></div>
<div class="kind"><span class="kind-label">録音条件（合成音に足すもの）</span><div class="clips">{rows[1]}</div></div>
<p class="heard" aria-live="polite">クリップを選ぶと予測モデルの点数を表示します。</p>
</article>"""
            )
    return "".join(out)


def pair(x: dict[str, float], fmt: str = "{:.2f}") -> str:
    return " / ".join(fmt.format(x[s]) for s in SEXES)


def swap_table(d: Path, variant: str) -> str:
    p = d / f"gap_{variant}.json"
    if not p.exists():
        return ""
    g = json.loads(p.read_text())
    h = g["hybrids"]
    rows = (
        ("R", "実録音", ""), ("S", "合成音（V1）", ""), ("S+envR", "合成音 + 録音の包絡", "毎フレーム"), ("R+envS", "録音 + 合成音の包絡", "毎フレーム"),
        ("S+envR:static", "合成音 + 固定の補正カーブ", "発話ごとに 1 本の EQ"), ("S+envR:coarse", "合成音 + 包絡の大まかな形だけ", "傾きとレベル、毎フレーム"),
        ("R|S:2.5-5k", "録音の 2.5〜5 kHz だけを合成音に", ""), ("R|S:0-1k", "録音の 0〜1 kHz だけを合成音に", ""), ("S|R:2.5-5k", "合成音の 2.5〜5 kHz だけを録音に", ""),
    )  # fmt: skip
    body = "".join(
        f'<tr><th>{esc(label)}{f'<span class="sub">{esc(sub)}</span>' if sub else ""}</th><td class="num">{100 * h[k]["wavlm"]["top1"]:.1f}</td><td class="num">{100 * h[k]["unispeech"]["top1"]:.1f}</td>'
        f'<td class="num">{h[k]["wavlm"]["own"]:.3f}</td><td class="num">{pair(h[k]["natural"])}</td><td class="num">{pair(h[k]["quality"])}</td></tr>'
        for k, label, sub in rows if k in h
    )  # fmt: skip
    return f"""<div class="scroll"><table>
<caption>JVS のテスト {g["clips"]} 文（8 話者）。「本人」は、登録した {g.get("enrolled", 100)} 話者（各話者の学習用の録音）の中で最も近い話者が本人だった割合 %。照合モデルは WavLM と UniSpeech-SAT。類似度は WavLM の本人とのコサイン類似度。NISQA は女性 / 男性。すべて 24 kHz、録音のレベルに揃えた音。</caption>
<thead><tr><th>音</th><th>本人 WavLM</th><th>本人 UniSpeech</th><th>類似度</th><th>NISQA 自然さ</th><th>NISQA 品質</th></tr></thead>
<tbody>{body}</tbody></table></div>"""


def low_table(d: Path, variant: str) -> str:
    p = d / f"gap_low_{variant}.json"
    if not p.exists():
        return ""
    g = json.loads(p.read_text())
    h = g["hybrids"]
    rows = (
        ("R", "実録音"), ("S", "合成音（V1）"), ("S|R:0-250", "合成音の 0〜250 Hz だけを録音に"), ("S|R:250-500", "合成音の 250〜500 Hz だけを録音に"),
        ("S|R:500-1000", "合成音の 500 Hz〜1 kHz だけを録音に"), ("S|R:0-250@unvoiced", "合成音の 0〜250 Hz を録音に、無声の区間だけ"),
        ("S|R:0-250@voiced", "同、有声の区間だけ"), ("S|R:0-250@silence", "同、無音の区間だけ"), ("R|S:0-250", "録音の 0〜250 Hz だけを合成音に"),
    )  # fmt: skip
    body = "".join(f'<tr><th>{esc(label)}</th><td class="num">{pair(h[k]["natural"])}</td><td class="num">{pair(h[k]["quality"])}</td></tr>' for k, label in rows if k in h)
    return f"""<div class="scroll"><table>
<caption>NISQA の自然さは、どの帯域のどの区間を聞いているか。JVS のテスト {g["clips"]} 文、女性 / 男性。区間は、録音のピッチ抽出が有声としたか、音素の対応付けがポーズとしたかで分けています。</caption>
<thead><tr><th>音</th><th>NISQA 自然さ</th><th>NISQA 品質</th></tr></thead>
<tbody>{body}</tbody></table></div>"""


def bench_tables(d: Path, labels: dict[str, str]) -> str:
    p = d / "bench_v1.json"
    if not p.exists():
        return ""
    b = json.loads(p.read_text())
    v = b["variants"]
    head = "<th>本人 WavLM</th><th>本人 UniSpeech</th><th>NISQA 自然さ</th><th>同、録音条件あり</th><th>NISQA 品質</th><th>同、録音条件あり</th><th>SQUIM</th><th>UTMOSv2</th>"
    body = ""
    for k, label in labels.items():
        if k not in v:
            continue
        c = v[k]
        who = "".join(f'<td class="num">{pair({s: 100 * c[m]["sexes"][s]["top1"] for s in SEXES}, "{:.0f}")}</td>' for m in gap.VERIFIERS) if "wavlm" in c else '<td class="num">–</td>' * 2
        mos = "".join(f'<td class="num">{pair(c[m][cond]) if cond in c[m] else "–"}</td>' for m in ("natural", "quality") for cond in ("dry", "room"))
        more = "".join(f'<td class="num">{pair(c[m]) if m in c else "–"}</td>' for m in ("squim", "utmos"))
        body += f"<tr><th>{esc(label)}</th>{who}{mos}{more}</tr>"
    out = f"""<div class="scroll"><table>
<caption>新しい採点表（<code>pmvc.bench</code>）での V1 の基準値。JVS のテスト {b["clips"]} 文、すべて女性 / 男性。録音も合成音も {b["rate"] // 1000} kHz、同じ発話レベル（{b["level"]:.0f} dB）に揃えてから採点しています。「録音条件あり」は、合成音に決まったハイパス・短い残響・ノイズフロアを通したもの（<code>pmvc.room</code>）。</caption>
<thead><tr><th>音</th>{head}</tr></thead><tbody>{body}</tbody></table></div>"""
    curve = "".join(
        f'<tr><th>{esc(labels[k])}・{"女性" if s == "F" else "男性"}</th><td class="num">{x["total"]:.1f}</td><td class="num">{x["common"]:.1f}</td><td class="num">{x["speaker"]:.1f}</td><td class="num">{x["vowel"]:.1f}</td><td class="num">{x["rest"]:.1f}</td>'
        + "".join(f'<td class="num">{c:+.0f}</td>' for c in x["curve"]) + "</tr>"
        for k in labels if k in v and "shape" in v[k] for s, x in v[k]["shape"].items()
    )  # fmt: skip
    if curve:
        bands = "".join(f"<th>{lo / 1000:g}〜{hi / 1000:g}k</th>" for lo, hi in CURVE)
        out += f"""<div class="scroll"><table>
<caption>母音のスペクトル包絡のずれ（録音 − 合成音、dB。フレームごとのレベルは除く）。左は 8 kHz 以下の rms と、その内訳（性別に共通、話者ごと、母音ごと、残り）。右は、性別に共通のカーブを帯域（Hz）ごとに平均したもの。正の値は合成音の不足です。</caption>
<thead><tr><th>音</th><th>全体</th><th>共通</th><th>話者</th><th>母音</th><th>残り</th>{bands}</tr></thead><tbody>{curve}</tbody></table></div>"""
    return out


def build(d: Path, variant: str, notes: str) -> str:
    clips: dict[str, str] = {}
    body = f"""<section aria-labelledby="h-swap"><h2 id="h-swap">1. 合成音は録音の何を欠いているか</h2>
{SWAP_TEXT}
{swap_table(d, variant)}
{ROOM_TEXT}
{low_table(d, variant)}
{cards(d, variant, clips)}
</section>
<section aria-labelledby="h-bench"><h2 id="h-bench">2. V1 の基準値（新しい採点表）</h2>
<p>これまでは合成音 48 kHz・録音 24 kHz を、それぞれのレベルのまま比べていました。帯域とレベルを揃えて取り直した値です。以後の変更は、この表と比べます。</p>
{bench_tables(d, {"real": "実録音", variant: "合成音（V1: 公開用エンコーダと 2 つの基準の声）", "encit": "話者ごとに 11 個の設定を当てはめた合成音"})}
</section>"""
    return shell(TITLE, LEDE, notes, body, CREDITS, clips)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "jvs")
    ap.add_argument("--variant", type=str, default="encv2")
    ap.add_argument("--notes", type=Path, default=None)
    ap.add_argument("--pages", type=Path, default=ROOT / "out" / "transplant")
    ap.add_argument("--out", type=str, default="listening_v2.html")
    a = ap.parse_args()
    page = build(a.dir, a.variant, a.notes.read_text() if a.notes else "<p>（所見は未記入）</p>")
    (a.pages / a.out).write_text(page)
    print(f"{a.pages / a.out}: {len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
