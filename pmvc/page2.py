"""The listening page of the work on the voice itself: what a render lacks of its recording, for the ear to check the judges against.

usage: python -m pmvc.page2 [dir] [--variant encv2] [--ladder DIR] [--forms FILE] [--reached KEY=VARIANT,...]
                            [--notes notes_v2.html] [--out listening_v2.html] [--pages out/transplant]

A fourth page beside `pmvc.page`'s three, in their form (its shell, its buttons). From <dir>
(JVS's test recordings and a variant's renders of them) it makes, for one sentence each of
SPEAKERS, a card: the recording, the render, the render with the recording's envelope in every
frame (`pmvc.gap`), and between the last two the steps of `pmvc.ladder` (its clips are read from
`--ladder`, the directory it wrote them to): the envelope mended by averages only, and frame by
frame but slowly (of the steps, HEARD_STEPS are on the cards); with `--forms` (a file of
`pmvc.form`'s SPECs) the track rendered again with the tract in the first of those forms; and
with `--reached` the renders that `pmvc.reach` found by searching the tracks, and those of
V1's tracks in the speakers' own tract shapes (`pmvc.form --write ... --shapes`). Under each
clip, how far its envelope is from the recording's in the vowels.
The tables are read from what `pmvc.ladder`, `pmvc.gap` and `pmvc.bench` wrote in <dir>:
ladder.json, gap_<variant>.json, gap_low_<variant>.json, bench_v1.json, and for the test of
how near the voice comes to a recording's envelope when asked, bench_static.json,
bench_reach.json and the searches' reach_*.json, and `pmvc.tract`'s tract_<variant>.json for
the chart of the engine's resonances (each left out if it is not there). JVS's terms allow only
some of its audio on a page: the page holds ten clips that carry a recording frame by frame
(the recordings, the renders with their envelope, and for two speakers the slow step); the
other steps are a render under gain curves that are averages over other clips. `--notes` is an
HTML fragment shown at the top. Writes <pages>/<out>.
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import gap
from .audio import Array, at_level, load
from .bench import CURVE, LEVEL, SEXES
from .encoder import LEAD, ids_in, render_track
from .form import Clip, formed, heard_error, settings
from .page import button, esc, shell
from .paths import ROOT

TITLE: str = "CHIMERA 声の改良の試聴"
LEDE: str = "物理モデル音声合成 CHIMERA の声そのものを良くしていく作業（V2）の試聴ページです。合成音のスペクトル包絡を録音にどこまで合わせれば録音に近く聞こえるかを、段階に分けた音で確かめます。ボタンを押すと再生し、下にその音の包絡のずれが出ます。"
CREDITS: str = "録音は JVS（東京大学 猿渡研究室。一部の音声だけを載せています。録音をフレームごとに使った音は 10 クリップです）。合成は CHIMERA（MIT License）。再生用に MP3 96 kbps へ変換し、音量をそろえています。"
# the speakers on the cards, and those who are also heard with the slow step
SPEAKERS: dict[str, str] = {"jvs093": "女性 jvs093", "jvs095": "女性 jvs095", "jvs097": "男性 jvs097", "jvs100": "男性 jvs100"}
WITH_SLOW: tuple[str, ...] = ("jvs093", "jvs100")
# a card's sentence is the speaker's test sentence nearest this long (seconds of speech)
LENGTH: float = 5.0
LABELS: dict[str, str] = {
    "R": "実録音",
    "S": "合成音（V1）",
    "S+envR": "フレームごとに合わせる（上限）",
    "sex": "① 性別に共通の補正 1 本",
    "speaker": "② 話者ごとの補正 1 本",
    "sound": "③ 音素ごとの補正（性別に共通）",
    "own": "④ 音素ごと・話者ごとの平均の補正",
    "slow": "⑤ フレームごと、ただしゆっくり",
    "form-a": "ⓐ 喉頭の上の管を長く狭く、声門の閉じ方を少し鋭く",
    "form-b": "ⓑ ⓐ に高域の損失を少し足す",
    "form-c": "ⓒ 包絡だけに当てはめた形（6 つとも）",
    "shape": "ⓓ ⓐ + 話者の声道の形（制御は V1 のまま）",
    "reach-a": "ⓔ ⓐ のまま、制御を探索",
    "reach-as": "ⓕ 話者の声道の形 + 制御を探索",
}
# the forms of `--forms`' lines, in their order, and those heard on the cards
FORMS: tuple[str, ...] = ("form-a", "form-b", "form-c")
HEARD_FORMS: tuple[str, ...] = ("form-a",)
# the steps of the ladder heard on the cards
HEARD_STEPS: tuple[str, ...] = ("S", "S+envR", "own", "slow")
# the searched renders (`pmvc.reach`'s and `pmvc.form --shapes`' variants), as `--reached` names them
REACHED: tuple[str, ...] = ("shape", "reach-a", "reach-as")
ROWS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("基準", ("R", "S", "S+envR")),
    ("フレームごとの補正を 0.1 秒の移動平均でならしたもの（前回、上限にかなり近いと確認）", ("slow",)),
    ("前回までに聞いた、時間で変わらない補正と形", ("own", "form-a")),
    ("今回: 話者ごとの声道の形と、制御の探索（エンジンをそのまま鳴らした音。補正は掛けていません）", REACHED),
)
# V1's values of the settings a form is fitted in (`pmvc.form`'s `--from`)
START: dict[str, str] = {
    "F": "form.layerLoss=0,glottis.closureSharpness=0.18,larynx.length=1.0,set.epilarynx=0,form.fossaLength=2.0,form.fossaArea=0.9",
    "M": "form.layerLoss=0,glottis.closureSharpness=0.2,larynx.length=1.0,set.epilarynx=0.78,form.fossaLength=2.0,form.fossaArea=0.9",
}
# what each step stands for, in the table
STEPS: tuple[tuple[str, str, str], ...] = (
    ("S", "合成音（V1）", ""),
    ("sex", "① 性別に共通の補正 1 本", "エンジンの形を全員ぶん直す（計画の Phase 2）"),
    ("speaker", "② 話者ごとの補正 1 本", "話者ごとの固定の設定"),
    ("sound", "③ 音素ごとの補正（性別に共通）", "どの音も平均的には正しいエンジン"),
    ("own", "④ 音素ごとの補正（話者ごと）", "話者ごとの声道の形。前後の音には依らない"),
    ("slow", "⑤ フレームごと、0.1 秒でならす", "制御列が録音をゆっくり追う"),
    ("S+envR", "フレームごとに合わせる", "制御列が録音をフレームごとに追う（上限）"),
)

NOW_TEXT = """<p>前回の試聴で、平均だけで作った補正（①〜④）と形だけの変更（ⓐ）は V1 とほとんど変わらず、フレームごとの補正をゆっくり掛けた音（⑤）だけが上限にかなり近い、と確認できました。つまり、合成音の包絡が録音を<b>時間に沿って</b>追わないと、近づいて聞こえません。</p>
<p>そこで今回は、補正を掛けるのをやめて、エンジンそのものがどこまで録音の包絡を追えるかを調べました。制御列（舌・顎・唇などの 14 チャネル）を 50 ms きざみで動かし、実際に鳴らしながら録音の包絡に寄せる探索です（<code>pmvc.reach</code>）。あわせて、エンジン（フォーク）に「話者ごとの声道の形」を足しました。声門から唇までの 12 か所で、声道の断面積を何倍にするかを話者ごとに決める設定です（時間では変わりません。既定では無効で、V1 の音は 1 サンプルも変わりません）。</p>
<p>カードの最後の段が今回の音です。どれも、エンジンをそのまま鳴らした音です。</p>
<ul>
<li><b>ⓓ</b>: ⓐ の形に、その話者の声道の形を足したもの。制御列は V1 のまま（エンコーダの出力そのまま）。</li>
<li><b>ⓔ</b>: ⓐ の形のまま、制御列だけをその文の録音に合わせて探索したもの。</li>
<li><b>ⓕ</b>: 話者の声道の形と、制御列の探索の両方。声道の形は、その話者の 3 文に共通の 1 組です。</li>
</ul>
<p>聞いていただきたいのは次の点です。</p>
<ul>
<li>ⓕ は、V1 や ⓐ より録音に近づいて聞こえますか。⑤（男女 1 人ずつにあります）や上限と比べて、どのくらいの位置ですか。</li>
<li>ⓕ に副作用はありますか（ざらつき、こもり、ブザーのような音、発音の崩れ）。包絡のずれだけを見て探索しているので、前回の ⓒ のような抜け道に入っていないかを耳で確かめたいです。</li>
<li>ⓓ（形だけ）と ⓕ の差、ⓔ（制御だけ）と ⓕ の差は聞こえますか。ⓓ が良ければ話者ごとの設定だけで済み、ⓕ まで要るならエンコーダの学習をやり直すことになります。</li>
</ul>"""
LADDER_TEXT = """<p>前回（2 回め）の試聴で使った表です。「合成音 + 録音の包絡」（上限）は包絡のずれを一度に全部直しているので、どこまで直せば近づいて聞こえるのかを段階に分けました。①〜④は補正カーブを<b>平均だけ</b>で作ったもの（その発話自身のフレームは使わず、同性のほかの話者や同じ話者のほかの発話の平均）、⑤はフレームごとの補正を 0.1 秒の移動平均でならしたものです。</p>
<p>試聴の結果は次のとおりでした。</p>
<ul>
<li>①〜④は、V1 より微小なニュアンスで近づいた気はするが、4 つの間にほぼ差がない。②と④の差もほとんど分からない。</li>
<li>⑤は上限にかなり近く、①〜④よりずっと良い。上限との違いは、一部の音（「なく」「だった」）の後ろで薄く二重に聞こえるものがある程度。</li>
</ul>
<p>照合モデルの答えも同じ向きです。平均だけで作った①〜④では本人率が上がらず、⑤で少し、フレームごとに合わせて初めて大きく上がります。</p>"""
FIRST_TEXT = """<p>最初の診断と、1 回めの試聴の結果です。この節に音はありません。</p>
<ul>
<li>「合成音 + 録音の包絡」は、4 枚のカードすべてで録音の次に良く、録音にかなり近く聞こえる。→ 包絡を録音に合わせることを、以後の作業の的にします。</li>
<li>ノイズフロアや短い残響を足しても、違いは聞こえない。→ NISQA の自然さがこれで上がるのは、耳には関係のない採点の癖でした。出力段（部屋とノイズ）は作りません。</li>
<li>予測モデルの点数（NISQA の自然さと品質、UTMOSv2）は、どれも聞こえ方と安定して合わない。→ 点数は報告するだけにして、合わせ込みません。</li>
<li>喉頭の上の管を狭めた音は、副作用はないが違いは少し分かる程度。「華やかさ」は「合成音 + 録音の包絡」のほうが圧倒的に良い。</li>
</ul>
<p>下の表は、録音と合成音の成分を入れ替えた音の、照合モデルと NISQA の点数です。</p>"""
REACH_TEXT = """<p>作業の最初に、いまのエンジンに「録音の包絡に合わせよ」と頼むとどこまで寄れるかを 2 通りで試したときの記録です。1 つは話者ごとの設定（体格の 4 つの値、声門、喉頭の上の管、喉頭・口蓋帆・舌の根のずれ）を包絡の距離だけで当てはめること。もう 1 つは、制御列を発話ごとに探索すること（変更を 1 か所ずつ試し、発話全体の距離で採点する方式）です。このときは「ほとんど寄れない」という結果でしたが、探索が弱かったためで、第 1 節の探索（包絡が局所的に決まることを使い、全部の箇所を同時に動かす方式）に置き換えました。この節に音はありません。</p>"""
LOW_TEXT = """<p>NISQA の自然さのモデルは、合成音に声と無関係なノイズフロアや短い残響を足すだけで点を上げます。どの帯域のどの区間を聞いているかを調べた表です。</p>"""


def chosen(d: Path, speaker: str, k: int) -> dict[str, Any]:
    """Of the speaker's test recordings, the one whose speech is `k`-th nearest LENGTH (so that
    the cards do not all say one sentence)."""
    metas = [json.loads((d / "meta" / f"{i}.json").read_text()) for i in ids_in(d, "test")]
    return sorted((m for m in metas if m["speaker"] == speaker), key=lambda m: abs(m["span"][1] - m["span"][0] - LENGTH))[k]


def versions(d: Path, variant: str, ladder: Path, m: dict[str, Any], forms: list[str], reached: dict[str, list[str]]) -> dict[str, Array]:
    """A card's clips, at one speech level: the recording, `pmvc.ladder`'s steps of its render,
    for each of `forms` (SPECs of `pmvc.form`) the track rendered again with the tract in it,
    and for each of `reached` the render that one of its variants has of the recording."""
    x = gap.pair(d, variant, m["id"])[0]

    def on_clock(y: Array) -> Array:
        y = resample_poly(y, 1, 48000 // gap.SR)[round(LEAD * gap.SR) :]
        y = np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]
        return at_level(np.roll(y, -gap.lag(x, y)), gap.SR, LEVEL)

    out = {"R": at_level(x, gap.SR, LEVEL)}
    for k in HEARD_STEPS:
        out[k] = load(ladder / k / f"{m['id']}.wav", gap.SR)[0]
    track = json.loads((d / variant / f"{m['id']}.tracks.json").read_text())
    for k, spec in zip(FORMS, forms):
        if k in HEARD_FORMS:
            out[k] = on_clock(render_track(formed(track, m["sex"], settings(spec))))
    for k, variants in reached.items():
        for v in variants:
            if (d / v / f"{m['id']}.wav").exists():
                out[k] = on_clock(load(d / v / f"{m['id']}.wav", 48000)[0])
    return out


def cards(d: Path, variant: str, ladder: Path, forms: list[str], reached: dict[str, list[str]], clips: dict[str, str], errors: dict[str, dict[str, list[tuple[float, float]]]]) -> str:
    """The cards. `errors` takes, for each kind of clip and each sex, the envelope's error of
    the cards' clips (as far as it is heard, and plain)."""
    metas = [json.loads((d / "meta" / f"{i}.json").read_text()) for i in ids_in(d, "test")]
    aligned: dict[str, Any] = json.loads((d / "aligned.json").read_text())
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for n, (speaker, label) in enumerate(SPEAKERS.items()):
            m = chosen(d, speaker, n)
            pitch = float(np.median([x["f0Median"] for x in metas if x["speaker"] == speaker]))
            heard = Clip(d, variant, m, pitch, aligned)
            made = versions(d, variant, ladder, m, forms, reached)
            paths: dict[str, Path] = {}
            said: dict[str, str] = {}
            for k, x in made.items():
                if k != "R":
                    lack = heard.lack(x)
                    plain = float(np.sqrt(((lack - lack.mean(axis=1, keepdims=True)) ** 2).mean()))
                    errors.setdefault(k, {}).setdefault(m["sex"], []).append((heard_error(heard.real, lack), plain))
                if k == "slow" and speaker not in WITH_SLOW:
                    continue
                paths[k] = Path(tmp) / f"{m['id']}_{len(paths)}.wav"
                sf.write(str(paths[k]), x * min(1.0, 0.98 / max(float(np.abs(x).max()), 1e-9)), gap.SR, subtype="PCM_16")
                said[k] = "録音そのもの" if k == "R" else f"母音の包絡のずれ {errors[k][m['sex']][-1][0]:.1f} dB（聞こえ方で重み付け。重みなしでは {plain:.1f} dB）"
            rows_html = "".join(
                f'<div class="kind"><span class="kind-label">{esc(title)}</span><div class="clips">'
                + "".join(button(clips, f"{m['id']}/{k}", paths[k], LABELS[k], said[k], "ref" if k == "R" else "", "包絡") for k in keys if k in paths)
                + "</div></div>"
                for title, keys in ROWS
                if any(k in paths for k in keys)
            )
            out.append(
                f"""<article class="sentence">
<p class="meta">{esc(m["id"])} · {esc(label)}</p>
<p class="text">{esc(m["text"])}</p>
{rows_html}
<p class="heard" aria-live="polite">クリップを選ぶと、録音との包絡のずれを表示します。</p>
</article>"""
            )
    return "".join(out)


def cards_table(d: Path, errors: dict[str, dict[str, list[tuple[float, float]]]]) -> str:
    """The envelope's error of the cards' own clips, a row a kind; and under it what the
    speakers' shapes do to the other test sentences (`pmvc.form --shapes`' form_shape.json)."""
    order = ("S", "own", "form-a", "shape", "reach-a", "reach-as", "slow", "S+envR")
    body = "".join(
        f'<tr><th>{esc(LABELS[k])}</th>' + "".join(f'<td class="num">{pair({s: float(np.mean([e[i] for e in errors[k][s]])) for s in SEXES}, "{:.1f}")}</td>' for i in (0, 1)) + "</tr>"
        for k in order if k in errors and all(s in errors[k] for s in SEXES)
    )  # fmt: skip
    out = f"""<div class="scroll"><table>
<caption>カードの 4 文（女性 2 文 / 男性 2 文）での、母音の包絡の録音とのずれ（dB、rms）。「重み付き」は各帯域を録音での大きさに応じて数えたもの、「重みなし」は 8 kHz までの全帯域を同じ重みで数えたもの。ⓕ の声道の形は、カードの文を含むその話者の 3 文に合わせてあります。</caption>
<thead><tr><th>音</th><th>ずれ（重み付き）</th><th>ずれ（重みなし）</th></tr></thead><tbody>{body}</tbody></table></div>"""
    p = d / "form_shape.json"
    if p.exists():
        tries = json.loads(p.read_text())["tries"]
        names = {"": "合成音（V1）"}
        for k in tries:
            if k:
                names[k] = "ⓓ ⓐ + 話者の声道の形" if k.endswith("+shapes") else "ⓐ"
        rows = "".join(
            f'<tr><th>{esc(names[k])}</th>'
            + "".join(f'<td class="num">{pair({s: m[s][key] for s in SEXES}, fmt)}</td>' for key, fmt in (("heard", "{:.1f}"), ("total", "{:.1f}"), ("quality", "{:.2f}"), ("natural", "{:.2f}"), ("own", "{:.3f}")) if all(key in m[s] for s in SEXES))
            + "</tr>"
            for k, m in tries.items()
        )
        out += f"""<div class="scroll"><table>
<caption>声道の形は、合わせるのに使っていない文でも効くか。カードの 4 話者のテスト全文（各 25 文、うち 3 文が形を合わせた文）を、制御列は V1 のまま鳴らした値です。女性 / 男性。類似度は WavLM の本人とのコサイン類似度（実録音は 0.98）。NISQA は参考です。</caption>
<thead><tr><th>音</th><th>ずれ（重み付き）</th><th>ずれ（重みなし）</th><th>NISQA 品質</th><th>NISQA 自然さ</th><th>類似度</th></tr></thead><tbody>{rows}</tbody></table></div>"""
    return out


def ladder_table(d: Path) -> str:
    p = d / "ladder.json"
    if not p.exists():
        return ""
    g = json.loads(p.read_text())
    h = g["steps"]
    opinions = all("utmos" in h[k] for k, _, _ in STEPS if k in h)
    body = ""
    for k, label, work in STEPS:
        if k not in h:
            continue
        c = h[k]
        body += (
            f'<tr><th>{esc(label)}{f'<span class="sub">{esc(work)}</span>' if work else ""}</th><td class="num">{pair({s: c["shape"][s]["total"] for s in SEXES}, "{:.1f}")}</td>'
            + "".join(f'<td class="num">{pair({s: 100 * c[m]["sexes"][s]["top1"] for s in SEXES}, "{:.0f}")}</td>' for m in gap.VERIFIERS)
            + f'<td class="num">{c["wavlm"]["own"]:.3f}</td>'
            + ("".join(f'<td class="num">{pair(c[m])}</td>' for m in ("natural", "utmos")) if opinions else "")
            + "</tr>"
        )
    more = "<th>NISQA 自然さ</th><th>UTMOSv2</th>" if opinions else ""
    return f"""<div class="scroll"><table>
<caption>JVS のテスト {g["clips"]} 文（8 話者）、すべて女性 / 男性。包絡のずれは、母音での録音との差（dB、rms。フレームごとのレベルは除く）。「本人」は、登録した 100 話者の中で最も近い話者が本人だった割合 %（実録音は 89〜90%）。類似度は WavLM の本人とのコサイン類似度。小さい字は、その段が何の作業の上限にあたるか。{"予測モデルの点数は参考です。" if opinions else ""}</caption>
<thead><tr><th>音</th><th>包絡のずれ</th><th>本人 WavLM</th><th>本人 UniSpeech</th><th>類似度</th>{more}</tr></thead>
<tbody>{body}</tbody></table></div>"""


# the settings of a form, as the page names them
FORM_KEYS: tuple[tuple[str, str, str], ...] = (
    ("larynx.length", "喉頭の上の管の長さ（体格が決める長さの何倍か）", "{:.2f}"),
    ("set.epilarynx", "喉頭の上の管の狭め（0〜1）", "{:.2f}"),
    ("glottis.closureSharpness", "声門の閉じ方の鋭さ（0〜1）", "{:.2f}"),
    ("form.layerLoss", "高域の損失（壁の第 2 層。0 は無し）", "{:.2f}"),
    ("form.fossaLength", "梨状窩の長さ cm（成人男性の体格で）", "{:.2f}"),
    ("form.fossaArea", "梨状窩の断面積 cm²（同）", "{:.2f}"),
)
FORM_TEXT = """<p>合成音の包絡のずれのうち、全員に共通の部分はエンジンの形から来ています。そこで、形の設定を実際に鳴らしながら録音の包絡に当てはめました（制御列は V1 のまま。当てはめはテストに入っていない 24 話者で行い、下の表はテストの 8 話者です）。動かしたのは、喉頭の上の管の長さと狭め、声門の閉じ方、梨状窩、そしてエンジン（フォーク）に新しく足した「周波数とともに増える損失」です。この損失は既定では無効で、V1 の音は 1 サンプルも変わりません。</p>
<p><b>最初の当てはめは、物差しの抜け道に入りました。</b>母音の包絡のずれだけを見て当てはめると、「強い損失 + 鋭い声門の閉じ方」という形（ⓒ）になり、ずれは 1.7 dB 縮みます。ところが予測モデルと照合モデルは、この音を V1 よりかなり悪いと判定します（下の表）。スペクトログラムで見ると、5 kHz より上の余分な共鳴の線は消えますが、録音でははっきりしている 2〜4.5 kHz のフォルマントまでならされていました。包絡のずれを全帯域同じ重みで数えていたので、録音でほとんど鳴っていない 5〜8 kHz を消すことが高く評価されてしまったのです。</p>
<p>そこで、包絡のずれを「録音で大きく鳴っている帯域ほど重く数える」ように直し（表の「聞こえ方で重み付け」）、損失を使わない控えめな形（ⓐ）を選び直しました。</p>
<p>前回の試聴の結果は次のとおりでした（ⓑとⓒの音は、いまのページからは外してあります）。</p>
<ul>
<li>ⓐは V1 よりほんの少し録音に近い。ただ、⑤や上限からはまだだいぶ遠い。</li>
<li>ⓒは強いざらつきとブザーのような音があり、使えない。包絡のずれは最も小さいのに耳では最悪で、包絡のずれだけでは形を選べないことが確かめられました。</li>
<li>ⓑ（ⓐ に高域の損失を少し）は V1 との差が非常に小さく、ⓐのほうが少し良い。</li>
</ul>"""


def form_section(d: Path, forms: list[str]) -> str:
    """The forms on the cards: their settings, and what the yardsticks make of them on the test."""
    test = d / "form_test.json"
    if not (forms and test.exists()):
        return ""
    tries = json.loads(test.read_text())["tries"]
    if not all(f in tries for f in forms):
        return ""
    names = ("ⓐ", "ⓑ", "ⓒ")
    now = [{s: {f"{part}.{key}": x for whose, part, key, x in settings(f) if whose in ("", s)} for s in SEXES} for f in forms]
    was = {s: {f"{part}.{key}": x for _, part, key, x in settings(START[s])} for s in SEXES}

    def cell(values: dict[str, dict[str, float]], k: str, fmt: str) -> str:
        return " / ".join(fmt.format(values[s].get(k, was[s][k])) for s in SEXES)

    rows = "".join(
        f"<tr><th>{esc(label)}</th><td class=\"num\">{cell(was, k, fmt)}</td>" + "".join(f'<td class="num">{cell(n, k, fmt)}</td>' for n in now) + "</tr>"
        for k, label, fmt in FORM_KEYS
    )
    out = f"""<div class="scroll"><table>
<caption>形の設定（女性の声 / 男性の声）。</caption>
<thead><tr><th>設定</th><th>V1</th>{"".join(f"<th>{n}</th>" for n in names[: len(forms)])}</tr></thead><tbody>{rows}</tbody></table></div>"""

    def line(label: str, m: dict[str, Any]) -> str:
        return (
            f"<tr><th>{esc(label)}</th>"
            + "".join(f'<td class="num">{pair({s: m[s][k] for s in SEXES}, fmt)}</td>' for k, fmt in (("heard", "{:.1f}"), ("total", "{:.1f}"), ("quality", "{:.2f}"), ("natural", "{:.2f}"), ("own", "{:.3f}")))
            + "</tr>"
        )

    short = ("ⓐ 喉頭の上の管と声門の閉じ方", "ⓑ ⓐ + 高域の損失を少し", "ⓒ 包絡だけに当てはめた形")
    body = line("合成音（V1）", tries[""]) + "".join(line(n, tries[f]) for n, f in zip(short, forms))
    out += f"""<div class="scroll"><table>
<caption>テストの 8 話者・200 文（当てはめには使っていません）、女性 / 男性。「ずれ」は母音の包絡の録音との差（dB、rms）。「重み付き」は各帯域を録音での大きさに応じて数えたもの、「重みなし」は 8 kHz までの全帯域を同じ重みで数えたもの（最初の当てはめが下げようとした値）。類似度は WavLM の本人とのコサイン類似度（実録音は 0.98）。NISQA は参考です。</caption>
<thead><tr><th>音</th><th>ずれ（重み付き）</th><th>ずれ（重みなし）</th><th>NISQA 品質</th><th>NISQA 自然さ</th><th>類似度</th></tr></thead><tbody>{body}</tbody></table></div>"""
    return out


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


def reach_tables(d: Path) -> str:
    """Whether the voice comes nearer a recording's envelope when asked: by its settings, and by its tracks."""
    out = ""
    p = d / "bench_static.json"
    if p.exists():
        b = json.loads(p.read_text())
        labels = {"enci": "性別と声の高さだけ（設定を当てはめる前）", "encit": "照合モデルに合わせた 11 個の設定", "encie": "録音の包絡に合わせた 12 個の設定"}
        body = "".join(
            f'<tr><th>{esc(label)}</th>' + "".join(f'<td class="num">{pair({s: 100 * c[m]["sexes"][s]["top1"] for s in SEXES}, "{:.0f}")}</td>' for m in gap.VERIFIERS)
            + f'<td class="num">{pair({s: c["wavlm"]["sexes"][s]["own"] for s in SEXES}, "{:.3f}")}</td><td class="num">{pair({s: c["shape"][s]["total"] for s in SEXES}, "{:.1f}")}</td>'
            f'<td class="num">{pair({s: c["shape"][s]["common"] for s in SEXES}, "{:.1f}")}</td><td class="num">{pair({s: c["shape"][s]["rest"] for s in SEXES}, "{:.1f}")}</td></tr>'
            for k, label in labels.items() if k in b["variants"] for c in [b["variants"][k]]
        )  # fmt: skip
        out += f"""<div class="scroll"><table>
<caption>話者ごとの設定（体格、声門、構えの癖）を当てはめたときの、テストの {b["clips"]} 文。すべて女性 / 男性。制御列は同じ（エンコーダ I）。「本人」は 100 話者の中で本人が選ばれた割合 %（実録音は {pair({s: 100 * b["variants"]["real"]["wavlm"]["sexes"][s]["top1"] for s in SEXES}, "{:.0f}")}）。包絡のずれは母音での dB（rms）。</caption>
<thead><tr><th>設定</th><th>本人 WavLM</th><th>本人 UniSpeech</th><th>類似度</th><th>包絡のずれ</th><th>うち共通</th><th>うち残り</th></tr></thead><tbody>{body}</tbody></table></div>"""
    rows = ""
    for name, label in (("reach_t", "包絡の項の重み 0.3"), ("reach_s", "包絡の項の重み 3.0")):
        q = d / f"{name}.json"
        if q.exists():
            r = json.loads(q.read_text())
            rows += f'<tr><th>{esc(label)}</th><td class="num">{len(r)}</td><td class="num">{np.mean([x["distance"][0] for x in r]):.3f} → {np.mean([x["distance"][1] for x in r]):.3f}</td><td class="num">{10 * np.mean([x["shape"][0] for x in r]):.1f} → {10 * np.mean([x["shape"][1] for x in r]):.1f}</td></tr>'
    if rows:
        out += f"""<div class="scroll"><table>
<caption>制御列を発話ごとに探索したとき（<code>pmvc.copysynth</code>）。始点は「照合モデルに合わせた 11 個の設定」の合成音。リスナー距離は、何を言ったかの距離（mHuBERT 第 8 層）。包絡の距離は、録音と合成音のメルスペクトル（40 帯域、8 kHz まで）のフレームごとの差の平均（dB）。</caption>
<thead><tr><th>探索</th><th>文数</th><th>リスナー距離</th><th>包絡の距離（dB）</th></tr></thead><tbody>{rows}</tbody></table></div>"""
    return out


VIZ_STYLE = """<style>
/* the charts' data colours: three hues that are told apart with any colour vision, stepped for each theme */
.viz { --series-1: #257de0; --series-2: #f47230; --series-3: #00b894; display: block; width: 100%; min-width: 640px; height: auto; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .viz { --series-1: #338ced; --series-2: #e76624; --series-3: #00a888; } }
:root[data-theme="dark"] .viz { --series-1: #338ced; --series-2: #e76624; --series-3: #00a888; }
.viz text { font-family: var(--font-body); font-size: 11px; fill: var(--muted); }
.viz .title { font-size: 12px; font-weight: 500; fill: var(--ink); }
.viz .grid { stroke: var(--line); stroke-width: 1; }
.viz .zero { stroke: var(--muted); stroke-width: 1; }
.viz .lack { fill: var(--ink); opacity: 0.06; }
.viz .hit { fill: transparent; }
.viz .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
</style>"""


def tract_chart(sex: str, freqs: list[float], lines: list[tuple[str, list[float]]], gap: list[float]) -> str:
    """One sex's chart: the engine's transfer function over the vowels (a line a setting, on
    one scale), and under it, on the same frequency axis, what the renders' envelopes lack of
    the recordings' in 250 Hz bands (columns)."""
    left, width, top_a, h_a, top_b, h_b = 48.0, 700.0, 46.0, 150.0, 244.0, 130.0
    span_a, span_b, top_hz = 30.0, 16.0, 8000.0

    def x(hz: float) -> float:
        return left + hz / top_hz * width

    def ya(db: float) -> float:
        return top_a + (span_a - db) / (2 * span_a) * h_a

    def yb(db: float) -> float:
        return top_b + (span_b - db) / (2 * span_b) * h_b

    f = np.array(freqs)
    ref = float(np.array(lines[0][1])[(f >= 500) & (f < 1000)].mean())
    out = [f'<svg class="viz" viewBox="0 0 760 404" role="img" aria-label="{"女性" if sex == "F" else "男性"}: エンジンの伝達関数と、録音との包絡の差">']
    out.append(f'<defs><clipPath id="clip-{sex}"><rect x="{left}" y="{top_a}" width="{width}" height="{h_a}"/></clipPath></defs>')
    # where the renders lack 6 dB or more, through both panels
    centres = [125.0 + 250.0 * i for i in range(len(gap))]
    for c, g in zip(centres, gap):
        if g >= 6.0 and c < top_hz:
            out.append(f'<rect class="lack" x="{x(c - 125):.1f}" y="{top_a}" width="{x(250) - x(0):.1f}" height="{top_b + h_b - top_a}"/>')
    out.append(f'<text class="title" x="{left}" y="14">{"女性" if sex == "F" else "男性"}: エンジンの伝達関数（母音の平均。0.5〜1 kHz の高さを 0 dB）</text>')
    for k, (label, _) in enumerate(lines):
        out.append(f'<line x1="{left + 190 * k}" y1="30" x2="{left + 190 * k + 22}" y2="30" class="line" style="stroke: var(--series-{k + 1})"/><text x="{left + 190 * k + 28}" y="34">{esc(label)}</text>')
    for db in (-20, 0, 20):
        out.append(f'<line class="{"zero" if db == 0 else "grid"}" x1="{left}" y1="{ya(db):.1f}" x2="{left + width}" y2="{ya(db):.1f}"/><text x="{left - 6}" y="{ya(db) + 4:.1f}" text-anchor="end">{f"{db:+d}" if db else "0"}</text>')
    for k, (_, level) in enumerate(lines):
        pts = " ".join(f"{x(hz):.1f},{ya(v - ref):.1f}" for hz, v in zip(freqs, level))
        out.append(f'<polyline class="line" clip-path="url(#clip-{sex})" style="stroke: var(--series-{k + 1})" points="{pts}"/>')
    out.append(f'<text class="title" x="{left}" y="{top_b - 12}">録音 − 合成音: 母音の包絡の差（dB。正は合成音の不足、網掛けは 6 dB 以上）</text>')
    for db in (-10, 0, 10):
        out.append(f'<line class="{"zero" if db == 0 else "grid"}" x1="{left}" y1="{yb(db):.1f}" x2="{left + width}" y2="{yb(db):.1f}"/><text x="{left - 6}" y="{yb(db) + 4:.1f}" text-anchor="end">{f"{db:+d}" if db else "0"}</text>')
    w, r = 14.0, 4.0
    for c, g in zip(centres, gap):
        if c >= top_hz:
            continue
        x0, y0, y1 = x(c) - w / 2, yb(0.0), yb(float(np.clip(g, -span_b, span_b)))
        q = min(r, abs(y1 - y0))
        e = q if g >= 0 else -q  # the rounded end is the far one from the baseline
        out.append(
            f'<path style="fill: var(--series-3)" d="M{x0:.1f},{y0:.1f} L{x0:.1f},{y1 + e:.1f} Q{x0:.1f},{y1:.1f} {x0 + q:.1f},{y1:.1f} L{x0 + w - q:.1f},{y1:.1f} Q{x0 + w:.1f},{y1:.1f} {x0 + w:.1f},{y1 + e:.1f} L{x0 + w:.1f},{y0:.1f} Z">'
            f"<title>{(c - 125) / 1000:g}〜{(c + 125) / 1000:g} kHz: {g:+.1f} dB</title></path>"
        )
        said = "、".join(f"{label} {float(np.array(level)[(f >= c - 125) & (f < c + 125)].mean()) - ref:+.0f} dB" for label, level in lines)
        out.append(f'<rect class="hit" x="{x(c - 125):.1f}" y="{top_a}" width="{x(250) - x(0):.1f}" height="{h_a}"><title>{(c - 125) / 1000:g}〜{(c + 125) / 1000:g} kHz: {esc(said)}</title></rect>')
    for khz in range(0, 9):
        out.append(f'<text x="{x(1000.0 * khz) + (6 if khz == 8 else 0):.1f}" y="{top_b + h_b + 16}" text-anchor="{"end" if khz == 8 else "middle"}">{khz}{" kHz" if khz == 8 else ""}</text>')
    out.append("</svg>")
    return "".join(out)


def tract_section(d: Path, variant: str, n: int) -> str:
    """What the common curve is made of: the engine's resonances against the bands the renders lack."""
    first, bench = d / f"tract_{variant}.json", d / "bench_v1.json"
    if not (first.exists() and bench.exists()):
        return ""
    t0 = json.loads(first.read_text())
    narrow = d / f"tract_{variant}_epilarynx1.json"
    t1 = json.loads(narrow.read_text()) if narrow.exists() else None
    shapes = json.loads(bench.read_text())["variants"][variant].get("shape", {})
    charts = ""
    for sex in SEXES:
        if sex not in t0["sexes"] or sex not in shapes:
            continue
        lines = [(f"V1（狭め {'0' if sex == 'F' else '0.78'}）", t0["sexes"][sex]["mean"])]
        if t1 and sex in t1["sexes"]:
            lines.append(("狭め 1.0", t1["sexes"][sex]["mean"]))
        charts += f'<figure><div class="scroll">{tract_chart(sex, t0["freqs"], lines, shapes[sex]["commonCurve"])}</div></figure>'
    peaks = "、".join(f"{'女性' if s == 'F' else '男性'} {' / '.join(f'{p / 1000:g}' for p in t0['sexes'][s]['peaks'] if 2000 <= p <= 6500)} kHz" for s in SEXES if s in t0["sexes"])
    return f"""<section aria-labelledby="h-tract"><h2 id="h-tract">{n}. 不足している帯域と、エンジンの共鳴の位置</h2>
<p>合成音の包絡は、声源のスペクトルに声道の共鳴が掛かったものです。エンジンには気道の周波数領域のモデル（シミュレーションと同じ要素を線形にしたもの）があり、ある瞬間の声道の形だけの効果（伝達関数）を、声源抜きで計算できます。V1 の制御列を鳴らしながら、母音の中央のフレームでこれを読み、平均しました（<code>pmvc.tract</code>、{t0["clips"]} 文）。</p>
{VIZ_STYLE}{charts}
<p class="small">上の線がエンジンの伝達関数、下の棒が録音と合成音の包絡の差（第 {n - 2} 節の表の「共通」のカーブを 250 Hz ごとに）。縦の網掛けは、合成音が 6 dB 以上不足している帯域です。棒や線の上にカーソルを置くと値が出ます。伝達関数の鋭い切れ込み（男性 4.2 kHz、女性 4.8 kHz）は、梨状窩の零点です。</p>
<p>合成音が不足している帯域は、エンジンの 3 番めの共鳴と、その上の共鳴のあいだの谷にあたります（2〜6.5 kHz の山の位置: {peaks}）。録音では、この谷のあたりに 3 番めと 4 番めのフォルマントが寄っています。エンジンの上の共鳴は高すぎる位置にあり、喉頭の上の管を最大まで狭めると少し下がって強くなりますが（橙の線）、谷は埋まりきりません。5 kHz より上で合成音が過剰なのは、伝達関数がそこで高いままだからです。次の作業（計画の Phase 2-1）は、この上の共鳴を下げられるように喉頭の上の管の長さと形を設定にすることです。形の変更は、この図で先に確かめてから音にします。</p>
</section>"""


def build(d: Path, variant: str, ladder: Path, forms: list[str], reached: dict[str, list[str]], notes: str) -> str:
    clips: dict[str, str] = {}
    errors: dict[str, dict[str, list[tuple[float, float]]]] = {}
    form = form_section(d, forms)
    made = cards(d, variant, ladder, forms if form else [], reached, clips, errors)
    body = f"""<section aria-labelledby="h-now"><h2 id="h-now">1. エンジンは録音の包絡を追えるか（話者ごとの声道の形と、制御の探索）</h2>
{NOW_TEXT}
{cards_table(d, errors)}
{made}
</section>
<section aria-labelledby="h-ladder"><h2 id="h-ladder">2. 包絡をどこまで合わせれば、録音に近く聞こえるか（前回の試聴）</h2>
{LADDER_TEXT}
{ladder_table(d)}
</section>"""
    if form:
        body += f"""<section aria-labelledby="h-form"><h2 id="h-form">3. エンジンの形を直す（最初の当てはめと、その失敗）</h2>
{FORM_TEXT}
{form}
</section>"""
    n = 4 if form else 3
    body += f"""<section aria-labelledby="h-swap"><h2 id="h-swap">{n}. 最初の診断と、1 回めの試聴の結果</h2>
{FIRST_TEXT}
{swap_table(d, variant)}
{LOW_TEXT}
{low_table(d, variant)}
</section>
<section aria-labelledby="h-bench"><h2 id="h-bench">{n + 1}. V1 の基準値（新しい採点表）</h2>
<p>これまでは合成音 48 kHz・録音 24 kHz を、それぞれのレベルのまま比べていました。帯域とレベルを揃えて取り直した値です。以後の変更は、この表と比べます。</p>
{bench_tables(d, {"real": "実録音", variant: "合成音（V1: 公開用エンコーダと 2 つの基準の声）", "encit": "話者ごとに 11 個の設定を当てはめた合成音"})}
</section>"""
    reach = reach_tables(d)
    if reach:
        body += f"""<section aria-labelledby="h-reach"><h2 id="h-reach">{n + 2}. 最初の試み: 話者ごとの設定と、発話全体で採点する探索</h2>
{REACH_TEXT}
{reach}
</section>"""
    body += tract_section(d, variant, n + 3)
    return shell(TITLE, LEDE, notes, body, CREDITS, clips)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "jvs")
    ap.add_argument("--variant", type=str, default="encv2")
    ap.add_argument("--ladder", type=Path, default=ROOT / "out" / "jvs" / "v2_probe" / "ladder", help="where pmvc.ladder wrote its clips")
    ap.add_argument("--forms", type=Path, default=None, help="a file of forms (pmvc.form's SPECs, a line each, up to three): their renders on the cards, and their section if <dir>/form_test.json has them")
    ap.add_argument("--reached", type=str, default="", help="the searched renders on the cards, each with the variants that may hold a card's clip: shape=VARIANT,reach-a=VARIANT,reach-as=VARIANT+VARIANT")
    ap.add_argument("--notes", type=Path, default=None)
    ap.add_argument("--pages", type=Path, default=ROOT / "out" / "transplant")
    ap.add_argument("--out", type=str, default="listening_v2.html")
    a = ap.parse_args()
    forms = [line for line in a.forms.read_text().splitlines() if line.strip()] if a.forms else []
    reached = {w.split("=")[0]: w.split("=")[1].split("+") for w in a.reached.split(",")} if a.reached else {}
    assert all(k in REACHED for k in reached), f"--reached takes {REACHED}"
    page = build(a.dir, a.variant, a.ladder, forms, reached, a.notes.read_text() if a.notes else "<p>（所見は未記入）</p>")
    (a.pages / a.out).write_text(page)
    print(f"{a.pages / a.out}: {len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
