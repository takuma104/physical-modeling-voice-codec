"""Figures for the technical report (English and Japanese labels).

    uv run python docs/report/make_figures.py --font-dir docs/report/fonts

Numbers: `pmvc.judge`'s and `pmvc.natural`'s summaries on the 100 test sentences of
Hi-Fi-CAPTAIN's Japanese man. For the first figure, the text-to-speech models taught with fewer
recordings (README, step 9): out/ja/results_tts.json (variants tts, tts2000, tts200, tts50,
tts20, and enci and real for the two reference lines). For the second, the control (step 11):
out/ja/results_ctl.json (the track: tts20n, tts50n, tts200n, tts2000n, tts; the neural
decoders' latents: ctl<latent><n>; the spectrogram with its vocoder tuned: ctlmel<n>vp; each
decoder's best: enci, copy_<latent>).

The drawing of the vocal tract is CHIMERA's own: `pmvc_engine.outline` gives the midsagittal
outlines its airway's areas are computed from, for the male body in one posture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pmvc_engine
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, Polygon
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline

# recordings the model was taught with, and what the judges make of its 100 test sentences:
# Whisper small and large-v3-turbo (% of phonemes wrong), NISQA's predicted quality
SIZES: list[int] = [20, 50, 200, 2000, 18655]
SMALL: list[float] = [13.84, 6.07, 1.31, 1.87, 0.83]
LARGE: list[float] = [7.73, 2.22, 0.50, 0.17, 0.17]
QUALITY: list[float] = [4.16, 4.53, 4.70, 4.75, 4.78]
# the encoder's renders of the recordings (the model's teacher) and the recordings themselves
ENCODER: dict[str, float] = {"small": 1.29, "large": 0.10, "quality": 4.75}
RECORDING: dict[str, float] = {"small": 0.48, "large": 0.08, "quality": 4.56}

# the first two slots of the categorical palette (validated for colour-vision deficiencies), and ink
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#dddcd6"
AQUA, VIOLET = "#1baf7a", "#6250d6"

# the control: what the model writes → Whisper large-v3-turbo (% of phonemes wrong) and NISQA of
# the models taught with SIZES recordings, and of the decoder at its best
CONTROL: dict[str, dict[str, list[float] | float]] = {
    "track": {"large": [8.75, 3.16, 0.71, 0.27, 0.17], "quality": [4.06, 4.5, 4.77, 4.81, 4.78], "best": 4.75},
    "mel": {"large": [7.71, 2.76, 0.58, 0.04, 0.42], "quality": [2.57, 3.24, 3.53, 3.64, 3.68], "best": 4.45},
    "melv": {"large": [7.19, 3.03, 0.56, 0.04, 0.29], "quality": [2.83, 3.76, 4.13, 4.52, 4.46]},
    "encodec": {"large": [6.57, 2.64, 0.67, 0.12, 0.08], "quality": [1.85, 2.1, 2.35, 3.27, 2.59], "best": 4.36},
    "mimi": {"large": [23.43, 11.25, 2.62, 0.33, 0.35], "quality": [2.62, 3.55, 3.5, 4.65, 3.31], "best": 4.81},
}
# a latent's colour and marker; dashed with open markers where its decoder was tuned
LOOKS: dict[str, tuple[str, str, bool]] = {"track": (BLUE, "o", False), "mel": (ORANGE, "s", False), "melv": (ORANGE, "s", True), "encodec": (AQUA, "^", False), "mimi": (VIOLET, "D", False)}

# the drawing of the vocal tract: air, flesh and bone
AIR, FLESH, BONE = "#dce9f8", "#f3cdbd", "#d4d3cd"
Points = NDArray[np.float64]
# a mid-open vowel with the passage to the nose open, so that every part shows
POSTURE: dict[str, float] = {"tbX": 2.05, "tbY": 6.15, "jaw": 1.05, "lipAp": 0.95, "lipProt": 0.25, "lipWidth": 0.6, "root": 0.15, "larynx": 0.05, "ttX": 5.8, "ttY": 6.2, "velum": 0.8}
# the back wall of the throat above where the airway's sections start, up into the nose
# (CHIMERA's fixed landmarks for the male body, `HumanVocalTract::new`)
BACK: list[tuple[float, float]] = [(-1.3, 5.6), (-1.38, 7.5), (-1.33, 8.9), (-1.05, 10.1), (-0.45, 10.9), (0.4, 11.3)]

TRACT: dict[str, dict[str, str]] = {
    "en": {
        "velum": "passage to the nose",
        "root": "tongue root",
        "larynx": "larynx height",
        "glottis": "glottal opening",
        "f0": "pitch",
        "lungs": "lung pressure",
        "lips": "lips",
        "jaw": "jaw",
        "body": "tongue body",
        "tip": "tongue tip",
        "lateral": "side passage, as for /l/",
        "encoder": "written by the encoder (16)",
        "pitch": "from a pitch tracker",
        "constant": "constant here:",
    },
    "ja": {
        "velum": "鼻への通路",
        "root": "舌の根",
        "larynx": "喉頭の高さ",
        "glottis": "声門の開き",
        "f0": "声の高さ",
        "lungs": "肺の圧力",
        "lips": "唇",
        "jaw": "顎",
        "body": "舌の本体",
        "tip": "舌の先",
        "lateral": "舌の側面の通路（/l/ など）",
        "encoder": "エンコーダが書く（16）",
        "pitch": "ピッチ抽出器から",
        "constant": "ここでは一定:",
    },
}
MONO: str = "DejaVu Sans Mono"


LABELS: dict[str, dict[str, Any]] = {
    "en": {
        "x": "recordings the model is taught with",
        "error": "phonemes wrong (%)",
        "quality": "predicted quality (NISQA)",
        "small": "Whisper small",
        "large": "Whisper large-v3-turbo",
        "encoder": "encoder I",
        "recording": "recordings",
        "minutes": ["1.5 min", "4 min", "15 min", "2.6 h", "24 h"],
        "large_axis": "phonemes wrong (%)\nWhisper large-v3-turbo",
        "names": {"track": "control track (CHIMERA)", "mel": "mel spectrogram", "melv": "mel, vocoder tuned", "encodec": "EnCodec", "mimi": "Mimi"},
        "best": "decoder's\nbest",
    },
    "ja": {
        "x": "学習に使う録音の数（文）",
        "error": "音素誤り率（%）",
        "quality": "予測品質（NISQA）",
        "small": "Whisper small",
        "large": "Whisper large-v3-turbo",
        "encoder": "エンコーダ I",
        "recording": "実録音",
        "minutes": ["1.5 分", "4 分", "15 分", "2.6 時間", "24 時間"],
        "large_axis": "音素誤り率（%）\nWhisper large-v3-turbo",
        "names": {"track": "制御列（CHIMERA）", "mel": "メルスペクトログラム", "melv": "メル、ボコーダを調整", "encodec": "EnCodec", "mimi": "Mimi"},
        "best": "デコーダ\nの上限",
    },
}


def figure(lang: str, out: Path) -> None:
    t = LABELS[lang]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.45, 3.6), sharex=True, gridspec_kw={"height_ratios": [1.5, 1.0], "hspace": 0.12})
    for ax in (top, bottom):
        ax.set_xscale("log")
        ax.grid(True, which="major", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
            ax.spines[side].set_linewidth(0.6)
        ax.tick_params(colors=MUTED, labelsize=7, width=0.6, length=3)
        ax.tick_params(which="minor", length=0)
    top.set_yscale("log")
    top.plot(SIZES, SMALL, color=BLUE, linewidth=1.5, marker="o", markersize=4.5, label=t["small"])
    top.plot(SIZES, LARGE, color=ORANGE, linewidth=1.5, marker="s", markersize=4.2, label=t["large"])
    # the teacher: dashes at the encoder's two rates, in the series' colours, named once in ink
    for key, colour in (("small", BLUE), ("large", ORANGE)):
        top.axhline(ENCODER[key], color=colour, linewidth=0.8, linestyle=(0, (4, 3)), alpha=0.8)
    top.text(16, ENCODER["small"] / 1.12, t["encoder"], fontsize=6.5, color=MUTED, ha="left", va="top")
    top.text(16, ENCODER["large"] * 1.12, t["encoder"], fontsize=6.5, color=MUTED, ha="left", va="bottom")
    top.set_ylim(0.06, 30)
    top.set_yticks([0.1, 0.3, 1, 3, 10])
    top.set_yticklabels(["0.1", "0.3", "1", "3", "10"])
    top.set_ylabel(t["error"], fontsize=7.5, color=INK)
    top.legend(fontsize=6.5, frameon=False, loc="upper right", handlelength=1.8, labelcolor=INK)

    bottom.plot(SIZES, QUALITY, color=INK, linewidth=1.5, marker="o", markersize=4.5)
    bottom.axhline(ENCODER["quality"], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    bottom.axhline(RECORDING["quality"], color=MUTED, linewidth=0.8, linestyle=(0, (1, 2)))
    bottom.text(16, ENCODER["quality"] + 0.03, t["encoder"], fontsize=6.5, color=MUTED, ha="left", va="bottom")
    bottom.text(600, RECORDING["quality"] - 0.04, t["recording"], fontsize=6.5, color=MUTED, ha="left", va="top")
    bottom.set_ylim(3.9, 5.0)
    bottom.set_yticks([4.0, 4.5, 5.0])
    bottom.set_ylabel(t["quality"], fontsize=7.5, color=INK)
    bottom.set_xlabel(t["x"], fontsize=7.5, color=INK)
    bottom.set_xlim(14, 30000)
    bottom.set_xticks(SIZES)
    bottom.set_xticklabels([f"{n:,}\n{m}" for n, m in zip(SIZES, t["minutes"])], fontsize=6.3)
    fig.align_ylabels([top, bottom])
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def control(lang: str, out: Path) -> None:
    """The same model writing four latents: what is understood of it, and how it sounds."""
    t = LABELS[lang]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.45, 4.1), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.1})
    for ax in (top, bottom):
        ax.set_xscale("log")
        ax.grid(True, which="major", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
            ax.spines[side].set_linewidth(0.6)
        ax.tick_params(colors=MUTED, labelsize=7, width=0.6, length=3)
        ax.tick_params(which="minor", length=0)
    top.set_yscale("log")
    for name, series in CONTROL.items():
        colour, marker, tuned = LOOKS[name]
        large, quality = series["large"], series["quality"]
        assert isinstance(large, list) and isinstance(quality, list)
        look: dict[str, Any] = {"color": colour, "linewidth": 1.5, "marker": marker, "markersize": 4.2}
        if tuned:
            look.update(linestyle=(0, (3, 1.6)), markerfacecolor="white", markeredgewidth=1.1)
        top.plot(SIZES[: len(large)], [max(v, 0.03) for v in large], label=t["names"][name], **look)
        bottom.plot(SIZES[: len(quality)], quality, **look)
        # the decoder at its best: a tick past the last point, in the series' colour
        if "best" in series:
            bottom.plot([90000, 170000], [series["best"]] * 2, color=colour, linewidth=2.2, solid_capstyle="butt")
    top.set_ylim(0.025, 40)
    top.set_yticks([0.03, 0.1, 0.3, 1, 3, 10, 30])
    top.set_yticklabels(["≤0.03", "0.1", "0.3", "1", "3", "10", "30"])
    top.set_ylabel(t["large_axis"], fontsize=7.5, color=INK)
    top.legend(fontsize=6.3, frameon=False, loc="upper right", handlelength=1.8, labelcolor=INK, borderaxespad=0.1)
    bottom.set_ylim(1.5, 5.05)
    bottom.set_yticks([2, 3, 4, 5])
    bottom.set_ylabel(t["quality"], fontsize=7.5, color=INK)
    bottom.set_xlabel(t["x"], fontsize=7.5, color=INK)
    bottom.set_xlim(14, 220000)
    bottom.set_xticks([*SIZES, 124000])
    bottom.set_xticklabels([*(f"{n:,}" for n in SIZES), t["best"]], fontsize=6.3)
    fig.align_ylabels([top, bottom])
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def pts(v: list[dict[str, float]]) -> Points:
    return np.array([[p["x"], p["y"]] for p in v])


def smooth(p: list[tuple[float, float]] | Points, n: int = 80) -> Points:
    """A smooth curve through points, by how far along the points they are."""
    q = np.asarray(p, dtype=np.float64)
    t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    return make_interp_spline(t, q, k=min(3, len(q) - 1))(np.linspace(0, t[-1], n))


def label(ax: Axes, at: tuple[float, float], to: tuple[float, float], title: str, names: list[tuple[str, str]], side: str, above: float | None = None) -> None:
    """A part's name with its channels under it (each with a dot in the colour of where its
    values come from), and a line from the name to the part at `to`. `side`: where the text
    is held, and the line leaves from the other end of the name, or with `above` from over
    the name, so far from its held end."""
    x, y = at
    title_text = ax.text(x, y, title, fontsize=6.3, color=INK, ha=side, va="bottom")
    ax.figure.canvas.draw()
    box = title_text.get_window_extent().transformed(ax.transData.inverted())
    for k, (name, colour) in enumerate(names):
        yy = y - 0.16 - 0.62 * k
        if side == "left":
            ax.plot([x + 0.16], [yy - 0.3], marker="o", ms=2.6, color=colour, mec="none", zorder=8, clip_on=False)
            ax.text(x + 0.44, yy, name, fontsize=6.0, color=INK, ha="left", va="top", family=MONO)
        else:
            text = ax.text(x, yy, name, fontsize=6.0, color=INK, ha="right", va="top", family=MONO)
            ax.figure.canvas.draw()
            ax.plot([text.get_window_extent().transformed(ax.transData.inverted()).x0 - 0.28], [yy - 0.3], marker="o", ms=2.6, color=colour, mec="none", zorder=8, clip_on=False)
    start = (box.x1 + 0.2 if side == "left" else box.x0 - 0.2, (box.y0 + box.y1) / 2)
    if above is not None:
        start = (x + above if side == "left" else x - above, box.y1 + 0.12)
    ax.plot([start[0], to[0]], [start[1], to[1]], color=MUTED, lw=0.45, zorder=7, solid_capstyle="round")
    ax.plot([to[0]], [to[1]], marker="o", ms=1.8, color=MUTED, mec="none", zorder=7)


def arrow(ax: Axes, a: tuple[float, float], b: tuple[float, float], both: bool = True, colour: str = MUTED) -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<|-|>" if both else "-|>", mutation_scale=4.0, lw=0.55, color=colour, zorder=9, shrinkA=0, shrinkB=0))


def tract(lang: str, out: Path) -> None:
    """The decoder's vocal tract (CHIMERA's outlines) and where each channel of the track acts."""
    t = TRACT[lang]
    f: dict[str, Any] = json.loads(pmvc_engine.outline("male", json.dumps(POSTURE)))
    fig, ax = plt.subplots(figsize=(3.45, 3.45 * 18.6 / 20.8))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-7.4, 13.4)
    ax.set_ylim(-5.4, 13.2)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    grid = f["grid"]
    outer, inner = pts([g["outer"] for g in grid]), pts([g["inner"] for g in grid])
    wall, tongue, velum = pts(f["outer"]), pts(f["tongue"]), pts(f["velum"])
    lip_u, lip_l = pts(f["upperLip"]), pts(f["lowerLip"])
    glottis = (f["glottis"]["x"], f["glottis"]["y"])
    centre = (f["tongueCenter"]["x"], f["tongueCenter"]["y"])
    tip = (f["tip"]["x"], f["tip"]["y"])
    roof = pts(f["nasalCavity"])
    back = wall[wall[:, 1] <= outer[0, 1]]  # the throat's back wall below the first section, bottom → top
    # the throat's back wall above the sections, and on over the nose to the nostril
    ceiling = smooth(np.concatenate([np.array(BACK), [[1.6, 11.32]], roof[roof[:, 0] > 2.9]]), 90)
    # the airway: the windpipe and larynx below the sections, then between the walls to the lips
    front = smooth([(0.22, -1.5), (0.16, glottis[1] - 0.6), (0.0, glottis[1]), (0.1, 1.5), (float(inner[0, 0]), float(inner[0, 1]))], 30)
    ax.add_patch(Polygon(np.concatenate([back, outer, inner[::-1], front[::-1]]), closed=True, fc=AIR, ec="none", zorder=1))
    # the nose: under the ceiling; over the soft palate (from its tip) and the bone of the hard one
    palate = wall[(wall[:, 0] >= roof[0, 0]) & (wall[:, 1] > 8.5) & (wall[:, 0] <= roof[-1, 0] - 0.25)]
    bone = np.concatenate([palate, (palate + np.array([0.0, 0.36]))[::-1]])
    above = ceiling[ceiling[:, 1] >= velum[10, 1] - 0.4]
    ax.add_patch(Polygon(np.concatenate([above, (palate + np.array([0.0, 0.36]))[::-1], velum[:11]]), closed=True, fc=AIR, ec="none", zorder=1))
    ax.add_patch(Polygon(bone, closed=True, fc="white", ec=INK, lw=0.55, zorder=4, joinstyle="round"))
    # the body of the jaw under the tongue (its sides, which reach up to the ears, are left out)
    jaw = pts(f["jawBone"])
    ax.plot(*np.concatenate([jaw[:3], [jaw[2] + 0.75 * (jaw[3] - jaw[2])]]).T, color=BONE, lw=3.2, solid_capstyle="round", solid_joinstyle="round", zorder=2)
    # flesh and teeth
    ax.add_patch(Polygon(tongue, closed=True, fc=FLESH, ec=INK, lw=0.7, zorder=3, joinstyle="round"))
    ax.add_patch(Polygon(velum, closed=True, fc=FLESH, ec=INK, lw=0.6, zorder=5, joinstyle="round"))
    ax.add_patch(Polygon(pts(f["upperTeeth"])[1:4], closed=True, fc="white", ec=INK, lw=0.55, zorder=4, joinstyle="round"))
    ax.add_patch(Polygon(pts(f["lowerTeeth"]), closed=True, fc="white", ec=INK, lw=0.55, zorder=4, joinstyle="round"))
    ax.plot(*pts(f["epiglottis"]).T, color=INK, lw=1.3, solid_capstyle="round", zorder=4)
    # the walls: the throat's back and the nose's ceiling, the palate's front, the windpipe's front
    ax.plot(*np.concatenate([back, ceiling[ceiling[:, 1] > back[-1, 1]]]).T, color=INK, lw=1.0, zorder=5)
    ax.plot(*wall[wall[:, 0] >= palate[-1, 0]].T, color=INK, lw=0.8, zorder=5)
    ax.plot(*front.T, color=INK, lw=0.8, zorder=5)
    for lip in (lip_u, lip_l):
        ax.plot(*smooth(lip, 60).T, color=INK, lw=0.8, zorder=4)
    # the vocal folds
    behind_glottis = float(back[np.argmin(np.abs(back[:, 1] - glottis[1])), 0])
    for a, b in ((behind_glottis, -0.66), (0.0, -0.36)):
        ax.plot([a, b], [glottis[1], glottis[1]], color=INK, lw=2.2, solid_capstyle="butt", zorder=6)
    # the face
    nose = smooth([(float(lip_u[-1, 0]), float(lip_u[-1, 1])), (7.45, 10.5), (8.25, 10.75), (8.5, 11.2), (8.15, 11.9), (7.6, 12.7), (7.35, 13.4)], 50)
    chin = smooth([(float(lip_l[-1, 0]), float(lip_l[-1, 1])), (5.95, 3.75), (5.8, 3.0), (5.0, 2.5), (3.5, 2.35), (2.2, 1.7), (1.6, 0.4), (1.55, -1.5)], 60)
    for line in (nose, chin):
        ax.plot(*line.T, color=MUTED, lw=0.7, zorder=2)

    enc, f0 = BLUE, ORANGE
    root = tongue[np.argmin(np.hypot(tongue[:, 0] - 0.2, tongue[:, 1] - 4.7))]
    left, right = -7.3, 13.3
    label(ax, (left, 10.9), (float(velum[5, 0]) + 0.25, float(velum[5, 1]) - 0.3), t["velum"], [("velum", enc)], "left")
    label(ax, (left, 8.3), (float(root[0]) + 0.2, float(root[1])), t["root"], [("root", enc)], "left")
    label(ax, (left, 5.7), (0.62, 1.7), t["larynx"], [("larynx", enc)], "left")
    label(ax, (left, 3.1), (-0.56, glottis[1] + 0.08), t["glottis"], [("abduction", enc)], "left")
    label(ax, (left, 0.5), (-0.56, glottis[1] - 0.08), t["f0"], [("f0", f0)], "left")
    label(ax, (left, -2.1), (-0.42, -1.25), t["lungs"], [("breath", enc)], "left")
    label(ax, (right, 12.3), (float(lip_u[8, 0]) + 0.02, float(lip_u[8, 1])), t["lips"], [("lipAp", enc), ("lipProt", enc), ("lipWidth", enc), ("labiodental", enc)], "right")
    label(ax, (right, 7.9), (tip[0] - 0.05, tip[1] + 0.02), t["tip"], [("ttX", enc), ("ttY", enc), ("ttCurl", enc)], "right")
    label(ax, (right, 4.4), (5.9, 3.5), t["jaw"], [("jaw", enc)], "right")
    label(ax, (3.1, 0.3), (centre[0] + 0.25, centre[1] - 1.25), t["body"], [("tbX", enc), ("tbY", enc)], "left", above=0.9)
    label(ax, (right, 1.2), (centre[0] + 2.0, centre[1] - 0.45), t["lateral"], [("lateral", enc)], "right")
    # how they move
    arrow(ax, (centre[0] - 0.85, centre[1] + 0.1), (centre[0] + 0.85, centre[1] + 0.1))
    arrow(ax, (centre[0], centre[1] - 0.75), (centre[0], centre[1] + 0.95))
    arrow(ax, (0.62, 0.95), (0.62, 2.45))
    arrow(ax, (-0.45, -2.45), (-0.45, -0.7), both=False, colour=BLUE)
    # where the values come from
    y = -4.0
    for x, colour, text in ((left + 0.16, enc, t["encoder"]), (2.6, f0, t["pitch"])):
        ax.plot([x], [y], marker="o", ms=2.6, color=colour, mec="none", clip_on=False)
        ax.text(x + 0.3, y, text, fontsize=6.0, color=INK, ha="left", va="center")
    y -= 0.8
    ax.plot([left + 0.16], [y], marker="o", ms=2.4, mfc="white", mec=MUTED, mew=0.5, clip_on=False)
    held = ax.text(left + 0.46, y, t["constant"], fontsize=6.0, color=INK, ha="left", va="center")
    fig.canvas.draw()
    ax.text(held.get_window_extent().transformed(ax.transData.inverted()).x1 + 0.25, y, "epilarynx ventricular dorsum closure", fontsize=6.0, color=MUTED, ha="left", va="center", family=MONO)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--font-dir", type=Path, default=None, help="where build.sh put the Noto fonts")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()
    japanese = a.font_dir is not None and (a.font_dir / "NotoSansJP-Regular.otf").exists()
    if japanese:
        for name in ("NotoSansJP-Regular.otf", "NotoSansJP-Bold.otf"):
            font_manager.fontManager.addfont(str(a.font_dir / name))
    plt.rcParams["axes.unicode_minus"] = False
    for lang in LABELS:
        # the Latin font as TrueType; the Japanese one's outlines are not TrueType, and only as
        # Type 3 does the figure keep its glyphs when the report's PDF is made
        plt.rcParams["pdf.fonttype"] = 3 if lang == "ja" and japanese else 42
        plt.rcParams["font.family"] = "Noto Sans JP" if lang == "ja" and japanese else "DejaVu Sans"
        figure(lang, a.out / f"fig_data_{lang}.pdf")
        control(lang, a.out / f"fig_control_{lang}.pdf")
        tract(lang, a.out / f"fig_tract_{lang}.pdf")
        print(a.out / f"fig_data_{lang}.pdf", a.out / f"fig_control_{lang}.pdf", a.out / f"fig_tract_{lang}.pdf")


if __name__ == "__main__":
    main()
