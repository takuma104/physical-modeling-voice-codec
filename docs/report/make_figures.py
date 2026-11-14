"""Figures for the technical report (English and Japanese labels).

    uv run python docs/report/make_figures.py --font-dir docs/report/fonts

Numbers: `pmvc.judge`'s and `pmvc.natural`'s summaries for the text-to-speech models taught with
fewer recordings (README, step 9): out/ja/results_tts.json, the 100 test sentences of
Hi-Fi-CAPTAIN's Japanese man (variants tts, tts2000, tts200, tts50, tts20, and enci and real
for the two reference lines).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

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

LABELS: dict[str, dict[str, str]] = {
    "en": {
        "x": "recordings the model is taught with",
        "error": "phonemes wrong (%)",
        "quality": "predicted quality (NISQA)",
        "small": "Whisper small",
        "large": "Whisper large-v3-turbo",
        "encoder": "encoder I",
        "recording": "recordings",
        "minutes": ["1.5 min", "4 min", "15 min", "2.6 h", "24 h"],
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
        print(a.out / f"fig_data_{lang}.pdf")


if __name__ == "__main__":
    main()
