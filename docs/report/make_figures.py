"""Figures for the technical report (English and Japanese labels).

    uv run python docs/report/make_figures.py --font-dir docs/report/fonts

Numbers: `pmvc.judge`'s and `pmvc.natural`'s summaries on the 100 test sentences of
Hi-Fi-CAPTAIN's Japanese man. For the first figure, the text-to-speech models taught with fewer
recordings (README, step 9): out/ja/results_tts.json (variants tts, tts2000, tts200, tts50,
tts20, and enci and real for the two reference lines). For the second, the control (step 11):
out/ja/results_ctl.json (the track: tts20n, tts50n, tts200n, tts2000n, tts; the neural
decoders' latents: ctl<latent><n>; each decoder's best: enci, copy_<latent>).
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
AQUA, VIOLET = "#1baf7a", "#6250d6"

# the control: what the model writes → Whisper large-v3-turbo (% of phonemes wrong) and NISQA of
# the models taught with SIZES recordings, and of the decoder at its best
CONTROL: dict[str, dict[str, list[float] | float]] = {
    "track": {"large": [8.75, 3.16, 0.71, 0.27, 0.17], "quality": [4.06, 4.5, 4.77, 4.81, 4.78], "best": 4.75},
    "mel": {"large": [7.71, 2.76, 0.58, 0.04, 0.42], "quality": [2.57, 3.24, 3.53, 3.64, 3.68], "best": 4.45},
    "encodec": {"large": [6.57, 2.64, 0.67, 0.12, 0.08], "quality": [1.85, 2.1, 2.35, 3.27, 2.59], "best": 4.36},
    "mimi": {"large": [23.43, 11.25, 2.62, 0.33, 0.35], "quality": [2.62, 3.55, 3.5, 4.65, 3.31], "best": 4.81},
}
LOOKS: dict[str, tuple[str, str]] = {"track": (BLUE, "o"), "mel": (ORANGE, "s"), "encodec": (AQUA, "^"), "mimi": (VIOLET, "D")}

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
        "large_axis": "phonemes wrong (%)\nWhisper large-v3-turbo",
        "names": {"track": "control track (CHIMERA)", "mel": "mel spectrogram", "encodec": "EnCodec", "mimi": "Mimi"},
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
        "names": {"track": "制御列（CHIMERA）", "mel": "メルスペクトログラム", "encodec": "EnCodec", "mimi": "Mimi"},
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
        colour, marker = LOOKS[name]
        large, quality = series["large"], series["quality"]
        assert isinstance(large, list) and isinstance(quality, list)
        top.plot(SIZES[: len(large)], [max(v, 0.03) for v in large], color=colour, linewidth=1.5, marker=marker, markersize=4.2, label=t["names"][name])
        bottom.plot(SIZES[: len(quality)], quality, color=colour, linewidth=1.5, marker=marker, markersize=4.2)
        # the decoder at its best: a tick past the last point, in the series' colour
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
        print(a.out / f"fig_data_{lang}.pdf", a.out / f"fig_control_{lang}.pdf")


if __name__ == "__main__":
    main()
