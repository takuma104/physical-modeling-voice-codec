"""Does a render's voice pass through a nose as the speaker's does? Harmonics by kind of phoneme.

A listener heard the women's renders as speech with a blocked nose. What a nose adds to a voice
is a low resonance, near 250-300 Hz: a nasal (m, n, the moraic nasal) is little but its first
harmonic at a woman's pitch, and a vowel beside a nasal has more of its first harmonic than one
away from it. This sets the renders against the recordings there.

usage: python -m pmvc.nasals [dir] [--variants enci,encis] [--split test] [--name nasals]

For the recordings of a split that `pmvc.align write` has aligned and each variant's renders of
them, over the middle frames of every phoneme (its first and last two frames are transitions),
with the recording's pitch (<dir>/pitch) saying where the harmonics lie:
  h2, h3   the second and the third harmonic against the first (dB) in vowels away from a nasal
  beside   the second harmonic against the first in vowels beside a nasal
  level    the nasals' level against that of the sentence's vowels away from a nasal (dB)
  nasal    the second harmonic against the first in nasals
Each is a sentence's mean, averaged over the women's sentences and over the men's (the vowels'
over the five vowels, so that each counts the same). The recordings' values are printed, and
for each variant its differences from them. Writes <dir>/<name>.json.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import stft

from .audio import FRAME_RATE, load
from .encoder import LEAD, ids_in
from .paths import ROOT

SR: int = 16000
FFT: int = 1024
VOWELS: str = "aiueo"
NASALS: set[str] = {"m", "n", "N", "ny", "my"}
# a harmonic is the strongest bin within this of its place (Hz)
NEAR: float = 20.0
# frames at each end of a phoneme that are left out
EDGE: int = 2
KEYS: tuple[str, ...] = ("h2", "h3", "beside", "level", "nasal")


def kinds(phones: list[str], frames: list[int]) -> NDArray[np.str_]:
    """For every frame of a recording (its own clock): the vowel it is the middle of, lower
    case away from a nasal and upper case beside one; "n" in the middle of a nasal; "." else."""
    out: list[str] = []
    for k, (p, n) in enumerate(zip(phones, frames)):
        beside = any(q in NASALS for q in phones[max(0, k - 1) : k + 2])
        kind = "n" if p in NASALS else (p.upper() if beside else p) if p in VOWELS else "."
        out += [kind if EDGE <= i < n - EDGE else "." for i in range(n)]
    return np.array(out)[round(LEAD * FRAME_RATE) :]


def measures(x: NDArray[np.floating], kind: NDArray[np.str_], f0: NDArray[np.floating]) -> dict[str, float]:
    """A clip on its recording's clock (16 kHz): the sentence's values of KEYS (nan where it has no such frames)."""
    _, _, s = stft(x, SR, nperseg=FFT, noverlap=FFT - SR // FRAME_RATE, boundary="zeros", padded=True)
    power = np.abs(s.T) ** 2  # frame j is centred j frames in
    hz = np.fft.rfftfreq(FFT, 1 / SR)
    rows: dict[str, list[NDArray[np.float64]]] = {}
    loud: dict[str, list[float]] = {"vowel": [], "nasal": []}
    for j in range(min(len(power), len(kind), len(f0))):
        if kind[j] == "." or f0[j] < 60:
            continue
        h = np.array([power[j][(hz > q * f0[j] - NEAR) & (hz < q * f0[j] + NEAR)].max() for q in (1, 2, 3)])
        rows.setdefault(str(kind[j]), []).append(10 * np.log10(h[1:] / h[0] + 1e-12))
        if kind[j] == "n" or kind[j] in VOWELS:
            loud["nasal" if kind[j] == "n" else "vowel"].append(float(power[j].sum()))
    mean = {k: np.mean(v, axis=0) for k, v in rows.items()}
    away = [mean[v] for v in VOWELS if v in mean]
    beside = [mean[v.upper()][0] for v in VOWELS if v.upper() in mean]
    return {
        "h2": float(np.mean([m[0] for m in away])) if away else np.nan,
        "h3": float(np.mean([m[1] for m in away])) if away else np.nan,
        "beside": float(np.mean(beside)) if beside else np.nan,
        "level": 10 * float(np.log10(np.mean(loud["nasal"]) / np.mean(loud["vowel"]))) if loud["nasal"] and loud["vowel"] else np.nan,
        "nasal": float(mean["n"][0]) if "n" in mean else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "jvs")
    ap.add_argument("--variants", type=str, default="enci")
    ap.add_argument("--split", choices=["test", "valid", "train", "all"], default="test")
    ap.add_argument("--name", type=str, default="nasals")
    a = ap.parse_args()
    d: Path = a.dir
    variants = a.variants.split(",")
    aligned = json.loads((d / "aligned.json").read_text())
    metas = [json.loads((d / "meta" / f"{i}.json").read_text()) for i in ids_in(d, a.split)]
    metas = [m for m in metas if m["id"] in aligned and (d / "pitch" / f"{m['id']}.npy").exists() and all((d / v / f"{m['id']}.wav").exists() for v in ("real", *variants))]
    got: dict[str, dict[str, list[dict[str, float]]]] = {}
    for m in metas:
        kind = kinds(aligned[m["id"]]["p"], aligned[m["id"]]["frames"])
        f0 = np.load(d / "pitch" / f"{m['id']}.npy")
        for v in ("real", *variants):
            x = load(d / v / f"{m['id']}.wav", SR)[0]
            # a render runs LEAD behind its recording
            got.setdefault(m["sex"], {}).setdefault(v, []).append(measures(x if v == "real" else x[round(LEAD * SR) :], kind, f0))
    out: dict[str, Any] = {"split": a.split, "sexes": {}}
    for sex, by in sorted(got.items()):
        mean = {v: {k: float(np.nanmean([r[k] for r in rows])) for k in KEYS} for v, rows in by.items()}
        out["sexes"][sex] = {"n": len(by["real"]), **mean}
        print(f"{sex}: {len(by['real'])} sentences" + " " * 12 + "  ".join(f"{k:>7s}" for k in KEYS))
        print(f"  {'real':10s} dB           " + "  ".join(f"{mean['real'][k]:+7.1f}" for k in KEYS))
        for v in variants:
            print(f"  {v:10s} minus real   " + "  ".join(f"{mean[v][k] - mean['real'][k]:+7.1f}" for k in KEYS))
    (d / f"{a.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
