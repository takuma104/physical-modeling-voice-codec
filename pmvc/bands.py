"""Is a render's voice as bright as the speaker's? Vowels and fricatives against the recording's, band by band.

A listener missed something in the women's renders and could only call it brilliance. A voice's
brilliance is what it has between some 2 and 5 kHz, and a consonant's what it has above that;
this sets the renders against the recordings there, and says whether a band is made of harmonics
(the voice) or of noise (the breath).

usage: python -m pmvc.bands [dir] [--variants enci,encib] [--split test] [--name bands]

For the recordings of a split that `pmvc.align write` has aligned and each variant's renders of
them, at 24 kHz, with the recording's pitch (<dir>/pitch) saying where the harmonics lie:
  vowels      the level of each band between EDGES over the middle frames of the vowels (their
              first and last two frames are transitions), against the vowels' whole level (dB)
  h2, h3      the second and the third harmonic against the first in those frames (dB)
  harmonic    there, in each band: the level at the harmonics (within a quarter of the pitch)
              against the level between them (dB)
  fricative   the voiceless fricatives' level against the sentence's vowels (dB)
  fricatives  and their level in each band, against the vowels' whole level
Each is a sentence's value, averaged over the women's sentences and over the men's. The
recordings' values are printed, and for each variant its differences from them. Writes
<dir>/<name>.json.
"""

import argparse
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import stft

from .audio import FRAME_RATE, load
from .encoder import LEAD, ids_in
from .paths import ROOT

SR: int = 24000
FFT: int = 1024
EDGES: tuple[int, ...] = (80, 200, 400, 630, 1000, 1600, 2500, 4000, 5500, 7500, 9500, 12000)
VOWELS: str = "aiueo"
FRICATIVES: set[str] = {"s", "sh", "h", "hy", "f"}
# a harmonic is the strongest bin within this of its place (Hz)
NEAR: float = 20.0
# frames at each end of a vowel that are left out
EDGE: int = 2
KEYS: tuple[str, ...] = ("vowels", "h2", "h3", "harmonic", "fricative", "fricatives")


def kinds(phones: list[str], frames: list[int]) -> NDArray[np.str_]:
    """For every frame of a recording (its own clock): "v" in the middle of a vowel, "s" in a
    voiceless fricative, "." else."""
    out: list[str] = []
    for p, n in zip(phones, frames):
        out += ["v" if EDGE <= i < n - EDGE else "." for i in range(n)] if p in VOWELS else ["s" if p in FRICATIVES else "."] * n
    return np.array(out)[round(LEAD * FRAME_RATE) :]


def measures(x: NDArray[np.floating], kind: NDArray[np.str_], f0: NDArray[np.floating]) -> dict[str, Any] | None:
    """A clip on its recording's clock (24 kHz): the sentence's values of KEYS (None if it has too few vowel frames)."""
    _, _, s = stft(x, SR, nperseg=FFT, noverlap=FFT - SR // FRAME_RATE, boundary="zeros", padded=True)
    power = np.abs(s.T) ** 2  # frame j is centred j frames in
    n = min(len(power), len(kind), len(f0))
    power, kind, f0 = power[:n], kind[:n], f0[:n]
    hz = np.fft.rfftfreq(FFT, 1 / SR)
    bands = [(hz >= lo) & (hz < hi) for lo, hi in pairwise(EDGES)]
    vowel = (kind == "v") & (f0 > 60)
    if vowel.sum() < 10:
        return None
    whole = power[vowel].sum(axis=1).mean()
    low: list[NDArray[np.float64]] = []
    # the level at the harmonics and between them, summed over the frames that have both in a band
    at, between = np.zeros(len(bands)), np.zeros(len(bands))
    for j in np.flatnonzero(vowel):
        h = np.array([power[j][(hz > q * f0[j] - NEAR) & (hz < q * f0[j] + NEAR)].max() for q in (1, 2, 3)])
        low.append(10 * np.log10(h[1:] / h[0] + 1e-12))
        near = np.abs(hz / f0[j] - np.round(hz / f0[j])) < 0.25
        for k, b in enumerate(bands):
            if (b & near).any() and (b & ~near).any():
                at[k] += power[j][b & near].mean()
                between[k] += power[j][b & ~near].mean()
    fric = power[kind == "s"]
    return {
        "vowels": [10 * float(np.log10(power[vowel][:, b].sum(axis=1).mean() / whole + 1e-12)) for b in bands],
        "h2": float(np.mean(low, axis=0)[0]),
        "h3": float(np.mean(low, axis=0)[1]),
        "harmonic": [10 * float(np.log10(x / y)) if y > 0 else np.nan for x, y in zip(at, between)],
        "fricative": 10 * float(np.log10(fric.sum(axis=1).mean() / whole)) if len(fric) >= 3 else np.nan,
        "fricatives": [10 * float(np.log10(fric[:, b].sum(axis=1).mean() / whole + 1e-12)) if len(fric) >= 3 else np.nan for b in bands],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "jvs")
    ap.add_argument("--variants", type=str, default="enci")
    ap.add_argument("--split", choices=["test", "valid", "train", "all"], default="test")
    ap.add_argument("--name", type=str, default="bands")
    a = ap.parse_args()
    d: Path = a.dir
    variants = a.variants.split(",")
    aligned = json.loads((d / "aligned.json").read_text())
    metas = [json.loads((d / "meta" / f"{i}.json").read_text()) for i in ids_in(d, a.split)]
    metas = [m for m in metas if m["id"] in aligned and (d / "pitch" / f"{m['id']}.npy").exists() and all((d / v / f"{m['id']}.wav").exists() for v in ("real", *variants))]
    got: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for m in metas:
        kind = kinds(aligned[m["id"]]["p"], aligned[m["id"]]["frames"])
        f0 = np.load(d / "pitch" / f"{m['id']}.npy")
        rows = {v: measures(load(d / v / f"{m['id']}.wav", SR)[0][0 if v == "real" else round(LEAD * SR) :], kind, f0) for v in ("real", *variants)}  # a render runs LEAD behind its recording
        if all(r is not None for r in rows.values()):
            for v, r in rows.items():
                got.setdefault(m["sex"], {}).setdefault(v, []).append(r)  # type: ignore[arg-type]
    out: dict[str, Any] = {"split": a.split, "edges": list(EDGES), "sexes": {}}
    head = " ".join(f"{lo:>5d}" for lo in EDGES[:-1])
    for sex, by in sorted(got.items()):
        mean = {v: {k: np.nanmean([r[k] for r in rows], axis=0) for k in KEYS} for v, rows in by.items()}
        out["sexes"][sex] = {"n": len(by["real"]), **{v: {k: np.round(x, 2).tolist() for k, x in m.items()} for v, m in mean.items()}}
        print(f"{sex}: {len(by['real'])} sentences; bands from (Hz)   {head} |    h2    h3 | fricative")
        for key in ("vowels", "harmonic", "fricatives"):
            tail = {"vowels": ("h2", "h3"), "harmonic": (), "fricatives": ("fricative",)}[key]
            for v in ("real", *variants):
                off = 0.0 if v == "real" else mean["real"]
                row = mean[v][key] - (off[key] if isinstance(off, dict) else 0.0)
                rest = [float(mean[v][k] - (off[k] if isinstance(off, dict) else 0.0)) for k in tail]
                print(f"  {key:10s} {v:8s} {'dB' if v == 'real' else 'minus real':10s}   " + " ".join(f"{x:+5.1f}" for x in row) + ("".join(" | " + " ".join(f"{x:+5.1f}" for x in rest)) if rest else ""))
    (d / f"{a.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
