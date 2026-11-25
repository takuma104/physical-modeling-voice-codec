"""Where are the tract's resonances? The engine's own transfer function over the vowels of a variant's tracks.

A render's spectral envelope is the source's spectrum shaped by the tract, and what a render
lacks of its recording (`pmvc.bench`'s common curve) could be either's. The engine has a
frequency-domain model of its airway (the simulation's elements, linearised), which says what
the tract alone does at any moment: `pmvc_engine.Voice.transfer`. Here a variant's tracks are
played, and in the middle of every vowel (the frames `pmvc.bands` takes) the transfer function
from the glottis to the radiated sound is read off, with no source in it.

usage: python -m pmvc.tract [dir] [--variant encv2] [--set epilarynx=1.0,...] [--per 4]
                            [--name tract_<variant>]

Writes <dir>/<name>.json: `freqs` (Hz), and for the women and for the men the mean level (dB,
against an arbitrary reference) over their vowel frames (`mean`), over each vowel's, and the
peaks and dips of the mean. `--set` holds channels of the tracks at a value in every frame (the
tube above the glottis narrower, say), to see what a setting does to the resonances before
anything is rendered. Needs <dir>/aligned.json (`pmvc.align write`).

On V1's renders this shows what the common curve is made of: the renders lack 7-15 dB where
the recordings have their third and fourth formants close together (men 2.5-3.4 kHz, women
3.3-4.6 kHz), and there the engine's transfer function has the valley between its third
resonance and a fourth that sits higher (men 3.5-3.9 kHz, women 4.4-5.0 kHz); above 5 kHz,
where the renders have too much, it stays high.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .audio import FRAME_RATE
from .bands import VOWELS, kinds
from .calibrate import make_voice
from .encoder import LEAD, ids_in
from .paths import ROOT

Array = NDArray[np.float64]

FREQS: Array = np.arange(100.0, 8001.0, 50.0)
# every STRIDE-th vowel frame is read (neighbouring frames say the same)
STRIDE: int = 2
SAMPLES: int = 48000 // FRAME_RATE


def read(track: dict[str, Any], held: dict[str, float], kind: NDArray[np.str_], phones: list[str]) -> tuple[Array, list[str]]:
    """The transfer function in the middle of the track's vowels (rows) and each row's vowel.
    `kind` and `phones` are on the recording's clock, which the track runs LEAD behind."""
    frames = np.array(track["frames"], dtype=np.float64)
    for key, x in held.items():
        frames[:, track["keys"].index(key)] = x
    v = make_voice(track, None)
    v.set_track(frames, 0)
    lead = round(LEAD * FRAME_RATE)
    rows, vowels = [], []
    for j in range(len(frames) - 1):
        v.render(SAMPLES)
        r = j - lead
        if 0 <= r < len(kind) and kind[r] == "v" and r % STRIDE == 0:
            rows.append(np.asarray(v.transfer(FREQS)))
            vowels.append(phones[r])
    return np.array(rows).reshape(-1, len(FREQS)), vowels


def turns(mean: Array) -> tuple[list[int], list[int]]:
    """Where a curve over FREQS has its peaks and its dips (Hz)."""
    inner = range(1, len(mean) - 1)
    return [int(FREQS[i]) for i in inner if mean[i - 1] < mean[i] > mean[i + 1]], [int(FREQS[i]) for i in inner if mean[i - 1] > mean[i] < mean[i + 1]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "jvs")
    ap.add_argument("--variant", type=str, default="encv2")
    ap.add_argument("--set", type=str, default="", help="channels held at a value in every frame: epilarynx=1.0,...")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--per", type=int, default=4, help="recordings of each speaker")
    ap.add_argument("--name", type=str, default="")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    d: Path = a.dir
    held = {w.split("=")[0]: float(w.split("=")[1]) for w in a.set.split(",")} if a.set else {}
    aligned: dict[str, Any] = json.loads((d / "aligned.json").read_text())
    metas = [json.loads((d / "meta" / f"{i}.json").read_text()) for i in ids_in(d, a.split)]
    count: dict[str, int] = {}
    kept: list[dict[str, Any]] = []
    for m in metas:
        if m["id"] in aligned and (d / a.variant / f"{m['id']}.tracks.json").exists():
            count[m["speaker"]] = count.get(m["speaker"], 0) + 1
            if count[m["speaker"]] <= a.per:
                kept.append(m)

    def one(m: dict[str, Any]) -> tuple[Array, list[str]]:
        phones: list[str] = []
        for p, n in zip(aligned[m["id"]]["p"], aligned[m["id"]]["frames"]):
            phones += [p] * n
        track = json.loads((d / a.variant / f"{m['id']}.tracks.json").read_text())
        return read(track, held, kinds(aligned[m["id"]]["p"], aligned[m["id"]]["frames"]), phones[round(LEAD * FRAME_RATE) :])

    with ThreadPoolExecutor(a.jobs) as pool:
        done = list(pool.map(one, kept))
    out: dict[str, Any] = {"variant": a.variant, "set": held, "clips": len(kept), "freqs": FREQS.tolist(), "sexes": {}}
    for sex in ("F", "M"):
        mine = [(rows, vowels) for (rows, vowels), m in zip(done, kept) if m["sex"] == sex and len(rows)]
        if not mine:
            continue
        rows = np.concatenate([r for r, _ in mine])
        said = np.array([x for _, vowels in mine for x in vowels])
        mean = rows.mean(axis=0)
        peaks, dips = turns(mean)
        out["sexes"][sex] = {
            "frames": int(len(rows)), "mean": np.round(mean, 2).tolist(), "peaks": peaks, "dips": dips,
            "vowels": {v: np.round(rows[said == v].mean(axis=0), 2).tolist() for v in VOWELS if (said == v).sum() >= 20},
        }  # fmt: skip
        print(f"{sex}: {len(rows)} vowel frames; the mean has its peaks at {peaks} Hz and its dips at {dips} Hz")
    name = a.name or f"tract_{a.variant}" + "".join(f"_{k}{x:g}" for k, x in held.items())
    (d / f"{name}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
