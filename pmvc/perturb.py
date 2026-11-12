"""Perturbation test: how does the voice break when its control track is wrong?

A control track (21 values, 100 times a second: 17 articulator positions, glottal abduction and
closure, breath, pitch) is a complete description of an utterance for CHIMERA: `pmvc-synth
drive` plays it back with no planner. Here the tracks of `pmvc.transplant`'s `f0` renders (the
most intelligible it makes) are damaged the way a codec or a generative model damages a
latent, and played back.

usage: python -m pmvc.perturb [dir] [--base f0] [--ids FA01_01,...] [--n 24] [--tag T]
                              [--conds drive,coded_4bit_25hz,...] [--jobs N]

Writes one directory of <id>.wav per condition under <dir> (default out/transplant):
  drive                  the track unchanged: what the round trip through a track costs
  white_snr<dB>          independent Gaussian noise on every frame and channel
  smooth_snr<dB>         Gaussian noise that varies slowly (below 4 Hz): an error that drifts
  quant_<bits>bit        each channel rounded to 2^bits levels over its range
  rate_<Hz>hz            the track kept at a lower frame rate, interpolated back
  coded_<bits>bit_<Hz>hz both: the lower frame rate's values rounded, then interpolated back
                         (only with `--conds`)

Noise is scaled per channel to a signal-to-noise ratio: how far the channel moves within an
utterance (its standard deviation over speech frames, pooled over all the tracks in <dir>),
over the noise's. Pitch is perturbed on a log
scale. Channels that never move in these tracks (the epilarynx, glottal closure) get no noise.

`--tag T` keeps a second test apart from the first: the directories are <T>_<condition> and
the listing <dir>/perturb_<T>.json (else perturb.json).
"""

import argparse
import json
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .audio import FRAME_RATE, Array
from .paths import ROOT
from .synth import drive

SNRS: tuple[int, ...] = (20, 10, 5, 0)
BITS: tuple[int, ...] = (5, 4, 3, 2)
RATES: tuple[float, ...] = (50.0, 25.0, 12.5)
# upper band edge of the slowly varying noise (Hz)
SMOOTH_HZ: float = 4.0


def conditions() -> list[str]:
    return (
        ["drive"]
        + [f"{kind}_snr{s}" for kind in ("white", "smooth") for s in SNRS]
        + [f"quant_{b}bit" for b in BITS]
        + [f"rate_{str(r).rstrip('0').rstrip('.').replace('.', 'p')}hz" for r in RATES]
    )


class Scale:
    """Each channel's spread and range in a set of tracks (pitch in log2). The spread is how far
    a channel moves within an utterance, over its speech frames and with each utterance's mean
    removed, so a man's and a woman's pitch levels do not count as signal. The range, which the
    quantiser's levels are laid over, is over every frame, the voice at rest included: over
    speech alone it starts at a breath of 0.75, and a quantised track then has the voice sound
    through what should be its silences."""

    def __init__(self, tracks: list[dict[str, Any]]) -> None:
        self.keys: list[str] = tracks[0]["keys"]
        self.f0: int = self.keys.index("f0")
        breath = self.keys.index("breath")
        frames = [self.to_domain(np.array(t["frames"])) for t in tracks]
        self.std: Array = np.concatenate([f[f[:, breath] > 0.5] - f[f[:, breath] > 0.5].mean(axis=0) for f in frames]).std(axis=0)
        pooled = np.concatenate(frames)
        self.lo: Array = np.percentile(pooled, 0.5, axis=0)
        self.hi: Array = np.percentile(pooled, 99.5, axis=0)

    def to_domain(self, x: Array) -> Array:
        y = x.copy()
        y[:, self.f0] = np.log2(y[:, self.f0])
        return y

    def from_domain(self, y: Array) -> Array:
        x = y.copy()
        x[:, self.f0] = 2.0 ** x[:, self.f0]
        return x


def at_rate(y: Array, hz: float, levels: int | None, scale: Scale) -> Array:
    """`y` kept at `hz` values a second (the mean of each block of frames, placed at the block's
    centre), each rounded to `levels` levels over its channel's range if given, and
    interpolated back."""
    step = round(FRAME_RATE / hz)
    n = len(y)
    centres = np.arange(0, n, step) + 0.5 * (min(step, n) - 1)
    means = np.stack([y[i : i + step].mean(axis=0) for i in range(0, n, step)])
    if levels is not None:
        span = np.where(scale.hi > scale.lo, scale.hi - scale.lo, 1.0)
        means = scale.lo + np.round(np.clip((means - scale.lo) / span, 0.0, 1.0) * levels) / levels * span
    return np.stack([np.interp(np.arange(n), centres, means[:, c]) for c in range(y.shape[1])], axis=1)


def perturb(x: Array, cond: str, scale: Scale, rng: np.random.Generator) -> Array:
    """The track `x` (frames × channels) under condition `cond`."""
    if cond == "drive":
        return x
    y = scale.to_domain(x)
    kind, arg = cond.split("_", 1)
    if kind in ("white", "smooth"):
        noise = rng.standard_normal(y.shape)
        if kind == "smooth":
            sos = butter(2, SMOOTH_HZ, btype="low", fs=FRAME_RATE, output="sos")
            noise = sosfiltfilt(sos, noise, axis=0)
            noise /= noise.std(axis=0, keepdims=True)
        y = y + noise * scale.std * 10.0 ** (-float(arg[3:]) / 20.0)
    elif kind == "quant":
        levels = 2 ** int(arg[:-3]) - 1
        span = np.where(scale.hi > scale.lo, scale.hi - scale.lo, 1.0)
        y = scale.lo + np.round(np.clip((y - scale.lo) / span, 0.0, 1.0) * levels) / levels * span
    elif kind == "rate":
        y = at_rate(y, float(arg[:-2].replace("p", ".")), None, scale)
    elif kind == "coded":
        # what a plain codec would send: `coded_<bits>bit_<Hz>hz`
        bits, hz = arg.split("_")
        y = at_rate(y, float(hz[:-2].replace("p", ".")), 2 ** int(bits[:-3]) - 1, scale)
    else:
        raise ValueError(cond)
    out = scale.from_domain(y)
    # what cannot be negative stays so (the engine clamps the articulators to their own ranges)
    for key in ("abduction", "closure", "breath"):
        c = scale.keys.index(key)
        out[:, c] = np.clip(out[:, c], 0.0, None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "transplant")
    ap.add_argument("--base", type=str, default="f0", help="the variant whose control tracks are perturbed")
    ap.add_argument("--ids", type=str, default="")
    ap.add_argument("--n", type=int, default=24, help="how many recordings (the first n ids, alternating sexes)")
    ap.add_argument("--tag", type=str, default="", help="a prefix for the conditions' directories and the listing")
    ap.add_argument("--conds", type=str, default="", help="comma-separated conditions instead of the standard set")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    d: Path = a.dir

    def name(cond: str) -> str:
        return f"{a.tag}_{cond}" if a.tag else cond

    paths = sorted((d / a.base).glob("*.tracks.json"))
    scale = Scale([json.loads(p.read_text()) for p in paths])
    if a.ids:
        ids = a.ids.split(",")
    else:
        all_ids = [p.name.split(".")[0] for p in paths]
        women, men = [i for i in all_ids if i[0] == "F"], [i for i in all_ids if i[0] == "M"]
        ids = [i for pair in zip(women, men) for i in pair][: a.n]
    conds = a.conds.split(",") if a.conds else conditions()
    for c in conds:
        (d / name(c)).mkdir(exist_ok=True)

    def one(job: tuple[str, str]) -> None:
        uid, cond = job
        track = json.loads((d / a.base / f"{uid}.tracks.json").read_text())
        # the same noise for a recording at every level, so only its size changes
        rng = np.random.default_rng(zlib.crc32(f"{uid}/{cond.split('_')[0]}".encode()))
        x = perturb(np.array(track["frames"]), cond, scale, rng)
        track["frames"] = np.round(x, 5).tolist()
        drive(track, d / name(cond) / f"{uid}.wav")

    with ThreadPoolExecutor(a.jobs) as pool:
        list(pool.map(one, [(uid, c) for uid in ids for c in conds]))
    (d / (f"perturb_{a.tag}.json" if a.tag else "perturb.json")).write_text(
        json.dumps({"ids": ids, "conditions": [name(c) for c in conds], "keys": scale.keys, "std": scale.std.tolist()}, indent=1)
    )
    print(f"{len(ids)} recordings × {len(conds)} conditions → {d}")
    print("channel spread within an utterance, over speech frames (pitch in octaves):")
    print("  " + "  ".join(f"{k} {s:.3f}" for k, s in zip(scale.keys, scale.std)))


if __name__ == "__main__":
    main()
