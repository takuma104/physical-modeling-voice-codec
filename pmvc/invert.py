"""Could the engine give a recording's envelope if only its tract stood right? Postures searched frame by frame.

A render's envelope is far from its recording's in every vowel frame (`pmvc.bench`: some 10 dB
rms), and most of that is neither common to all renders nor a speaker's or a vowel's own: it
changes from frame to frame (`pmvc.ladder`). Is that because the tracks put the articulators in
the wrong places, or because no place of the articulators gives this engine's tract the
recording's envelope? The engine's own frequency-domain model answers without rendering: the
tract's transfer function in a posture (`pmvc_engine.Voice.transfers`), which predicts what a
change of the tract does to a render's envelope (`pmvc.tract`). A render's envelope in a frame
is its source's spectrum shaped by the tract as it stood there; with the tract in another
posture, or of another form, it would be the same envelope with the two transfer functions'
difference added. So for each vowel frame the posture is searched (each frame alone, from the
track's own posture, within the articulators' ranges) whose predicted envelope is nearest the
recording's, in shape (the level is free; with `--tilt`, a slope over frequency too: what the
source could give).

usage: python -m pmvc.invert [dir] [--variant encv2] [--per 3] [--stride 3] [--larynx length=1.3,...]
                             [--form fossaLength=1.6,...] [--tilt] [--stay 0.0] [--fix velum,...]
                             [--evals 640] [--name invert]

What is left is a bound, and a generous one: nothing asks the postures of neighbouring frames to
be near each other, or a posture to be one a speaker would take (`--stay` weighs the distance
moved from the track's posture, in units of the articulators' ranges, against the error in dB).
`--larynx` and `--form` give the tract another form (`pmvc_voice::Player`'s `set_larynx` and
`set_form`): what the envelope's error would be with the track's own postures in that form,
and what could be reached in it. Prints and writes to <dir>/<name>.json, for each sex, the
envelope's error (`pmvc.bench`'s terms) as it is, in the form given, and reached; and how far
the articulators went. The postures the tracks' renders pass through are kept in
<dir>/<variant>/<id>.poses.npy.
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
from .bench import CURVE, SEXES, STEP, TOP, clip, shape, vowel_frames, vowel_rows
from .calibrate import make_voice
from .encoder import LEAD, ids_in
from .paths import ROOT
from .tract import FREQS, SAMPLES

Array = NDArray[np.float64]

# CHIMERA's articulators, in its order, and their ranges (`control/articulators.rs`)
NAMES: tuple[str, ...] = ("jaw", "tbX", "tbY", "ttX", "ttY", "ttCurl", "root", "lipAp", "lipProt", "lipWidth", "labiodental", "velum", "larynx", "epilarynx", "ventricular", "lateral", "dorsum")
LOW: Array = np.array([-0.1, 1.0, 4.8, 3.5, 5.0, 0.0, -1.0, -0.4, -0.2, 0.0, 0.0, -0.25, -1.6, 0.0, 0.0, 0.0, 0.0])
HIGH: Array = np.array([2.6, 4.8, 9.2, 8.0, 10.8, 1.0, 1.0, 2.4, 1.5, 1.0, 1.0, 1.4, 1.6, 1.0, 1.0, 1.0, 1.0])
# the articulators that no vowel is made with stay where the track has them
FIXED: tuple[str, ...] = ("labiodental", "ventricular", "lateral")
# a generation of the search: candidates a frame, and how many of the best the next one is drawn around
CANDIDATES: int = 16
BEST: int = 4
# the spread the search starts with, as a share of each articulator's range
SPREAD: float = 0.08
# a transfer function is not followed below this (dB under its frame's highest band): a closed tract
DEPTH: float = 60.0
# bands of STEP (as `pmvc.bench` takes the envelope's error in) as means over FREQS
CENTRES: Array = np.arange(0.0, TOP, STEP) + STEP / 2
BANDS: Array = np.stack([((FREQS >= c - STEP / 2) & (FREQS < c + STEP / 2)).astype(float) for c in CENTRES])
BANDS = BANDS / BANDS.sum(axis=1, keepdims=True)


def poses_of(track: dict[str, Any]) -> Array:
    """The articulators at the start of each frame of a track, as its render has them (frames, 17)."""
    v = make_voice(track, None)
    frames = np.array(track["frames"], dtype=np.float64)
    v.set_track(frames, 0)
    out = []
    for _ in range(len(frames) - 1):
        out.append(np.asarray(v.pose))
        v.render(SAMPLES)
    return np.array(out)


def in_bands(levels: Array) -> Array:
    """Transfer functions (…, FREQS) in the bands (…, bands), none more than DEPTH under its highest."""
    b = levels @ BANDS.T
    return np.maximum(b, b.max(axis=-1, keepdims=True) - DEPTH)


def shaped(a: Array, tilt: bool) -> Array:
    """`a` (…, bands) without its level, and with `tilt` without its slope over frequency."""
    a = a - a.mean(axis=-1, keepdims=True)
    if tilt:
        u = CENTRES - CENTRES.mean()
        a = a - (a @ u)[..., None] / (u @ u) * u
    return a


class Clip:
    """A recording's vowel frames: what its render's envelope lacks there (dB, in bands), the
    postures the render stood in, and the transfer functions of those postures in V1's form."""

    def __init__(self, d: Path, variant: str, m: dict[str, Any], pitch: float, aligned: dict[str, Any], stride: int) -> None:
        self.meta = m
        uid = m["id"]
        self.track: dict[str, Any] = json.loads((d / variant / f"{uid}.tracks.json").read_text())
        kept = d / variant / f"{uid}.poses.npy"
        if kept.exists():
            poses = np.load(kept)
        else:
            poses = poses_of(self.track)
            np.save(kept, poses.astype(np.float32))
        kind, f0, phones = vowel_frames(d, uid, aligned)
        rows, vowels, at = vowel_rows(clip(d, "real", uid), clip(d, variant, uid), pitch, kind, f0, phones)
        lead = round(LEAD * FRAME_RATE)
        keep = [i for i in range(0, len(at), stride) if at[i] + lead < len(poses)]
        self.lack: Array = rows[keep]
        self.vowels = [vowels[i] for i in keep]
        self.poses: Array = poses[[at[i] + lead for i in keep]].astype(np.float64)
        self.own: Array = in_bands(np.asarray(make_voice(self.track, None).transfers(self.poses, FREQS))) if len(keep) else np.zeros((0, len(CENTRES)))


def search(c: Clip, track: dict[str, Any], free: NDArray[np.bool_], tilt: bool, stay: float, evals: int, seed: int) -> tuple[Array, Array, Array]:
    """For each of a clip's frames: what its envelope would lack with the track's posture in
    this form, what it lacks in the best posture found, and that posture."""
    v = make_voice(track, None)
    n = len(c.poses)
    if n == 0:
        return c.lack, c.lack, c.poses
    span = (HIGH - LOW)[free]

    def lacking(poses: Array) -> Array:
        return c.lack[:, None, :] - (in_bands(np.asarray(v.transfers(poses.reshape(-1, len(NAMES)), FREQS))).reshape(n, -1, len(CENTRES)) - c.own[:, None, :])

    def cost(poses: Array, lack: Array) -> Array:
        moved = (((poses - c.poses[:, None, :])[:, :, free] / span) ** 2).mean(axis=-1)
        return np.sqrt((shaped(lack, tilt) ** 2).mean(axis=-1)) + stay * np.sqrt(moved)

    first = lacking(c.poses[:, None, :])
    best_pose, best_lack, best_cost = c.poses.copy(), first[:, 0].copy(), cost(c.poses[:, None, :], first)[:, 0]
    rng = np.random.default_rng(seed)
    mean, sigma = c.poses[:, free].copy(), np.tile(SPREAD * span, (n, 1))
    for _ in range(max(1, evals // CANDIDATES)):
        drawn = np.clip(mean[:, None, :] + sigma[:, None, :] * rng.standard_normal((n, CANDIDATES, int(free.sum()))), LOW[free], HIGH[free])
        poses = np.repeat(c.poses[:, None, :], CANDIDATES, axis=1)
        poses[:, :, free] = drawn
        lack = lacking(poses)
        costs = cost(poses, lack)
        order = np.argsort(costs, axis=1)
        top = order[:, 0]
        rows = np.arange(n)
        better = costs[rows, top] < best_cost
        best_pose[better], best_lack[better], best_cost[better] = poses[rows, top][better], lack[rows, top][better], costs[rows, top][better]
        elite = np.take_along_axis(drawn, order[:, :BEST, None], axis=1)
        # the next generation is drawn around the best few and the best so far
        mean = 0.5 * elite.mean(axis=1) + 0.5 * best_pose[:, free]
        sigma = np.maximum(0.7 * sigma + 0.3 * 1.5 * elite.std(axis=1), 0.004 * span)
    return first[:, 0], best_lack, best_pose


def pairs(text: str) -> dict[str, float]:
    return {w.split("=")[0]: float(w.split("=")[1]) for w in text.split(",")} if text else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "jvs")
    ap.add_argument("--variant", type=str, default="encv2")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--per", type=int, default=3, help="recordings of each speaker")
    ap.add_argument("--stride", type=int, default=3, help="every so-many-th vowel frame of a recording")
    ap.add_argument("--larynx", type=pairs, default={}, help="the tube above the glottis: length=1.3,narrow=0.3,...")
    ap.add_argument("--form", type=pairs, default={}, help="the tract's form: fossaLength=1.6,pharynxAlpha=2.4,...")
    ap.add_argument("--tilt", action="store_true", help="a slope of the envelope over frequency is free too")
    ap.add_argument("--stay", type=float, default=0.0, help="what moving every free articulator across its whole range costs (dB)")
    ap.add_argument("--fix", type=str, default="", help="more articulators that stay where the track has them")
    ap.add_argument("--evals", type=int, default=640, help="postures tried for each frame")
    ap.add_argument("--name", type=str, default="invert")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    d: Path = a.dir
    aligned: dict[str, Any] = json.loads((d / "aligned.json").read_text())
    metas = [json.loads((d / "meta" / f"{i}.json").read_text()) for i in ids_in(d, a.split)]
    metas = [m for m in metas if m["id"] in aligned and all((d / v / f"{m['id']}.wav").exists() for v in ("real", a.variant)) and (d / a.variant / f"{m['id']}.tracks.json").exists()]
    pitch = {s: float(np.median([m["f0Median"] for m in metas if m["speaker"] == s])) for s in {m["speaker"] for m in metas}}
    count: dict[str, int] = {}
    kept: list[dict[str, Any]] = []
    for m in metas:
        count[m["speaker"]] = count.get(m["speaker"], 0) + 1
        if count[m["speaker"]] <= a.per:
            kept.append(m)
    fixed = (*FIXED, *filter(None, a.fix.split(",")))
    free = np.array([k not in fixed for k in NAMES])
    with ThreadPoolExecutor(a.jobs) as pool:
        clips = list(pool.map(lambda m: Clip(d, a.variant, m, pitch[m["speaker"]], aligned, a.stride), kept))
        found = list(pool.map(lambda job: search(job[1], {**job[1].track, "larynx": {**job[1].track.get("larynx", {}), **a.larynx}, "form": {**job[1].track.get("form", {}), **a.form}}, free, a.tilt, a.stay, a.evals, job[0]), enumerate(clips)))
    out: dict[str, Any] = {"variant": a.variant, "clips": len(kept), "larynx": a.larynx, "form": a.form, "tilt": a.tilt, "stay": a.stay, "evals": a.evals, "free": [k for k in NAMES if k not in fixed], "sexes": {}}
    print(f"{len(kept)} recordings, {sum(len(c.lack) for c in clips)} vowel frames; the envelope's error (dB rms; of which common to the sex, a speaker's, a vowel's, the rest)")
    for sex in SEXES:
        mine = [(c, f) for c, f in zip(clips, found) if c.meta["sex"] == sex and len(c.lack)]
        if not mine:
            continue
        speakers = [c.meta["speaker"] for c, _ in mine for _ in c.vowels]
        vowels = [x for c, _ in mine for x in c.vowels]
        cell: dict[str, Any] = {}
        for k, rows in (("is", [c.lack for c, _ in mine]), ("form", [f[0] for _, f in mine]), ("reached", [f[1] for _, f in mine])):
            lack = np.concatenate(rows)
            if a.tilt:  # the slope is the source's to give: it is no part of what is left
                lack = shaped(lack, True)
            cell[k] = shape(lack, speakers, vowels)
            s = cell[k]
            # where the error is: its rms in each band of the curve, all of it and what changes from frame to frame
            level = shaped(lack, False)
            moving = level - level.mean(axis=0)
            for name, part in (("byBand", level), ("movingByBand", moving)):
                s[name] = [float(np.sqrt((part[:, (CENTRES >= lo) & (CENTRES < hi)] ** 2).mean())) for lo, hi in CURVE]
            print(f"  {sex} {k:8s} {s['total']:5.1f}  ({s['common']:.1f}, {s['speaker']:.1f}, {s['vowel']:.1f}, {s['rest']:.1f})   curve " + " ".join(f"{x:+5.1f}" for x in s["curve"]) + "   rms by band " + " ".join(f"{x:4.1f}" for x in s["byBand"]))
        moved = np.concatenate([(f[2] - c.poses) / (HIGH - LOW) for c, f in mine])
        cell["moved"] = {k: [float(moved[:, i].mean()), float(np.abs(moved[:, i]).mean())] for i, k in enumerate(NAMES) if free[i]}
        print("    moved (mean, mean size; shares of the range): " + "  ".join(f"{k} {x[0]:+.2f} {x[1]:.2f}" for k, x in cell["moved"].items()))
        out["sexes"][sex] = cell
    (d / f"{a.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
