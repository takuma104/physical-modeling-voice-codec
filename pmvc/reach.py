"""Can the engine give a recording's envelope when its track asks for it, frame by frame? Tracks searched through the engine.

The ear puts a render next to its recording only when its envelope follows the recording's
from frame to frame (`pmvc.ladder`'s `slow` step, a mean over 0.1 s, is already near; no step
made of averages is). So the question is no longer what form the tract should have, but whether
the engine, in the form it has, comes to the recording's envelope when the track is free to go
where it must. Here the track itself is searched, on renders.

What makes that affordable is that an envelope is local: what the track does at one moment
shows in the render's envelope within some tens of milliseconds of it. So one render tries a
change at many moments at once. The track's change is a set of knots KNOT frames apart, each
a vector over the channels in FREE (in units of how far each moves in speech), with straight
lines between them. A candidate moves every COLOURS-th knot along a random direction of its
own; the candidates of a batch are rendered side by side, and each moved knot then takes the
mix of its directions that brings the envelope of the frames around it nearest the
recording's if the envelope changes in a straight line over so short a way (`search`). The
next batch takes the next knots. (A first version kept, for each knot, the mean of the draws
that lowered its error; it came as far in twice as many batches.)

With `--shape N` the tracks of a speaker are searched together and share a shape of the tract
that is searched with them: the fork's area profile (`pmvc_voice::Player::set_form`'s
`profile`), the logarithms of what the tract's area is multiplied by at N places from the
glottis to the lips, the same in every frame. In V1's form the search lowers the error only by
what a mean curve would (the engine's upper resonances stand at the same frequencies whatever
the track does); with a shape of the speaker's own, the error from frame to frame falls too.

How far the search can be trusted: on a target made by moving the track's own knots by half a
unit (`--astray 0.5 --stay 0`), where the answer exists, the error falls from 4.6 dB to 1.9
(four fifths of its variance), and no further: what a search reaches of a recording is a
bound from below on what the engine can.

Renders to be compared should draw their noise alike (`--form airway.noiseDraws=3`, the fork's
setting; the self-test above was run so). The engine as it is draws one random number a noise
source a sample, and the constriction's source joins the list when the narrowest section
passes under 0.9 cm², so a change that moves that moment by one control step shifts the breath
noise of all that follows: two renders of tracks that are all but the same then differ by 1.1
to 1.4 dB from frame to frame (0.004 dB with the setting), and the self-test stops at 2.5 dB.
The searches of recordings written up in the plan were run without the setting.

The error of a frame the recording is voiced in is `pmvc.form`'s: the envelope's distance in
bands, each band counted by how loud the recording is in it, the level the two differ by taken
out; and LEVEL times the square of that level, against what it is over the utterance, so that
the voice does not go faint to look right. In the other frames with sound (the consonants'
noises) a candidate is held to the render it started from, by the same error. STAY weighs how
far a knot has gone from the track.

usage: python -m pmvc.reach [dir] [--variant encv2] [--ids ID,...] [--per 1] [--form SPEC] [--shape 12]
                            [--out reach] [--iters 40] [--free jaw,tbX,...] [--judges out/jvsall]
                            [--name reach] [--stay 0.5] [--astray 0.5]

Writes <dir>/<out>/<id>.tracks.json and .wav (48 kHz), <id>.log.json (the error by iteration,
how far each channel went) and <dir>/<name>.json: for each recording the envelope's error in
the vowels before and after (`pmvc.form`'s two: as far as it is heard, and plain), and with
`--judges` what NISQA and a verifier make of the two renders. `--form` is a SPEC of
`pmvc.form`: the form the tracks are searched in. A file of a search with `--shape` is what
`pmvc.form --shapes` takes, to render other tracks in the speakers' shapes.

An envelope's error alone has misled a fit before (`pmvc.form`): what this search reaches is to
be heard, not only measured.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly

from . import gap
from .audio import FRAME_RATE, load
from .encoder import LEAD, ids_in, render_track
from .form import BANDS, DB, FAINT, LOUD, Clip, Judges, formed, heard_error, settings
from .invert import HIGH, LOW, NAMES
from .ladder import QUIET, phones_of
from .paths import ROOT

Array = NDArray[np.float64]

SR: int = gap.SR
# the channels a search may move, and the unit each moves by (about its spread in speech)
FREE: dict[str, float] = {
    "jaw": 0.35, "tbX": 0.6, "tbY": 0.45, "ttX": 0.25, "ttY": 1.0, "root": 0.25, "lipAp": 0.5, "lipProt": 0.5, "lipWidth": 0.25,
    "velum": 0.15, "larynx": 0.22, "epilarynx": 0.2, "abduction": 0.08, "breath": 0.2,
}  # fmt: skip
# what the channels that are no articulator may be
BOUNDS: dict[str, tuple[float, float]] = {"abduction": (0.0, 1.0), "breath": (0.0, 1.3)}
# frames between knots
KNOT: int = 5
# a batch changes every COLOURS-th knot
COLOURS: int = 3
# what a knot's change shows in: the frames this long after it (s), and this far to either side (s)
AFTER: float = 0.01
AROUND: float = 0.07
# a knot is searched if the recording is voiced in this share of the frames it shows in
SHARE: float = 0.5
# how long a knot's draws are at first (units, over all channels together), and the least and the most
STEP: float = 0.6
STEP_RANGE: tuple[float, float] = (0.1, 1.5)
# a step takes a knot at most this many times as far as its draws are long
FAR: float = 1.5
# a step that gained under POOR of what it promised shortens the knot's draws, one over GOOD lengthens them
POOR: float = 0.25
GOOD: float = 0.6
GROW: float = 1.3
SHRINK: float = 0.6
# the damping of a step: this share of the mean square of the draws' effects, added to each
DAMP: float = 0.05
# the tract's own shape: how long its draws are at first (logarithms of area, over all places together), the
# least and the most; the most a place's logarithm may be; what a unit of it costs (dB squared a unit squared)
SHAPE_STEP: float = 0.5
SHAPE_STEP_RANGE: tuple[float, float] = (0.1, 1.5)
SHAPE_MOST: float = 1.5
SHAPE_STAY: float = 1.0
# what a unit of distance from the track costs (dB squared a unit squared)
STAY: float = 0.5
# what the level's error costs against the shape's, and the most it is taken to be (dB)
LEVEL: float = 0.25
LEVEL_MOST: float = 20.0
# a frame has sound if its level is over this share of the clip's highest
SOUND: float = 0.03


class Target:
    """A recording, the track its render came from, and how far a render of a changed track is
    from the recording, frame by frame."""

    def __init__(self, d: Path, variant: str, m: dict[str, Any], pitch: float, aligned: dict[str, Any], spec: str, free: list[str], astray: float = 0.0) -> None:
        self.meta = m
        uid = m["id"]
        self.track: dict[str, Any] = formed(json.loads((d / variant / f"{uid}.tracks.json").read_text()), m["sex"], settings(spec))
        self.base = np.array(self.track["frames"], dtype=np.float64)
        keys: list[str] = self.track["keys"]
        self.free = [keys.index(k) for k in free]
        self.unit = np.array([FREE[k] for k in free])
        self.low = np.array([LOW[NAMES.index(k)] if k in NAMES else BOUNDS[k][0] for k in free])
        self.high = np.array([HIGH[NAMES.index(k)] if k in NAMES else BOUNDS[k][1] for k in free])
        self.x = load(d / "real" / f"{uid}.wav", SR)[0]
        self.cut = round(min(gap.CUT_MAX, gap.CUT_SHARE / pitch) * SR)
        self.late = 0
        first = self.render(self.base)
        self.late = gap.lag(self.x, first)
        first = np.roll(first, -self.late)
        self.knots = np.arange(0, len(self.base), KNOT)
        self.f0: Array = np.load(d / "pitch" / f"{uid}.npy")
        self.phones = phones_of(aligned, uid, len(gap.spectrum(self.x)))
        self.listen(first)
        if astray:
            # a test of the search: the recording is a render of the track with the knots that
            # are searched gone astray
            self.x = self.render(self.frames(astray * np.random.default_rng(1).normal(size=(len(self.knots), len(free))) * self.searched[:, None]))
            self.listen(first)

    def listen(self, first: Array) -> None:
        """What a render is asked for, from the recording and the track's own render `first`."""
        self.real, level = self.heard(self.x)
        n = len(self.real)
        j = np.minimum((np.arange(n) * gap.HOP / SR * FRAME_RATE).round().astype(int), len(self.f0) - 1)
        speech = np.array([p not in QUIET for p in self.phones])
        # the frames a render is asked to be the recording in, and those it is held to itself in
        self.asked: NDArray[np.bool_] = (self.f0[j] > 0) & speech & (level > SOUND * level.max())
        self.own, own_level = self.heard(first)
        self.held: NDArray[np.bool_] = ~self.asked & (own_level > SOUND * own_level.max())
        self.goal = np.where(self.asked[:, None], self.real, self.own)
        w = 10.0 ** (LOUD * (self.goal - self.goal.max(axis=1, keepdims=True)) / 20.0) + FAINT
        self.weight = w / w.sum(axis=1, keepdims=True)
        self.counted = (self.asked | self.held).astype(float)
        lack = self.goal - self.own
        # the level a render differs from the recording by over the utterance, as it started
        self.level0 = float(np.median((self.weight * lack).sum(axis=1)[self.asked])) if self.asked.any() else 0.0
        # the frames of the spectrum each knot's change shows in
        at = (self.knots / FRAME_RATE - LEAD - self.late / SR + AFTER) * SR / gap.HOP
        self.shows = np.clip(1.0 - np.abs(np.arange(n)[None, :] - at[:, None]) / (AROUND * SR / gap.HOP), 0.0, None)
        total = self.shows.sum(axis=1)
        self.searched: NDArray[np.bool_] = (self.shows @ self.asked.astype(float)) >= SHARE * np.maximum(total, 1e-9)
        self.searched &= total > 0
        self.shows = self.shows * self.counted[None, :]
        self.shows = self.shows / np.maximum(self.shows.sum(axis=1, keepdims=True), 1e-9)

    def heard(self, y: Array) -> tuple[Array, Array]:
        """A clip's envelope in bands (dB) and its level, frame by frame."""
        mag = np.abs(gap.spectrum(y))
        return gap.envelope(mag, self.cut) @ BANDS.T * DB, mag.sum(axis=1)

    def frames(self, moved: Array) -> Array:
        """The track with the knots moved by `moved` (knots, channels; in units)."""
        x = self.base.copy()
        at = np.arange(len(x))
        for c, (k, unit, lo, hi) in enumerate(zip(self.free, self.unit, self.low, self.high)):
            # a channel the track holds outside its range stays where it is until it is moved
            x[:, k] = np.where(np.interp(at, self.knots, moved[:, c]) == 0.0, x[:, k], np.clip(x[:, k] + unit * np.interp(at, self.knots, moved[:, c]), lo, hi))
        return x

    def shaped(self, frames: Array, shape: Array | None = None) -> dict[str, Any]:
        """The track with these frames, and with the tract in a shape of its own (`shape`: the
        logarithms of what its area is multiplied by, glottis to lips)."""
        form = {**self.track.get("form", {}), "profile": shape.tolist()} if shape is not None and len(shape) else self.track.get("form")
        return {**self.track, "frames": frames, **({"form": form} if form else {})}

    def render(self, frames: Array, shape: Array | None = None) -> Array:
        """The render of a track, on the recording's clock."""
        y = resample_poly(render_track(self.shaped(frames, shape)), 1, 48000 // SR)[round(LEAD * SR) :]
        y = np.pad(y, (0, max(0, len(self.x) - len(y))))[: len(self.x)]
        return np.roll(y, -self.late)

    def errors(self, y: Array) -> Array:
        """The error of each frame of a render (dB squared): the envelope's shape against the
        frame's goal, and the level's."""
        lack = self.goal - self.heard(y)[0]
        level = (self.weight * lack).sum(axis=1)
        shape = (self.weight * (lack - level[:, None]) ** 2).sum(axis=1)
        return (shape + LEVEL * np.clip(level - self.level0, -LEVEL_MOST, LEVEL_MOST) ** 2) * self.counted

    def error(self, e: Array) -> float:
        """The frames' errors as one number (dB rms over the frames the recording is voiced in)."""
        return float(np.sqrt(e[self.asked].mean())) if self.asked.any() else 0.0


class Knots:
    """Where a search has a track's knots, and how it is getting on with each."""

    def __init__(self, t: Target) -> None:
        k, c = len(t.knots), len(t.free)
        self.moved = np.zeros((k, c))
        # how far each knot may go: to the channels' bounds (a channel the track has outside them stays free to come back)
        at_knots = t.base[t.knots][:, t.free]
        self.room = np.minimum((t.low - at_knots) / t.unit, 0.0), np.maximum((t.high - at_knots) / t.unit, 0.0)
        self.step = np.full(k, STEP)
        self.before, self.promised = np.full(k, np.nan), np.full(k, np.nan)


def squares(t: Target, env: Array) -> tuple[Array, Array, Array]:
    """A render's envelope against its goal: each frame's lack in bands with the level out, the
    level's own error, and the frame's error (dB squared)."""
    lack = t.goal - env
    level = (t.weight * lack).sum(axis=1)
    shape = lack - level[:, None]
    off = np.clip(level - t.level0, -LEVEL_MOST, LEVEL_MOST)
    return shape, off, ((t.weight * shape**2).sum(axis=1) + LEVEL * off**2) * t.counted


def step_within(a: Array, y: Array, dirs: Array, at: Array, stay: float, far: float) -> tuple[Array, float]:
    """The mix of `dirs` (candidates, dimensions; their effects on the squares' roots are the
    rows of `a`) that brings the roots `y` nearest nought, if effects add up in a straight
    line: damped, held to `at` by `stay` a unit squared, and no longer than `far`. Returns the
    move and the sum of squares it promises."""
    normal = a @ a.T
    normal += DAMP * np.trace(normal) / len(normal) * np.eye(len(normal)) + stay * dirs @ dirs.T
    mix = np.linalg.solve(normal, a @ y - stay * dirs @ at)
    mix = mix / max(float(np.linalg.norm(mix @ dirs)) / far, 1.0)
    return mix @ dirs, float(((y - mix @ a) ** 2).sum())


def knots_batch(t: Target, s: Knots, colour: int, shape: Array, pool: ThreadPoolExecutor, jobs: int, stay: float, rng: np.random.Generator) -> float:
    """One batch of a track's search: every COLOURS-th knot takes a step. Returns the error
    of the track as the batch found it."""
    k, c = s.moved.shape
    mine = np.flatnonzero(t.searched & (np.arange(k) % COLOURS == colour))
    draws = rng.normal(size=(jobs - 1, len(mine), c)) / np.sqrt(c)
    tries = [s.moved.copy() for _ in range(jobs)]
    for p in range(1, jobs):
        tries[p][mine] = np.clip(s.moved[mine] + s.step[mine, None] * draws[p - 1], s.room[0][mine], s.room[1][mine])
    env = np.stack(list(pool.map(lambda m: t.heard(t.render(t.frames(m), shape))[0], tries)))
    lack, off, e = squares(t, env[0])
    cost = t.shows @ e + stay * (s.moved**2).sum(axis=1)
    for kn in np.flatnonzero(~np.isnan(s.before)):
        gained = (s.before[kn] - cost[kn]) / max(s.before[kn] - s.promised[kn], 1e-9)
        s.step[kn] = float(np.clip(s.step[kn] * (GROW if gained > GOOD else SHRINK if gained < POOR else 1.0), *STEP_RANGE))
    s.before[:] = np.nan
    change = env[1:] - env[0]
    mean = (t.weight * change).sum(axis=2)
    for kn in mine:
        at = np.flatnonzero(t.shows[kn] > 0)
        r, w = np.sqrt(t.shows[kn, at]), np.sqrt(t.weight[at])
        # the draws' effects on the roots of the frames' squares (rows), and the roots as they are
        a = np.concatenate([(r[None, :, None] * w[None] * (change[:, at] - mean[:, at, None])).reshape(jobs - 1, -1), np.sqrt(LEVEL) * r[None, :] * mean[:, at]], axis=1)
        y = np.concatenate([(r[:, None] * w * lack[at]).ravel(), np.sqrt(LEVEL) * r * off[at]])
        move, left = step_within(a, y, np.stack([m[kn] for m in tries[1:]]) - s.moved[kn], s.moved[kn], stay, FAR * s.step[kn])
        to = np.clip(s.moved[kn] + move, s.room[0][kn], s.room[1][kn])
        s.before[kn], s.promised[kn] = cost[kn], left + stay * float((to**2).sum())
        s.moved[kn] = to
    return t.error(e)


def shape_batch(ts: list[Target], ss: list[Knots], shape: Array, step: float, pool: ThreadPoolExecutor, jobs: int, rng: np.random.Generator) -> tuple[Array, float]:
    """One step of the tract's own shape, which the tracks of a speaker share: the mix of
    random changes of the shape that brings the frames the recordings are voiced in nearest
    them, over all the tracks. The step is kept if the renders are nearer for it. Returns the
    shape and how long the next draws are to be."""
    draws = rng.normal(size=(jobs - 1, len(shape))) / np.sqrt(len(shape))
    tries = [shape, *(np.clip(shape + step * z, -SHAPE_MOST, SHAPE_MOST) for z in draws)]
    envs = list(pool.map(lambda job: job[0].heard(job[0].render(job[0].frames(job[1].moved), job[2]))[0], [(t, s, x) for t, s in zip(ts, ss) for x in tries]))
    rows, roots, now = [], [], 0.0
    for i, t in enumerate(ts):
        env = np.stack(envs[i * jobs : (i + 1) * jobs])
        lack, off, e = squares(t, env[0])
        change = (env[1:] - env[0])[:, t.asked]
        mean = (t.weight[t.asked] * change).sum(axis=2)
        w = np.sqrt(t.weight[t.asked])
        rows.append(np.concatenate([(w[None] * (change - mean[:, :, None])).reshape(jobs - 1, -1), np.sqrt(LEVEL) * mean], axis=1))
        roots.append(np.concatenate([(w * lack[t.asked]).ravel(), np.sqrt(LEVEL) * off[t.asked]]))
        now += float(e[t.asked].sum())
    frames = sum(int(t.asked.sum()) for t in ts)
    move, left = step_within(np.concatenate(rows, axis=1), np.concatenate(roots), np.stack(tries[1:]) - shape, shape, SHAPE_STAY * frames, FAR * step)
    to = np.clip(shape + move, -SHAPE_MOST, SHAPE_MOST)
    after = sum(float(squares(t, env)[2][t.asked].sum()) for t, env in zip(ts, pool.map(lambda job: job[0].heard(job[0].render(job[0].frames(job[1].moved), to))[0], list(zip(ts, ss)))))
    gained = (now - after) / max(now - left, 1e-9)
    step = float(np.clip(step * (GROW if gained > GOOD else SHRINK if gained < POOR else 1.0), *SHAPE_STEP_RANGE))
    return (to if after < now else shape), step


def search(ts: list[Target], iters: int, pool: ThreadPoolExecutor, jobs: int, stay: float = STAY, points: int = 0, seed: int = 0) -> tuple[list[Array], Array, list[float]]:
    """The knots' moves that bring the renders of a speaker's tracks nearest their recordings,
    with `points` the shape of the tract that they share (its area's logarithms at so many
    places), and the error by iteration.

    A batch renders a track as it is and with every COLOURS-th knot moved along a random
    direction, one direction a candidate. A frame's error is a sum of squares (the envelope's
    distance in each band, and the level's), and over a step as short as the draws the
    envelope changes about in a straight line with the track: so each knot takes the mix of
    its draws that would bring the envelope of the frames around it nearest their goal if the
    line held (a Gauss-Newton step within the draws' span, damped, and no farther than FAR
    draws). The next batch shows what the step gained against what it promised, and the
    knot's draws become longer or shorter by that. After the knots, the shape takes a step
    of the same kind over all frames of all the tracks (`shape_batch`); the consonants' frames
    are then held to what the tracks sound like in the new shape."""
    rng = np.random.default_rng(seed)
    ss = [Knots(t) for t in ts]
    shape, shape_step = np.zeros(points), SHAPE_STEP
    log: list[float] = []
    for it in range(iters):
        errors = [[knots_batch(t, s, colour, shape, pool, jobs, stay, rng) for colour in range(COLOURS)][0] for t, s in zip(ts, ss)]
        log.append(float(np.sqrt(np.mean(np.square(errors)))))
        if points:
            was = shape
            shape, shape_step = shape_batch(ts, ss, shape, shape_step, pool, jobs, rng)
            if shape is not was:
                for t, first in zip(ts, pool.map(lambda t: t.render(t.base, shape), ts)):
                    t.listen(first)
                for s in ss:
                    s.before[:] = np.nan
        print(f"  {it + 1:3d}: {log[-1]:.2f} dB  (draws {np.mean([s.step[t.searched].mean() for t, s in zip(ts, ss)]):.2f} long)" + (f"  shape {' '.join(f'{x:+.2f}' for x in shape)}" if points else ""), flush=True)
    log.append(float(np.sqrt(np.mean([t.error(t.errors(t.render(t.frames(s.moved), shape))) ** 2 for t, s in zip(ts, ss)]))))
    return [s.moved for s in ss], shape, log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "jvs")
    ap.add_argument("--variant", type=str, default="encv2")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--ids", type=str, default="", help="the recordings to search (comma-separated), whatever their split")
    ap.add_argument("--per", type=int, default=1, help="recordings of each speaker, if no --ids")
    ap.add_argument("--form", type=str, default="", help="the form the tracks are searched in (a SPEC of pmvc.form)")
    ap.add_argument("--shape", type=int, default=0, help="also search a shape of the tract that a speaker's tracks share: the area's logarithm at so many places from the glottis to the lips")
    ap.add_argument("--out", type=str, default="reach")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--free", type=str, default=",".join(FREE), help="the channels to move")
    ap.add_argument("--stay", type=float, default=STAY, help="what a unit of distance from the track costs (dB squared a unit squared)")
    ap.add_argument("--astray", type=float, default=0.0, help="a test of the search: in place of the recording, a render of the track with every knot moved at random by this much (units)")
    ap.add_argument("--judges", type=Path, default=None, help="also ask NISQA and a verifier of the renders before and after; the directory whose training recordings stand for the speakers (out/jvsall)")
    ap.add_argument("--name", type=str, default="")
    ap.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    d: Path = a.dir
    aligned: dict[str, Any] = json.loads((d / "aligned.json").read_text())
    every = [json.loads((d / "meta" / f"{i}.json").read_text()) for i in ids_in(d, "all")]
    pitch = {s: float(np.median([m["f0Median"] for m in every if m["speaker"] == s])) for s in {m["speaker"] for m in every}}
    if a.ids:
        kept = [m for m in every if m["id"] in a.ids.split(",")]
    else:
        count: dict[str, int] = {}
        kept = []
        for m in (m for m in every if m["id"] in ids_in(d, a.split) and m["id"] in aligned):
            count[m["speaker"]] = count.get(m["speaker"], 0) + 1
            if count[m["speaker"]] <= a.per:
                kept.append(m)
    free = a.free.split(",")
    (d / a.out).mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    before: list[Array] = []
    after: list[Array] = []
    # the tracks of a speaker are searched together if they are to share a shape, else each alone
    groups: dict[str, list[dict[str, Any]]] = {}
    for m in kept:
        groups.setdefault(m["speaker"] if a.shape else m["id"], []).append(m)
    kept = [m for group in groups.values() for m in group]
    with ThreadPoolExecutor(a.jobs) as pool:
        for name, group in groups.items():
            ts = list(pool.map(lambda m: Target(d, a.variant, m, pitch[m["speaker"]], aligned, a.form, free, a.astray), group))
            for t in ts:
                print(f"{t.meta['id']}: {len(t.knots)} knots, {int(t.searched.sum())} searched, {int(t.asked.sum())} frames asked, {int(t.held.sum())} held", flush=True)
            starts = list(pool.map(lambda t: t.render(t.base), ts))
            moves, shape, log = search(ts, a.iters, pool, a.jobs, a.stay, a.shape)
            for t, moved, start in zip(ts, moves, starts):
                m = t.meta
                frames = t.frames(moved)
                track = {**t.shaped(frames, shape), "frames": np.round(frames, 5).tolist()}
                (d / a.out / f"{m['id']}.tracks.json").write_text(json.dumps(track))
                sf.write(str(d / a.out / f"{m['id']}.wav"), render_track(track), 48000, subtype="PCM_16")
                # the vowels' envelope error as the tables have it, before and after
                c = Clip(d, a.variant, m, pitch[m["speaker"]], aligned)
                if a.astray:
                    c.x, c.real = t.x, c.envelope(t.x)
                said: dict[str, list[float]] = {"heard": [], "plain": []}
                for y, kept_in in ((start, before), (t.render(frames, shape), after)):
                    kept_in.append(y)
                    lack = c.lack(y)
                    said["heard"].append(heard_error(c.real, lack))
                    said["plain"].append(float(np.sqrt(((lack - lack.mean(axis=1, keepdims=True)) ** 2).mean())))
                went = moved[t.searched]
                row = {
                    "id": m["id"], "sex": m["sex"], "speaker": m["speaker"], **said, "shape": shape.tolist(),
                    "went": {k: [float(went[:, i].mean()), float(np.sqrt((went[:, i] ** 2).mean()))] for i, k in enumerate(free)},
                }  # fmt: skip
                (d / a.out / f"{m['id']}.log.json").write_text(json.dumps({**row, "iterations": log}))
                rows.append(row)
                print(f"{m['id']}: vowels heard {said['heard'][0]:.2f} → {said['heard'][1]:.2f}, plain {said['plain'][0]:.2f} → {said['plain'][1]:.2f}", flush=True)
                print("  went (mean, rms; units): " + "  ".join(f"{k} {x[0]:+.2f} {x[1]:.2f}" for k, x in row["went"].items()), flush=True)
            print(f"{name}: searched error {log[0]:.2f} → {log[-1]:.2f} dB" + (f"; shape {' '.join(f'{x:+.2f}' for x in shape)}" if a.shape else ""), flush=True)
    out: dict[str, Any] = {"variant": a.variant, "form": a.form, "free": free, "iters": a.iters, "shape": a.shape, "clips": rows}
    if a.judges and rows:
        judges = Judges(a.judges, a.jobs)
        clips = [type("Heard", (), {"meta": m})() for m in kept]
        out["judges"] = {"before": judges.of(clips, before), "after": judges.of(clips, after)}  # type: ignore[arg-type]
        for when, by_sex in out["judges"].items():
            print(f"{when}: " + "   ".join(f"{sex} NISQA natural {v['natural']:.2f} quality {v['quality']:.2f} own {v['own']:.3f} top-1 {100 * v['top1']:.0f} %" for sex, v in by_sex.items()))
    for sex in ("F", "M"):
        mine = [r for r in rows if r["sex"] == sex]
        if mine:
            print(f"{sex} ({len(mine)}): vowels heard {np.mean([r['heard'][0] for r in mine]):.2f} → {np.mean([r['heard'][1] for r in mine]):.2f}, plain {np.mean([r['plain'][0] for r in mine]):.2f} → {np.mean([r['plain'][1] for r in mine]):.2f} dB")
    (d / f"{a.name or a.out}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
