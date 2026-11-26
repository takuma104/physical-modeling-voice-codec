"""Can the engine give a recording's envelope when it is asked to? Tracks, and a speaker's own voice, searched through the engine.

The ear puts a render next to its recording only when its envelope follows the recording's
from frame to frame (`pmvc.ladder`'s `slow` step, a mean over 0.1 s, is already near; no step
made of averages is). So the question is whether the engine comes to the recording's envelope
when the track is free to go where it must, and when its tract and source may be a speaker's
own. Here both are searched, on renders.

What makes that affordable is that an envelope is local: what the track does at one moment
shows in the render's envelope within some tens of milliseconds of it. So one render tries a
change at many moments at once. The track's change is a set of knots KNOT frames apart, each
a vector over the channels in FREE (in units of how far each moves in speech), with straight
lines between them. A candidate moves every COLOURS-th knot along a random direction of its
own; the candidates of a batch are rendered side by side, and each moved knot then takes the
mix of its directions that brings the envelope of the frames around it nearest the
recording's if the envelope changes in a straight line over so short a way (`knots_batch`).
The next batch takes the next knots. (A first version kept, for each knot, the mean of the
draws that lowered its error; it came as far in twice as many batches.)

The voice (`--shape N`, `--statics KEY,...`) is what does not change from frame to frame: the
tract's shape (the fork's area profile, `pmvc_voice::Player::set_form`'s `profile`: the
logarithms of what the tract's area is multiplied by at N places from the glottis to the lips)
and settings among STATICS (how sharply the glottis closes, its open quotient, the length of
the tube above it). The tracks of a speaker (`--share speaker`), of a sex or of all are
searched together and share one voice, which takes a step of the same kind after the knots'
(`voice_batch`). With `--still` the tracks stay as they are and only the voice is searched:
what a speaker's reference recordings can give without an encoder that writes other tracks.
`--from` names files of earlier searches: each track starts in the voice they have for its
speaker, or for its sex.

In V1's form the search of the tracks lowers the error only by what a mean curve would (the
engine's upper resonances stand at the same frequencies whatever the track does); with a
shape of the speaker's own, the error from frame to frame falls too.

The error of a frame the recording is voiced in is `pmvc.form`'s: the envelope's distance in
bands, each band counted by how loud the recording is in it, the level the two differ by taken
out; and LEVEL times the square of that level, against what it is over the utterance, so that
the voice does not go faint to look right. In the other frames with sound (the consonants'
noises) a candidate is held to the render it started from, by the same error. STAY weighs how
far a knot has gone from the track.

Two things keep a search from what no speaker would do (the first searches did both, and
were heard: a creak and a lip's flutter where a nasal's closure had been opened to a slit,
and a lisp where the tongue's tip stood high through the vowels). The tongue's tip is not
moved (MOVED; with it the error is 0.1 dB less). A track is moved only inside its vowels (`Target.vowel`: the aligner's phones; the consonants, the moraic nasal and the
glides stay as the encoder wrote them). And the mouth may not become much narrower than the
track had it (`Target.slits`: the oral tube's narrowest section under KEEP of what it was, or
under FLOOR, costs SLIT a squared logarithm), which a knot's step and the voice's weigh like
any other error.

How far the search can be trusted: on a target made by moving the track's own knots by half a
unit (`--astray 0.5 --stay 0`), where the answer exists, the error falls from 3.7 dB to 1.9
(seven tenths of its variance; before the vowels and the mouth were guarded, from 4.6 to 1.9),
and no further: what a search reaches of a recording is a bound from below on what the engine
can.

Renders to be compared should draw their noise alike (`--form airway.noiseDraws=3`, the fork's
setting; the self-test above was run so). The engine as it is draws one random number a noise
source a sample, and the constriction's source joins the list when the narrowest section
passes under 0.9 cm², so a change that moves that moment by one control step shifts the breath
noise of all that follows: two renders of tracks that are all but the same then differ by 1.1
to 1.4 dB from frame to frame (0.004 dB with the setting), and the self-test stops at 2.5 dB.

usage: python -m pmvc.reach [dir] [--variant encv2] [--ids ID,... | --ids-file FILE] [--split test] [--per 1]
                            [--form SPEC] [--shape 12] [--statics KEY,...] [--share speaker|sex|all]
                            [--still] [--from FILE,...] [--free jaw,tbX,...] [--out reach]
                            [--iters 16] [--judges out/jvsall] [--name reach] [--stay 0.5] [--astray 0.5]

Writes <dir>/<out>/<id>.tracks.json and .wav (48 kHz), <id>.log.json and <dir>/<name>.json:
for each recording the envelope's error in the vowels before and after (`pmvc.form`'s two: as
far as it is heard, and plain), the voices found (`voices`: by speaker, or by sex; what
`pmvc.form --shapes` and `--from` take), and with `--judges` what NISQA and a verifier make of
the renders before and after. `--form` is a SPEC of `pmvc.form`: the form the tracks are
searched in.

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
from .bands import VOWELS
from .calibrate import make_voice
from .encoder import LEAD, ids_in, render_track
from .form import BANDS, DB, FAINT, LOUD, Clip, Judges, formed, heard_error, settings, shaped, voice_for, voices_in
from .invert import HIGH, LOW, NAMES
from .ladder import QUIET, phones_of
from .paths import ROOT

Array = NDArray[np.float64]

SR: int = gap.SR
HOP: int = 48000 // FRAME_RATE
# the channels a search may move, and the unit each moves by (about its spread in speech)
FREE: dict[str, float] = {
    "jaw": 0.35, "tbX": 0.6, "tbY": 0.45, "ttX": 0.25, "ttY": 1.0, "root": 0.25, "lipAp": 0.5, "lipProt": 0.5, "lipWidth": 0.25,
    "velum": 0.15, "larynx": 0.22, "epilarynx": 0.2, "abduction": 0.08, "breath": 0.2,
}  # fmt: skip
# of them, those a search moves unless told otherwise: not the tongue's tip, which the first
# searches raised through the vowels (heard as a lisp), and which is worth 0.1 dB
MOVED: tuple[str, ...] = tuple(k for k in FREE if k not in ("ttX", "ttY"))
# what the channels that are no articulator may be
BOUNDS: dict[str, tuple[float, float]] = {"abduction": (0.0, 1.0), "breath": (0.0, 1.3)}
# the settings a voice may have besides its tract's shape: the unit each moves by, and its bounds
STATICS: dict[str, tuple[float, float, float]] = {
    "glottis.closureSharpness": (0.1, 0.05, 1.0),
    "glottis.openQuotient": (0.05, 0.3, 0.9),
    "larynx.length": (0.05, 0.8, 1.5),
}
# the engine's own values of settings a track may not name
OWN: dict[str, float] = {"larynx.length": 1.0}
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
# a voice: how long its draws are at first (logarithms of area and units of settings, all together), the
# least and the most; the most a place's logarithm may be; what a unit of it costs (dB squared a unit squared)
VOICE_STEP: float = 0.5
VOICE_STEP_RANGE: tuple[float, float] = (0.1, 1.5)
SHAPE_MOST: float = 1.5
VOICE_STAY: float = 1.0
# what a unit of distance from the track costs (dB squared a unit squared)
STAY: float = 0.5
# what the level's error costs against the shape's, and the most it is taken to be (dB)
LEVEL: float = 0.25
LEVEL_MOST: float = 20.0
# a frame has sound if its level is over this share of the clip's highest
SOUND: float = 0.03
# the mouth's narrowest section may be this share of what the track had there, or FLOOR (cm²) if that is
# less; under it, SLIT (dB squared) a squared natural logarithm
KEEP: float = 0.8
FLOOR: float = 0.6
SLIT: float = 60.0


class Target:
    """A recording, the track its render came from, and how far a render of a changed track,
    or of the track in another voice, is from the recording, frame by frame."""

    def __init__(self, d: Path, variant: str, m: dict[str, Any], pitch: float, aligned: dict[str, Any], spec: str, free: list[str], points: int = 0, statics: list[str] | None = None, start: dict[str, Any] | None = None, astray: float = 0.0) -> None:
        self.meta = m
        uid = m["id"]
        self.track: dict[str, Any] = shaped(formed(json.loads((d / variant / f"{uid}.tracks.json").read_text()), m["sex"], settings(spec)), start)
        self.base = np.array(self.track["frames"], dtype=np.float64)
        keys: list[str] = self.track["keys"]
        self.free = [keys.index(k) for k in free]
        self.unit = np.array([FREE[k] for k in free])
        self.low = np.array([LOW[NAMES.index(k)] if k in NAMES else BOUNDS[k][0] for k in free])
        self.high = np.array([HIGH[NAMES.index(k)] if k in NAMES else BOUNDS[k][1] for k in free])
        # the voice: the tract's shape as the track starts with it, and the settings that may change
        had = self.track.get("form", {}).get("profile") or []
        self.points = points
        self.profile = np.interp(np.linspace(0.0, 1.0, points), np.linspace(0.0, 1.0, len(had)), had) if points and len(had) > 1 else np.zeros(points)
        self.statics = statics or []
        self.started = np.array([self.track.get(k.split(".")[0], {}).get(k.split(".")[1], OWN.get(k, 0.0)) for k in self.statics])
        units = np.array([STATICS[k][0] for k in self.statics])
        self.voice_unit = np.concatenate([np.ones(points), units])
        self.voice_room = (
            np.concatenate([-SHAPE_MOST - self.profile, (np.array([STATICS[k][1] for k in self.statics]) - self.started) / units]),
            np.concatenate([SHAPE_MOST - self.profile, (np.array([STATICS[k][2] for k in self.statics]) - self.started) / units]),
        )
        # a track is moved inside its vowels only: a frame of the track counts as one if it and its neighbours are
        said: list[str] = []
        for p, n in zip(aligned[uid]["p"], aligned[uid]["frames"]):
            said += [p] * n
        is_vowel = np.array([p in VOWELS for p in said[: len(self.base)]] + [False] * max(0, len(self.base) - len(said)), dtype=float)
        inside = np.minimum(np.minimum(is_vowel, np.roll(is_vowel, 1)), np.roll(is_vowel, -1))
        self.vowel: Array = np.convolve(inside, [0.25, 0.5, 0.25], "same")
        self.x = load(d / "real" / f"{uid}.wav", SR)[0]
        self.cut = round(min(gap.CUT_MAX, gap.CUT_SHARE / pitch) * SR)
        self.late = 0
        self.knots = np.arange(0, len(self.base), KNOT)
        first, narrow = self.sound(self.base)
        self.late = gap.lag(self.x, first)
        first = np.roll(first, -self.late)
        # how narrow the mouth may become, frame by frame of the track
        self.least: Array = np.minimum(KEEP * narrow, FLOOR)
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
        self.searched &= (total > 0) & (self.vowel[self.knots] > 0)
        self.shows = self.shows * self.counted[None, :]
        self.shows = self.shows / np.maximum(self.shows.sum(axis=1, keepdims=True), 1e-9)
        # the frames of the track each knot moves, as shares
        near = np.clip(1.0 - np.abs(np.arange(len(self.base))[None, :] - self.knots[:, None]) / KNOT, 0.0, None)
        self.moves = near / np.maximum(near.sum(axis=1, keepdims=True), 1e-9)

    def heard(self, y: Array) -> tuple[Array, Array]:
        """A clip's envelope in bands (dB) and its level, frame by frame."""
        mag = np.abs(gap.spectrum(y))
        return gap.envelope(mag, self.cut) @ BANDS.T * DB, mag.sum(axis=1)

    def frames(self, moved: Array) -> Array:
        """The track with the knots moved by `moved` (knots, channels; in units), inside its vowels."""
        x = self.base.copy()
        at = np.arange(len(x))
        for c, (k, unit, lo, hi) in enumerate(zip(self.free, self.unit, self.low, self.high)):
            by = np.interp(at, self.knots, moved[:, c]) * self.vowel
            # a channel the track holds outside its range stays where it is until it is moved
            x[:, k] = np.where(by == 0.0, x[:, k], np.clip(x[:, k] + unit * by, lo, hi))
        return x

    def voice(self, v: Array | None) -> dict[str, Any]:
        """The voice `v` steps away from the one the track started in: the tract's shape and the settings."""
        v = np.zeros(self.points + len(self.statics)) if v is None else v
        values = np.clip(self.started + self.voice_unit[self.points :] * v[self.points :], [STATICS[k][1] for k in self.statics], [STATICS[k][2] for k in self.statics])
        return {"profile": (self.profile + v[: self.points]).tolist(), "set": {k: float(x) for k, x in zip(self.statics, values)}}

    def shaped(self, frames: Array, v: Array | None = None) -> dict[str, Any]:
        """The track with these frames, in the voice `v` steps away from its own."""
        return {**(shaped(self.track, self.voice(v)) if v is not None and len(v) else self.track), "frames": frames}

    def sound(self, frames: Array, v: Array | None = None) -> tuple[Array, Array]:
        """The render of a track on the recording's clock, and the oral tube's narrowest
        section (cm²) at the start of each frame of the track."""
        voice = make_voice(self.shaped(frames, v), None)
        voice.set_track(frames, 0)
        out, watched = voice.render_watched(len(frames) - 1, HOP)
        narrow = np.append(np.asarray(watched), watched[-1])
        y = resample_poly(np.asarray(out, dtype=np.float64), 1, 48000 // SR)[round(LEAD * SR) :]
        y = np.pad(y, (0, max(0, len(self.x) - len(y))))[: len(self.x)]
        return np.roll(y, -self.late), narrow

    def render(self, frames: Array, v: Array | None = None) -> Array:
        return self.sound(frames, v)[0]

    def judged(self, frames: Array, v: Array | None = None) -> tuple[Array, Array]:
        """What a search goes by of a track's render: its envelope in bands, frame by frame,
        and the roots of what the mouth's narrowness costs, frame by frame of the track."""
        y, narrow = self.sound(frames, v)
        return self.heard(y)[0], self.slits(narrow)

    def slits(self, narrow: Array) -> Array:
        """For each frame of the track, the root of what the mouth's narrowness costs there:
        nothing while the narrowest section is over what it may be."""
        return np.sqrt(SLIT) * np.where(self.least > 1e-3, np.maximum(0.0, np.log(np.maximum(self.least, 1e-3) / np.maximum(narrow, 1e-4))), 0.0)

    def errors(self, y: Array) -> Array:
        """The error of each frame of a render (dB squared): the envelope's shape against the
        frame's goal, and the level's."""
        return squares(self, self.heard(y)[0])[2]

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


def knots_batch(t: Target, s: Knots, colour: int, voice: Array, pool: ThreadPoolExecutor, jobs: int, stay: float, rng: np.random.Generator) -> float:
    """One batch of a track's search: every COLOURS-th knot takes a step. Returns the error
    of the track as the batch found it."""
    k, c = s.moved.shape
    mine = np.flatnonzero(t.searched & (np.arange(k) % COLOURS == colour))
    draws = rng.normal(size=(jobs - 1, len(mine), c)) / np.sqrt(c)
    tries = [s.moved.copy() for _ in range(jobs)]
    for p in range(1, jobs):
        tries[p][mine] = np.clip(s.moved[mine] + s.step[mine, None] * draws[p - 1], s.room[0][mine], s.room[1][mine])
    made = list(pool.map(lambda m: t.judged(t.frames(m), voice), tries))
    env, slit = np.stack([e for e, _ in made]), np.stack([q for _, q in made])
    lack, off, e = squares(t, env[0])
    cost = t.shows @ e + t.moves @ slit[0] ** 2 + stay * (s.moved**2).sum(axis=1)
    for kn in np.flatnonzero(~np.isnan(s.before)):
        gained = (s.before[kn] - cost[kn]) / max(s.before[kn] - s.promised[kn], 1e-9)
        s.step[kn] = float(np.clip(s.step[kn] * (GROW if gained > GOOD else SHRINK if gained < POOR else 1.0), *STEP_RANGE))
    s.before[:] = np.nan
    change = env[1:] - env[0]
    mean = (t.weight * change).sum(axis=2)
    for kn in mine:
        at = np.flatnonzero(t.shows[kn] > 0)
        by = np.flatnonzero(t.moves[kn] > 0)
        r, w, q = np.sqrt(t.shows[kn, at]), np.sqrt(t.weight[at]), np.sqrt(t.moves[kn, by])
        # the draws' effects on the roots of the frames' squares (rows), and the roots as they are:
        # the envelope's shape, the level, and the mouth's narrowness
        a = np.concatenate([(r[None, :, None] * w[None] * (change[:, at] - mean[:, at, None])).reshape(jobs - 1, -1), np.sqrt(LEVEL) * r[None, :] * mean[:, at], q[None, :] * (slit[0, by] - slit[1:, by])], axis=1)
        y = np.concatenate([(r[:, None] * w * lack[at]).ravel(), np.sqrt(LEVEL) * r * off[at], q * slit[0, by]])
        move, left = step_within(a, y, np.stack([m[kn] for m in tries[1:]]) - s.moved[kn], s.moved[kn], stay, FAR * s.step[kn])
        to = np.clip(s.moved[kn] + move, s.room[0][kn], s.room[1][kn])
        s.before[kn], s.promised[kn] = cost[kn], left + stay * float((to**2).sum())
        s.moved[kn] = to
    return t.error(e)


def voice_batch(ts: list[Target], ss: list[Knots], voice: Array, step: float, pool: ThreadPoolExecutor, jobs: int, rng: np.random.Generator) -> tuple[Array, float, float]:
    """One step of the voice that tracks share (the tract's shape and the settings): the mix
    of random changes of it that brings the frames the recordings are voiced in nearest them,
    over all the tracks, without making a mouth too narrow. The step is kept if the renders
    are nearer for it. Returns the voice, how long the next draws are to be, and the tracks'
    error as the batch found it."""
    lo, hi = ts[0].voice_room
    draws = rng.normal(size=(jobs - 1, len(voice))) / np.sqrt(len(voice))
    tries = [voice, *(np.clip(voice + step * z, lo, hi) for z in draws)]
    made = list(pool.map(lambda job: job[0].judged(job[0].frames(job[1].moved), job[2]), [(t, s, x) for t, s in zip(ts, ss) for x in tries]))

    def cost(t: Target, env: Array, slit: Array) -> float:
        return float(squares(t, env)[2][t.asked].sum() + (t.vowel * slit**2).sum())

    rows, roots, now, errors = [], [], 0.0, []
    for i, t in enumerate(ts):
        mine = made[i * jobs : (i + 1) * jobs]
        env = np.stack([e for e, _ in mine])
        slit = np.stack([q for _, q in mine]) * np.sqrt(t.vowel)
        lack, off, e = squares(t, env[0])
        change = (env[1:] - env[0])[:, t.asked]
        mean = (t.weight[t.asked] * change).sum(axis=2)
        w = np.sqrt(t.weight[t.asked])
        rows.append(np.concatenate([(w[None] * (change - mean[:, :, None])).reshape(jobs - 1, -1), np.sqrt(LEVEL) * mean, slit[0] - slit[1:]], axis=1))
        roots.append(np.concatenate([(w * lack[t.asked]).ravel(), np.sqrt(LEVEL) * off[t.asked], slit[0]]))
        now += float(e[t.asked].sum() + (slit[0] ** 2).sum())
        errors.append(t.error(e))
    frames = sum(int(t.asked.sum()) for t in ts)
    move, left = step_within(np.concatenate(rows, axis=1), np.concatenate(roots), np.stack(tries[1:]) - voice, voice, VOICE_STAY * frames, FAR * step)
    to = np.clip(voice + move, lo, hi)
    after = sum(cost(t, env, slit) for t, (env, slit) in zip(ts, pool.map(lambda job: job[0].judged(job[0].frames(job[1].moved), to), list(zip(ts, ss)))))
    gained = (now - after) / max(now - left, 1e-9)
    step = float(np.clip(step * (GROW if gained > GOOD else SHRINK if gained < POOR else 1.0), *VOICE_STEP_RANGE))
    return (to if after < now else voice), step, float(np.sqrt(np.mean(np.square(errors))))


def search(ts: list[Target], iters: int, pool: ThreadPoolExecutor, jobs: int, stay: float = STAY, still: bool = False, seed: int = 0) -> tuple[list[Array], Array, list[float]]:
    """The knots' moves that bring the renders of tracks nearest their recordings, the voice
    that they share (steps from the one they started in; with `still`, the voice alone is
    searched and the tracks stay), and the error by iteration.

    A batch renders a track as it is and with every COLOURS-th knot moved along a random
    direction, one direction a candidate. A frame's error is a sum of squares (the envelope's
    distance in each band, the level's, and what the mouth's narrowness costs), and over a
    step as short as the draws the envelope changes about in a straight line with the track:
    so each knot takes the mix of its draws that would bring the envelope of the frames around
    it nearest their goal if the line held (a Gauss-Newton step within the draws' span,
    damped, and no farther than FAR draws). The next batch shows what the step gained against
    what it promised, and the knot's draws become longer or shorter by that. After the knots,
    the voice takes a step of the same kind over all frames of all the tracks (`voice_batch`);
    the consonants' frames are then held to what the tracks sound like in the new voice."""
    rng = np.random.default_rng(seed)
    ss = [Knots(t) for t in ts]
    voice, voice_step = np.zeros(ts[0].points + len(ts[0].statics)), VOICE_STEP
    log: list[float] = []
    for it in range(iters):
        if not still:
            errors = [[knots_batch(t, s, colour, voice, pool, jobs, stay, rng) for colour in range(COLOURS)][0] for t, s in zip(ts, ss)]
            log.append(float(np.sqrt(np.mean(np.square(errors)))))
        if len(voice):
            was = voice
            voice, voice_step, error = voice_batch(ts, ss, voice, voice_step, pool, jobs, rng)
            if still:
                log.append(error)
            if voice is not was:
                for t, first in zip(ts, pool.map(lambda t: t.render(t.base, voice), ts)):
                    t.listen(first)
                for s in ss:
                    s.before[:] = np.nan
        print(
            f"  {it + 1:3d}: {log[-1]:.2f} dB" + ("" if still else f"  (draws {np.mean([s.step[t.searched].mean() for t, s in zip(ts, ss)]):.2f} long)")
            + (f"  voice {' '.join(f'{x:+.2f}' for x in voice)}" if len(voice) else ""),
            flush=True,
        )  # fmt: skip
    log.append(float(np.sqrt(np.mean([t.error(t.errors(t.render(t.frames(s.moved), voice))) ** 2 for t, s in zip(ts, ss)]))))
    return [s.moved for s in ss], voice, log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "jvs")
    ap.add_argument("--variant", type=str, default="encv2")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--ids", type=str, default="", help="the recordings to search (comma-separated), whatever their split")
    ap.add_argument("--ids-file", type=Path, default=None, help="a file of recordings' names (comma-separated), of which --per of each speaker are taken")
    ap.add_argument("--per", type=int, default=1, help="recordings of each speaker, if no --ids")
    ap.add_argument("--form", type=str, default="", help="the form the tracks are searched in (a SPEC of pmvc.form)")
    ap.add_argument("--shape", type=int, default=0, help="also search a shape of the tract that tracks share: the area's logarithm at so many places from the glottis to the lips")
    ap.add_argument("--statics", type=str, default="", help=f"also search settings that tracks share (comma-separated, of {', '.join(STATICS)})")
    ap.add_argument("--share", choices=["speaker", "sex", "all"], default="speaker", help="whose tracks share a voice")
    ap.add_argument("--still", action="store_true", help="leave the tracks as they are and search the voice alone")
    ap.add_argument("--from", dest="start", type=str, default="", help="files of earlier searches (comma-separated): each track starts in the voice they have for its speaker, or for its sex")
    ap.add_argument("--out", type=str, default="reach")
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--free", type=str, default=",".join(MOVED), help=f"the channels to move (of {', '.join(FREE)})")
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
        among = set(a.ids_file.read_text().strip().split(",")) if a.ids_file else set(ids_in(d, a.split))
        count: dict[str, int] = {}
        kept = []
        for m in (m for m in every if m["id"] in among and m["id"] in aligned):
            count[m["speaker"]] = count.get(m["speaker"], 0) + 1
            if count[m["speaker"]] <= a.per:
                kept.append(m)
    free = a.free.split(",")
    statics = [k for k in a.statics.split(",") if k]
    shared = bool(a.shape or statics)
    assert shared or not a.still, "--still searches a voice: give --shape or --statics"
    starts = voices_in(a.start)
    (d / a.out).mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    voices: dict[str, dict[str, Any]] = {}
    before: list[Array] = []
    after: list[Array] = []
    # the tracks that share a voice are searched together, the others each alone
    groups: dict[str, list[dict[str, Any]]] = {}
    for m in kept:
        groups.setdefault(m["id"] if not shared else {"speaker": m["speaker"], "sex": m["sex"], "all": "all"}[a.share], []).append(m)
    kept = [m for group in groups.values() for m in group]
    with ThreadPoolExecutor(a.jobs) as pool:
        for name, group in groups.items():
            ts = list(pool.map(lambda m: Target(d, a.variant, m, pitch[m["speaker"]], aligned, a.form, free, a.shape, statics, voice_for(starts, m), a.astray), group))
            for t in ts:
                print(f"{t.meta['id']}: {len(t.knots)} knots, {int(t.searched.sum())} searched, {int(t.asked.sum())} frames asked, {int(t.held.sum())} held", flush=True)
            first = list(pool.map(lambda t: t.render(t.base), ts))
            moves, voice, log = search(ts, a.iters, pool, a.jobs, a.stay, a.still)
            if shared:
                voices[name] = ts[0].voice(voice)
            for t, moved, start in zip(ts, moves, first):
                m = t.meta
                frames = t.frames(moved)
                track = {**t.shaped(frames, voice), "frames": np.round(frames, 5).tolist()}
                (d / a.out / f"{m['id']}.tracks.json").write_text(json.dumps(track))
                sf.write(str(d / a.out / f"{m['id']}.wav"), render_track(track), 48000, subtype="PCM_16")
                # the vowels' envelope error as the tables have it, before and after
                c = Clip(d, a.variant, m, pitch[m["speaker"]], aligned)
                if a.astray:
                    c.x, c.real = t.x, c.envelope(t.x)
                said: dict[str, list[float]] = {"heard": [], "plain": []}
                for y, kept_in in ((start, before), (t.render(frames, voice), after)):
                    kept_in.append(y)
                    lack = c.lack(y)
                    said["heard"].append(heard_error(c.real, lack))
                    said["plain"].append(float(np.sqrt(((lack - lack.mean(axis=1, keepdims=True)) ** 2).mean())))
                went = moved[t.searched]
                row = {
                    "id": m["id"], "sex": m["sex"], "speaker": m["speaker"], "group": name, **said,
                    "went": {k: [float(went[:, i].mean()), float(np.sqrt((went[:, i] ** 2).mean()))] for i, k in enumerate(free)} if len(went) else {},
                }  # fmt: skip
                (d / a.out / f"{m['id']}.log.json").write_text(json.dumps({**row, "iterations": log}))
                rows.append(row)
                print(f"{m['id']}: vowels heard {said['heard'][0]:.2f} → {said['heard'][1]:.2f}, plain {said['plain'][0]:.2f} → {said['plain'][1]:.2f}", flush=True)
                if not a.still:
                    print("  went (mean, rms; units): " + "  ".join(f"{k} {x[0]:+.2f} {x[1]:.2f}" for k, x in row["went"].items()), flush=True)
            print(f"{name}: searched error {log[0]:.2f} → {log[-1]:.2f} dB" + (f"; voice {json.dumps({'profile': [round(x, 2) for x in voices[name]['profile']], 'set': {k: round(x, 3) for k, x in voices[name]['set'].items()}})}" if shared else ""), flush=True)
    out: dict[str, Any] = {"variant": a.variant, "form": a.form, "free": free, "iters": a.iters, "still": a.still, "share": a.share, "voices": voices, "clips": rows}
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
