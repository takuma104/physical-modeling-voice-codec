"""Closing the loop on timing: move the planner's phone boundaries until the render's voicing
starts and stops where the recording's does.

The planner's segments are gestures, not sounds. Told that a /k/ lasts 60 ms and the vowel after
it 50 ms, CHIMERA opens the glottis for its fixed 50 ms of aspiration and the vowel is gone. So
the recording's phone times, given to the planner as they are, do not come out as the
recording's timing. The onsets and offsets of voicing are landmarks both signals have, and a
pitch tracker finds them to a frame, so each one in the recording that sits at a phone boundary
is used to correct that boundary: render, measure how late the render's landmark is, move the
boundary that much earlier, and repeat.

A landmark only pulls a boundary where voicing can start or stop: an onset needs an obstruent
or silence before it, an offset one after it (a tracker's dropout inside a vowel is not a
landmark). And a boundary that does not answer is put back and left alone: the speaker
devoiced a /ð/ that CHIMERA voices, say, and no amount of moving that boundary changes it.
"""

from dataclasses import dataclass

import numpy as np

from .audio import FRAME_RATE, Array
from .phones import STOPS, VOWELS, Seg

# A recording's landmark belongs to a phone boundary at most this far from it (s).
NEAR: float = 0.045
# The render's landmark is looked for at most this far from where it should be (s).
REACH: float = 0.12
# Share of the measured error corrected per round.
GAIN: float = 0.8
# Errors up to this (s) are left alone: the landmarks are known to a 10 ms frame.
TOLERANCE: float = 0.015
# A boundary moves at most this far from where the aligner put it (s).
CAP: float = 0.07
# A move that takes less than this share of itself off the error did not work.
RESPONSE: float = 0.3
# The shortest a stop may be planned (s), its aspiration being planned after it; other phones
# keep at least `KEEP` of their aligned duration, and this much.
MIN_STOP: float = 0.02
MIN_DUR: float = 0.03
KEEP: float = 0.5

SONORANTS: frozenset[str] = VOWELS | {"m", "n", "ŋ", "l", "ɫ", "ɹ", "w", "j", "ɾ"}
VOICED_OBSTRUENTS: frozenset[str] = frozenset(["b", "d", "g", "v", "ð", "z", "ʒ", "dʒ"])
Key = tuple[int, int]


@dataclass(frozen=True)
class Edge:
    """A voicing onset (+1) or offset (-1) at `time` seconds."""

    time: float
    sign: int


def voiced_mask(f0: Array, min_run: int = 3) -> Array:
    """Which frames are voiced, with runs shorter than `min_run` frames merged into their
    surroundings (a tracker's dropouts and blips are not landmarks)."""
    v = f0 > 0
    for value in (False, True):
        edges = np.flatnonzero(np.diff(np.concatenate([[not value], v == value, [not value]]).astype(int)))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a < min_run and a > 0 and b < len(v):
                v[a:b] = not value
    return v


def edges(f0: Array) -> list[Edge]:
    v = voiced_mask(f0).astype(int)
    idx = np.flatnonzero(np.diff(v))
    return [Edge((i + 1) / FRAME_RATE, int(v[i + 1] - v[i])) for i in idx]


def _can_change(left: Seg | None, right: Seg | None, sign: int) -> bool:
    """Whether voicing can start (sign +1) or stop (-1) between these phones (None: silence)."""
    before, after = (left, right) if sign > 0 else (right, left)
    # `after` is the voiced side, `before` the side that may be voiceless
    return after is not None and (after.ipa in SONORANTS or after.ipa in VOICED_OBSTRUENTS) and (
        before is None or before.ipa not in SONORANTS
    )


def landmark_targets(phrases: list[list[Seg]], real_f0: Array) -> dict[Key, Edge]:
    """The recording's voicing landmarks, each given to the nearest phone boundary where voicing
    can change that way. A boundary is (phrase, index): index 0 is the phrase's start, index k
    the end of its k-th phone. Landmarks far from any such boundary are left out."""
    bs: list[tuple[Key, float, Seg | None, Seg | None]] = []
    for p, ph in enumerate(phrases):
        for k in range(len(ph) + 1):
            left, right = (ph[k - 1] if k > 0 else None), (ph[k] if k < len(ph) else None)
            bs.append(((p, k), right.start if right else ph[-1].end, left, right))
    out: dict[Key, tuple[float, Edge]] = {}
    for e in edges(real_f0):
        near = [(abs(t - e.time), key) for key, t, left, right in bs if _can_change(left, right, e.sign)]
        if near:
            d, key = min(near)
            if d <= NEAR and (key not in out or d < out[key][0]):
                out[key] = (d, e)
    return {key: e for key, (_, e) in out.items()}


def landmark_errors(targets: dict[Key, Edge], synth_f0: Array, shift: float) -> dict[Key, float]:
    """How late (s) the render's landmark is for each target (the render runs `shift` behind
    the recording). Targets the render has no such landmark near are left out."""
    es = edges(synth_f0)
    out: dict[Key, float] = {}
    for key, target in targets.items():
        same = [e.time - shift - target.time for e in es if e.sign == target.sign]
        if same:
            err = min(same, key=abs)
            if abs(err) <= REACH:
                out[key] = err
    return out


class BoundaryFit:
    """Phone boundaries under correction. `update` takes the errors measured on a render of
    `phrases` and moves the boundaries for the next render; with `move=False` it only takes
    back the last moves that did not work, which settles the fit."""

    def __init__(self, phrases: list[list[Seg]], earliest: float) -> None:
        self.segs = phrases
        self.orig: list[Array] = [np.array([ph[0].start, *(s.end for s in ph)]) for ph in phrases]
        self.b: list[Array] = [o.copy() for o in self.orig]
        self.earliest = earliest
        self.frozen: set[Key] = set()
        # the error a boundary's last move answered, and the move
        self.last: dict[Key, tuple[float, float]] = {}

    @property
    def phrases(self) -> list[list[Seg]]:
        return [
            [Seg(s.ipa, float(b[k]), float(b[k + 1]), s.stress, s.word_end) for k, s in enumerate(ph)]
            for ph, b in zip(self.segs, self.b)
        ]

    def update(self, errors: dict[Key, float], move: bool = True) -> None:
        for key, (prev, move) in list(self.last.items()):
            err = errors.get(key)
            if err is None or abs(err) > abs(prev) - RESPONSE * abs(move):
                self.b[key[0]][key[1]] -= move
                self.frozen.add(key)
        self.last.clear()
        for key, err in errors.items():
            if not move or key in self.frozen or abs(err) < TOLERANCE:
                continue
            p, k = key
            to = float(np.clip(self.b[p][k] - GAIN * err, self.orig[p][k] - CAP, self.orig[p][k] + CAP))
            self.last[key] = (err, to - self.b[p][k])
            self.b[p][k] = to
        floor = self.earliest
        for ph, b, o in zip(self.segs, self.b, self.orig):
            b[0] = max(b[0], floor)
            for k, s in enumerate(ph):
                shortest = MIN_STOP if s.ipa in STOPS else max(MIN_DUR, KEEP * (o[k + 1] - o[k]))
                b[k + 1] = max(b[k + 1], b[k] + min(shortest, o[k + 1] - o[k]))
            # the next breath group needs its lead-in after this one ends
            floor = b[-1] + 0.1
