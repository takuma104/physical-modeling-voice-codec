"""Time-warped control tracks: the speaker's timing without the planner.

Given the speaker's phone durations, CHIMERA's planner squashes the short phones: its gestures
have fixed costs. But it says the same phones well on its own rule durations (`pmvc.transplant`'s
`f0` variant), and that render's control track can be played back at any speed. So the track is
warped in time, phone by phone, onto the recording's timeline and driven with no planner: every
closure and release still happens, only faster or slower. The pitch channel is the speaker's
contour.

usage: python -m pmvc.warp [dir] [--rounds 3] [--jobs N] [--ids FA01_01,...]

Writes <dir>/warp/<id>.wav and <id>.tracks.json (dir defaults to out/transplant, as
`pmvc.transplant` left it). The render runs on the recording's timeline, `shift` behind it.

The warp is anchored at the phone boundaries. Where the recording has a voicing onset or offset
at a boundary and the rule render has the matching one, the anchor ties those two instants
together rather than the two boundaries, since a planned boundary and an aligner's are not the
same instant of the sound (the planner's stop ends at its release, the aligner's at the end of
its aspiration). After each render the anchors are corrected by where the landmarks came out.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .audio import FRAME_RATE, Array, fill_pitch, load, pitch_track
from .landmarks import GAIN, REACH, TOLERANCE, Edge, Key, edges, landmark_errors, landmark_targets
from .paths import ROOT
from .phones import Seg
from .synth import drive
from .transplant import F0_ADVANCE

# Anchors are kept at least this far apart on both timelines (s).
MIN_STEP: float = 0.01


def spec_phrases(spec: dict[str, Any]) -> list[list[Seg]]:
    """The phrases of a `pmvc-synth render` spec with explicit durations, on the voice clock."""
    out: list[list[Seg]] = []
    for ph in spec["phrases"]:
        t = ph["t0"]
        segs: list[Seg] = []
        for s in ph["segments"]:
            segs.append(Seg(s["ipa"], t, t + s["dur"], s["stress"], s["wordEnd"]))
            t += s["dur"]
        out.append(segs)
    return out


def boundary_keys(phrases: list[list[Seg]]) -> list[tuple[Key, float]]:
    return [((p, k), (ph[k].start if k < len(ph) else ph[-1].end)) for p, ph in enumerate(phrases) for k in range(len(ph) + 1)]


def nearest_edge(es: list[Edge], sign: int, t: float) -> float | None:
    same = [e.time for e in es if e.sign == sign and abs(e.time - t) <= REACH]
    return min(same, key=lambda x: abs(x - t)) if same else None


def monotone(src: Array, dst: Array) -> tuple[Array, Array]:
    """The anchor pairs that advance on both timelines (a pair that does not is dropped)."""
    keep = [0]
    for i in range(1, len(src)):
        if src[i] >= src[keep[-1]] + MIN_STEP and dst[i] >= dst[keep[-1]] + MIN_STEP:
            keep.append(i)
    return src[keep], dst[keep]


def warp_track(frames: Array, src: Array, dst: Array, n: int) -> Array:
    """`frames` (on the timeline of `src`) resampled so that what was at src[i] is at dst[i];
    before the first and after the last anchor the track runs at its own speed."""
    t = np.arange(n) / FRAME_RATE
    at = np.interp(t, dst, src)
    at = np.where(t < dst[0], src[0] + (t - dst[0]), at)
    at = np.where(t > dst[-1], src[-1] + (t - dst[-1]), at)
    x = np.clip(at * FRAME_RATE, 0, len(frames) - 1)
    i = np.floor(x).astype(int)
    j = np.minimum(i + 1, len(frames) - 1)
    a = (x - i)[:, None]
    return frames[i] * (1 - a) + frames[j] * a


def warp(uid: str, d: Path, rounds: int) -> dict[str, Any]:
    meta = json.loads((d / "meta" / f"{uid}.json").read_text())
    shift: float = meta["shift"]
    sex: str = meta["sex"]
    track = json.loads((d / "f0" / f"{uid}.tracks.json").read_text())
    frames = np.array(track["frames"])
    rule = spec_phrases(json.loads((d / "f0" / f"{uid}.spec.json").read_text()))
    real = spec_phrases(json.loads((d / "durf0" / f"{uid}.spec.json").read_text()))
    real_f0 = np.load(d / "pitch" / f"{uid}.npy")
    x, sr = load(d / "real" / f"{uid}.wav")
    n = int(round((len(x) / sr + shift + 0.2) * FRAME_RATE))

    # the recording's voicing landmarks at its phone boundaries (recording time), and the rule
    # render's landmark at the same boundary
    unshifted = [[Seg(s.ipa, s.start - shift, s.end - shift, s.stress, s.word_end) for s in ph] for ph in real]
    targets = landmark_targets(unshifted, real_f0)
    y, rate = load(d / "f0" / f"{uid}.wav")
    rule_edges = edges(pitch_track(y, rate, sex))
    src_list: list[float] = []
    dst_list: list[float] = []
    index: dict[Key, int] = {}
    for (key, t_rule), (_, t_real) in zip(boundary_keys(rule), boundary_keys(real)):
        if key in targets:
            hit = nearest_edge(rule_edges, targets[key].sign, t_rule)
            if hit is not None:
                index[key] = len(src_list)
                t_rule, t_real = hit, targets[key].time + shift
        src_list.append(t_rule)
        dst_list.append(t_real)
    src, dst = np.array(src_list), np.array(dst_list)

    # the speaker's contour on the voice clock, as in the `durf0` variant
    contour = fill_pitch(real_f0)
    pitch = np.interp(np.arange(n) / FRAME_RATE - shift + F0_ADVANCE, np.arange(len(contour)) / FRAME_RATE, contour)
    f0_col = track["keys"].index("f0")
    wav = d / "warp" / f"{uid}.wav"

    def render() -> dict[Key, float]:
        s, t = monotone(src, dst)
        out = warp_track(frames, s, t, n)
        out[:, f0_col] = pitch
        drive({**track, "frames": np.round(out, 5).tolist()}, wav)
        z, r = load(wav)
        return landmark_errors(targets, pitch_track(z, r, sex), shift)

    def score(errors: dict[Key, float]) -> float:
        return (sum(abs(e) for e in errors.values()) + REACH * (len(targets) - len(errors))) / max(1, len(targets))

    errors = render()
    first = score(errors)
    best, best_dst = first, dst.copy()
    for _ in range(rounds):
        for key, err in errors.items():
            if key in index and abs(err) >= TOLERANCE:
                dst[index[key]] -= GAIN * err
        errors = render()
        if score(errors) < best:
            best, best_dst = score(errors), dst.copy()
    if not np.array_equal(dst, best_dst):
        dst = best_dst
        render()
    return {"id": uid, "landmarks": len(targets), "anchored": len(index), "landmarkError": [round(1000 * first, 1), round(1000 * best, 1)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "transplant")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--ids", type=str, default="")
    a = ap.parse_args()
    d: Path = a.dir
    ids = a.ids.split(",") if a.ids else sorted(p.stem for p in (d / "meta").glob("*.json"))
    (d / "warp").mkdir(exist_ok=True)
    with ThreadPoolExecutor(a.jobs) as pool:
        rows = list(pool.map(lambda uid: warp(uid, d, a.rounds), ids))
    err = np.array([r["landmarkError"] for r in rows])
    (d / "warp.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} recordings → {d / 'warp'}; landmark error {err[:, 0].mean():.1f} → {err[:, 1].mean():.1f} ms")


if __name__ == "__main__":
    main()
