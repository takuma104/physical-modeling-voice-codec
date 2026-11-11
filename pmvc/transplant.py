"""Prosody transplant: CHIMERA says a real recording's sentence with the speaker's phones and
some or all of the speaker's prosody, its articulation still planned by rule.

usage: python -m pmvc.transplant [--n 200] [--seed 0] [--out out/transplant] [--jobs N] [--ids FA01_01,...]

For each chosen TSP recording it writes, under <out>:
  real/<id>.wav     the recording
  rule/<id>.wav     CHIMERA as shipped (`chimera render text`): its own G2P, durations, intonation
  f0/<id>.wav       the speaker's phones and pitch contour, on the durations CHIMERA's rules give
                    those phones (the contour is stretched phone by phone to fit)
  floor/<id>.wav    the speaker's durations where CHIMERA can say them: no vowel under 70 ms, no
                    consonant under 50 ms; the contour stretched to fit
  durf0/<id>.wav    the speaker's durations and pitch contour, as they are
  fit/<id>.wav      the same, with the phone boundaries moved until the render's voicing starts
                    and stops where the recording's does (pmvc.landmarks)
  <variant>/<id>.tracks.json   the control track of the f0 and fit renders
  meta/<id>.json    what was transplanted; pitch/<id>.npy, the speaker's pitch track (0 = unvoiced)

`durf0` and `fit` run on the recording's timeline, `shift` seconds (in meta) behind it, since
the planner needs room before the first phone. The others have their own timing.
"""

import argparse
import json
import math
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .audio import FRAME_RATE, Array, fill_pitch, load, pitch_track
from .landmarks import REACH, BoundaryFit, landmark_errors, landmark_targets
from .paths import ROOT
from .phones import DIPHTHONGS, VOWELS, Seg, boundary_times, phrase_specs, relayout, to_phrases, to_segments
from .synth import plan_spec, render_spec, render_text
from .tsp import Utterance, load_utterances

# The first phone starts no earlier than this on the voice clock (s): the planner's 60 ms
# breath lead, time for the breath to rise, and room for the fit to start the phrase earlier.
START_MIN: float = 0.2
# The earliest a phrase may start on the voice clock (s): the planner's lead.
START_FLOOR: float = 0.07
# Rounds of boundary correction for the `fit` variant.
FIT_ROUNDS: int = 4
# The voice glides toward its pitch target with a 30 ms time constant, so the contour is
# given this much early (s) and arrives on time.
F0_ADVANCE: float = 0.03
# The shortest phones of the `floor` variant (s). A third of the speakers' unstressed vowels
# are under 50 ms; CHIMERA's rules give none under 60.
FLOOR_VOWEL: float = 0.07
FLOOR_CONSONANT: float = 0.05

VARIANTS: tuple[str, ...] = ("real", "rule", "f0", "floor", "durf0", "fit")
# the variants on the recording's timeline
ALIGNED: tuple[str, ...] = ("durf0", "fit")


def choose(utts: dict[str, Utterance], n: int, seed: int) -> list[Utterance]:
    """`n` recordings, half from women and half from men, spread over the speakers."""
    rng = random.Random(seed)
    ok = [u for u in utts.values() if all(p.label not in ("spn", "sil", "") for p in u.phones)]
    out: list[Utterance] = []
    for sex in ("F", "M"):
        by_speaker: dict[str, list[Utterance]] = {}
        for u in sorted(ok, key=lambda u: u.id):
            if u.sex == sex:
                by_speaker.setdefault(u.speaker, []).append(u)
        for group in by_speaker.values():
            rng.shuffle(group)
        # round-robin over the speakers
        speakers = sorted(by_speaker)
        k = 0
        while sum(u.sex == sex for u in out) < n // 2 and any(by_speaker.values()):
            group = by_speaker[speakers[k % len(speakers)]]
            if group:
                out.append(group.pop())
            k += 1
    return sorted(out, key=lambda u: u.id)


def warp_contour(contour: Array, real: list[list[Seg]], new: list[list[Seg]]) -> Array:
    """The recording's contour on another timing of the same phones: each phone's stretch of
    the contour is stretched or squeezed to that phone's new span."""
    n = math.ceil((new[-1][-1].end + 1.0) * FRAME_RATE)
    at = np.interp(np.arange(n) / FRAME_RATE, boundary_times(new), boundary_times(real))
    return np.interp(at, np.arange(len(contour)) / FRAME_RATE, contour)


def transplant(utt: Utterance, out: Path) -> dict[str, Any]:
    x, sr = load(utt.wav)
    f0 = pitch_track(x, sr, utt.sex)
    contour = fill_pitch(f0)
    median = float(np.median(f0[f0 > 0]))
    phrases = to_phrases(to_segments(utt))
    shift = math.ceil(max(0.0, START_MIN - phrases[0][0].start) * FRAME_RATE) / FRAME_RATE

    def spec(ph: list[list[Seg]], pitch: Array | None, tail: float = 0.3, durations: bool = True) -> dict[str, Any]:
        s: dict[str, Any] = {
            "body": utt.body,
            "source": {"f0": round(median, 2)},
            "phrases": phrase_specs(ph, shift, durations),
            "tail": round(tail, 3),
        }
        if pitch is not None:
            # on the voice clock: `shift` later than the phrases' own times, less the glide's lag
            lead = round((shift - F0_ADVANCE) * FRAME_RATE)
            track = np.concatenate([np.full(max(0, lead), pitch[0]), pitch[max(0, -lead) :]])
            s["f0Track"] = {"rate": FRAME_RATE, "values": [round(float(v), 2) for v in track]}
        return s

    def voicing(variant: str) -> Array:
        y, rate = load(out / variant / f"{utt.id}.wav")
        return pitch_track(y, rate, utt.sex)

    shutil.copyfile(utt.wav, out / "real" / f"{utt.id}.wav")
    render_text(utt.text, utt.body, out / "rule" / f"{utt.id}.wav")

    # f0: the durations the planner's rules give these phones (phrases planned far apart, so
    # that their rule durations cannot collide), then the phrases laid out with the speaker's
    # pauses and the contour stretched onto them
    apart = [[Seg(s.ipa, s.start + 30.0 * n, s.end + 30.0 * n, s.stress, s.word_end) for s in ph] for n, ph in enumerate(phrases)]
    planned = iter(plan_spec(spec(apart, None, durations=False), out / "f0" / utt.id)["segments"])
    rule_durs: list[list[float]] = []
    for ph in phrases:
        rule_durs.append([])
        for s in ph:
            parts = [next(planned) for _ in range(2 if s.ipa in DIPHTHONGS else 1)]
            rule_durs[-1].append(parts[-1]["t1"] - parts[0]["t0"])
    ruled = relayout(phrases, rule_durs)
    render_spec(spec(ruled, warp_contour(contour, phrases, ruled)), out / "f0" / f"{utt.id}.wav", tracks=True)

    # floor: the speaker's durations, none shorter than CHIMERA can say
    floored = relayout(phrases, [[max(s.dur, FLOOR_VOWEL if s.ipa in VOWELS else FLOOR_CONSONANT) for s in ph] for ph in phrases])
    render_spec(spec(floored, warp_contour(contour, phrases, floored)), out / "floor" / f"{utt.id}.wav")

    # durf0: the speaker's timing as it is, the render as long as the recording
    def tail(ph: list[list[Seg]]) -> float:
        return max(0.3, len(x) / sr - ph[-1][-1].end)

    info = render_spec(spec(phrases, contour, tail(phrases)), out / "durf0" / f"{utt.id}.wav")

    # fit: the recording's voicing landmarks, each at a phone boundary, pull that boundary
    targets = landmark_targets(phrases, f0)
    fit = BoundaryFit(phrases, START_FLOOR - shift)

    def measure(variant: str) -> tuple[dict[tuple[int, int], float], float]:
        """The landmark errors of a render and their mean (s); a landmark the render lacks
        counts as one at the reach."""
        errors = landmark_errors(targets, voicing(variant), shift)
        total = sum(abs(e) for e in errors.values()) + REACH * (len(targets) - len(errors))
        return errors, total / max(1, len(targets))

    errors, before = measure("durf0")
    after = before
    for n in range(FIT_ROUNDS + 1):
        # the last round moves nothing: it only takes back what did not work
        fit.update(errors, move=n < FIT_ROUNDS)
        info = render_spec(spec(fit.phrases, contour, tail(fit.phrases)), out / "fit" / f"{utt.id}.wav", tracks=True)
        errors, after = measure("fit")
    if after >= before:
        info = render_spec(spec(phrases, contour, tail(phrases)), out / "fit" / f"{utt.id}.wav", tracks=True)
        after = before

    np.save(out / "pitch" / f"{utt.id}.npy", f0)
    meta = {
        "id": utt.id,
        "speaker": utt.speaker,
        "sex": utt.sex,
        "body": utt.body,
        "text": utt.text,
        "shift": shift,
        "aligned": list(ALIGNED),
        "f0Median": median,
        "phrases": len(phrases),
        # the recording's first and last phone boundaries (s, recording time)
        "span": [phrases[0][0].start, phrases[-1][-1].end],
        "ipa": " | ".join("".join(s.ipa for s in ph) for ph in phrases),
        # voicing landmarks used by the fit, and their mean error (ms) before and after it
        "landmarks": len(targets),
        "landmarkError": [round(1000 * before, 1), round(1000 * after, 1)],
        "peak": info["peak"],
        "nonFinite": info["nonFinite"],
    }
    (out / "meta" / f"{utt.id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "transplant")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--ids", type=str, default="", help="comma-separated recording ids instead of a random choice")
    a = ap.parse_args()
    utts = load_utterances()
    chosen = [utts[i] for i in a.ids.split(",")] if a.ids else choose(utts, a.n, a.seed)
    for d in (*VARIANTS, "meta", "pitch"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.jobs) as pool:
        metas = list(pool.map(lambda u: transplant(u, a.out), chosen))
    bad = [m["id"] for m in metas if m["nonFinite"]]
    print(f"{len(metas)} recordings → {a.out}" + (f"; non-finite samples in {bad}" if bad else ""))


if __name__ == "__main__":
    main()
