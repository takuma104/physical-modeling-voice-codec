"""How much of a recording's envelope must a render have? The envelope's error mended in steps.

`pmvc.gap`'s hybrid `S+envR` (a render with its recording's envelope in every frame) is the
render that the ear puts next to the recording. It mends everything at once: what every render
of the engine has wrong, what this speaker's tract does differently, how each sound is shaped,
and what happens from frame to frame. Here the same gain is given in steps, each made of
averages only, so that each step is what one kind of work could reach at best:
  S          the render as it is
  sex        one gain curve for a sex (the mean over the sex's other speakers): an engine whose
             form is right for everyone, and nothing else changed
  speaker    one curve for the speaker (the mean over the speaker's other clips): static
             settings of a speaker
  sound      a curve for each phone, of the sex (its other speakers): an engine in which every
             sound is right on average
  own        a curve for each phone, of the speaker (the speaker's other clips): the speaker's
             tract, each sound shaped as this speaker shapes it, whatever stands around it
  slow       the frames' own gain, but only as fast as a mean over SLOW seconds follows
  S+envR     the frames' own gain
A clip's own frames are in none of its averages (the speaker's other clips, the sex's other
speakers), so a step gives nothing that a setting could not have known. Phones are the
aligner's (`pmvc.align write`), a phone's curve cross-fades into the next over FADE seconds,
and a phone heard in fewer than FEW frames takes the step below (`own` falls to `speaker`,
`sound` to `sex`).

usage: python -m pmvc.ladder [dir] --variant encv2 --enrol out/jvsall --out DIR [--name ladder] [--per 25]
                             [--no-speaker] [--opinions]

Judges: the envelope's error in the vowels (`pmvc.bench`'s), the two verifiers among the
enrolled speakers, and with `--opinions` NISQA's two models and UTMOSv2. Writes the clips to
<out>/<step>/ and the table to <dir>/<name>.json.
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

from . import gap
from .audio import FRAME_RATE, at_level
from .bench import LEVEL, SEXES, shape, shape_rows, vowel_frames
from .encoder import LEAD, ids_in, pick_device
from .paths import ROOT

Array = NDArray[np.float64]

SR: int = gap.SR
STEPS: tuple[str, ...] = ("S", "sex", "speaker", "sound", "own", "slow", "S+envR")
FADE: float = 0.04
SLOW: float = 0.1
FEW: int = 40
QUIET: tuple[str, ...] = ("sil", "pau")


def phones_of(aligned: dict[str, Any], uid: str, n: int) -> list[str]:
    """The phone at each of a clip's `n` spectrum frames (the recording's clock)."""
    at: list[str] = []
    for p, k in zip(aligned[uid]["p"], aligned[uid]["frames"]):
        at += [p] * k
    at = at[round(LEAD * FRAME_RATE) :]  # the alignment is on the voice's clock
    j = np.minimum((np.arange(n) * gap.HOP / SR * FRAME_RATE).round().astype(int), len(at) - 1)
    return [at[i] for i in j]


def over_time(a: Array, seconds: float) -> Array:
    """Each column of `a` (frames, bins) as its mean over `seconds` around each frame."""
    k = max(1, round(seconds * SR / gap.HOP))
    padded = np.pad(a, ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
    c = np.cumsum(np.pad(padded, ((1, 0), (0, 0))), axis=0)
    return (c[k:] - c[:-k]) / k


class Heard:
    """A recording and its render taken apart: the render's spectrum, the gain that gives it
    the recording's envelope (frames, bins; log), the frames where both have sound, each
    frame's phone, and the gain's sums over each phone's frames."""

    def __init__(self, d: Path, variant: str, m: dict[str, Any], pitch: float, aligned: dict[str, Any]) -> None:
        self.meta = m
        x, y, _ = gap.pair(d, variant, m["id"])
        self.x, self.n = x, len(x)
        sx, self.sy = gap.spectrum(x), gap.spectrum(y)
        cut = round(min(gap.CUT_MAX, gap.CUT_SHARE / pitch) * SR)
        limit = gap.LIMIT / 20 * np.log(10)
        self.diff: NDArray[np.float32] = np.clip(gap.envelope(np.abs(sx), cut) - gap.envelope(np.abs(self.sy), cut), -limit, limit).astype(np.float32)
        level_x, level_y = np.abs(sx).sum(axis=1), np.abs(self.sy).sum(axis=1)
        self.both: NDArray[np.bool_] = (level_x > 0.03 * level_x.max()) & (level_y > 0.03 * level_y.max())
        self.phones = phones_of(aligned, m["id"], len(self.diff))
        self.sums: dict[str, tuple[Array, int]] = {}
        said = np.array(self.phones)
        for p in sorted(set(self.phones) - set(QUIET)):
            rows = self.diff[(said == p) & self.both]
            if len(rows):
                self.sums[p] = (rows.sum(axis=0, dtype=np.float64), len(rows))


class Sums:
    """Sums of gains by phone, over a set of clips; means of them with some clips left out."""

    def __init__(self, clips: list[Heard]) -> None:
        self.by: dict[str, tuple[Array, int]] = {}
        for c in clips:
            for p, (s, k) in c.sums.items():
                t = self.by.get(p)
                self.by[p] = (s.copy(), k) if t is None else (t[0] + s, t[1] + k)

    def mean(self, less: "Sums | None" = None) -> tuple[Array, dict[str, Array]]:
        """The mean gain over all frames and over each phone's (those with FEW frames or
        more), without the frames in `less`."""
        left = {p: (s - less.by[p][0], k - less.by[p][1]) if less is not None and p in less.by else (s, k) for p, (s, k) in self.by.items()}
        total = sum(k for _, k in left.values())
        whole = sum(s for s, _ in left.values()) / max(total, 1)
        return np.asarray(whole), {p: s / k for p, (s, k) in left.items() if k >= FEW}


def gain(c: Heard, whole: Array, by_phone: dict[str, Array] | None) -> Array:
    """A gain for each frame of a clip from averages: `whole` everywhere, or each frame's
    phone's curve where there is one, cross-faded."""
    if by_phone is None:
        return np.tile(whole, (len(c.diff), 1))
    return over_time(np.stack([by_phone.get(p, whole) for p in c.phones]), FADE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "jvs")
    ap.add_argument("--variant", type=str, default="encv2")
    ap.add_argument("--enrol", type=Path, default=ROOT / "out" / "jvsall", help="a directory whose training recordings stand for the speakers")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", type=str, default="ladder")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--per", type=int, default=25, help="recordings of each speaker")
    ap.add_argument("--no-speaker", action="store_true")
    ap.add_argument("--opinions", action="store_true", help="also ask NISQA's two models and UTMOSv2")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    d: Path = a.dir
    aligned: dict[str, Any] = json.loads((d / "aligned.json").read_text())
    metas = [json.loads((d / "meta" / f"{i}.json").read_text()) for i in ids_in(d, a.split)]
    metas = [m for m in metas if m["id"] in aligned and all((d / v / f"{m['id']}.wav").exists() for v in ("real", a.variant))]
    count: dict[str, int] = {}
    kept: list[dict[str, Any]] = []
    for m in metas:
        count[m["speaker"]] = count.get(m["speaker"], 0) + 1
        if count[m["speaker"]] <= a.per:
            kept.append(m)
    pitch = {s: float(np.median([m["f0Median"] for m in kept if m["speaker"] == s])) for s in count}
    with ThreadPoolExecutor(a.jobs) as pool:
        clips = list(pool.map(lambda m: Heard(d, a.variant, m, pitch[m["speaker"]], aligned), kept))
    print(f"{len(clips)} recordings of {len(count)} speakers")

    of_speaker = {s: Sums([c for c in clips if c.meta["speaker"] == s]) for s in count}
    of_sex = {x: Sums([c for c in clips if c.meta["sex"] == x]) for x in SEXES}

    def path_of(step: str, uid: str) -> Path:
        return a.out / step / f"{uid}.wav"

    def make(c: Heard) -> dict[str, tuple[Array, list[str]]]:
        m = c.meta
        mine = Sums([c])
        sex_whole, sex_phone = of_sex[m["sex"]].mean(of_speaker[m["speaker"]])
        own_whole, own_phone = of_speaker[m["speaker"]].mean(mine)
        diff = c.diff.astype(np.float64)
        gains: dict[str, Array | None] = {
            "S": None, "sex": gain(c, sex_whole, None), "speaker": gain(c, own_whole, None),
            "sound": gain(c, sex_whole, sex_phone), "own": gain(c, own_whole, own_phone),
            "slow": over_time(diff, SLOW), "S+envR": diff,
        }  # fmt: skip
        kind, f0, phones = vowel_frames(d, m["id"], aligned)
        rows: dict[str, tuple[Array, list[str]]] = {}
        for step in STEPS:
            g = gains[step]
            y = gap.signal(c.sy if g is None else c.sy * np.exp(g), c.n)
            rows[step] = shape_rows(c.x, y, pitch[m["speaker"]], kind, f0, phones)
            y = at_level(y, SR, LEVEL)
            p = path_of(step, m["id"])
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), y * min(1.0, 0.98 / max(float(np.abs(y).max()), 1e-9)), SR, subtype="PCM_16")
        return rows

    with ThreadPoolExecutor(a.jobs) as pool:
        made = list(pool.map(make, clips))

    out: dict[str, Any] = {"variant": a.variant, "clips": len(kept), "steps": {step: {} for step in STEPS}}
    for step in STEPS:
        cell = out["steps"][step]["shape"] = {}
        for x in SEXES:
            part = [(r[step], c.meta) for r, c in zip(made, clips) if c.meta["sex"] == x and len(r[step][0])]
            if part:
                cell[x] = shape(
                    np.concatenate([r for (r, _), _ in part]),
                    [m["speaker"] for (r, _), m in part for _ in range(len(r))],
                    [v for (_, vowels), _ in part for v in vowels],
                )
    who = [m["speaker"] for m in kept]
    path = {step: [path_of(step, m["id"]) for m in kept] for step in STEPS}
    for short, model in {} if a.no_speaker else gap.VERIFIERS.items():
        from .audio import load
        from .whose import Verifier

        verifier = Verifier(pick_device(), model)
        means, sex = gap.enrolled(a.enrol, verifier, short, a.jobs)
        with ThreadPoolExecutor(a.jobs) as pool:
            for step in STEPS:
                heard = pool.map(lambda p: load(p, 16000)[0], path[step])
                out["steps"][step][short] = gap.identify(np.stack([verifier.vector(x) for x in heard]), who, means, sex)
        del verifier
    (d / f"{a.name}.json").write_text(json.dumps(out, indent=1))
    if a.opinions:
        from .natural import score
        from .opinions import utmos

        for step in STEPS:
            rows = score(path[step])
            marks = {"quality": [r["quality"] for r in rows], "natural": [r["natural"] for r in rows], "utmos": utmos(path[step])}
            for k, values in marks.items():
                out["steps"][step][k] = {x: float(np.mean([v for v, m in zip(values, kept) if m["sex"] == x])) for x in SEXES}
        (d / f"{a.name}.json").write_text(json.dumps(out, indent=1))

    print(f"{'':10s}{'envelope error F / M (dB)':28s}" + ("" if a.no_speaker else f"{'top-1 WavLM F / M':20s}{'UniSpeech-SAT F / M':22s}{'own (WavLM)':12s}") + ("NISQA natural   quality     UTMOSv2" if a.opinions else ""))
    for step in STEPS:
        h = out["steps"][step]
        line = f"{step:10s}" + f"{h['shape']['F']['total']:5.1f} / {h['shape']['M']['total']:4.1f}".ljust(28)
        if not a.no_speaker:
            for short in gap.VERIFIERS:
                line += f"{100 * h[short]['sexes']['F']['top1']:5.1f} / {100 * h[short]['sexes']['M']['top1']:5.1f}".ljust(21)
            line += f"{h['wavlm']['own']:.3f}".ljust(12)
        if a.opinions:
            line += "   ".join(f"{h[k]['F']:.2f} / {h[k]['M']:.2f}" for k in ("natural", "quality", "utmos"))
        print(line)


if __name__ == "__main__":
    main()
