"""The scoreboard for work on the voice: what a variant's renders are to their recordings, on the same terms for both.

Results so far set renders at 48 kHz against recordings at 24 kHz, each at the level it came
with, and that alone moved the judges: NISQA's quality score for the women's renders rises
from 3.85 to 4.35 when they are cut to the recordings' band. Here every clip is brought to
SR, to the recording's clock (a render's lead is cut and what little it still runs behind is
taken out) and to one speech level before any judge hears it.

usage: python -m pmvc.bench [dir] --variants encv2,encit [--enrol out/jvsall] [--name bench]
                            [--split test] [--per 25] [--judged results_rel] [--opinions]
                            [--keep DIR]

For the recordings of a split (`real`) and each variant's renders of them, which must be on
the recording's clock (an encoder's, or its tracks under another voice):
  who       two speaker-verification models (`pmvc.whose`): `top1`, the share of clips whose
            nearest speaker among those enrolled is their own; `own`, the cosine to the
            clip's speaker; `other`, to the other speakers of the sex. The enrolled are the
            speakers of `--enrol`, each standing in the mean vector of the training
            recordings there (none of them a test sentence). Without `--enrol`, left out.
            A verifier is deaf to a fixed colouring of the sound (`pmvc.gap`): what it
            follows is the envelope as it moves.
  natural, quality
            NISQA's two models (`pmvc.natural`), on the clip as it is (`dry`) and through a
            fixed recording condition (`room`: `pmvc.room`'s high-pass, tail and noise floor;
            the recordings are left as they were recorded). The naturalness model rewards a
            floor and a tail whatever the voice, so both are given.
  squim, utmos
            `pmvc.opinions`' two other predictors of a listener's score (`--opinions`; slow)
  shape     the spectral envelope in the vowels (`pmvc.gap`'s envelope, the frames
            `pmvc.bands` takes for a vowel's middle), recording minus render with each
            frame's level taken out, in 250 Hz bands under 8 kHz: the rms of the error (dB),
            and how much of it is common to all speakers of a sex (the voice's own colour),
            a speaker's own, a vowel's, a speaker's vowel's, and the rest; and the common
            curve itself over CURVE's bands. Needs <dir>/aligned.json (`pmvc.align write`).
  harmonic  how far the harmonics stand above what lies between them in the vowels, band by
            band (`pmvc.bands`), render minus recording (dB); `h2`, the second harmonic
            against the first, likewise
  judged    from a results file of `pmvc.judge` (`--judged`): Whisper large-v3-turbo's error
            rate and the phone mismatch, over the same clips
All by sex. Writes <dir>/<name>.json and prints the table. `--keep` leaves the clips as the
judges heard them in a directory (<variant>/ and <variant>+room/).
"""

import argparse
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from . import gap
from .audio import FRAME_RATE, Array, at_level, load
from .bands import EDGES, kinds, measures
from .encoder import LEAD, ids_in, pick_device, split_in
from .paths import ROOT
from .room import recorded

SR: int = gap.SR
LEVEL: float = -26.0
# the envelope's error is taken in bands this wide (Hz) under TOP
STEP: float = 250.0
TOP: float = 8000.0
# the bands the common curve is printed over (Hz)
CURVE: tuple[tuple[float, float], ...] = ((0, 500), (500, 1000), (1000, 2500), (2500, 3500), (3500, 4750), (4750, 7000), (7000, 8000))
SEXES: tuple[str, ...] = ("F", "M")


def clip(d: Path, variant: str, uid: str) -> Array:
    """A recording, or a variant's render of it on the recording's clock, at SR and LEVEL."""
    x = load(d / "real" / f"{uid}.wav", SR)[0] if variant == "real" else gap.pair(d, variant, uid)[1]
    return at_level(x, SR, LEVEL)


def vowel_frames(d: Path, uid: str, aligned: dict[str, Any]) -> tuple[NDArray[np.str_], Array, list[str]]:
    """For a recording's 10 ms frames (its own clock): `pmvc.bands`' kind, the pitch, the phone."""
    kind = kinds(aligned[uid]["p"], aligned[uid]["frames"])
    phones: list[str] = []
    for p, n in zip(aligned[uid]["p"], aligned[uid]["frames"]):
        phones += [p] * n
    return kind, np.load(d / "pitch" / f"{uid}.npy"), phones[round(LEAD * FRAME_RATE) :]


def vowel_rows(x: Array, y: Array, pitch: float, kind: NDArray[np.str_], f0: Array, phones: list[str]) -> tuple[Array, list[str], list[int]]:
    """The envelope of the recording `x` minus that of the render `y` (dB) in bands of STEP,
    for the frames in the middle of a vowel; each such frame's vowel; and which of the
    recording's 10 ms frames they are."""
    cut = round(min(gap.CUT_MAX, gap.CUT_SHARE / pitch) * SR)
    ex, ey = gap.envelope(np.abs(gap.spectrum(x)), cut), gap.envelope(np.abs(gap.spectrum(y)), cut)
    hz = np.fft.rfftfreq(gap.FFT, 1 / SR)
    bands = [(hz >= lo) & (hz < lo + STEP) for lo in np.arange(0.0, TOP, STEP)]
    rows, vowels, at = [], [], []
    for j in range(min(len(kind), len(f0), len(phones))):
        k = round(j / FRAME_RATE * SR / gap.HOP)
        if kind[j] == "v" and f0[j] > 60 and k < len(ex):
            diff = (ex[k] - ey[k]) * 20 / np.log(10)
            rows.append([diff[b].mean() for b in bands])
            vowels.append(phones[j])
            at.append(j)
    return np.array(rows).reshape(-1, len(bands)), vowels, at


def shape_rows(x: Array, y: Array, pitch: float, kind: NDArray[np.str_], f0: Array, phones: list[str]) -> tuple[Array, list[str]]:
    """`vowel_rows` without the frames' numbers."""
    rows, vowels, _ = vowel_rows(x, y, pitch, kind, f0, phones)
    return rows, vowels


def shape(rows: Array, speakers: list[str], vowels: list[str]) -> dict[str, Any]:
    """The envelope's error over one sex's frames, split by what it is common to."""
    a = rows - rows.mean(axis=1, keepdims=True)  # each frame's level out: the shape
    who, what = np.array(speakers), np.array(vowels)
    common = a.mean(axis=0)
    r = a - common
    own = {s: r[who == s].mean(axis=0) for s in sorted(set(speakers))}
    of_speaker = np.stack([own[s] for s in speakers])
    r = r - of_speaker
    by_vowel = {v: r[what == v].mean(axis=0) for v in sorted(set(vowels))}
    of_vowel = np.stack([by_vowel[v] for v in vowels])
    r = r - of_vowel
    cells = {(s, v): r[(who == s) & (what == v)].mean(axis=0) for s in own for v in by_vowel if ((who == s) & (what == v)).sum() > 5}
    of_both = np.stack([cells.get((s, v), np.zeros(a.shape[1])) for s, v in zip(speakers, vowels)])

    def rms(z: Array) -> float:
        return float(np.sqrt((z**2).mean()))

    centres = np.arange(0.0, TOP, STEP) + STEP / 2
    return {
        "frames": int(len(a)),
        "total": rms(a), "common": rms(np.tile(common, (len(a), 1))), "speaker": rms(of_speaker), "vowel": rms(of_vowel), "speakerVowel": rms(of_both), "rest": rms(r - of_both),
        "curve": [float(common[(centres >= lo) & (centres < hi)].mean()) for lo, hi in CURVE],
        "commonCurve": np.round(common, 2).tolist(),
    }  # fmt: skip


def by_sex(values: list[float], metas: list[dict[str, Any]]) -> dict[str, float]:
    return {s: float(np.mean([v for v, m in zip(values, metas) if m["sex"] == s])) for s in SEXES if any(m["sex"] == s for m in metas)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "jvs")
    ap.add_argument("--variants", type=str, required=True)
    ap.add_argument("--enrol", type=Path, default=None, help="a directory whose training recordings stand for the speakers (out/jvsall)")
    ap.add_argument("--name", type=str, default="bench")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--per", type=int, default=25, help="recordings of each speaker")
    ap.add_argument("--judged", type=str, default="", help="a results file of pmvc.judge in <dir>, by its name")
    ap.add_argument("--opinions", action="store_true", help="also ask SQUIM and UTMOSv2")
    ap.add_argument("--keep", type=Path, default=None, help="where to leave the clips as the judges heard them")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    d: Path = a.dir
    variants = a.variants.split(",")
    aligned: dict[str, Any] = json.loads((d / "aligned.json").read_text()) if (d / "aligned.json").exists() else {}
    metas = [json.loads((d / "meta" / f"{i}.json").read_text()) for i in ids_in(d, a.split)]
    metas = [m for m in metas if all((d / v / f"{m['id']}.wav").exists() for v in ("real", *variants))]
    count: dict[str, int] = {}
    kept: list[dict[str, Any]] = []
    for m in metas:
        count[m["speaker"]] = count.get(m["speaker"], 0) + 1
        if count[m["speaker"]] <= a.per:
            kept.append(m)
    pitch = {s: float(np.median([m["f0Median"] for m in kept if m["speaker"] == s])) for s in count}
    print(f"{len(kept)} recordings of {len(count)} speakers, at {SR} Hz and {LEVEL:.0f} dB")
    conditions = ("dry", "room")
    out: dict[str, Any] = {"clips": len(kept), "rate": SR, "level": LEVEL, "variants": {v: {} for v in ("real", *variants)}}

    with tempfile.TemporaryDirectory() as tmp:
        work = a.keep or Path(tmp)
        paths: dict[tuple[str, str], list[Path]] = {}
        rows: dict[str, list[tuple[Array, list[str]]]] = {}
        heard: dict[str, list[dict[str, Any] | None]] = {}

        def make(job: tuple[str, dict[str, Any]]) -> tuple[tuple[Array, list[str]] | None, dict[str, Any] | None]:
            v, m = job
            y = clip(d, v, m["id"])
            for cond in conditions if v != "real" else ("dry",):
                p = work / (v if cond == "dry" else f"{v}+room") / f"{m['id']}.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                z = y if cond == "dry" else recorded(y, SR)
                sf.write(str(p), z * min(1.0, 0.98 / max(float(np.abs(z).max()), 1e-9)), SR, subtype="PCM_16")
            if m["id"] not in aligned:
                return None, None
            kind, f0, phones = vowel_frames(d, m["id"], aligned)
            row = shape_rows(clip(d, "real", m["id"]), y, pitch[m["speaker"]], kind, f0, phones) if v != "real" else None
            return row, measures(y, kind, f0)

        with ThreadPoolExecutor(a.jobs) as pool:
            for v in ("real", *variants):
                done = list(pool.map(make, [(v, m) for m in kept]))
                rows[v], heard[v] = [r for r, _ in done if r is not None], [h for _, h in done]
                for cond in conditions if v != "real" else ("dry",):
                    paths[v, cond] = [work / (v if cond == "dry" else f"{v}+room") / f"{m['id']}.wav" for m in kept]

        # the envelope and the harmonics
        in_aligned = [m for m in kept if m["id"] in aligned]
        for v in variants if aligned else ():
            cell = out["variants"][v]
            cell["shape"] = {}
            for s in SEXES:
                part = [(r, vowels, m) for (r, vowels), m in zip(rows[v], in_aligned) if m["sex"] == s and len(r)]
                if part:
                    cell["shape"][s] = shape(np.concatenate([r for r, _, _ in part]), [m["speaker"] for r, _, m in part for _ in range(len(r))], [x for _, vowels, _ in part for x in vowels])
            both = [(h, hr, m) for h, hr, m in zip(heard[v], heard["real"], kept) if h is not None and hr is not None]
            cell["harmonic"] = {s: np.round(np.nanmean([np.array(h["harmonic"]) - np.array(hr["harmonic"]) for h, hr, m in both if m["sex"] == s], axis=0), 2).tolist() for s in SEXES}
            cell["h2"] = {s: float(np.mean([h["h2"] - hr["h2"] for h, hr, m in both if m["sex"] == s])) for s in SEXES}

        # who
        if a.enrol:
            from .whose import Verifier

            who = [m["speaker"] for m in kept]
            for short, model in gap.VERIFIERS.items():
                verifier = Verifier(pick_device(), model)
                means, sex = gap.enrolled(a.enrol, verifier, short, a.jobs)
                out["enrolled"] = len(means)
                with ThreadPoolExecutor(a.jobs) as pool:
                    for v in ("real", *variants):
                        clips = pool.map(lambda p: load(p, 16000)[0], paths[v, "dry"])
                        out["variants"][v][short] = gap.identify(np.stack([verifier.vector(x) for x in clips]), who, means, sex)
                del verifier

        # the predictors of a listener's score
        from .natural import score

        for (v, cond), ps in paths.items():
            got = score(ps)
            for measure in ("quality", "natural"):
                out["variants"][v].setdefault(measure, {})[cond] = by_sex([r[measure] for r in got], kept)
        # what there is so far, in case the slow judges below are stopped
        (d / f"{a.name}.json").write_text(json.dumps(out, indent=1))
        if a.opinions:
            from .opinions import squim, utmos

            enrol = a.enrol or d
            pool_ = [json.loads(p.read_text()) for p in sorted((enrol / "meta").glob("*.json"))]
            pool_ = [m for m in pool_ if split_in(enrol, m["id"]) == "train" and m["speaker"] not in count and (enrol / "real" / f"{m['id']}.wav").exists()]
            references = [enrol / "real" / f"{next(m for m in pool_ if m['sex'] == s)['id']}.wav" for s in SEXES if any(m["sex"] == s for m in pool_)]
            for v in ("real", *variants):
                if references:
                    out["variants"][v]["squim"] = by_sex(squim(paths[v, "dry"], references), kept)
                out["variants"][v]["utmos"] = by_sex(utmos(paths[v, "dry"]), kept)

    # what pmvc.judge counted
    if a.judged:
        by_id = {r["id"]: r["variants"] for r in json.loads((d / f"{a.judged}.json").read_text())["utterances"]}
        for v in ("real", *variants):
            cells = [(by_id[m["id"]][v], m) for m in kept if m["id"] in by_id and v in by_id[m["id"]]]
            if cells:
                out["variants"][v]["judged"] = {
                    "whisper": {s: sum(c["errors2"] for c, m in cells if m["sex"] == s) / max(1, sum(c["units"] for c, m in cells if m["sex"] == s)) for s in SEXES},
                    **({"phones": {s: sum(c["phoneErrors"] for c, m in cells if m["sex"] == s) / max(1, sum(c["phoneUnits"] for c, m in cells if m["sex"] == s)) for s in SEXES}} if v != "real" else {}),
                }

    (d / f"{a.name}.json").write_text(json.dumps(out, indent=1))

    def pair_of(x: dict[str, float] | None, fmt: str = "{:.2f}") -> str:
        return " / ".join(fmt.format(x[s]) for s in SEXES if s in x) if x else "-"

    print("\nwomen / men")
    head = f"{'':10s}" + (f"{'top1 % WavLM':>15s}{'UniSpeech':>15s}{'own (WavLM)':>15s}" if a.enrol else "") + f"{'natural':>13s}{'in a room':>13s}{'quality':>13s}{'in a room':>13s}"
    print(head + (f"{'SQUIM':>13s}{'UTMOSv2':>13s}" if a.opinions else "") + (f"{'Whisper %':>13s}{'phones %':>14s}" if a.judged else ""))
    for v in ("real", *variants):
        c = out["variants"][v]
        line = f"{v:10s}"
        if a.enrol:
            line += "".join(f"{pair_of({s: 100 * c[k]['sexes'][s]['top1'] for s in SEXES}, '{:.1f}'):>15s}" for k in gap.VERIFIERS)
            line += f"{pair_of({s: c['wavlm']['sexes'][s]['own'] for s in SEXES}, '{:.3f}'):>15s}"
        for measure in ("natural", "quality"):
            line += f"{pair_of(c[measure]['dry']):>13s}{pair_of(c[measure].get('room')):>13s}"
        if a.opinions:
            line += f"{pair_of(c.get('squim')):>13s}{pair_of(c.get('utmos')):>13s}"
        if a.judged:
            j = c.get("judged", {})
            line += f"{pair_of({s: 100 * x for s, x in j['whisper'].items()} if j else None, '{:.1f}'):>13s}{pair_of({s: 100 * x for s, x in j['phones'].items()} if 'phones' in j else None, '{:.1f}'):>14s}"
        print(line)
    if aligned:
        print("\nthe envelope in the vowels, recording minus render (dB)")
        print(f"{'':12s}{'rms':>6s}{'common':>8s}{'speaker':>8s}{'vowel':>7s}{'both':>6s}{'rest':>6s}   the common curve over " + " ".join(f"{lo / 1000:g}-{hi / 1000:g}k" for lo, hi in CURVE))
        for v in variants:
            for s, x in out["variants"][v]["shape"].items():
                print(f"{v:10s}{s:2s}{x['total']:6.1f}{x['common']:8.1f}{x['speaker']:8.1f}{x['vowel']:7.1f}{x['speakerVowel']:6.1f}{x['rest']:6.1f}   " + " ".join(f"{c:+6.1f}" for c in x["curve"]))
        print("\nharmonics above what lies between them, render minus recording (dB), in bands from (Hz) " + " ".join(f"{lo}" for lo, _ in pairwise(EDGES)) + " | h2")
        for v in variants:
            for s in SEXES:
                print(f"{v:10s}{s:2s}" + " ".join(f"{x:+5.1f}" for x in out["variants"][v]["harmonic"][s]) + f" | {out['variants'][v]['h2'][s]:+5.1f}")


if __name__ == "__main__":
    main()
