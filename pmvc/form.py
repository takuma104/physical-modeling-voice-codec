"""What does another form of the tract do to the renders' envelopes? Forms tried, or fitted, on renders themselves.

The envelope of a render is far from its recording's (`pmvc.bench`), and part of that is the
engine's form: where its tract resonates and how sharply, whatever the track asks of the
articulators. The engine's frequency-domain model predicts a form's effect only roughly (it
has no vibrating glottis, whose opening damps the upper resonances; `pmvc.invert`). So here a
form is judged by what it sounds like: the variant's tracks are rendered again in the form
(some seconds for dozens of recordings), and the envelope's error in the vowels is measured as
`pmvc.bench` measures it.

usage: python -m pmvc.form [dir] [--variant encv2] [--split test | --ids FILE] [--per 6] [--try SPEC ...]
                           [--judges out/jvsall] [--name form]
       python -m pmvc.form [dir] --fit KEY=LO:HI[,KEY=LO:HI...] [--from SPEC] [--sex F] [--rounds 3] ...
       python -m pmvc.form [dir] --write VARIANT --try SPEC [--speakers FILE] [--split test]

A SPEC is settings separated by commas, each `part.key=value`: `form.viscous=4`,
`larynx.length=1.2` (`pmvc_voice::Player`'s `set_form` and `set_larynx`), `glottis.chink=0.02`,
`set.epilarynx=0.5` (a channel held at a value in every frame) or `shift.root=0.2` (a value
added to a channel in every frame). A setting can be for one sex's voices only: `F:larynx.length=1.3`.
`--try` prints, for each SPEC and each sex, the envelope's error: as far as it is heard
(`heard`: each band counted by how loud the recording is in it) and plain (dB rms over the
vowels' frames and all bands of 250 Hz up to 8 kHz alike; of which common to the sex, a
speaker's, a vowel's, the rest), and the common curve by band. `--judges` asks NISQA's two
models and a verifier too: the first fit, by the plain error alone, found a form (a strong
loss at the walls with a sharp closure) whose error was 1.7 dB less and which these judges put
far below V1 (NISQA quality 4.0 to 2.7, the men's similarity to themselves 0.91 to 0.80); in
its spectrograms the upper formants are smeared out. `--fit` searches the settings named,
each within its bounds, for the least error (`--by`: heard, or the plain total; of one sex's
voices with `--sex`, else of both together): one setting at a time by golden section,
`--rounds` times round. Both write <dir>/<name>.json. `--write` leaves the
split's tracks and renders in the form of the one SPEC as a variant of their own
(<dir>/<VARIANT>/<id>.tracks.json and .wav), for `pmvc.bench` and the other judges. With
`--speakers` (a file of `pmvc.speaker fit`), each speaker's renders are also under the settings
fitted to the speaker, on top of the form.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly

from . import gap
from .audio import FRAME_RATE, load
from .bench import CURVE, SEXES, STEP, TOP, shape, vowel_frames
from .encoder import LEAD, ids_in, render_track
from .paths import ROOT
from .speaker import settled

Array = NDArray[np.float64]

SR: int = gap.SR
HZ: Array = np.fft.rfftfreq(gap.FFT, 1 / SR)
BANDS: Array = np.stack([((HZ >= lo) & (HZ < lo + STEP)).astype(float) for lo in np.arange(0.0, TOP, STEP)])
BANDS = BANDS / BANDS.sum(axis=1, keepdims=True)
DB: float = 20.0 / np.log(10.0)
# `heard_error`: the power of a band's level that it counts by, and what every band counts by at the least
LOUD: float = 0.6
FAINT: float = 0.02


def settings(spec: str) -> list[tuple[str, str, str, float]]:
    """A SPEC as (sex or "", part, key, value)."""
    out = []
    for w in filter(None, spec.split(",")):
        name, value = w.split("=")
        sex, name = name.split(":") if ":" in name else ("", name)
        part, key = name.split(".")
        out.append((sex, part, key, float(value)))
    return out


def formed(track: dict[str, Any], sex: str, spec: list[tuple[str, str, str, float]]) -> dict[str, Any]:
    """A track under settings: those for its speaker's sex and those for both."""
    t = dict(track)
    frames: Array | None = None
    for whose, part, key, x in spec:
        if whose not in ("", sex):
            continue
        if part in ("set", "shift"):
            frames = np.array(t["frames"], dtype=np.float64) if frames is None else frames
            k = t["keys"].index(key)
            frames[:, k] = x if part == "set" else frames[:, k] + x
        else:
            t[part] = {**t.get(part, {}), key: x}
    if frames is not None:
        t["frames"] = frames.tolist()
    return t


class Clip:
    """A recording's vowel frames: its envelope there (dB, in bands), and how to take a render's."""

    def __init__(self, d: Path, variant: str, m: dict[str, Any], pitch: float, aligned: dict[str, Any]) -> None:
        self.meta = m
        uid = m["id"]
        self.track: dict[str, Any] = json.loads((d / variant / f"{uid}.tracks.json").read_text())
        self.x = load(d / "real" / f"{uid}.wav", SR)[0]
        self.cut = round(min(gap.CUT_MAX, gap.CUT_SHARE / pitch) * SR)
        kind, f0, phones = vowel_frames(d, uid, aligned)
        n = len(self.x) // gap.HOP
        at = [j for j in range(min(len(kind), len(f0), len(phones))) if kind[j] == "v" and f0[j] > 60 and round(j / FRAME_RATE * SR / gap.HOP) < n]
        self.frames = np.array([round(j / FRAME_RATE * SR / gap.HOP) for j in at], dtype=int)
        self.vowels = [phones[j] for j in at]
        self.real = self.envelope(self.x)

    def envelope(self, x: Array) -> Array:
        """The envelope of `x` (on the recording's clock) in the vowel frames, in bands (dB)."""
        mag = np.abs(gap.spectrum(x))[self.frames]
        return gap.envelope(mag, self.cut) @ BANDS.T * DB if len(mag) else np.zeros((0, len(BANDS)))

    def render(self, sex_spec: list[tuple[str, str, str, float]], own: dict[str, float] | None = None) -> Array:
        """The track's render in a form, on the recording's clock; with `own`, under the
        speaker's own settings on top of the form (`pmvc.speaker`'s)."""
        track = formed(self.track, self.meta["sex"], sex_spec)
        y = resample_poly(render_track(settled(track, own) if own else track), 1, 48000 // SR)[round(LEAD * SR) :]
        y = np.pad(y, (0, max(0, len(self.x) - len(y))))[: len(self.x)]
        return np.roll(y, -gap.lag(self.x, y))

    def lack(self, y: Array) -> Array:
        """What a render lacks of the recording, in the vowel frames (dB, in bands)."""
        return self.real - self.envelope(y)


def heard_error(real: Array, lack: Array) -> float:
    """The envelope's error as far as it is heard (dB rms): each band of each frame counts by
    how loud the recording is there (its level against the frame's loudest band, to the power
    LOUD: a band 20 dB down counts a quarter), with a little weight (FAINT) left for the bands
    where the recording has next to nothing; the level the two differ by, weighted likewise,
    is taken out of each frame. The plain rms counts every band of 250 Hz up to 8 kHz alike,
    and a render that loses the band a voice is strong in (2-4.5 kHz) while it loses the
    faint lines above it comes out ahead by it."""
    w = 10.0 ** (LOUD * (real - real.max(axis=1, keepdims=True)) / 20.0) + FAINT
    w = w / w.sum(axis=1, keepdims=True)
    d = lack - (w * lack).sum(axis=1, keepdims=True)
    return float(np.sqrt((w * d**2).sum(axis=1).mean()))


class Judges:
    """The other judges of a form's renders, for a warning: NISQA's two models, and how like
    the speaker a verifier finds a render (the cosine to the speaker's enrolled mean, and
    whether the speaker is the nearest of all enrolled). None of them is fitted to."""

    def __init__(self, enrol: Path, jobs: int) -> None:
        from .encoder import pick_device
        from .whose import Verifier

        self.verifier = Verifier(pick_device(), gap.VERIFIERS["wavlm"])
        self.means, self.sex = gap.enrolled(enrol, self.verifier, "wavlm", jobs)

    def of(self, clips: list[Clip], renders: list[Array]) -> dict[str, dict[str, float]]:
        import tempfile

        import soundfile as sf

        from .audio import at_level
        from .bench import LEVEL
        from .natural import score

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for c, y in zip(clips, renders):
                z = at_level(y, SR, LEVEL)
                paths.append(Path(tmp) / f"{c.meta['id']}.wav")
                sf.write(str(paths[-1]), z * min(1.0, 0.98 / max(float(np.abs(z).max()), 1e-9)), SR, subtype="PCM_16")
            rows = score(paths)
            who = gap.identify(np.stack([self.verifier.vector(load(p, 16000)[0]) for p in paths]), [c.meta["speaker"] for c in clips], self.means, self.sex)
        out: dict[str, dict[str, float]] = {}
        for sex in SEXES:
            mine = [r for r, c in zip(rows, clips) if c.meta["sex"] == sex]
            if mine:
                out[sex] = {"natural": float(np.mean([r["natural"] for r in mine])), "quality": float(np.mean([r["quality"] for r in mine])), "own": who["sexes"][sex]["own"], "top1": who["sexes"][sex]["top1"]}
        return out


def measure(clips: list[Clip], spec: str, pool: ThreadPoolExecutor, speakers: dict[str, Any] | None = None, judges: Judges | None = None) -> dict[str, Any]:
    """The envelope's error of the renders in a form, for each sex; with `speakers`
    (`pmvc.speaker`'s file's), each speaker's renders under the speaker's own settings too;
    with `judges`, what they make of the renders as well."""
    s = settings(spec)
    renders = list(pool.map(lambda c: c.render(s, (speakers or {}).get(c.meta["speaker"], {}).get("settings")), clips))
    rows = list(pool.map(lambda job: job[0].lack(job[1]), zip(clips, renders)))
    said_by = judges.of(clips, renders) if judges else {}
    out: dict[str, Any] = {}
    for sex in SEXES:
        mine = [(c, r) for c, r in zip(clips, rows) if c.meta["sex"] == sex and len(r)]
        if mine:
            lack = np.concatenate([r for _, r in mine])
            out[sex] = shape(lack, [c.meta["speaker"] for c, r in mine for _ in range(len(r))], [v for c, _ in mine for v in c.vowels])
            level = lack - lack.mean(axis=1, keepdims=True)
            centres = np.arange(0.0, TOP, STEP) + STEP / 2
            out[sex]["byBand"] = [float(np.sqrt((level[:, (centres >= lo) & (centres < hi)] ** 2).mean())) for lo, hi in CURVE]
            out[sex]["heard"] = heard_error(np.concatenate([c.real for c, _ in mine]), lack)
            out[sex].update(said_by.get(sex, {}))
    return out


def said(spec: str, m: dict[str, Any]) -> str:
    return "\n".join(
        f"  {sex} heard {m[sex]['heard']:5.2f}   plain {m[sex]['total']:5.2f}  ({m[sex]['common']:.1f}, {m[sex]['speaker']:.1f}, {m[sex]['vowel']:.1f}, {m[sex]['rest']:.1f})   curve "
        + " ".join(f"{x:+5.1f}" for x in m[sex]["curve"]) + "   rms by band " + " ".join(f"{x:4.1f}" for x in m[sex]["byBand"])
        + (f"   NISQA natural {m[sex]['natural']:.2f} quality {m[sex]['quality']:.2f}  own {m[sex]['own']:.3f} top-1 {100 * m[sex]['top1']:.0f} %" if "natural" in m[sex] else "")
        for sex in SEXES if sex in m
    )  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "jvs")
    ap.add_argument("--variant", type=str, default="encv2")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--ids", type=Path, default=None, help="a file of recordings' names (comma-separated) to take, whatever their split")
    ap.add_argument("--per", type=int, default=6, help="recordings of each speaker")
    ap.add_argument("--try", dest="tries", type=str, action="append", default=[], help="a form to measure (may be given more than once)")
    ap.add_argument("--fit", type=str, default="", help="settings to search, with their bounds: form.viscous=1:20,F:larynx.length=0.8:1.6")
    ap.add_argument("--from", dest="start", type=str, default="", help="the settings a fit starts from, and those it holds")
    ap.add_argument("--sex", type=str, default="", help="fit to the error of this sex's voices only")
    ap.add_argument("--by", choices=["heard", "total"], default="heard", help="what a fit lowers: the error as far as it is heard (bands weighted by the recording's loudness), or the plain rms over all bands")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=7, help="renders for each setting in a round")
    ap.add_argument("--speakers", type=Path, default=None, help="pmvc.speaker's file: each speaker's renders under the speaker's own settings, on top of a --try's form (those tries are named `SPEC +speakers`)")
    ap.add_argument("--judges", type=Path, default=None, help="also ask NISQA and a verifier of each --try's renders; the directory whose training recordings stand for the speakers (out/jvsall)")
    ap.add_argument("--write", type=str, default="", help="a variant to leave the tracks and renders of the one --try in")
    ap.add_argument("--name", type=str, default="form")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    d: Path = a.dir
    aligned: dict[str, Any] = json.loads((d / "aligned.json").read_text())
    metas = [json.loads((d / "meta" / f"{i}.json").read_text()) for i in (a.ids.read_text().strip().split(",") if a.ids else ids_in(d, a.split))]
    metas = [m for m in metas if m["id"] in aligned and (d / "real" / f"{m['id']}.wav").exists() and (d / a.variant / f"{m['id']}.tracks.json").exists()]
    pitch = {s: float(np.median([m["f0Median"] for m in metas if m["speaker"] == s])) for s in {m["speaker"] for m in metas}}
    fitted: dict[str, Any] = json.loads(a.speakers.read_text())["speakers"] if a.speakers else {}
    if a.write:
        import soundfile as sf

        spec = settings(a.tries[0])
        (d / a.write).mkdir(exist_ok=True)

        def write(m: dict[str, Any]) -> None:
            track = formed(json.loads((d / a.variant / f"{m['id']}.tracks.json").read_text()), m["sex"], spec)
            if m["speaker"] in fitted:
                track = settled(track, fitted[m["speaker"]]["settings"])
            (d / a.write / f"{m['id']}.tracks.json").write_text(json.dumps(track))
            sf.write(str(d / a.write / f"{m['id']}.wav"), render_track(track), 48000, subtype="PCM_16")

        with ThreadPoolExecutor(a.jobs) as pool:
            list(pool.map(write, metas))
        print(f"{len(metas)} tracks of {a.variant} in the form {a.tries[0]} → {d / a.write}")
        return
    count: dict[str, int] = {}
    kept: list[dict[str, Any]] = []
    for m in metas:
        count[m["speaker"]] = count.get(m["speaker"], 0) + 1
        if count[m["speaker"]] <= a.per and (not a.sex or not a.fit or m["sex"] == a.sex):
            kept.append(m)
    out: dict[str, Any] = {"variant": a.variant, "clips": len(kept), "tries": {}}
    with ThreadPoolExecutor(a.jobs) as pool:
        clips = list(pool.map(lambda m: Clip(d, a.variant, m, pitch[m["speaker"]], aligned), kept))
        print(f"{len(clips)} recordings, {sum(len(c.frames) for c in clips)} vowel frames; the envelope's error (dB rms; of which common to the sex, a speaker's, a vowel's, the rest)")
        judges = Judges(a.judges, a.jobs) if a.judges else None
        for spec in ["", *a.tries]:
            out["tries"][spec] = measure(clips, spec, pool, None, judges)
            print(spec or "as it is")
            print(said(spec, out["tries"][spec]), flush=True)
            if spec and fitted:
                out["tries"][f"{spec} +speakers"] = measure(clips, spec, pool, fitted, judges)
                print(f"{spec} +speakers")
                print(said(spec, out["tries"][f"{spec} +speakers"]))
        if a.fit:
            bounds = {w.split("=")[0]: tuple(float(x) for x in w.split("=")[1].split(":")) for w in a.fit.split(",")}
            now = {f"{(sex + ':') if sex else ''}{part}.{key}": x for sex, part, key, x in settings(a.start)}

            def error(values: dict[str, float]) -> float:
                m = measure(clips, ",".join(f"{k}={x:.5g}" for k, x in values.items()), pool)
                return float(np.sqrt(np.mean([m[s][a.by] ** 2 for s in m])))

            # a setting that is not in the start is at the middle of its bounds until its turn
            for k, (lo, hi) in bounds.items():
                now.setdefault(k, 0.5 * (lo + hi))
            best = error(now)
            print(f"fit from {best:.3f}: " + ", ".join(f"{k}={x:.4g}" for k, x in now.items()))
            g = (np.sqrt(5.0) - 1) / 2
            for r in range(a.rounds):
                for k, (lo, hi) in bounds.items():
                    lo, hi = max(lo, now[k] - (hi - lo) * 0.5 ** r), min(hi, now[k] + (hi - lo) * 0.5 ** r)
                    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
                    e1, e2 = error({**now, k: x1}), error({**now, k: x2})
                    for _ in range(a.steps - 2):
                        if e1 < e2:
                            hi, x2, e2 = x2, x1, e1
                            x1 = hi - g * (hi - lo)
                            e1 = error({**now, k: x1})
                        else:
                            lo, x1, e1 = x1, x2, e2
                            x2 = lo + g * (hi - lo)
                            e2 = error({**now, k: x2})
                    x, e = (x1, e1) if e1 < e2 else (x2, e2)
                    if e < best:
                        now[k], best = x, e
                    print(f"  round {r + 1} {k}: {now[k]:.4g}  → {best:.3f}", flush=True)
            spec = ",".join(f"{k}={x:.4g}" for k, x in now.items())
            out["fit"] = {"spec": spec, "error": best, "measure": measure(clips, spec, pool)}
            print(f"fitted: {spec}")
            print(said(spec, out["fit"]["measure"]))
    (d / f"{a.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
