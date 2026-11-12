"""Calibrate the voice's fixed settings for speech: the gains of its three turbulence noise
sources (glottis, constrictions, teeth), its source's pressure, tension, breathiness,
brightness and roughness, and two parameters of the glottis the source's settings reach only
part of the range of: the open quotient (at most 0.78 by `tension`, where a woman's voice is
nearer 0.9) and the vibration's amplitude.

CHIMERA's settings were tuned by ear and against a singer, under guards that protect singing;
its own speech judge reports sibilants 10-17 dB too quiet in the female voice and a source short
of energy at 1-4 kHz. These are properties of the decoder, the same for every utterance, so they
are fitted once per body: the control tracks of `pmvc.warp` (the speaker's timing) are replayed
under candidate settings and scored by `pmvc.listener` against the recordings.

usage: python -m pmvc.calibrate [dir] --sex F|M [--n 24] [--evals 250] [--tracks warp] [--prefix P]
                                [--by listener|nisqa] [--start JSON] [--name calibration-<sex>]
                                [--jobs N]

Fits on `n` recordings of that sex (every k-th, so the speakers are mixed; `--prefix` keeps to
ids that start with it) and writes <dir>/<name>.json: the settings, and the scores before and
after on the recordings it was fitted on and on the others. `--tracks` names the variant whose
tracks are replayed, an encoder's as well as `pmvc.warp`'s.

`--by` is what the settings are fitted to:
  listener  the listener's feature distance to the recordings and the text's CTC loss. This
            moves little: the listener hears what is said, not how it sounds.
  nisqa     the quality NISQA predicts for the renders (`pmvc.natural`), less GUARD times the
            feature distance to the recordings as a multilingual listener hears it, so that a
            voice that sounds better does not say something else. This is the one for a voice
            that is understood but sounds wrong, as CHIMERA's female voice does: the same
            tracks an octave lower score 4.4, at a woman's pitch 2.9.
`pmvc.encoder encode --voice <dir>/<name>.json` then renders the recordings of speakers with a
high voice with the settings (`pmvc.pitchmix` says from which pitch they help).
"""

import argparse
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pmvc_engine
import torch
from scipy.optimize import minimize

from .audio import Array, load
from .listener import FEATURE_RATE, LISTEN_SR, Listener, distance, to16k
from .paths import ROOT

# name, default, lower, upper. The noise gains multiply CHIMERA's own.
PARAMS: tuple[tuple[str, float, float, float], ...] = (
    ("noiseGlottis", 1.0, 0.25, 8.0),
    ("noiseConstriction", 1.0, 0.25, 8.0),
    ("noiseTeeth", 1.0, 0.25, 8.0),
    ("pressure", 8.0, 5.0, 14.0),
    ("tension", 0.5, 0.02, 0.9),
    ("breathiness", 0.15, 0.02, 0.5),
    ("brightness", 0.25, 0.02, 0.8),
    ("roughness", 0.1, 0.001, 0.3),
    # set on the glottis directly (`pmvc_engine.Voice.set_glottis`); the defaults are what the
    # source's own defaults make them
    ("openQuotient", 0.6, 0.45, 0.97),
    ("amplitudeScale", 1.0, 0.3, 2.0),
)
# the source's settings among them, and the glottis's; the rest are the noise gains
SOURCE: tuple[str, ...] = ("pressure", "tension", "breathiness", "brightness", "roughness")
GLOTTIS: tuple[str, ...] = ("openQuotient", "amplitudeScale")
# with `--by nisqa`: what one unit of the listener's feature distance costs in opinion score,
# and the listener that hears it (`pmvc.yardstick` chose it)
GUARD: float = 5.0
GUARD_LISTENER: tuple[str, int] = ("utter-project/mHuBERT-147", 8)
# what one unit of CTC loss per character is worth against one unit of feature distance
NLL_WEIGHT: float = 0.1


@dataclass
class Item:
    """A recording, the control track that copies it, and what the listener heard in it."""

    uid: str
    text: str
    track: dict[str, Any]
    frames: Array
    # samples (48 kHz) the render runs behind the recording: the track's `shift` if it has one
    # (`pmvc.encoder`), else the meta's
    shift: int
    # the recording's speech in feature frames
    span: tuple[int, int]
    feats: torch.Tensor
    chars: list[int]


def settings(theta: Array) -> dict[str, float]:
    """The settings for an optimiser's point (each coordinate is the log of a ratio to the default)."""
    return {name: float(np.clip(default * np.exp(x), lo, hi)) for (name, default, lo, hi), x in zip(PARAMS, theta)}


def mixed(fitted: dict[str, float], w: float) -> dict[str, float]:
    """The settings a share `w` of the way from CHIMERA's own to `fitted`, in the fit's own
    coordinates: each setting's ratio to its default raised to `w`."""
    return {name: float(np.clip(default * (fitted[name] / default) ** w, lo, hi)) for name, default, lo, hi in PARAMS}


def make_voice(track: dict[str, Any], s: dict[str, float] | None) -> pmvc_engine.Voice:
    """A voice for a control track, under these settings (None: the track's own, which are
    CHIMERA's unless its `source`, `noise` and `glottis` say otherwise)."""
    source = dict(track["source"])
    if s:
        source.update({k: s[k] for k in SOURCE if k in s})
    v = pmvc_engine.Voice(json.dumps(track["body"]), json.dumps(source), track["keys"], track["rate"], track.get("speed", 6.0))
    gains = (s["noiseGlottis"], s["noiseConstriction"], s["noiseTeeth"]) if s else track.get("noise")
    if gains:
        g = v.noise_gains
        v.noise_gains = (g[0] * gains[0], g[1] * gains[1], g[2] * gains[2])
    glottis = {k: s[k] for k in GLOTTIS if k in s} if s else track.get("glottis")
    if glottis:
        v.set_glottis(json.dumps(glottis))
    return v


def render(item: Item, s: dict[str, float] | None) -> Array:
    v = make_voice(item.track, s)
    v.set_track(item.frames, 0)
    return np.asarray(v.render((len(item.frames) - 1) * (pmvc_engine.SAMPLE_RATE // int(item.track["rate"]))))


def load_items(d: Path, ids: list[str], listener: Listener, variant: str = "warp") -> list[Item]:
    items: list[Item] = []
    for uid in ids:
        meta = json.loads((d / "meta" / f"{uid}.json").read_text())
        track = json.loads((d / variant / f"{uid}.tracks.json").read_text())
        real, _ = load(d / "real" / f"{uid}.wav", LISTEN_SR)
        span = (int(meta["span"][0] * FEATURE_RATE), int(meta["span"][1] * FEATURE_RATE))
        items.append(
            Item(uid, meta["text"], track, np.array(track["frames"]), round(track.get("shift", meta["shift"]) * 48000), span,
                 listener.hear(real).feats[0], listener.chars(meta["text"]))
        )  # fmt: skip
    return items


def score(listener: Listener, item: Item, x48: Array) -> tuple[float, float]:
    """The listener's feature distance to the recording over its speech, and the CTC loss of
    the sentence per character, for a render of `item`."""
    heard = listener.hear(to16k(x48[item.shift :]))
    a, b = item.span
    dist = float(distance(item.feats[a:b], heard.feats[0][a:b]).mean())
    return dist, listener.nll(heard.logp[0], item.chars)


def evaluate(listener: Listener, items: list[Item], s: dict[str, float] | None, jobs: int) -> tuple[float, float]:
    """Mean feature distance and CTC loss over `items` under settings `s`."""
    with ThreadPoolExecutor(jobs) as pool:
        audio = list(pool.map(lambda it: render(it, s), items))
    scores = np.array([score(listener, it, x) for it, x in zip(items, audio)])
    return float(scores[:, 0].mean()), float(scores[:, 1].mean())


def heard_quality(listener: Listener, items: list[Item], s: dict[str, float] | None, jobs: int) -> tuple[float, float]:
    """NISQA's mean quality for `items` rendered under settings `s`, and the listener's mean
    feature distance to their recordings."""
    import soundfile as sf

    from .natural import score as nisqa

    with ThreadPoolExecutor(jobs) as pool:
        audio = list(pool.map(lambda it: render(it, s), items))
    with tempfile.TemporaryDirectory() as tmp:
        wavs = [Path(tmp) / f"{k}.wav" for k in range(len(items))]
        for wav, x in zip(wavs, audio):
            sf.write(str(wav), x, pmvc_engine.SAMPLE_RATE, subtype="PCM_16")
        quality = float(np.mean([q["quality"] for q in nisqa(wavs, natural=False)]))
    return quality, float(np.mean([score(listener, it, x)[0] for it, x in zip(items, audio)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "transplant")
    ap.add_argument("--sex", choices=["F", "M"], required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--evals", type=int, default=250)
    ap.add_argument("--tracks", type=str, default="warp", help="the variant whose tracks are replayed")
    ap.add_argument("--prefix", type=str, default="", help="only recordings whose ids start with this")
    ap.add_argument("--by", choices=["listener", "nisqa"], default="listener", help="what the settings are fitted to")
    ap.add_argument("--name", type=str, default="", help="the file to write, without .json (calibration-<sex>)")
    ap.add_argument("--start", type=str, default="{}", help="settings to start the search from, as JSON (CHIMERA's own where it does not say)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    d: Path = a.dir
    ids = sorted(p.name.split(".")[0] for p in (d / a.tracks).glob(f"{a.prefix}*.tracks.json"))
    ids = [i for i in ids if json.loads((d / "meta" / f"{i}.json").read_text())["sex"] == a.sex]
    step = max(1, len(ids) // a.n)
    fit_ids = ids[::step][: a.n]
    held_ids = [i for i in ids if i not in fit_ids]
    listener = Listener(None, *GUARD_LISTENER) if a.by == "nisqa" else Listener()
    fit, held = load_items(d, fit_ids, listener, a.tracks), load_items(d, held_ids, listener, a.tracks)

    def scores(items: list[Item], s: dict[str, float] | None) -> tuple[float, float]:
        """The two numbers the fit goes by: quality and distance, or distance and CTC loss."""
        return heard_quality(listener, items, s, a.jobs) if a.by == "nisqa" else evaluate(listener, items, s, a.jobs)

    def objective(theta: Array) -> float:
        first, second = scores(fit, settings(theta))
        return GUARD * second - first if a.by == "nisqa" else first + NLL_WEIGHT * second

    start: dict[str, float] = json.loads(a.start)
    x0 = np.array([np.log(start.get(name, default) / default) for name, default, _, _ in PARAMS])
    # a first simplex that moves every setting by a third
    simplex = np.vstack([x0, *(x0 + np.eye(len(PARAMS))[i] * 0.3 for i in range(len(PARAMS)))])
    r = minimize(objective, x0, method="Nelder-Mead", options={"maxfev": a.evals, "initial_simplex": simplex, "xatol": 0.01, "fatol": 1e-4})
    best = settings(r.x)
    names = ("quality", "distance") if a.by == "nisqa" else ("distance", "nll")
    report: dict[str, Any] = {"sex": a.sex, "by": a.by, "tracks": a.tracks, "settings": best, "fitIds": fit_ids, "evals": int(r.nfev)}
    for name, items in (("fit", fit), ("held", held)):
        if items:
            before, after = scores(items, None), scores(items, best)
            report[name] = {"n": len(items), names[0]: [before[0], after[0]], names[1]: [before[1], after[1]]}
            print(f"{name:5s} ({len(items)}): {names[0]} {before[0]:.3f} → {after[0]:.3f}   {names[1]} {before[1]:.3f} → {after[1]:.3f}")
    print("settings: " + ", ".join(f"{k} {v:.3g}" for k, v in best.items()))
    (d / f"{a.name or f'calibration-{a.sex}'}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
