"""Calibrate the voice's fixed settings for speech: the gains of its three turbulence noise
sources (glottis, constrictions, teeth) and its source's pressure, tension, breathiness and
brightness.

CHIMERA's settings were tuned by ear and against a singer, under guards that protect singing;
its own speech judge reports sibilants 10-17 dB too quiet in the female voice and a source short
of energy at 1-4 kHz. These are properties of the decoder, the same for every utterance, so they
are fitted once per body: the control tracks of `pmvc.warp` (the speaker's timing) are replayed
under candidate settings and scored by `pmvc.listener` against the recordings.

usage: python -m pmvc.calibrate [dir] --sex F|M [--n 24] [--evals 250] [--jobs N]

Fits on `n` recordings of that sex (every k-th, so the speakers are mixed) and writes
<dir>/calibration-<sex>.json: the settings, and the listener's scores before and after on the
recordings it was fitted on and on the others.
"""

import argparse
import json
import os
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
    ("tension", 0.5, 0.2, 0.9),
    ("breathiness", 0.15, 0.02, 0.5),
    ("brightness", 0.25, 0.02, 0.8),
)
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


def make_voice(track: dict[str, Any], s: dict[str, float] | None) -> pmvc_engine.Voice:
    """A voice for a control track, under these settings (None: CHIMERA's own)."""
    source = dict(track["source"])
    if s:
        source.update({k: s[k] for k in ("pressure", "tension", "breathiness", "brightness")})
    v = pmvc_engine.Voice(json.dumps(track["body"]), json.dumps(source), track["keys"], track["rate"], track.get("speed", 6.0))
    if s:
        g = v.noise_gains
        v.noise_gains = (g[0] * s["noiseGlottis"], g[1] * s["noiseConstriction"], g[2] * s["noiseTeeth"])
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "transplant")
    ap.add_argument("--sex", choices=["F", "M"], required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--evals", type=int, default=250)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    d: Path = a.dir
    ids = sorted(p.stem for p in (d / "meta").glob(f"{a.sex}*.json"))
    step = max(1, len(ids) // a.n)
    fit_ids = ids[::step][: a.n]
    held_ids = [i for i in ids if i not in fit_ids]
    listener = Listener()
    fit, held = load_items(d, fit_ids, listener), load_items(d, held_ids, listener)

    def objective(theta: Array) -> float:
        dist, nll = evaluate(listener, fit, settings(theta), a.jobs)
        return dist + NLL_WEIGHT * nll

    x0 = np.zeros(len(PARAMS))
    # a first simplex that moves every setting by a third
    simplex = np.vstack([x0, *(np.eye(len(PARAMS))[i] * 0.3 for i in range(len(PARAMS)))])
    r = minimize(objective, x0, method="Nelder-Mead", options={"maxfev": a.evals, "initial_simplex": simplex, "xatol": 0.01, "fatol": 1e-4})
    best = settings(r.x)
    report: dict[str, Any] = {"sex": a.sex, "settings": best, "fitIds": fit_ids, "evals": int(r.nfev)}
    for name, items in (("fit", fit), ("held", held)):
        before, after = evaluate(listener, items, None, a.jobs), evaluate(listener, items, best, a.jobs)
        report[name] = {"n": len(items), "distance": [before[0], after[0]], "nll": [before[1], after[1]]}
        print(f"{name:5s} ({len(items)}): distance {before[0]:.3f} → {after[0]:.3f}   CTC loss {before[1]:.3f} → {after[1]:.3f}")
    print("settings: " + ", ".join(f"{k} {v:.3g}" for k, v in best.items()))
    (d / f"calibration-{a.sex}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
