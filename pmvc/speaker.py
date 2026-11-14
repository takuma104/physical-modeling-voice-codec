"""Fit the voice to a speaker: the body and the glottis under which the speaker's control
tracks sound most like the speaker.

Of a speaker the voice is told the sex, which picks one of two bodies, and the median pitch;
a speaker-verification model then cannot tell its renders of one speaker from those of another
of the same sex (`pmvc.whose`). But CHIMERA's body is continuous (from the adult male to the
adult female, and from adult to child: the size of the tract and how much of it is throat), and
so are the settings of its source that differ between people. Here they are fitted for each
speaker, a handful of numbers to go with the speaker's tracks:
  female, child   the body (0 to 1 each; the sex gives 0 or 1 and no child)
  openQuotient    the share of a cycle the glottis is open (pressed to breathy)
  brightness      how sharply the glottis closes (the slope of the source's spectrum)
  tension         the folds' tension (what else of the pulse follows from it)

usage: python -m pmvc.speaker fit [dir] [--tracks enci] [--split train] [--n 6] [--of 20]
                                  [--only female,child] [--evals 90] [--name speakers]
                                  [--seconds 4] [--at-once 3] [--speakers a,b]

For each speaker of <dir>: the first `--seconds` of `--n` of the split's recordings (spread
over them) are rendered from
their tracks (`--tracks`, an encoder's) under candidate settings, starting from the tracks' own,
and a verification model (`pmvc.whose`'s) says how close the renders are to the speaker: the
cosine between a render's vector and the mean vector of `--of` of the speaker's recordings.
Each setting is first tried on its own over its range, then all are searched together
(`--evals` renders of the recordings in all). The settings kept are those with the highest
mean cosine, less GUARD for every unit by which the
renders move away from their recordings as `pmvc.listener` hears them: a voice more like the
speaker's must not say something else. `--only` fits these settings and leaves the others as
the tracks have them. Writes <dir>/<name>.json, a speaker at a time (speakers already in it are
not fitted again):
  speakers.<name>.settings   the fitted settings
  speakers.<name>.fit        the cosine and the listener's distance before and after, on the
                             fitted recordings
`pmvc.encoder revoice --speakers <file>` renders a variant's tracks under them; whether the
voice is then the speaker's is for `pmvc.whose` to say on recordings that were not fitted, and
for the ear.
"""

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import minimize

from .audio import load
from .encoder import ids_in, pick_device, render_track
from .listener import FEATURE_RATE, Listener, distance, to16k
from .paths import ROOT
from .whose import SR, Verifier

# a setting, the part of a track it is in, its range, and what it is where a track does not
# say (CHIMERA's own; the open quotient is what the source's defaults make it)
PARAMS: tuple[tuple[str, str, float, float, float], ...] = (
    ("female", "body", 0.0, 1.0, 0.0),
    ("child", "body", 0.0, 0.5, 0.0),
    ("openQuotient", "glottis", 0.45, 0.95, 0.6),
    ("brightness", "source", 0.02, 0.8, 0.25),
    ("tension", "source", 0.02, 0.9, 0.5),
)
# what one unit of the listener's distance to the recordings costs in cosine, and the listener
GUARD: float = 2.0
GUARD_LISTENER: tuple[str, int] = ("utter-project/mHuBERT-147", 8)
# where in its range each setting is tried on its own before the search
SCAN: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)


def settled(track: dict[str, Any], s: dict[str, float]) -> dict[str, Any]:
    """`track` with a speaker's settings in place of its own."""
    out = {**track, "body": dict(track["body"]), "source": dict(track["source"]), "glottis": dict(track.get("glottis", {}))}
    for name, part, _, _, _ in PARAMS:
        if name in s:
            out[part][name] = round(float(s[name]), 5)
    if not out["glottis"]:
        del out["glottis"]
    return out


def own(track: dict[str, Any]) -> dict[str, float]:
    """The settings a track has."""
    return {name: float(track.get(part, {}).get(name, default)) for name, part, _, _, default in PARAMS}


class Judges:
    """The verification model and the listener, one clip at a time whoever asks."""

    def __init__(self, device: str) -> None:
        self.verifier = Verifier(device)
        self.listener = Listener(device, *GUARD_LISTENER)
        self.lock = threading.Lock()

    def vector(self, x16: NDArray[np.floating]) -> NDArray[np.float32]:
        with self.lock:
            return self.verifier.vector(x16)

    def heard(self, x16: NDArray[np.floating]) -> torch.Tensor:
        with self.lock:
            return self.listener.hear(x16).feats[0].cpu()


def chosen(ids: list[str], of: int, n: int) -> tuple[list[str], list[str]]:
    """Of a speaker's recordings, those that say who the speaker is (`of`, spread over them all)
    and, among them, those to render (`n`)."""
    told = ids[:: max(1, len(ids) // of)][:of]
    return told, told[:: max(1, len(told) // n)][:n]


def fit_speaker(d: Path, name: str, ids: list[str], a: argparse.Namespace, judges: Judges, pool: ThreadPoolExecutor) -> dict[str, Any]:
    told, fitted = chosen(ids, a.of, a.n)
    target = np.stack([judges.vector(load(d / "real" / f"{uid}.wav", SR)[0]) for uid in told]).mean(axis=0)
    target /= np.linalg.norm(target)
    metas = [json.loads((d / "meta" / f"{uid}.json").read_text()) for uid in fitted]
    tracks = [json.loads((d / a.tracks / f"{uid}.tracks.json").read_text()) for uid in fitted]
    for t in tracks:
        # the first seconds of the recording are enough to hear who speaks, and cost that much less
        t["frames"] = np.array(t["frames"])[: round((t["shift"] + a.seconds) * t["rate"])]
    spans = [(int(m["span"][0] * FEATURE_RATE), min(int(m["span"][1] * FEATURE_RATE), int(a.seconds * FEATURE_RATE))) for m in metas]
    recorded = [judges.heard(load(d / "real" / f"{uid}.wav", SR)[0]) for uid in fitted]
    start = own(tracks[0])
    free = [k for k, p in enumerate(PARAMS) if not a.only or p[0] in a.only.split(",")]

    def settings(u: NDArray[np.float64]) -> dict[str, float]:
        s = dict(start)
        for k, x in zip(free, u):
            name, _, lo, hi, _ = PARAMS[k]
            s[name] = float(lo + np.clip(x, 0.0, 1.0) * (hi - lo))
        return s

    def scores(s: dict[str, float]) -> tuple[float, float]:
        """How like the speaker the renders are under these settings, and how far from their recordings."""
        audio = list(pool.map(lambda t: render_track(settled(t, s)), tracks))
        like, far = [], []
        for t, x, (b0, b1), rec in zip(tracks, audio, spans, recorded):
            y = to16k(np.asarray(x[round(t["shift"] * 48000) :], dtype=np.float32))
            like.append(float(judges.vector(y) @ target))
            far.append(float(distance(rec[b0:b1], judges.heard(y)[b0:b1]).mean()))
        return float(np.mean(like)), float(np.mean(far))

    before = scores(start)

    def cost(u: NDArray[np.float64]) -> float:
        like, far = scores(settings(u))
        return -like + GUARD * max(0.0, far - before[1])

    u = np.array([(start[PARAMS[k][0]] - PARAMS[k][2]) / (PARAMS[k][3] - PARAMS[k][2]) for k in free])
    least, tries = -before[0], 0
    # first each setting on its own over its whole range, the others as they stand: a search
    # that only steps from where it is stays in the nearest dip
    for i in range(len(free)):
        for x in SCAN:
            v = u.copy()
            v[i] = x
            c = cost(v)
            tries += 1
            if c < least:
                least, u = c, v
    # then all together, from there: a first simplex that moves each by a tenth of its range, inwards
    simplex = np.vstack([u, *(u + np.eye(len(free))[i] * (0.1 if u[i] < 0.8 else -0.1) for i in range(len(free)))])
    r = minimize(cost, u, method="Nelder-Mead", bounds=[(0.0, 1.0)] * len(free), options={"maxfev": max(len(free) + 2, a.evals - tries), "initial_simplex": simplex, "xatol": 0.01, "fatol": 1e-4})
    if r.fun < least:
        least, u = float(r.fun), r.x
    best = settings(u) if least < -before[0] else start
    after = scores(best)
    return {"settings": {k: round(v, 4) for k, v in best.items()}, "fit": {"like": [before[0], after[0]], "listener": [before[1], after[1]], "evals": tries + int(r.nfev)}, "ids": fitted}


def fit(a: argparse.Namespace) -> None:
    d: Path = a.dir
    by: dict[str, list[str]] = {}
    for uid in ids_in(d, a.split):
        by.setdefault(json.loads((d / "meta" / f"{uid}.json").read_text())["speaker"], []).append(uid)
    path = d / f"{a.name}.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {"tracks": a.tracks, "only": a.only, "evals": a.evals, "speakers": {}}
    names = [n for n in sorted(by) if n not in out["speakers"] and (not a.speakers or n in a.speakers.split(","))]
    judges = Judges(pick_device())
    lock = threading.Lock()
    print(f"{len(names)} speakers to fit ({len(out['speakers'])} already are)", flush=True)

    def one(name: str) -> None:
        got = fit_speaker(d, name, by[name], a, judges, pool)
        with lock:
            out["speakers"][name] = got
            path.write_text(json.dumps(out, indent=1))
            f = got["fit"]
            print(f"{name}  like {f['like'][0]:.3f} → {f['like'][1]:.3f}  listener {f['listener'][0]:.3f} → {f['listener'][1]:.3f}  ({f['evals']} tries)  " + " ".join(f"{k} {v:.2f}" for k, v in got["settings"].items()), flush=True)

    with ThreadPoolExecutor(a.jobs) as pool, ThreadPoolExecutor(a.at_once) as speakers:
        list(speakers.map(one, names))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit")
    f.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "jvsall")
    f.add_argument("--tracks", type=str, default="enci", help="the variant whose tracks are rendered")
    f.add_argument("--split", choices=["test", "valid", "train", "all"], default="train")
    f.add_argument("--n", type=int, default=6, help="recordings of a speaker to render")
    f.add_argument("--of", type=int, default=20, help="recordings of a speaker that say who the speaker is")
    f.add_argument("--only", type=str, default="", help="fit only these settings, comma-separated")
    f.add_argument("--evals", type=int, default=90)
    f.add_argument("--seconds", type=float, default=4.0, help="how much of each recording is rendered")
    f.add_argument("--name", type=str, default="speakers")
    f.add_argument("--speakers", type=str, default="", help="only these speakers, comma-separated")
    f.add_argument("--at-once", type=int, default=3, help="speakers fitted at a time")
    f.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    fit(a)


if __name__ == "__main__":
    main()
