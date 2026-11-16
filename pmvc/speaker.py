"""Fit the voice to a speaker: the body, the glottis and the habits of posture under which
the speaker's control tracks sound most like the speaker.

Of a speaker the voice is told the sex, which picks one of two bodies, and the median pitch;
a speaker-verification model then cannot tell its renders of one speaker from those of another
of the same sex (`pmvc.whose`). But CHIMERA's body is continuous, and so are the settings of
its source that differ between people. Here they are fitted for each speaker, a handful of
numbers to go with the speaker's tracks:
  female, child   the body within CHIMERA's own family (0 to 1 each: from the adult male to the
                  adult female, from adult to child; the sex gives 0 or 1 and no child)
  scale, pharynx, tongueRadius, lipWidth
                  the body's four numbers themselves, which the family moves together: the
                  tract's size (1 the adult male's, and larger than his too), how much of it
                  is throat, the tongue body's radius and the lips' width (cm)
  openQuotient    the share of a cycle the glottis is open (pressed to breathy)
  brightness      how sharply the glottis closes (the slope of the source's spectrum)
  tension         the folds' tension (what else of the pulse follows from it)
  closureSharpness
                  how abruptly the glottis closes (0 to 1), set on the glottis itself: the
                  level of a voice above 1.6 kHz, with the balance of its lowest harmonics
                  left alone. The source's tension and brightness make it unless a track or
                  a fit says it (`closure` below measures it)
  epilarynx       how narrow the tube above the glottis is held (0 to 1; the encoder leaves it
                  at 0): the ring of a voice around 3 kHz
  larynxShift, velumShift, rootShift
                  what is added to the encoder's channel in every frame: a larynx carried
                  higher or lower (cm), a little more or less opening to the nose (cm²), the
                  tongue root further forward or back (cm)

usage: python -m pmvc.speaker fit [dir] [--tracks enci] [--split train] [--n 6] [--of 20]
                                  [--only female,child] [--start FILE] [--evals 90]
                                  [--name speakers] [--seconds 4] [--at-once 3] [--speakers a,b]

For each speaker of <dir>: the first `--seconds` of `--n` of the split's recordings (spread
over them) are rendered from their tracks (`--tracks`, an encoder's) under candidate settings,
starting from the tracks' own (or `--start`'s, an earlier fit's file), and a verification
model (`pmvc.whose`'s) says how close the renders are to the speaker: the cosine between a
render's vector and the mean vector of `--of` of the speaker's recordings. Each setting is
first tried on its own over its range and then a little to either side of where that left
it, then all are searched together (`--evals` renders of the recordings in all). The settings
kept are those with the highest mean cosine, less GUARD for every unit by which the renders
move away from their recordings as `pmvc.listener` hears them: a voice more like the speaker's
must not say something else. `--only` fits these settings and leaves the others as they are;
the body is fitted either within the family (female, child) or by its four numbers. Writes
<dir>/<name>.json, a speaker at a time (speakers already in it are not fitted again):
  speakers.<name>.settings   the fitted settings
  speakers.<name>.fit        the cosine and the listener's distance before and after, on the
                             fitted recordings
`pmvc.encoder revoice --speakers <file>` renders a variant's tracks under them; whether the
voice is then the speaker's is for `pmvc.whose` to say on recordings that were not fitted, and
for the ear.

usage: python -m pmvc.speaker closure [dir] [--tracks enci] [--split train] [--n 6] [--of 20]
                                      [--start FILE] [--sex F] [--value X] [--seconds 4]
                                      [--name speakers_closure] [--speakers a,b]

How abruptly a speaker's glottis closes, measured and not left to a verification model. Fitted
to one, the women's voices got a smaller open quotient: the second and third harmonic rose
against the first, which the ear takes for a buzz, and the band the voice lacked against its
recordings (some 8 dB between 1.6 and 5.5 kHz in the women's vowels) was only half filled. The
closure's sharpness fills that band and leaves the lowest harmonics as they are. For each
speaker (of `--sex`, if given) the same recordings as `fit` renders are rendered with the
sharpness at each of SHARP, under the speaker's settings in `--start`; in the frames the
recording is voiced in, the level of each band of PRESENCE against the frame's whole level is
compared with the recording's, and the sharpness kept is where the renders' bands are as strong
as the recordings' on average (between the tried values, along a straight line). `--value`
gives every such speaker this sharpness and measures nothing. Writes <dir>/<name>.json like
`fit` does (the other speakers of `--start` as they are there), with
  speakers.<name>.closure    the sharpnesses tried and the renders' level against the
                             recordings' at each (dB)
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
from scipy.signal import stft

from .audio import FRAME_RATE, load
from .encoder import ids_in, pick_device, render_track
from .listener import FEATURE_RATE, Listener, distance, to16k
from .paths import ROOT
from .whose import SR, Verifier

# a setting, where in a track it is (a part of the track; or a channel of its frames that is
# `set` to it or `shift`ed by it), its range, and what it is where a track does not say
# (CHIMERA's own; the open quotient is what the source's defaults make it; the body's four
# numbers are what its family gives; the closure's sharpness is the source's to make, and no
# setting of a track that does not say it)
PARAMS: tuple[tuple[str, str, float, float, float], ...] = (
    ("female", "body", 0.0, 1.0, 0.0),
    ("child", "body", 0.0, 0.5, 0.0),
    ("scale", "body", 0.80, 1.12, float("nan")),
    ("pharynx", "body", 0.76, 1.10, float("nan")),
    ("tongueRadius", "body", 2.05, 2.40, float("nan")),
    ("lipWidth", "body", 3.8, 5.0, float("nan")),
    ("openQuotient", "glottis", 0.45, 0.95, 0.6),
    ("brightness", "source", 0.02, 0.8, 0.25),
    ("tension", "source", 0.02, 0.9, 0.5),
    ("closureSharpness", "glottis", 0.0, 1.0, float("nan")),
    ("epilarynx", "set", 0.0, 1.0, 0.0),
    ("larynxShift", "shift", -0.6, 0.6, 0.0),
    ("velumShift", "shift", -0.1, 0.3, 0.0),
    ("rootShift", "shift", -0.5, 0.5, 0.0),
)
# the body's four numbers for CHIMERA's adult male, adult female and child
PRESETS: dict[str, tuple[float, float, float]] = {"scale": (1.0, 0.9, 0.74), "pharynx": (1.0, 0.88, 0.8), "tongueRadius": (2.25, 2.2, 2.15), "lipWidth": (4.6, 4.4, 4.0)}
# what one unit of the listener's distance to the recordings costs in cosine, and the listener
GUARD: float = 2.0
GUARD_LISTENER: tuple[str, int] = ("utter-project/mHuBERT-147", 8)
# where in its range each setting is tried on its own before the search, and then how far
# to either side of where that left it
SCAN: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
NEAR: tuple[float, ...] = (-0.1, 0.1)
# `closure`: the bands (Hz) whose level in voiced frames the closure's sharpness is measured
# by, the sharpnesses tried, and the analysis window (samples at SR)
PRESENCE: tuple[tuple[float, float], ...] = ((1600.0, 2500.0), (2500.0, 4000.0), (4000.0, 5500.0))
SHARP: tuple[float, ...] = (0.0, 0.15, 0.3, 0.45, 0.6, 0.8)
WINDOW: int = 512


def family(female: float, child: float) -> dict[str, float]:
    """The body's four numbers for a member of CHIMERA's family, as its `BodyMix` blends them."""
    w = ((1 - child) * (1 - female), (1 - child) * female, child)
    return {k: sum(a * b for a, b in zip(w, v)) for k, v in PRESETS.items()}


def settled(track: dict[str, Any], s: dict[str, float]) -> dict[str, Any]:
    """`track` with a speaker's settings in place of its own; its frames with the speaker's
    habits of posture (as a list if they were one)."""
    out = {**track, "body": dict(track["body"]), "source": dict(track["source"]), "glottis": dict(track.get("glottis", {}))}
    frames: NDArray[np.float64] | None = None
    for name, part, _, _, default in PARAMS:
        if name not in s:
            continue
        if part in ("set", "shift"):
            if part == "shift" and s[name] == default:
                continue
            if frames is None:
                frames = np.array(track["frames"], dtype=np.float64)
            k = track["keys"].index(name.removesuffix("Shift"))
            frames[:, k] = s[name] if part == "set" else frames[:, k] + s[name]
        else:
            out[part][name] = round(float(s[name]), 5)
    if not out["glottis"]:
        del out["glottis"]
    if frames is not None:
        out["frames"] = frames if isinstance(track["frames"], np.ndarray) else np.round(frames, 5).tolist()
    return out


def own(track: dict[str, Any]) -> dict[str, float]:
    """The settings a track has."""
    body = track["body"]
    of_family = family(float(body.get("female", 0.0)), float(body.get("child", 0.0)))
    s: dict[str, float] = {}
    for name, part, _, _, default in PARAMS:
        if part == "set":
            s[name] = float(np.mean(np.asarray(track["frames"])[:, track["keys"].index(name)]))
        elif part == "shift":
            s[name] = default
        else:
            x = float(track.get(part, {}).get(name, of_family.get(name, default)))
            if not np.isnan(x):
                s[name] = x
    return s


def presence(x: NDArray[np.floating], voiced: NDArray[np.bool_]) -> NDArray[np.float64]:
    """A clip on its recording's clock (at SR): the level of each band of PRESENCE against
    the whole level (dB), over the frames marked voiced."""
    _, _, spec = stft(x, SR, nperseg=WINDOW, noverlap=WINDOW - SR // FRAME_RATE, boundary="zeros", padded=True)
    power = np.abs(spec.T) ** 2  # frame j is centred j frames in
    n = min(len(power), len(voiced))
    power = power[:n][voiced[:n]]
    hz = np.fft.rfftfreq(WINDOW, 1 / SR)
    return np.array([10 * np.log10(power[:, (hz >= lo) & (hz < hi)].sum(axis=1).mean() / power.sum(axis=1).mean() + 1e-12) for lo, hi in PRESENCE])


def crossing(xs: tuple[float, ...], ys: list[float]) -> float:
    """Where the line through (xs, ys) first reaches 0 from below: the first x if it starts
    above, the last if it never gets there."""
    if ys[0] >= 0.0:
        return xs[0]
    for x0, x1, y0, y1 in zip(xs, xs[1:], ys, ys[1:]):
        if y1 >= 0.0:
            return x0 + (x1 - x0) * -y0 / (y1 - y0)
    return xs[-1]


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


def fit_speaker(d: Path, name: str, ids: list[str], a: argparse.Namespace, judges: Judges, pool: ThreadPoolExecutor, before_this: dict[str, float] | None = None) -> dict[str, Any]:
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
    if before_this is not None:
        # an earlier fit's settings; the body's four numbers are its family's unless it had them
        start.update(family(before_this.get("female", start["female"]), before_this.get("child", start["child"])))
        start.update(before_this)
    only = a.only.split(",") if a.only else ["female", "child", "openQuotient", "brightness", "tension"]
    free = [k for k, p in enumerate(PARAMS) if p[0] in only]
    within = any(name in only for name in ("female", "child"))
    assert not (within and any(name in only for name in PRESETS)), "the body is fitted within its family or by its four numbers, not both"

    def settings(u: NDArray[np.float64]) -> dict[str, float]:
        s = dict(start)
        for k, x in zip(free, u):
            name, _, lo, hi, _ = PARAMS[k]
            s[name] = float(lo + np.clip(x, 0.0, 1.0) * (hi - lo))
        if within:
            s.update(family(s["female"], s["child"]))
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

    # a setting the tracks do not have starts at the low end of its range
    u = np.clip(np.array([(start.get(PARAMS[k][0], PARAMS[k][2]) - PARAMS[k][2]) / (PARAMS[k][3] - PARAMS[k][2]) for k in free]), 0.0, 1.0)
    least, tries = -before[0], 0
    # first each setting on its own over its whole range, the others as they stand (a search
    # that only steps from where it is stays in the nearest dip), then a little to either side
    for spots in (SCAN, NEAR):
        for i in range(len(free)):
            for x in spots:
                v = u.copy()
                v[i] = x if spots is SCAN else float(np.clip(u[i] + x, 0.0, 1.0))
                if abs(v[i] - u[i]) < 1e-9:
                    continue
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
    out: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {"tracks": a.tracks, "only": a.only, "evals": a.evals, "start": str(a.start or ""), "speakers": {}}
    earlier: dict[str, Any] = json.loads(a.start.read_text())["speakers"] if a.start else {}
    names = [n for n in sorted(by) if n not in out["speakers"] and (not a.speakers or n in a.speakers.split(","))]
    judges = Judges(pick_device())
    lock = threading.Lock()
    print(f"{len(names)} speakers to fit ({len(out['speakers'])} already are)", flush=True)

    def one(name: str) -> None:
        got = fit_speaker(d, name, by[name], a, judges, pool, earlier[name]["settings"] if name in earlier else None)
        with lock:
            out["speakers"][name] = got
            path.write_text(json.dumps(out, indent=1))
            f = got["fit"]
            print(f"{name}  like {f['like'][0]:.3f} → {f['like'][1]:.3f}  listener {f['listener'][0]:.3f} → {f['listener'][1]:.3f}  ({f['evals']} tries)  " + " ".join(f"{k} {v:.2f}" for k, v in got["settings"].items()), flush=True)

    with ThreadPoolExecutor(a.jobs) as pool, ThreadPoolExecutor(a.at_once) as speakers:
        list(speakers.map(one, names))


def closure(a: argparse.Namespace) -> None:
    d: Path = a.dir
    by: dict[str, list[str]] = {}
    sex: dict[str, str] = {}
    for uid in ids_in(d, a.split):
        meta = json.loads((d / "meta" / f"{uid}.json").read_text())
        by.setdefault(meta["speaker"], []).append(uid)
        sex[meta["speaker"]] = meta["sex"]
    earlier: dict[str, Any] = json.loads(a.start.read_text())["speakers"] if a.start else {}
    out: dict[str, Any] = {"tracks": a.tracks, "start": str(a.start or ""), "sex": a.sex, "speakers": dict(earlier)}
    names = [n for n in sorted(by) if (not a.sex or sex[n] == a.sex) and (not a.speakers or n in a.speakers.split(","))]

    def one(name: str) -> dict[str, Any]:
        fitted = chosen(by[name], a.of, a.n)[1]
        start: dict[str, float] = dict(earlier[name]["settings"]) if name in earlier else {}
        if a.value is not None:
            return {"settings": {**start, "closureSharpness": a.value}, "ids": []}
        tracks = [json.loads((d / a.tracks / f"{uid}.tracks.json").read_text()) for uid in fitted]
        n = round(a.seconds * FRAME_RATE)
        for t in tracks:
            t["frames"] = np.array(t["frames"])[: round(t["shift"] * t["rate"]) + n]
        voiced = [np.load(d / "pitch" / f"{uid}.npy")[:n] > 0 for uid in fitted]
        recorded = [presence(load(d / "real" / f"{uid}.wav", SR)[0][: round(a.seconds * SR)], v) for uid, v in zip(fitted, voiced)]
        level: list[float] = []
        for x in SHARP:
            under = {**start, "closureSharpness": x}
            audio = list(pool.map(lambda t, s=under: render_track(settled(t, s)), tracks))
            got = [presence(to16k(np.asarray(y[round(t["shift"] * 48000) :], dtype=np.float32)), v) for t, y, v in zip(tracks, audio, voiced)]
            level.append(float(np.mean([g - r for g, r in zip(got, recorded)])))
        best = round(crossing(SHARP, level), 3)
        print(f"{name} {sex[name]}  sharpness {best:.2f}   " + "  ".join(f"{x:.2f}: {y:+.1f}" for x, y in zip(SHARP, level)), flush=True)
        return {"settings": {**start, "closureSharpness": best}, "closure": {"tried": list(SHARP), "level": [round(y, 2) for y in level]}, "ids": fitted}

    with ThreadPoolExecutor(a.jobs) as pool:
        for name in names:
            out["speakers"][name] = {**earlier.get(name, {}), **one(name)}
            (d / f"{a.name}.json").write_text(json.dumps(out, indent=1))
    print(f"{len(names)} speakers → {d / f'{a.name}.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit")
    c = sub.add_parser("closure")
    for p in (f, c):
        p.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "jvsall")
        p.add_argument("--tracks", type=str, default="enci", help="the variant whose tracks are rendered")
        p.add_argument("--split", choices=["test", "valid", "train", "all"], default="train")
        p.add_argument("--n", type=int, default=6, help="recordings of a speaker to render")
        p.add_argument("--of", type=int, default=20, help="recordings of a speaker that say who the speaker is")
        p.add_argument("--start", type=Path, default=None, help="an earlier fit's file: each speaker's search starts from its settings")
        p.add_argument("--seconds", type=float, default=4.0, help="how much of each recording is rendered")
        p.add_argument("--speakers", type=str, default="", help="only these speakers, comma-separated")
        p.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    c.add_argument("--sex", choices=["F", "M"], default=None, help="only the speakers of this sex")
    c.add_argument("--value", type=float, default=None, help="this sharpness for every such speaker, nothing measured")
    c.add_argument("--name", type=str, default="speakers_closure")
    f.add_argument("--only", type=str, default="", help="fit these settings, comma-separated (without it: female, child, openQuotient, brightness, tension)")
    f.add_argument("--evals", type=int, default=90)
    f.add_argument("--name", type=str, default="speakers")
    f.add_argument("--at-once", type=int, default=3, help="speakers fitted at a time")
    a = ap.parse_args()
    {"fit": fit, "closure": closure}[a.command](a)


if __name__ == "__main__":
    main()
