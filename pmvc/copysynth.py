"""Copy synthesis: fit a control track to a recording, by local search through the simulator.

The start is `pmvc.warp`'s track: the right phones by rule, on the speaker's timing. From there
the track is changed a little at a time, each change rendered and kept if `pmvc.listener` hears
the result closer to the recording. Nothing is differentiated: CHIMERA is a time-domain
simulation with closures, bursts and turbulence. What makes the search affordable is that the
voice can be forked (`pmvc_engine`): the track is rendered once with a snapshot of the whole
physical state every 50 ms, and a change is tried by continuing from the snapshot before it for
a few hundred milliseconds and splicing that into the render.

usage: python -m pmvc.copysynth [dir] [--ids ... | --n 40 | --all] [--iters 40] [--proposals 96]
                                [--base warp] [--out copy] [--calibrated] [--nll-weight 0.1]
                                [--shape-weight 0] [--listener MODEL --layer N] [--also KEY=STEP,...]
                                [--scale DIR] [--shard K/N] [--resume] [--jobs N]

The objective is the listener's feature distance to the recording over its speech, plus
`NLL_WEIGHT` × the CTC loss of the sentence's text per character. Each iteration draws
`proposals` changes spread over the utterance:
  bump    one to four channels pushed up or down over 60-240 ms (a Hann window)
  shift   everything but pitch moved earlier or later around a point, by up to a few frames
tries them all in parallel, scores each on the whole utterance, and takes the ones that lower
the objective, best first, as long as they do not overlap. The track is then rendered again
with all of them; if together they do not lower the objective, only the best one is kept, or
none.

Writes <dir>/<out>/<id>.wav, <id>.tracks.json (with the settings it was rendered under) and
<id>.log.json (the objective before and after, and by iteration), and <dir>/<out>.json, the
logs' summaries. With `--calibrated` the voice runs under `pmvc.calibrate`'s settings for its
body, and <dir>/<out>-start/<id>.wav is the starting track rendered under them, so that the
search's own gain can be told from the calibration's.

A search takes about a minute and only a third of it uses every core (the rest is the listener
and one sequential render per iteration), so a corpus is best split over processes: `--all`
takes every recording with a starting track, speaker after speaker in turn, `--shard K/N`
every N-th of them from the K-th, and `--resume` leaves alone the ones already written.

`--shape-weight` adds a third term: how far the render's spectral envelope is from the
recording's, as the mean distance between their log mel spectra (40 bands to 8 kHz), frame
against frame over the speech, with the level the two differ by over the utterance taken out,
in units of 10 dB. The listener's features leave the speaker out by design, so a search on
them alone asks for the words and not for the voice; with this term it also asks the voice to
resonate as the speaker's does, sound by sound, which is what a speaker-verification model goes
by (`pmvc.gap`). It is the test of how near the engine can come to a recording's envelope when
asked. `--listener` and `--layer` name another model for the first term (one that writes no
characters has no second: `--nll-weight 0`). `--also` lets the search move channels that the
rule's tracks hold still, each by steps of the size given (`epilarynx=0.15`).

The steps a search takes are sized by how far each channel moves in <dir>/warp's tracks. For
recordings from elsewhere (`pmvc.recordings`), which have no such tracks, `--scale` names a
directory that does; and with `--nll-weight 0` nothing here reads the sentence's text.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .audio import FRAME_RATE, Array, log_mel
from .calibrate import NLL_WEIGHT, Item, load_items, make_voice
from .listener import Listener, distance
from .paths import ROOT
from .perturb import Scale

HOP: int = 48000 // FRAME_RATE
# the same at the listener's rate
HOP16: int = HOP // 3
# frames between snapshots of the voice
SNAP: int = 5
# frames rendered past a change, for the voice to settle back onto the unchanged track
TAIL: int = 12
# a change must lower the objective by this much to be taken
MIN_GAIN: float = 0.0005
BUMP_WIDTHS: tuple[int, ...] = (3, 6, 12)
SHIFT_WIDTHS: tuple[int, ...] = (6, 12, 20)


@dataclass
class Proposal:
    """A changed copy of the track, and the frames that differ (inclusive)."""

    frames: Array
    first: int
    last: int
    gain: float = 0.0
    # the whole utterance with the change spliced in, at the listener's rate
    heard: Array = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


class Search:
    def __init__(self, item: Item, settings: dict[str, float] | None, listener: Listener, sigma: Array, seed: int, jobs: int, nll_weight: float = NLL_WEIGHT, shape_weight: float = 0.0) -> None:
        self.item, self.listener, self.jobs, self.nll_weight, self.shape_weight = item, listener, jobs, nll_weight, shape_weight
        self.x = item.frames.copy()
        self.n = len(self.x)
        self.keys: list[str] = item.track["keys"]
        self.rng = np.random.default_rng(seed)
        self.sigma = sigma
        # channels the search may change: those that move in speech, but not pitch (the speaker's)
        self.free = [c for c, k in enumerate(self.keys) if sigma[c] > 0 and k not in ("f0", "closure")]
        self.movable = [c for c, k in enumerate(self.keys) if k != "f0"]
        self.audio = np.zeros((self.n - 1) * HOP, dtype=np.float32)
        self.audio16 = np.zeros(0, dtype=np.float32)
        self.snaps: dict[int, Any] = {}
        self.scale = 1.0
        v = make_voice(item.track, settings)
        v.set_track(self.x, 0)
        self.snaps[0] = v
        self.rebuild(0)
        self.value = self.objectives([self.audio16])[0]

    def rebuild(self, first: int) -> None:
        """Render the track again from the snapshot before frame `first`, refreshing the
        snapshots and the audio after it."""
        k = (max(0, first - 2) // SNAP) * SNAP
        v = self.snaps[k].fork()
        v.set_track(self.x, k)
        for f in range(k, self.n - 1, SNAP):
            self.snaps[f] = v.fork()
            n = min(SNAP, self.n - 1 - f) * HOP
            self.audio[f * HOP : f * HOP + n] = v.render(n)
        self.audio16 = resample_poly(self.audio, 1, 3).astype(np.float32)

    def shape(self, clip: Array) -> float:
        """How far a render's envelope (16 kHz, on the recording's clock) is from the
        recording's over its speech, in units of 10 dB."""
        assert self.item.mel is not None
        a, b = 2 * self.item.span[0], 2 * self.item.span[1]
        mine = log_mel(clip.astype(np.float64))
        n = min(len(mine), len(self.item.mel), b)
        diff = self.item.mel[a:n] - mine[a:n]
        return float(np.abs(diff - diff.mean()).mean()) * 2.0 / np.log(10.0)

    def objectives(self, clips: list[Array]) -> list[tuple[float, float, float, float]]:
        """The objective, feature distance, CTC loss and envelope distance of whole-utterance renders (16 kHz)."""
        it = self.item
        a, b = it.span
        out: list[tuple[float, float, float, float]] = []
        for i in range(0, len(clips), 48):
            part = [c[it.shift // 3 :] for c in clips[i : i + 48]]
            heard = self.listener.hear(np.stack(part))
            with ThreadPoolExecutor(self.jobs) as pool:
                shapes = list(pool.map(self.shape, part)) if self.shape_weight else [0.0] * len(part)
            for feats, logp, shape in zip(heard.feats, heard.logp, shapes):
                dist = float(distance(it.feats[a:b], feats[a:b]).mean())
                nll = self.listener.nll(logp, it.chars) if self.nll_weight else 0.0
                out.append((dist + self.nll_weight * nll + self.shape_weight * shape, dist, nll, shape))
        return out

    def propose(self, centre: int) -> Proposal:
        x = self.x.copy()
        if self.rng.random() < 0.7:
            w = int(self.rng.choice(BUMP_WIDTHS))
            cs = self.rng.choice(self.free, size=1 if self.rng.random() < 0.5 else int(self.rng.integers(2, 5)), replace=False)
            lo, hi = max(0, centre - w), min(self.n - 1, centre + w)
            window = 0.5 + 0.5 * np.cos(np.pi * (np.arange(lo, hi + 1) - centre) / (w + 1))
            for c in cs:
                x[lo : hi + 1, c] += window * self.rng.normal() * self.scale * self.sigma[c]
        else:
            w = int(self.rng.choice(SHIFT_WIDTHS))
            lo, hi = max(0, centre - w), min(self.n - 1, centre + w)
            t = np.arange(lo, hi + 1)
            # read the track up to `by` frames away around the centre: later (the events come
            # earlier) or earlier
            by = self.rng.uniform(0.5, 2.5) * self.scale * self.rng.choice([-1.0, 1.0])
            at = np.clip(t + by * (0.5 + 0.5 * np.cos(np.pi * (t - centre) / (w + 1))), 0, self.n - 1)
            for c in self.movable:
                x[lo : hi + 1, c] = np.interp(at, np.arange(self.n), self.x[:, c])
        for key, floor, ceiling in (("abduction", 0.0, None), ("breath", 0.0, 1.3)):
            c = self.keys.index(key)
            x[:, c] = np.clip(x[:, c], floor, ceiling)
        return Proposal(x, lo, hi)

    def trial(self, p: Proposal) -> None:
        """Render the change from the snapshot before it and splice it into the utterance."""
        k = (max(0, p.first - 2) // SNAP) * SNAP
        e = min(self.n - 1, p.last + TAIL)
        v = self.snaps[k].fork()
        v.set_track(p.frames, k)
        seg = np.asarray(v.render((e - k) * HOP))
        # back onto the unchanged render over the last 5 ms
        fade = np.linspace(0.0, 1.0, 240, dtype=np.float32)
        seg[-240:] = seg[-240:] * (1 - fade) + self.audio[e * HOP - 240 : e * HOP] * fade
        # resample the changed stretch with a frame of the unchanged render on either side, so
        # the splice at the listener's rate has no edge of its own
        a, b = max(0, k - 1), min(self.n - 1, e + 1)
        chunk = self.audio[a * HOP : b * HOP].copy()
        chunk[(k - a) * HOP : (e - a) * HOP] = seg
        chunk16 = resample_poly(chunk, 1, 3).astype(np.float32)
        p.heard = self.audio16.copy()
        p.heard[k * HOP16 : e * HOP16] = chunk16[(k - a) * HOP16 : (e - a) * HOP16]

    def step(self, proposals: int) -> tuple[int, int]:
        """One iteration. Returns how many changes lowered the objective and how many were kept."""
        # centres spread over the speech (the span is in the listener's frames, twice as long),
        # with a random phase per iteration
        lead = self.item.shift // HOP
        a, b = 2 * self.item.span[0] + lead, min(self.n - 1, 2 * self.item.span[1] + lead)
        centres = (a + (np.arange(proposals) + self.rng.random()) * (b - a) / proposals).astype(int)
        ps = [self.propose(int(c)) for c in centres]
        with ThreadPoolExecutor(self.jobs) as pool:
            list(pool.map(self.trial, ps))
        for p, (value, _, _, _) in zip(ps, self.objectives([p.heard for p in ps])):
            p.gain = self.value[0] - value
        better = sorted((p for p in ps if p.gain > MIN_GAIN), key=lambda p: -p.gain)
        taken: list[Proposal] = []
        for p in better:
            if all(p.last + TAIL < q.first - 2 or q.last + TAIL < p.first - 2 for q in taken):
                taken.append(p)
        # all of them together, or failing that the best alone
        for chosen in (taken, taken[:1]) if len(taken) > 1 else (taken,):
            if not chosen:
                break
            # rebuilding replaces snapshots and never changes one, so the old state is these
            old = (self.x.copy(), self.audio.copy(), self.audio16, dict(self.snaps))
            for p in chosen:
                self.x[p.first : p.last + 1] = p.frames[p.first : p.last + 1]
            self.rebuild(min(p.first for p in chosen))
            value = self.objectives([self.audio16])[0]
            if value[0] < self.value[0]:
                self.value = value
                return len(better), len(chosen)
            self.x, self.audio, self.audio16, self.snaps = old
        self.scale = max(0.15, self.scale * 0.9)
        return len(better), 0


def copy(item: Item, settings: dict[str, float] | None, listener: Listener, sigma: Array, a: argparse.Namespace, d: Path) -> dict[str, Any]:
    s = Search(item, settings, listener, sigma, seed=0, jobs=a.jobs, nll_weight=a.nll_weight, shape_weight=a.shape_weight)
    start = s.value
    if settings:
        sf.write(str(d / f"{a.out}-start" / f"{item.uid}.wav"), s.audio, 48000, subtype="PCM_16")
    log: list[dict[str, float]] = []
    for it in range(a.iters):
        better, kept = s.step(a.proposals)
        log.append({"iter": it, "objective": s.value[0], "distance": s.value[1], "nll": s.value[2], "shape": s.value[3], "better": better, "kept": kept, "scale": s.scale})
    end = s.value
    stem = d / a.out / item.uid
    sf.write(f"{stem}.wav", s.audio, 48000, subtype="PCM_16")
    track = {**item.track, "frames": np.round(s.x, 5).tolist(), "settings": settings}
    row = {"id": item.uid, "objective": [start[0], end[0]], "distance": [start[1], end[1]], "nll": [start[2], end[2]], "shape": [start[3], end[3]], "kept": int(sum(r["kept"] for r in log))}
    Path(f"{stem}.log.json").write_text(json.dumps({**row, "iterations": log}))
    # the track last and at once: it is what says the recording is done
    tmp = Path(f"{stem}.tracks.json.tmp")
    tmp.write_text(json.dumps(track))
    tmp.replace(f"{stem}.tracks.json")
    print(f"{item.uid}: objective {start[0]:.3f} → {end[0]:.3f}  distance {start[1]:.3f} → {end[1]:.3f}  CTC loss {start[2]:.2f} → {end[2]:.2f}  envelope {10 * start[3]:.1f} → {10 * end[3]:.1f} dB  ({row['kept']} changes)", flush=True)
    return row


def in_turns(ids: list[str]) -> list[str]:
    """Recording ids (speaker, list, sentence: `FA01_03`) with the speakers taking turns: each
    speaker's first, then each one's second, ... so that any part of a corpus has them all."""
    place: dict[str, int] = {}
    turn: dict[str, int] = {}
    for i in sorted(ids):
        place[i] = turn[i[:2]] = turn.get(i[:2], -1) + 1
    return sorted(ids, key=lambda i: (place[i], i[:2]))


def summarise(d: Path, out: str) -> list[dict[str, Any]]:
    """<dir>/<out>.json: every finished search's objective before and after, from its log."""
    rows: list[dict[str, Any]] = []
    for p in sorted((d / out).glob("*.log.json")):
        if p.with_name(p.name.replace(".log.json", ".tracks.json")).exists():
            log = json.loads(p.read_text())
            rows.append({k: v for k, v in log.items() if k != "iterations"})
    tmp = d / f"{out}.json.{os.getpid()}"
    tmp.write_text(json.dumps(rows, indent=1))
    tmp.replace(d / f"{out}.json")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "transplant")
    ap.add_argument("--ids", type=str, default="")
    ap.add_argument("--n", type=int, default=40, help="how many recordings (the first n ids of the perturbation test)")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--proposals", type=int, default=96)
    ap.add_argument("--base", type=str, default="warp")
    ap.add_argument("--out", type=str, default="copy")
    ap.add_argument("--calibrated", action="store_true", help="run the voice under pmvc.calibrate's settings")
    ap.add_argument("--nll-weight", type=float, default=NLL_WEIGHT)
    ap.add_argument("--shape-weight", type=float, default=0.0, help="what a unit (10 dB) of envelope distance to the recording costs")
    ap.add_argument("--listener", type=str, default="", help="another model than pmvc.listener's for the feature distance")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--also", type=str, default="", help="channels the rule holds still that the search may move, with their step: epilarynx=0.15,...")
    ap.add_argument("--scale", type=Path, default=None, help="tracks whose spread sets the step sizes, if not <dir>/warp")
    ap.add_argument("--all", action="store_true", help="every recording with a starting track")
    ap.add_argument("--shard", type=str, default="0/1", help="K/N: every N-th recording from the K-th")
    ap.add_argument("--resume", action="store_true", help="skip the recordings already written")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    d: Path = a.dir
    if a.ids:
        ids = a.ids.split(",")
    elif a.all:
        ids = in_turns(sorted(p.name.split(".")[0] for p in (d / a.base).glob("*.tracks.json")))
    else:
        ids = json.loads((d / "perturb.json").read_text())["ids"][: a.n]
    k, n = (int(v) for v in a.shard.split("/"))
    ids = ids[k::n]
    if a.resume:
        ids = [i for i in ids if not (d / a.out / f"{i}.tracks.json").exists()]
    for name in (a.out, f"{a.out}-start") if a.calibrated else (a.out,):
        (d / name).mkdir(exist_ok=True)
    listener = Listener(model=a.listener, layer=a.layer) if a.listener else Listener()
    # how far each channel moves within an utterance of speech by rule (`pmvc.warp`'s tracks,
    # whatever the search starts from)
    sample = sorted((a.scale or d / ("warp" if (d / "warp").exists() else a.base)).glob("*.tracks.json"))[::5]
    scale = Scale([json.loads(p.read_text()) for p in sample])
    sigma = scale.std
    for word in a.also.split(",") if a.also else ():
        sigma[scale.keys.index(word.split("=")[0])] = float(word.split("=")[1])
    for uid in ids:
        (item,) = load_items(d, [uid], listener, a.base)
        settings = json.loads((d / f"calibration-{uid[0]}.json").read_text())["settings"] if a.calibrated else None
        copy(item, settings, listener, sigma, a, d)
    rows = summarise(d, a.out)
    r = np.array([[x["objective"], x["distance"], x["nll"]] for x in rows])
    print(f"{len(rows)} recordings: objective {r[:, 0, 0].mean():.3f} → {r[:, 0, 1].mean():.3f}, distance {r[:, 1, 0].mean():.3f} → {r[:, 1, 1].mean():.3f}, CTC loss {r[:, 2, 0].mean():.2f} → {r[:, 2, 1].mean():.2f}")


if __name__ == "__main__":
    main()
