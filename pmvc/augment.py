"""Renders of changed tracks for the encoder to learn from: pairs that are exact, off the rule's path.

An encoder taught on renders of the rule's tracks hears real speech well enough to write tracks
for it (`pmvc.encoder`), but every track it has seen is the rule's: one way to say each phone,
in two voices. Here each track is changed before it is rendered, so that the encoder also learns
what a different articulation, pitch, tempo and voice quality sound like.

usage: python -m pmvc.augment [dir] [--base warp] [--out aug] [--split train] [--seed 0] [--jobs N]

For every recording of the split with a `--base` track, writes <dir>/<out>/<id>.wav and
<id>.tracks.json (with the settings it was rendered under). The changes, drawn per recording:
  articulation   slow noise (under 4 Hz) on every channel but pitch, 8 to 20 dB under the
                 channel's spread within an utterance; the source's channels only during speech
  tempo          the whole track played 0.85 to 1.2 times as fast
  pitch          the contour moved by up to 3 semitones either way
  voice          the source's pressure, tension, breathiness and brightness and the three noise
                 gains, each within about a third of CHIMERA's own
"""

import argparse
import json
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .audio import FRAME_RATE, Array
from .calibrate import PARAMS, make_voice
from .encoder import split_of
from .paths import ROOT
from .perturb import Scale, perturb

SNR_DB: tuple[float, float] = (8.0, 20.0)
TEMPO: tuple[float, float] = (0.85, 1.2)
SEMITONES: float = 3.0
# the settings are drawn within this factor of CHIMERA's own, either way
VOICE: float = 1.35
SOURCE_KEYS: tuple[str, ...] = ("abduction", "closure", "breath")


def change(track: dict[str, Any], scale: Scale, rng: np.random.Generator) -> tuple[Array, dict[str, float]]:
    """A changed copy of a track's frames, and the settings to render it under."""
    x = np.array(track["frames"])
    keys: list[str] = track["keys"]
    f0, breath = keys.index("f0"), keys.index("breath")
    # tempo: the track read faster or slower
    at = np.arange(0.0, len(x) - 1, rng.uniform(*TEMPO))
    i = np.floor(at).astype(int)
    w = (at - i)[:, None]
    x = x[i] * (1 - w) + x[i + 1] * w
    # articulation: `perturb`'s slow noise, taken back where it does not belong
    y = perturb(x, f"smooth_snr{rng.uniform(*SNR_DB):.2f}", scale, rng)
    y[:, f0] = x[:, f0] * 2.0 ** (rng.uniform(-SEMITONES, SEMITONES) / 12.0)
    y[:, breath] = np.minimum(y[:, breath], 1.3)
    silent = x[:, breath] < 0.05
    for k in SOURCE_KEYS:
        y[silent, keys.index(k)] = x[silent, keys.index(k)]
    settings = {name: float(np.clip(default * VOICE ** rng.uniform(-1.0, 1.0), lo, hi)) for name, default, lo, hi in PARAMS}
    return y, settings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "transplant")
    ap.add_argument("--base", type=str, default="warp")
    ap.add_argument("--out", type=str, default="aug")
    ap.add_argument("--split", choices=["train", "valid", "test", "all"], default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    d: Path = a.dir
    paths = sorted((d / a.base).glob("*.tracks.json"))
    # the spread of the rule's tracks, whatever is changed
    scale = Scale([json.loads(p.read_text()) for p in sorted((d / ("warp" if (d / "warp").exists() else a.base)).glob("*.tracks.json"))[::5]])
    ids = [p.name.split(".")[0] for p in paths if a.split in ("all", split_of(p.name))]
    (d / a.out).mkdir(exist_ok=True)

    def one(uid: str) -> None:
        track = json.loads((d / a.base / f"{uid}.tracks.json").read_text())
        rng = np.random.default_rng(zlib.crc32(f"{uid}/{a.out}/{a.seed}".encode()))
        frames, settings = change(track, scale, rng)
        track = {**track, "frames": np.round(frames, 5).tolist(), "settings": settings}
        v = make_voice(track, settings)
        v.set_track(frames, 0)
        x = np.asarray(v.render((len(frames) - 1) * (48000 // FRAME_RATE)))
        sf.write(str(d / a.out / f"{uid}.wav"), x, 48000, subtype="PCM_16")
        (d / a.out / f"{uid}.tracks.json").write_text(json.dumps(track))

    with ThreadPoolExecutor(a.jobs) as pool:
        list(pool.map(one, ids))
    print(f"{len(ids)} recordings → {d / a.out}")


if __name__ == "__main__":
    main()
