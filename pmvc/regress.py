"""Does the voice still sound as V1's did? A fixed control track rendered under V1's voices, sample for sample.

From V2 on the engine is a fork (third-party/chimera-vocal-synth, branch pmvc-v2), and every
change to it is a setting that defaults to what the engine did before. This is the check of
that promise: voices/regress/track.json (the first seconds of the release encoder's track for
a TSP recording, a man's sentence) is rendered under `voices/v1.json`'s low voice and its high
one, and each render's samples (16 bit) must hash to what voices/regress/v1.json says.

usage: python -m pmvc.regress [--bless]

`--bless` writes the hashes instead of checking them: only for a change that is meant to move
V1's sound (none so far), and then the commit says so. The models and the demo published as V1
were made with these samples.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .encoder import Decoder, render_track
from .paths import ROOT

HOME: Path = ROOT / "voices" / "regress"
VOICES: Path = ROOT / "voices" / "v1.json"


def renders() -> dict[str, str]:
    """The hash of the track's render under each of the pair's voices."""
    voice: dict[str, Any] = json.loads(VOICES.read_text())
    decoder = Decoder(family={k: voice[k] for k in ("low", "high", "bright", "dull") if k in voice}, wall_air=False, closed_area=1e-6)
    track: dict[str, Any] = json.loads((HOME / "track.json").read_text())
    out: dict[str, str] = {}
    for name, female, pitch in (("low", 0.0, 1.0), ("high", 1.0, 2.0 ** (9 / 12))):
        f0 = track["keys"].index("f0")
        frames = [[*row[:f0], row[f0] * pitch, *row[f0 + 1 :]] for row in track["frames"]]
        t = decoder.settle({**track, "frames": frames, "body": {"female": female, "child": 0.0}, "source": {"f0": round(track["source"]["f0"] * pitch, 2)}}, 0.0)
        x = np.round(np.clip(render_track(t), -1.0, 1.0) * 32767.0).astype("<i2")
        out[name] = hashlib.sha256(x.tobytes()).hexdigest()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bless", action="store_true")
    a = ap.parse_args()
    got, path = renders(), HOME / "v1.json"
    if a.bless:
        path.write_text(json.dumps(got, indent=1) + "\n")
        print(f"blessed: {got}")
        return
    want: dict[str, str] = json.loads(path.read_text())
    for name, h in got.items():
        print(f"{name}: {'the same samples' if h == want[name] else 'DIFFERENT samples'} ({h[:12]})")
    sys.exit(0 if got == want else 1)


if __name__ == "__main__":
    main()
