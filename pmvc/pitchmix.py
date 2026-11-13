"""How much of a fitted voice suits a speaker? It goes by the speaker's pitch, not the sex.

`pmvc.calibrate --by nisqa` fits the voice's settings to one woman's recordings. This replays
each speaker's tracks with the settings a share w of the way from CHIMERA's own to the fitted
ones (in the fit's coordinates: each setting's ratio to its default raised to w) and has NISQA
score the renders (`pmvc.natural`).

usage: python -m pmvc.pitchmix DIR:VARIANT:PREFIX ... [--voice out/fit/voice-F2.json] [--n 10]
                               [--shares 0,0.25,0.5,0.75,1,1.25]

Each argument names a speaker: the recordings of DIR whose ids start with PREFIX, and the
variant whose tracks are replayed; `n` of them are used. Writes <voice>.pitchmix.json and prints
the table, the speakers by median pitch.

With encoder H's tracks, speakers whose median pitch is 150 Hz or lower are best left with
CHIMERA's settings (the fitted ones cost them up to 0.8 of a point), and those from 175 Hz up do
better with the fitted ones, the men among them too. `pmvc.encoder`'s `--voice` goes by that.
"""

import argparse
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .calibrate import make_voice, mixed
from .natural import score
from .paths import ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("speakers", type=str, nargs="+", help="DIR:VARIANT:PREFIX")
    ap.add_argument("--voice", type=Path, default=ROOT / "out" / "fit" / "voice-F2.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--shares", type=str, default="0,0.25,0.5,0.75,1,1.25")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    fitted: dict[str, float] = json.loads(a.voice.read_text())["settings"]
    shares = [float(w) for w in a.shares.split(",")]
    jobs: list[tuple[str, Path, float]] = []
    for spec in a.speakers:
        d, variant, prefix = spec.split(":")
        tracks = sorted((Path(d) / variant).glob(f"{prefix}*.tracks.json"))
        jobs += [(prefix, p, w) for p in tracks[:: max(1, len(tracks) // a.n)][: a.n] for w in shares]

    with tempfile.TemporaryDirectory() as tmp:

        def render(job: tuple[int, tuple[str, Path, float]]) -> tuple[Path, float]:
            k, (_, path, w) = job
            track: dict[str, Any] = json.loads(path.read_text())
            # the track as CHIMERA's own voice would play it, whatever voice it was written for
            # (but with what it says of the walls' air, which is no part of a voice)
            walls = {key: v for key, v in track.get("airway", {}).items() if key == "volumeSource"}
            track = {**{key: v for key, v in track.items() if key not in ("noise", "glottis", "airway")}, "source": {"f0": track["source"]["f0"]}, **({"airway": walls} if walls else {})}
            v = make_voice(track, mixed(fitted, w) if w else None)
            frames = np.array(track["frames"])
            v.set_track(frames, 0)
            wav = Path(tmp) / f"{k}.wav"
            sf.write(str(wav), np.asarray(v.render((len(frames) - 1) * 480)), 48000, subtype="PCM_16")
            return wav, float(track["source"]["f0"])

        with ThreadPoolExecutor(a.jobs) as pool:
            rendered = list(pool.map(render, enumerate(jobs)))
        quality = [s["quality"] for s in score([wav for wav, _ in rendered], natural=False)]

    by: dict[str, dict[str, Any]] = {}
    for (prefix, _, w), (_, f0), q in zip(jobs, rendered, quality):
        row = by.setdefault(prefix, {"pitch": [], "quality": {}})
        row["pitch"].append(f0)
        row["quality"].setdefault(w, []).append(q)
    table: list[dict[str, Any]] = [{"speaker": name, "pitch": float(np.median(r["pitch"])), "quality": {str(w): float(np.mean(q)) for w, q in r["quality"].items()}} for name, r in by.items()]
    table.sort(key=lambda r: r["pitch"])
    a.voice.with_suffix(".pitchmix.json").write_text(json.dumps({"shares": shares, "n": a.n, "speakers": table}, indent=1))
    print("speaker    pitch  " + "  ".join(f"w={w:<5g}" for w in shares))
    for r in table:
        print(f"{r['speaker']:9s} {r['pitch']:5.0f}  " + "  ".join(f"{r['quality'][str(w)]:7.2f}" for w in shares))


if __name__ == "__main__":
    main()
