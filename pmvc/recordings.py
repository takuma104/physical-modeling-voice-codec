"""Recordings from outside TSP, laid out as `pmvc.encoder encode` and `pmvc.judge` read them.

usage: python -m pmvc.recordings [dir] --set hjf --lang ja --sex F --wavs DIR --text FILE [--n 100]

`--text` has a line per recording: the WAV's name without `.wav`, a colon or a space, and the
sentence (JSUT's `transcript_utf8.txt`, Hi-Fi-CAPTAIN's `eval.txt`). Takes the first `n` lines
that have a WAV in `--wavs` and writes, under <dir> (default out/ja):
  real/<set>_<k>.wav    the recording
  pitch/<set>_<k>.npy   its pitch track (0 = unvoiced); the tracker's range goes by the sex
  meta/<set>_<k>.json   the sentence and its language, the speaker's sex and median pitch (all
                        the voice is told of the speaker), and the span of the speech: the
                        first and last frame within 40 dB of the loudest
Nothing here knows the language: there is no alignment and no render by rule.
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .audio import FRAME_RATE, MEL_SR, load, pitch_track, speech_span
from .paths import ROOT
from .transplant import START_MIN


def sentences(path: Path) -> list[tuple[str, str]]:
    """(name, sentence) for each line of a transcript file."""
    out: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = re.split(r"[:\s]", line.strip(), maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            out.append((parts[0], parts[1].strip()))
    return out


def take(d: Path, uid: str, wav: Path, name: str, lang: str, sex: str, text: str, split: str = "") -> dict[str, Any]:
    """Lays out one recording. `split` (train, valid or test) is written to its meta when a
    corpus is to be learned from; without it `pmvc.encoder` goes by TSP's speakers."""
    x, sr = load(wav)
    f0 = pitch_track(x, sr, sex)
    a, b = speech_span(load(wav, MEL_SR)[0])
    shutil.copyfile(wav, d / "real" / f"{uid}.wav")
    np.save(d / "pitch" / f"{uid}.npy", f0)
    meta = {
        "id": uid,
        "speaker": name,
        "sex": sex,
        "lang": lang,
        "text": text,
        "source": wav.name,
        # how far a render runs behind the recording when its track does not say: no variant
        # here is without a track of its own, and the encoder's run this far behind
        "shift": START_MIN,
        "aligned": [],
        "f0Median": float(np.median(f0[f0 > 0])),
        "span": [a / FRAME_RATE, min(b / FRAME_RATE, len(x) / sr)],
    }
    if split:
        meta["split"] = split
    (d / "meta" / f"{uid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "ja")
    ap.add_argument("--set", type=str, required=True, help="a name for these recordings: their ids start with it")
    ap.add_argument("--lang", type=str, required=True, help="the language, as Whisper names it (ja, en)")
    ap.add_argument("--sex", choices=["F", "M"], required=True)
    ap.add_argument("--wavs", type=Path, required=True)
    ap.add_argument("--text", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    d: Path = a.dir
    for sub in ("real", "pitch", "meta"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    found = [(name, text) for name, text in sentences(a.text) if (a.wavs / f"{name}.wav").exists()][: a.n]
    for k, (name, text) in enumerate(found, 1):
        take(d, f"{a.set}_{k:03d}", a.wavs / f"{name}.wav", a.set, a.lang, a.sex, text)
    print(f"{len(found)} recordings → {d}")


if __name__ == "__main__":
    main()
