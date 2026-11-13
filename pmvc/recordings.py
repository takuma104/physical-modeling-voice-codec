"""Recordings from outside TSP, laid out as `pmvc.encoder encode` and `pmvc.judge` read them.

usage: python -m pmvc.recordings [dir] --set hjf --lang ja --sex F --wavs DIR --text FILE [--n 100]
                                 [--speaker NAME] [--split train|valid|test] [--link] [--jobs N]

`--text` has a line per recording: the WAV's name without `.wav`, a colon or a space, and the
sentence (JSUT's `transcript_utf8.txt`, Hi-Fi-CAPTAIN's `eval.txt`). Takes the first `n` lines
that have a WAV in `--wavs` (all of them with `--n 0`) and writes, under <dir> (default out/ja):
  real/<set>_<k>.wav    the recording (with `--link`, a link to it: for a corpus to learn from)
  pitch/<set>_<k>.npy   its pitch track (0 = unvoiced); the tracker's range goes by the sex
  meta/<set>_<k>.json   the sentence and its language, the speaker's sex and median pitch (all
                        the voice is told of the speaker), and the span of the speech: the
                        first and last frame within 40 dB of the loudest
Nothing here knows the language: there is no alignment and no render by rule.

A corpus to learn from (`pmvc.tts`) is brought in set by set, each with its `--split` and all
under one `--speaker`: the speaker's name is what the voice goes by (`pmvc.encoder`), and it is
the set's name when not given.
"""

import argparse
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
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


def take(d: Path, uid: str, wav: Path, name: str, lang: str, sex: str, text: str, split: str = "", link: bool = False) -> dict[str, Any]:
    """Lays out one recording. `split` (train, valid or test) is written to its meta when a
    corpus is to be learned from; without it `pmvc.encoder` goes by TSP's speakers."""
    x, sr = load(wav)
    f0 = pitch_track(x, sr, sex)
    a, b = speech_span(load(wav, MEL_SR)[0])
    to = d / "real" / f"{uid}.wav"
    to.unlink(missing_ok=True)
    if link:
        to.symlink_to(wav.resolve())
    else:
        shutil.copyfile(wav, to)
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
    ap.add_argument("--n", type=int, default=100, help="how many recordings; 0 for all")
    ap.add_argument("--speaker", type=str, default="", help="the speaker's name, if not the set's")
    ap.add_argument("--split", choices=["", "train", "valid", "test"], default="", help="the split the recordings are in, for a corpus to learn from")
    ap.add_argument("--link", action="store_true", help="link to the recordings instead of copying them")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    d: Path = a.dir
    for sub in ("real", "pitch", "meta"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    found = [(name, text) for name, text in sentences(a.text) if (a.wavs / f"{name}.wav").exists()][: a.n or None]
    digits = max(3, len(str(len(found))))
    with ProcessPoolExecutor(a.jobs) as pool:
        tasks = [pool.submit(take, d, f"{a.set}_{k:0{digits}d}", a.wavs / f"{name}.wav", a.speaker or a.set, a.lang, a.sex, text, a.split, a.link) for k, (name, text) in enumerate(found, 1)]
        for t in tasks:
            t.result()
    print(f"{len(found)} recordings → {d}")


if __name__ == "__main__":
    main()
