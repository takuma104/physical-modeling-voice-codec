"""Emilia-YODAS (He et al., 2025: speech from videos whose makers put them under a Creative
Commons licence, cut into clips of one speaker, each with what a recogniser wrote of it and a
quality score; CC BY 4.0) laid out for the encoder to learn from and be tested on.

What an encoder is taught with decides who may have its weights. JVS may not be passed on and
is for research; Emilia-YODAS asks only that it be named. Its clips are of people talking, not
reading: fillers, laughter, a room, sometimes music the makers' separation left behind. An
encoder needs no text, so what the recogniser got wrong does not matter to it; a text-to-speech
model (`pmvc.tts`) would learn those mistakes.

usage: python -m pmvc.emilia --tars DIR --lang ja [--out out/emja] [--shards 3] [--speakers 300]
                             [--per 6] [--held 12] [--valid-clips 0] [--draw 3] [--quality 3.3]
                             [--seconds 3,10] [--seed 0] [--jobs N]

`--tars` is the language's directory of the corpus as it is distributed (JA-B000000.tar, ...:
for each clip an MP3 and a JSON with `text`, `duration`, `speaker`, `dnsmos`); the first
`--shards` of them are read, and nothing is unpacked to disk but the clips taken. A clip is
taken if it lasts `--seconds`, its DNSMOS is `--quality` or more and its text is in the
language's script (kana for Japanese with no Latin letter in it, Latin letters alone for
English: clips in another language are filed under each). Of a video, one speaker counts (the
one with most such clips: the others share the room, and sometimes the sentence); of the
speakers with `--per` clips or more, `--speakers` are for training and `--held` each for
validation and for the test, `--per` clips of each, all drawn by `--seed`: `--draw` times as many
speakers are drawn as are wanted, and of them high voices and low ones are taken in turn (the
corpus has more men; see the sex below).

For a text-to-speech model, which can only speak in voices it has heard, speakers are not held
out (`--held 0`) and each has many clips (`--per 40`); `--valid-clips` of them are for
validation. A clip whose text has a digit is left out: how it was read is not written.

Writes what `pmvc.recordings` writes under <out>, with the split in each meta; the ids are the
corpus's own (JA_<video>_W000012) and a speaker's name is the corpus's. The corpus does not say
a speaker's sex. The meta's is a guess from the pitch (F from SEX_PITCH Hz up, over the
speaker's clips), which sets only the pitch tracker's range: a voice that goes by the pitch
(`pmvc.encoder --voice voices/v1.json`) asks for nothing else.
"""

import argparse
import io
import json
import os
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .audio import pitch_track
from .paths import ROOT
from .recordings import take

# the median pitch (Hz) from which a speaker's meta says F
SEX_PITCH: float = 165.0
# what a clip's text must have, and must not, to be in the language
SCRIPT: dict[str, tuple[str, str]] = {"ja": (r"[ぁ-んァ-ヶ]", r"[A-Za-z0-9０-９]"), "en": (r"[A-Za-z]", r"[0-9]|[^\x00-\x7f’‘“”—–…]")}


def in_language(text: str, lang: str) -> bool:
    has, has_not = SCRIPT[lang]
    return bool(re.search(has, text)) and not re.search(has_not, text)


def listed(tar: Path) -> list[dict[str, Any]]:
    """The JSON of every clip in a tar."""
    out: list[dict[str, Any]] = []
    with tarfile.open(tar) as t:
        for m in t:
            if m.name.endswith(".json"):
                f = t.extractfile(m)
                assert f is not None
                out.append({**json.load(f), "tar": tar.name})
    return out


def candidates(clips: list[dict[str, Any]], a: argparse.Namespace) -> list[list[dict[str, Any]]]:
    """`--per` clips of each speaker that may be taken, the speakers in the order drawn: `--draw`
    times as many as are wanted, for as many high voices as low ones to be picked among them."""
    lo, hi = (float(x) for x in a.seconds.split(","))
    good = [c for c in clips if lo <= c["duration"] <= hi and c["dnsmos"] >= a.quality and in_language(c["text"], a.lang)]
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for c in good:
        by_speaker.setdefault(c["speaker"], []).append(c)
    # a video's speaker with most clips
    by_video: dict[str, str] = {}
    for name in sorted(by_speaker):
        video = name.rsplit("_SPEAKER", 1)[0]
        if video not in by_video or len(by_speaker[name]) > len(by_speaker[by_video[video]]):
            by_video[video] = name
    names = sorted(n for n in by_video.values() if len(by_speaker[n]) >= a.per)
    rng = np.random.default_rng(a.seed)
    names = [names[i] for i in rng.permutation(len(names))][: round(a.draw * (a.speakers + 2 * a.held))]
    assert len(names) > 2 * a.held, f"only {len(names)} speakers with {a.per} clips: read more shards"
    out: list[list[dict[str, Any]]] = []
    for name in names:
        mine = sorted(by_speaker[name], key=lambda c: c["_id"])
        out.append([mine[i] for i in sorted(rng.permutation(len(mine))[: a.per])])
    return out


def balanced(names: list[str], sex: dict[str, str], n: int) -> list[str]:
    """The first `n` of `names` with women and men in turn, as far as there are both."""
    left = {s: [name for name in names if sex[name] == s] for s in "FM"}
    out: list[str] = []
    while len(out) < n and (left["F"] or left["M"]):
        for s in "FM":
            if left[s] and len(out) < n:
                out.append(left[s].pop(0))
    return out


def first_pitch(job: tuple[Path, bytes]) -> float:
    """Decodes a clip to a WAV (mono, as sampled) and says its median pitch in a man's range."""
    wav, data = job
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    sf.write(str(wav), x, sr, subtype="PCM_16")
    f0 = pitch_track(x.astype(np.float64), sr, "M")
    return float(np.median(f0[f0 > 0])) if (f0 > 0).any() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tars", type=Path, required=True)
    ap.add_argument("--lang", choices=sorted(SCRIPT), required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--speakers", type=int, default=300)
    ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--held", type=int, default=12)
    ap.add_argument("--valid-clips", type=int, default=0, help="of each speaker to learn from, so many clips are for validation")
    ap.add_argument("--draw", type=float, default=3.0, help="how many times the speakers wanted are drawn, to pick high and low voices in turn from")
    ap.add_argument("--quality", type=float, default=3.3)
    ap.add_argument("--seconds", type=str, default="3,10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    d: Path = a.out or ROOT / "out" / f"em{a.lang}"
    for sub in ("real", "pitch", "meta", "staging"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    tars = sorted(a.tars.glob("*.tar"))[: a.shards]
    with ProcessPoolExecutor(min(a.jobs, len(tars))) as pool:
        clips = [c for part in pool.map(listed, tars) for c in part]
    drawn = candidates(clips, a)
    wanted = {c["_id"]: c for mine in drawn for c in mine}
    print(f"{len(clips)} clips in {len(tars)} tars; {len(drawn)} speakers drawn", flush=True)
    # the clips' MP3s, a tar at a time
    data: dict[str, bytes] = {}
    for tar in tars:
        with tarfile.open(tar) as t:
            for m in t:
                uid = m.name.rsplit(".", 1)[0]
                if m.name.endswith(".mp3") and uid in wanted:
                    f = t.extractfile(m)
                    assert f is not None
                    data[uid] = f.read()
    with ProcessPoolExecutor(a.jobs) as pool:
        heard = dict(zip(sorted(data), pool.map(first_pitch, [(d / "staging" / f"{uid}.wav", data[uid]) for uid in sorted(data)], chunksize=8)))
        # a speaker counts if every clip has a pitch; the sex is a guess from it
        whole = [mine for mine in drawn if all(heard.get(c["_id"], 0.0) > 0 for c in mine)]
        sex = {mine[0]["speaker"]: "F" if float(np.median([heard[c["_id"]] for c in mine])) >= SEX_PITCH else "M" for mine in whole}
        names = [mine[0]["speaker"] for mine in whole]
        split: dict[str, str] = {}
        for part, n in (("test", a.held), ("valid", a.held), ("train", a.speakers)):
            for name in balanced([name for name in names if name not in split], sex, n):
                split[name] = part
        taken = [c for mine in whole for c in mine if c["speaker"] in split]
        # of a speaker to learn from, the last clips are for validation
        part_of = {c["_id"]: "valid" if split[c["speaker"]] == "train" and k >= len(mine) - a.valid_clips else split[c["speaker"]] for mine in whole for k, c in enumerate(mine) if c["speaker"] in split}
        tasks = [pool.submit(take, d, c["_id"], d / "staging" / f"{c['_id']}.wav", c["speaker"], a.lang, sex[c["speaker"]], c["text"], part_of[c["_id"]]) for c in taken]
        metas = [t.result() for t in tasks]
    for c, meta in zip(taken, metas):
        (d / "meta" / f"{c['_id']}.json").write_text(json.dumps({**meta, "source": f"Emilia-YODAS {c['tar']}", "licence": "CC BY 4.0", "dnsmos": c["dnsmos"]}, ensure_ascii=False, indent=1))
    for wav in (d / "staging").glob("*.wav"):
        wav.unlink()
    (d / "staging").rmdir()
    by = {part: sum(part_of[c["_id"]] == part for c in taken) for part in ("train", "valid", "test")}
    women = sum(sex[name] == "F" for name in split)
    print(f"{len(taken)} recordings of {len(split)} speakers → {d}: {by}; {women} speakers have a high voice")


if __name__ == "__main__":
    main()
