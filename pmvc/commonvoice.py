"""Common Voice (Mozilla: sentences read aloud by volunteers and checked by others, CC0) laid out
for a text-to-speech model to learn from.

What a text-to-speech model needs of its recordings is that the text is what was said. Common
Voice's is: a speaker reads a sentence off the screen, and a clip is `validated` when other
volunteers heard that sentence in it. The rooms and microphones are whatever the volunteers
had, which a model that writes a control track for a simulated voice does not pass on. Many
speakers say which sex they are, so the voice need not be guessed (`pmvc.sexes`).

usage: python -m pmvc.commonvoice --corpus DIR [--lang ja] [--out out/cvja] [--speakers 300]
                                  [--per 16] [--valid-clips 1] [--seconds 2.5,10] [--seed 0]
                                  [--jobs N]

`--corpus` is the language's directory of a release as it unpacks (cv-corpus-27.0-2026-09-11/ja:
`validated.tsv`, `clip_durations.tsv`, `clips/*.mp3`). A clip is taken if it is validated with
no vote against it, lasts `--seconds` and its sentence is in the language's script with no
digit (`pmvc.emilia.in_language`). Of the speakers who say they are women and of those who say
they are men (a clip with no voice in it, by the pitch tracker, is dropped), `--speakers` each
are drawn among those with `--per` such clips or more, and
`--per` clips of each; the last `--valid-clips` of a speaker's are for validation, the others
to learn from (speakers are not held out: a model can only speak in voices it has heard).

Writes what `pmvc.recordings` writes under <out>; the ids are cv<lang>_<the clip's number> and
a speaker's name is cv_ and the first twelve characters of the corpus's id for them.
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .emilia import in_language
from .paths import ROOT
from .recordings import take

SEXES: dict[str, str] = {"female_feminine": "F", "male_masculine": "M"}


def decoded(job: tuple[Path, Path]) -> bool:
    """An MP3 as a WAV (mono, as sampled); whether it could be read."""
    mp3, wav = job
    try:
        x, sr = sf.read(str(mp3), dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError):
        return False
    sf.write(str(wav), x.mean(axis=1), sr, subtype="PCM_16")
    return len(x) > sr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--lang", type=str, default="ja")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--speakers", type=int, default=300, help="of each sex")
    ap.add_argument("--per", type=int, default=16)
    ap.add_argument("--valid-clips", type=int, default=1)
    ap.add_argument("--seconds", type=str, default="2.5,10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    d: Path = a.out or ROOT / "out" / f"cv{a.lang}"
    for sub in ("real", "pitch", "meta", "staging"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    lo, hi = (float(x) for x in a.seconds.split(","))
    with open(a.corpus / "clip_durations.tsv", encoding="utf-8") as f:
        lasts = {r["clip"]: float(r["duration[ms]"]) / 1000.0 for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}
    with open(a.corpus / "validated.tsv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]
    good = [r for r in rows if r["gender"] in SEXES and r["down_votes"] == "0" and lo <= lasts.get(r["path"], 0.0) <= hi and in_language(r["sentence"], a.lang)]
    by: dict[str, list[dict[str, Any]]] = {}
    for r in good:
        by.setdefault(r["client_id"], []).append(r)
    rng = np.random.default_rng(a.seed)
    taken: list[tuple[dict[str, Any], str]] = []
    for label, sex in SEXES.items():
        names = sorted(k for k, v in by.items() if v[0]["gender"] == label and len(v) >= a.per)
        names = [names[i] for i in rng.permutation(len(names))][: a.speakers]
        for name in names:
            mine = sorted(by[name], key=lambda r: r["path"])
            mine = [mine[i] for i in sorted(rng.permutation(len(mine))[: a.per])]
            taken += [(r, "valid" if k >= len(mine) - a.valid_clips else "train") for k, r in enumerate(mine)]
        print(f"{sex}: {len(names)} speakers of {sum(v[0]['gender'] == label for v in by.values())} who say so", flush=True)

    def uid(r: dict[str, Any]) -> str:
        return f"cv{a.lang}_{r['path'].rsplit('_', 1)[1].split('.')[0]}"

    with ProcessPoolExecutor(a.jobs) as pool:
        read = list(pool.map(decoded, [(a.corpus / "clips" / r["path"], d / "staging" / f"{uid(r)}.wav") for r, _ in taken], chunksize=16))
        taken = [t for t, ok in zip(taken, read) if ok]
        tasks = [pool.submit(take, d, uid(r), d / "staging" / f"{uid(r)}.wav", f"cv_{r['client_id'][:12]}", a.lang, SEXES[r["gender"]], r["sentence"].strip(), split) for r, split in taken]
        metas = [t.result() for t in tasks]
    for (r, _), meta in zip(taken, metas):
        if not np.isfinite(meta["f0Median"]):
            # a clip in which the pitch tracker finds no voice (a volunteer's silence, or noise) is no recording to learn from
            for path in (d / "meta" / f"{meta['id']}.json", d / "real" / f"{meta['id']}.wav", d / "pitch" / f"{meta['id']}.npy"):
                path.unlink(missing_ok=True)
            continue
        (d / "meta" / f"{meta['id']}.json").write_text(json.dumps({**meta, "source": f"Common Voice {a.corpus.parent.name} {r['path']}", "licence": "CC0-1.0"}, ensure_ascii=False, indent=1))
    taken = [t for t, meta in zip(taken, metas) if np.isfinite(meta["f0Median"])]
    for wav in (d / "staging").glob("*.wav"):
        wav.unlink()
    (d / "staging").rmdir()
    parts = {part: sum(split == part for _, split in taken) for part in ("train", "valid")}
    print(f"{len(taken)} recordings of {len({r['client_id'] for r, _ in taken})} speakers → {d}: {parts}")


if __name__ == "__main__":
    main()
