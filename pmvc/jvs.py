"""The JVS corpus (Takamichi et al., 2019: a hundred Japanese speakers, each reading the same
hundred sentences, 24 kHz) laid out for the encoder to learn from and be tested on.

usage: python -m pmvc.jvs [--jvs ~/datasets/jvs/jvs_ver1] [--out out/jvs] [--per 10] [--jobs N]
       python -m pmvc.jvs --all [--out out/jvsall]

Everyone reads the same `parallel100` sentences, so speakers and sentences are both held out:
  test   the last four women and four men (jvs093-jvs100) reading sentences 76-100, which
         nobody reads in training or validation
  valid  the three women and three men before them (jvs087-jvs092), ten of sentences 1-75 each
  train  the other 86 speakers, `--per` of sentences 1-75 each: a different stretch of them for
         each speaker, so that every sentence is read by about as many
Writes what `pmvc.recordings` writes under <out>, with the split in each meta; the ids are
<speaker>_<sentence> (jvs001_012). The sentences average 7.5 s, so ten a speaker is two hours.

`--all` is for a text-to-speech model (`pmvc.tts`), which can only speak in voices it has
heard: all hundred speakers, and all of what each read normally. Sentences are held out,
speakers are not:
  test   everyone's sentences 76-100, the encoder's test sentences (for jvs093-jvs100 these
         are its test recordings, with the same ids)
  valid  everyone's sentences 71-75
  train  everyone's sentences 1-70, and the thirty sentences of `nonpara30` each speaker read
         alone (ids <speaker>_n01 to _n30): 3,000 sentences nobody else reads
The recordings are linked to, not copied.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from .paths import ROOT
from .recordings import sentences, take

TEST: tuple[str, ...] = ("jvs093", "jvs094", "jvs095", "jvs096", "jvs097", "jvs098", "jvs099", "jvs100")
VALID: tuple[str, ...] = ("jvs087", "jvs088", "jvs089", "jvs090", "jvs091", "jvs092")
# sentences 1..HELD_FROM-1 are for training and validation, the rest for the test
HELD_FROM: int = 76
VALID_PER: int = 10
# with `--all`: the first of the sentences everyone reads for validation
VALID_FROM: int = 71


def plan(speakers: list[str], per: int) -> list[tuple[str, int, str]]:
    """(speaker, sentence number, split) for every recording used."""
    out: list[tuple[str, int, str]] = []
    turn = 0
    for name in speakers:
        if name in TEST:
            out += [(name, k, "test") for k in range(HELD_FROM, 101)]
            continue
        n = VALID_PER if name in VALID else per
        out += [(name, 1 + (turn + j) % (HELD_FROM - 1), "valid" if name in VALID else "train") for j in range(n)]
        turn += n
    return out


def everything(jvs: Path, speakers: list[str]) -> list[tuple[str, Path, str, str]]:
    """(id, recording, sentence, split) for all that every speaker read normally."""
    out: list[tuple[str, Path, str, str]] = []
    for name in speakers:
        d = jvs / name / "parallel100"
        for key, text in sentences(d / "transcripts_utf8.txt"):
            k = int(key.split("_")[1])
            out.append((f"{name}_{k:03d}", d / "wav24kHz16bit" / f"{key}.wav", text, "test" if k >= HELD_FROM else "valid" if k >= VALID_FROM else "train"))
        d = jvs / name / "nonpara30"
        # the transcript lists more sentences than the speaker read
        read = [(key, text) for key, text in sentences(d / "transcripts_utf8.txt") if (d / "wav24kHz16bit" / f"{key}.wav").exists()]
        out += [(f"{name}_n{k:02d}", d / "wav24kHz16bit" / f"{key}.wav", text, "train") for k, (key, text) in enumerate(read, 1)]
    return [job for job in out if job[1].exists()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jvs", type=Path, default=Path("~/datasets/jvs/jvs_ver1").expanduser())
    ap.add_argument("--out", type=Path, default=None, help="out/jvs, or out/jvsall with --all")
    ap.add_argument("--per", type=int, default=10, help="sentences from each training speaker")
    ap.add_argument("--all", action="store_true", help="every speaker and all they read normally, split by sentence: for pmvc.tts")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    a.out = a.out or ROOT / "out" / ("jvsall" if a.all else "jvs")
    for sub in ("real", "pitch", "meta"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    sex = {line.split()[0]: line.split()[1] for line in (a.jvs / "gender_f0range.txt").read_text().splitlines()[1:]}
    if a.all:
        jobs = everything(a.jvs, sorted(sex))
        with ProcessPoolExecutor(a.jobs) as workers:
            for task in [workers.submit(take, a.out, uid, wav, uid.split("_")[0], "ja", sex[uid.split("_")[0]], text, split, True) for uid, wav, text, split in jobs]:
                task.result()
        print(f"{a.out}: " + ", ".join(f"{sum(j[3] == s for j in jobs)} {s}" for s in ("train", "valid", "test")))
        return

    def one(job: tuple[str, int, str]) -> str:
        name, k, split = job
        d = a.jvs / name / "parallel100"
        text = dict(sentences(d / "transcripts_utf8.txt")).get(f"VOICEACTRESS100_{k:03d}")
        wav = d / "wav24kHz16bit" / f"VOICEACTRESS100_{k:03d}.wav"
        # a few speakers lack a sentence or two
        if text is None or not wav.exists():
            return ""
        take(a.out, f"{name}_{k:03d}", wav, name, "ja", sex[name], text, split)
        return split

    with ThreadPoolExecutor(a.jobs) as pool:
        done = list(pool.map(one, plan(sorted(sex), a.per)))
    print(f"{a.out}: " + ", ".join(f"{done.count(s)} {s}" for s in ("train", "valid", "test")) + (f"; {done.count('')} missing" if "" in done else ""))


if __name__ == "__main__":
    main()
