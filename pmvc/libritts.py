"""LibriTTS-R (Koizumi et al., 2023: LibriTTS's audiobook sentences with their sound restored,
24 kHz, CC BY 4.0) laid out for an English text-to-speech model to learn from.

usage: python -m pmvc.libritts --corpus DIR [--subset train-clean-100] [--out out/lten]
                               [--per 40] [--valid-clips 2] [--seconds 3,10] [--seed 0] [--jobs N]

`--corpus` is the corpus as it unpacks (LibriTTS_R: SPEAKERS.txt, and <subset>/<speaker>/
<chapter>/<utterance>.wav with its .normalized.txt beside it). Of every speaker of the subset,
`--per` sentences that last `--seconds` are drawn (all, if the speaker has fewer); the last
`--valid-clips` are for validation, the others to learn from. Speakers are not held out: a
model can only speak in voices it has heard. SPEAKERS.txt says each speaker's sex, which picks
the voice (`pmvc.encoder --voice voices/v1.json`).

Writes what `pmvc.recordings` writes under <out>, the recordings as links; the ids are the
corpus's own (<speaker>_<chapter>_<paragraph>_<sentence>) and a speaker's name is lt_<speaker>.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from .paths import ROOT
from .recordings import take


def sexes(corpus: Path, subset: str) -> dict[str, str]:
    """The sex of each speaker of a subset, as SPEAKERS.txt has it."""
    out: dict[str, str] = {}
    for line in (corpus / "SPEAKERS.txt").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if not line.startswith(";") and len(cells) >= 3 and cells[2] == subset and cells[1] in ("F", "M"):
            out[cells[0]] = cells[1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--subset", type=str, default="train-clean-100")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "lten")
    ap.add_argument("--per", type=int, default=40)
    ap.add_argument("--valid-clips", type=int, default=2)
    ap.add_argument("--seconds", type=str, default="3,10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    d: Path = a.out
    for sub in ("real", "pitch", "meta"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    lo, hi = (float(x) for x in a.seconds.split(","))
    sex = sexes(a.corpus, a.subset)
    rng = np.random.default_rng(a.seed)

    def lasting(wav: Path) -> float:
        info = sf.info(str(wav))
        return info.frames / info.samplerate

    jobs: list[tuple[str, Path, str, str, str]] = []
    with ThreadPoolExecutor(a.jobs) as pool:
        for name in sorted(sex, key=int):
            wavs = sorted((a.corpus / a.subset / name).glob("*/*.wav"))
            good = [w for w, t in zip(wavs, pool.map(lasting, wavs)) if lo <= t <= hi and w.with_suffix(".normalized.txt").exists()]
            mine = [good[i] for i in sorted(rng.permutation(len(good))[: a.per])]
            for k, w in enumerate(mine):
                text = w.with_suffix(".normalized.txt").read_text(encoding="utf-8").strip()
                jobs.append((w.stem, w, name, text, "valid" if k >= len(mine) - a.valid_clips else "train"))
    with ProcessPoolExecutor(a.jobs) as workers:
        tasks = [workers.submit(take, d, uid, w, f"lt_{name}", "en", sex[name], text, split, True) for uid, w, name, text, split in jobs]
        metas = [t.result() for t in tasks]
    kept = 0
    for meta in metas:
        path = d / "meta" / f"{meta['id']}.json"
        if not np.isfinite(meta["f0Median"]):
            for p in (path, d / "real" / f"{meta['id']}.wav", d / "pitch" / f"{meta['id']}.npy"):
                p.unlink(missing_ok=True)
            continue
        path.write_text(json.dumps({**meta, "source": f"LibriTTS-R {a.subset}", "licence": "CC BY 4.0"}, ensure_ascii=False, indent=1))
        kept += 1
    women = sum(s == "F" for s in sex.values())
    print(f"{kept} recordings of {len(sex)} speakers ({women} women) → {d}")


if __name__ == "__main__":
    main()
