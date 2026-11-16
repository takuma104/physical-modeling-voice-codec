"""Which body should a speaker get, when a corpus does not say the speaker's sex?

The voice has a male and a female body, and a speaker's meta says which (`sex`). The pitch does
not: in JVS the men's median pitch goes up to 196 Hz and the women's begins at 212 Hz, in TSP
the women's begins at 173 Hz, and people talking (Emilia-YODAS) fill the range between. So the
pitch decides only where nobody doubts it, and the others are placed by how their voice sounds.

usage: python -m pmvc.sexes DIR [DIR ...] [--sure 135,235] [--per 6] [--check] [--model MODEL]

For every speaker of the directories (as `pmvc.recordings`, `pmvc.jvs` or `pmvc.emilia` laid
them out), `--per` recordings are heard by a speaker-verification model (`pmvc.whose`'s) and
their vectors averaged. The speakers whose median pitch is under the first of `--sure` are
men and those over the second are women; the mean vector of each of the two groups stands for
the sex, and a speaker between is of the sex whose vector is nearer (by cosine).

Each speaker's metas then say that sex, and where it changed, the recordings' pitch is tracked
again in that sex's range (`pmvc.audio.pitch_track`), which the meta's median follows.

`--check` changes nothing: for directories whose metas are right (JVS, TSP), it says how many
of the speakers between the two pitches are placed as the meta has them, and how many the
pitch alone would have placed, at its best single threshold.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .audio import load, pitch_track
from .encoder import pick_device
from .whose import MODEL, SR, Verifier


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--sure", type=str, default="135,235", help="the median pitch (Hz) under which a speaker is a man and over which a woman, whatever the voice")
    ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    lo, hi = (float(x) for x in a.sure.split(","))
    metas: dict[tuple[Path, str], list[dict[str, Any]]] = {}
    for d in a.dirs:
        for p in sorted((d / "meta").glob("*.json")):
            m = json.loads(p.read_text())
            metas.setdefault((d, m["speaker"]), []).append(m)
    who = sorted(metas, key=lambda k: (str(k[0]), k[1]))
    pitch = np.array([float(np.median([m["f0Median"] for m in metas[k]])) for k in who])
    verifier = Verifier(pick_device(), a.model)
    heard = [(k, m) for k in who for m in metas[k][:: max(1, len(metas[k]) // a.per)][: a.per]]
    with ThreadPoolExecutor(a.jobs) as pool:
        clips = pool.map(lambda j: load(j[0][0] / "real" / f"{j[1]['id']}.wav", SR)[0], heard)
        vectors = [verifier.vector(x) for x in clips]
    mean: NDArray[np.float64] = np.stack([np.mean([v for (k, _), v in zip(heard, vectors) if k == key], axis=0) for key in who])
    mean /= np.linalg.norm(mean, axis=1, keepdims=True)
    men, women = mean[pitch < lo].mean(axis=0), mean[pitch > hi].mean(axis=0)
    lean = mean @ (women / np.linalg.norm(women)) - mean @ (men / np.linalg.norm(men))
    sex = np.where(pitch < lo, "M", np.where(pitch > hi, "F", np.where(lean > 0, "F", "M")))
    between = (pitch >= lo) & (pitch <= hi)
    print(f"{len(who)} speakers: {int((pitch < lo).sum())} under {lo:g} Hz, {int((pitch > hi).sum())} over {hi:g} Hz, {int(between.sum())} between, of whom {int((sex[between] == 'F').sum())} placed as women")
    if a.check:
        said = np.array([metas[k][0]["sex"] for k in who])
        right = sex == said
        best = max((float(np.mean((pitch[between] > t) == (said[between] == "F"))), float(t)) for t in np.arange(lo, hi + 1.0))
        print(f"between the two pitches: {int(right[between].sum())} of {int(between.sum())} as the metas say ({100 * right[between].mean():.1f} %); the pitch alone, at its best threshold ({best[1]:.0f} Hz), {100 * best[0]:.1f} %")
        print(f"outside them the pitch is right for {int(right[~between].sum())} of {int((~between).sum())}")
        for k, s, t, p, x in zip(who, sex, said, pitch, lean):
            if s != t:
                print(f"  {k[0].name} {k[1]}: {p:.0f} Hz, leans {x:+.3f}, placed {s}, meta {t}")
        return
    changed = 0
    for k, s in zip(who, sex):
        d = k[0]
        for m in metas[k]:
            if m["sex"] == s:
                continue
            x, sr = load(d / "real" / f"{m['id']}.wav")
            f0 = pitch_track(x, sr, str(s))
            np.save(d / "pitch" / f"{m['id']}.npy", f0)
            m.update({"sex": str(s), "f0Median": float(np.median(f0[f0 > 0])) if (f0 > 0).any() else m["f0Median"]})
            (d / "meta" / f"{m['id']}.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
            changed += 1
    for d in a.dirs:
        mine = [s for k, s in zip(who, sex) if k[0] == d]
        print(f"{d}: {mine.count('F')} women and {mine.count('M')} men")
    print(f"{changed} recordings' metas changed")


if __name__ == "__main__":
    main()
