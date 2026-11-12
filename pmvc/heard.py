"""What the listener makes of a results file's renders: the two terms `pmvc.copysynth` searches on.

usage: python -m pmvc.heard [dir] [--name results]

For every variant of <dir>/<name>.json that has control tracks (so a render on the recording's
timeline), adds to each recording `distance`, the listener's feature distance to the recording
over its speech (1 − cosine, frame against frame), and `ctc`, the CTC loss of the sentence per
character; and their means to the summary. Whisper's word errors say whether a render can be
understood; these say how close it is to what the speaker did, and when.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audio import load
from .calibrate import load_items, score
from .listener import Listener
from .paths import ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "transplant")
    ap.add_argument("--name", type=str, default="results")
    a = ap.parse_args()
    d: Path = a.dir
    path = d / f"{a.name}.json"
    results: dict[str, Any] = json.loads(path.read_text())
    listener = Listener()
    for row in results["utterances"]:
        for v, m in row["variants"].items():
            if (d / v / f"{row['id']}.tracks.json").exists():
                (item,) = load_items(d, [row["id"]], listener, v)
                x, _ = load(d / v / f"{row['id']}.wav")
                m["distance"], m["ctc"] = score(listener, item, x)
    for group, pick in (("all", "FM"), ("female", "F"), ("male", "M")):
        sel = [r for r in results["utterances"] if r["sex"] in pick]
        for v in sel[0]["variants"] if sel else []:
            for key in ("distance", "ctc"):
                vals = [r["variants"][v][key] for r in sel if key in r["variants"][v]]
                if vals:
                    results["summary"][group][v][key] = round(float(np.mean(vals)), 3)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    for v, s in results["summary"]["all"].items():
        if isinstance(s, dict) and "distance" in s:
            print(f"  {v:13s} distance {s['distance']:.3f}   CTC loss {s['ctc']:.2f}")


if __name__ == "__main__":
    main()
