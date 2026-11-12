"""How the clips of a results file sound to NISQA (Mittag et al., 2021; third-party/NISQA, as it is).

Whisper and the phone recogniser of `pmvc.judge` say whether a render can be understood and
which sounds it got wrong. NISQA predicts, from the signal alone, the opinion score listeners
would give it: `nisqa.tar` the overall quality of a speech sample and four dimensions of it,
`nisqa_tts.tar` the naturalness of synthesised speech. Of the two it is the quality score that
agrees with the ear here: CHIMERA's male voice scores as high as the men's recordings (4.6),
its female voice far below the women's (3.0 against 4.8), with the same encoder.

usage: python -m pmvc.natural [dir] [--name results] [--jobs N]

Adds to every variant of every recording in <dir>/<name>.json (as `pmvc.judge` wrote it):
  quality        NISQA v2's overall quality, a mean opinion score from 1 to 5
  noisiness, coloration, discontinuity, loudness   its four dimensions, on the same scale
  natural        NISQA-TTS's naturalness
and their means to the summary, which is redone as `pmvc.judge --again` does it.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from .judge import finish
from .paths import ROOT

NISQA: Path = ROOT / "third-party" / "NISQA"
# NISQA's own names for what its two models predict, and ours
QUALITY: dict[str, str] = {"mos_pred": "quality", "noi_pred": "noisiness", "col_pred": "coloration", "dis_pred": "discontinuity", "loud_pred": "loudness"}
NATURAL: dict[str, str] = {"mos_pred": "natural"}
MEASURES: tuple[str, ...] = (*QUALITY.values(), *NATURAL.values())


@contextlib.contextmanager
def _whole_checkpoints() -> Iterator[None]:
    """NISQA's checkpoints carry their settings beside the weights, which `torch.load` no longer
    reads unless told to; they are the files of the repository this was pointed at."""
    load = torch.load
    torch.load = lambda *a, **k: load(*a, **{**k, "weights_only": False})  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = load  # type: ignore[assignment]


def score(wavs: list[Path], batch: int = 16, natural: bool = True) -> list[dict[str, float]]:
    """NISQA's predictions for each file: quality and its dimensions, and naturalness if asked."""
    import pandas as pd

    sys.path.insert(0, str(NISQA))
    from nisqa.NISQA_model import nisqaModel  # type: ignore[import-not-found]

    out: list[dict[str, float]] = [{} for _ in wavs]
    with tempfile.TemporaryDirectory() as tmp, _whole_checkpoints():
        pd.DataFrame({"wav": [str(p.resolve()) for p in wavs]}).to_csv(Path(tmp) / "files.csv", index=False)
        for weights, names in (("nisqa.tar", QUALITY), ("nisqa_tts.tar", NATURAL))[: 2 if natural else 1]:
            args = {"mode": "predict_csv", "pretrained_model": str(NISQA / "weights" / weights), "csv_file": "files.csv", "csv_deg": "wav", "data_dir": tmp,
                    "output_dir": None, "num_workers": 0, "bs": batch, "ms_channel": None, "tr_bs_val": batch, "tr_num_workers": 0}  # fmt: skip
            # it prints every row
            with contextlib.redirect_stdout(io.StringIO()):
                table = nisqaModel(args).predict()
            for theirs, ours in names.items():
                for row, value in zip(out, table[theirs].to_numpy()):
                    row[ours] = float(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "ja")
    ap.add_argument("--name", type=str, default="results")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    a = ap.parse_args()
    d: Path = a.dir
    rows: list[dict[str, Any]] = json.loads((d / f"{a.name}.json").read_text())["utterances"]
    todo = [(r["variants"][v], d / v / f"{r['id']}.wav") for r in rows for v in r["variants"] if "quality" not in r["variants"][v]]
    for (cell, _), scores in zip(todo, score([wav for _, wav in todo])):
        cell.update(scores)
    finish(d, a.name, rows, list(rows[0]["variants"]), a.jobs)


if __name__ == "__main__":
    main()
