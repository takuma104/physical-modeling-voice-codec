"""Which model should the listener be? Candidates compared on sentences two speakers read.

`pmvc.listener` hears with an English recogniser. For English that distance agrees with the
judges; for Japanese it does not: a search that halves it leaves Whisper and the phone
recogniser of `pmvc.judge` unmoved, and the phone recogniser finds renders of Japanese much
further from their recordings than another speaker is (41 % against 24 %; in English 8 % and
6 %). A listener fit to teach Japanese by should hear that too.

usage: python -m pmvc.yardstick [dir] [--variant encn] [--models MODEL:layer,layer;...] [--name yardstick]
                                [--jobs N]

For each candidate (a model on the Hugging Face hub and one of its layers) and each language
of <dir> (default out/ja, as `pmvc.judge` left it), over the sentences two speakers read:
  other     the distance (1 − cosine) between the two recordings, along their best time warp
  render    the distance between a recording and `--variant`'s render of it, the same way
  aligned   that distance frame against frame, as the closed loop and the search take it
  ratio     render / other: 1 if a render is as near as another person
  d'        (mean render − mean other) / the spread of both
  judges    rank correlations over <dir>/results.json's Japanese rows and renders, between the
            aligned distance and what the phone recogniser and Whisper large-v3-turbo found
  search    for the recordings of <dir>/results_search.json, the aligned distance before and
            after a search toward the default listener: what that search gained in this
            candidate's hearing
Writes <dir>/<name>.json and prints the table.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import spearmanr

from .audio import dtw, load
from .encoder import pick_device
from .listener import FEATURE_RATE, LISTEN_SR, MODEL, Listener, to16k
from .paths import ROOT

# the English recogniser, a HuBERT trained on 147 languages, two models trained on Japanese alone,
# and the multilingual wav2vec2 the phone recogniser of `pmvc.judge` was made from
CANDIDATES: str = (
    f"{MODEL}:6,9,12;utter-project/mHuBERT-147:6,8,9,10,12;sarulab-speech/hubert-base-jtube:6,8,9,10,12;"
    "reazon-research/japanese-wav2vec2-base:6,8,9,10,12;facebook/wav2vec2-large-xlsr-53:8,12,16,20"
)
# variants whose renders the judges' findings are set against
JUDGED: tuple[str, ...] = ("sim", "enc", "encn")

Feats = NDArray[np.float32]


@torch.no_grad()
def layers_of(listener: Listener, x16k: NDArray[np.floating], layers: list[int]) -> Feats:
    """Unit features of `layers` for one clip, (layers, frames, width)."""
    x = torch.as_tensor(x16k, dtype=torch.float32, device=listener.device)[None]
    if listener.normalise:
        x = (x - x.mean()) / (x.std() + 1e-7)
    states = listener.model(x, output_hidden_states=True).hidden_states
    return torch.stack([torch.nn.functional.normalize(states[k][0].float(), dim=-1) for k in layers]).cpu().numpy()


def warped(pair: tuple[Feats, Feats]) -> float:
    """1 − cosine between two sequences of unit features along their best time warp."""
    a, b = (x.astype(np.float64) for x in pair)
    _, ia, ib = dtw(a, b)
    return float(1.0 - (a[ia] * b[ib]).sum(axis=1).mean())


def aligned(a: Feats, b: Feats) -> float:
    n = min(len(a), len(b))
    return float(1.0 - (a[:n] * b[:n]).sum(axis=1).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "ja")
    ap.add_argument("--variant", type=str, default="encn")
    ap.add_argument("--models", type=str, default=CANDIDATES)
    ap.add_argument("--name", type=str, default="yardstick")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    d: Path = a.dir
    rows: list[dict[str, Any]] = json.loads((d / "results.json").read_text())["utterances"]
    by_id = {r["id"]: r for r in rows}
    spans = {r["id"]: json.loads((d / "meta" / f"{r['id']}.json").read_text())["span"] for r in rows}
    searched: list[str] = [r["id"] for r in json.loads((d / "results_search.json").read_text())["utterances"]] if (d / "results_search.json").exists() else []
    langs = sorted({r["lang"] for r in rows})
    device = pick_device()
    out: list[dict[str, Any]] = []
    for spec in a.models.split(";"):
        name, _, wanted = spec.rpartition(":")
        layers = [int(k) for k in wanted.split(",")]
        listener = Listener(device, name)

        def speech(uid: str, variant: str, listener: Listener = listener, layers: list[int] = layers) -> Feats:
            """A clip's features over the recording's speech; a render runs its track's shift behind."""
            s0, s1 = (int(t * FEATURE_RATE) for t in spans[uid])
            if variant == "real":
                return layers_of(listener, load(d / "real" / f"{uid}.wav", LISTEN_SR)[0], layers)[:, s0:s1]
            shift = json.loads((d / variant / f"{uid}.tracks.json").read_text())["shift"]
            return layers_of(listener, to16k(load(d / variant / f"{uid}.wav")[0][round(shift * 48000) :]), layers)[:, s0:s1]

        real = {r["id"]: speech(r["id"], "real") for r in rows}
        render = {v: {r["id"]: speech(r["id"], v) for r in rows if v == a.variant or r["lang"] == "ja"} for v in dict.fromkeys([a.variant, *JUDGED])}
        after = {uid: speech(uid, "encncopy") for uid in searched}
        del listener
        with ProcessPoolExecutor(a.jobs) as pool:
            for k, layer in enumerate(layers):
                row: dict[str, Any] = {"model": name, "layer": layer}
                for lang in langs:
                    pairs = [r for r in rows if r["lang"] == lang and "other" in r]
                    other = np.array(list(pool.map(warped, [(real[r["id"]][k], real[r["other"]["id"]][k]) for r in pairs], chunksize=8)))
                    rend = np.array(list(pool.map(warped, [(real[r["id"]][k], render[a.variant][r["id"]][k]) for r in pairs], chunksize=8)))
                    flat = [aligned(real[r["id"]][k], render[a.variant][r["id"]][k]) for r in pairs]
                    row[lang] = {
                        "n": len(pairs), "other": float(other.mean()), "render": float(rend.mean()), "aligned": float(np.mean(flat)),
                        "ratio": float(rend.mean() / other.mean()), "dprime": float((rend.mean() - other.mean()) / np.sqrt(0.5 * (rend.var() + other.var()))),
                    }  # fmt: skip
                # against the judges, over every Japanese recording and judged render
                dist, phones, whisper = [], [], []
                for v in JUDGED:
                    for uid, feats in render[v].items():
                        m = by_id[uid]["variants"][v]
                        dist.append(aligned(real[uid][k], feats[k]))
                        phones.append(m["phoneErrors"] / m["phoneUnits"])
                        whisper.append(min(m["errors2"], m["units"]) / m["units"])
                row["judges"] = {"phones": float(spearmanr(dist, phones)[0]), "whisper": float(spearmanr(dist, whisper)[0]), "n": len(dist)}
                if searched:
                    before = float(np.mean([aligned(real[uid][k], render[a.variant][uid][k]) for uid in searched]))
                    row["search"] = [before, float(np.mean([aligned(real[uid][k], after[uid][k]) for uid in searched]))]
                out.append(row)
                cols = "  ".join(f"{lang}: other {row[lang]['other']:.3f} render {row[lang]['render']:.3f} (aligned {row[lang]['aligned']:.3f}) ratio {row[lang]['ratio']:.2f} d' {row[lang]['dprime']:.2f}" for lang in langs)
                more = f"  judges: phones {row['judges']['phones']:.2f} whisper {row['judges']['whisper']:.2f}" + (f"  search {row['search'][0]:.3f} → {row['search'][1]:.3f}" if searched else "")
                print(f"{name.split('/')[-1]:28s} layer {layer:2d}  {cols}{more}", flush=True)
        (d / f"{a.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
