"""Whose voice is a render's? What of a speaker survives the codec.

The voice is told two things of a speaker, the sex and the median pitch; whatever else makes
people sound different has to be in the control track, or it is lost. A speaker-verification
model (WavLM with an x-vector head, trained on VoxCeleb; it has heard no Japanese and none of
this) says how much is left: it turns a clip into a vector that is close for clips of one
speaker.

usage: python -m pmvc.whose [dir] [--variants enci,tts] [--split test] [--name whose] [--per 25]

For the recordings of a split and each variant's renders of them, with every speaker's
recordings standing for the speaker (the mean of their vectors; a clip's own sentence is left
out of its own speaker's mean, so that the words do not help):
  own      the cosine between a clip's vector and its speaker's
  other    the same against the other speakers of the same sex, averaged
  top1     the share of clips whose nearest speaker is their own, among all the speakers
  top1sex  the same among the speakers of the same sex
For the recordings themselves these say what the model can tell at best. Writes
<dir>/<name>.json. `--per` takes so many recordings of each speaker.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .audio import load
from .encoder import ids_in, pick_device
from .paths import ROOT

MODEL: str = "microsoft/wavlm-base-plus-sv"
SR: int = 16000


class Verifier:
    def __init__(self, device: str) -> None:
        from transformers import WavLMForXVector, logging

        logging.set_verbosity_error()
        self.device = device
        self.model = WavLMForXVector.from_pretrained(MODEL).to(device).eval()  # type: ignore[arg-type]

    @torch.no_grad()
    def vector(self, x: NDArray[np.floating]) -> NDArray[np.float32]:
        """A clip (16 kHz) as a unit vector."""
        a = torch.as_tensor(x, dtype=torch.float32, device=self.device)[None]
        e = self.model((a - a.mean()) / (a.std() + 1e-7)).embeddings[0]
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


def measure(vectors: NDArray[np.float32], real: NDArray[np.float32], speakers: list[str], sexes: list[str], sentences: list[str]) -> dict[str, float]:
    """`vectors` (clips, width) against the speakers that `real` (the recordings' vectors, one
    for each clip, in the same order) stands for."""
    names = sorted(set(speakers))
    sex = {s: x for s, x in zip(speakers, sexes)}
    who = np.array([names.index(s) for s in speakers])
    own, other, top1, top1sex = [], [], [], []
    for i, v in enumerate(vectors):
        # every speaker's mean without this clip's sentence
        keep = np.array([t != sentences[i] for t in sentences])
        means = np.stack([real[(who == k) & keep].mean(axis=0) for k in range(len(names))])
        cos = means @ v / (np.linalg.norm(means, axis=1) + 1e-9)
        same = np.array([sex[n] == sexes[i] for n in names])
        rivals = same & (np.arange(len(names)) != who[i])
        own.append(cos[who[i]])
        other.append(cos[rivals].mean())
        top1.append(int(cos.argmax()) == who[i])
        top1sex.append(int(np.where(same, cos, -np.inf).argmax()) == who[i])
    return {"own": float(np.mean(own)), "other": float(np.mean(other)), "top1": float(np.mean(top1)), "top1sex": float(np.mean(top1sex))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "jvs")
    ap.add_argument("--variants", type=str, default="enci,tts")
    ap.add_argument("--split", choices=["test", "valid", "train", "all"], default="test")
    ap.add_argument("--name", type=str, default="whose")
    ap.add_argument("--per", type=int, default=25, help="recordings of each speaker")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    d: Path = a.dir
    variants = a.variants.split(",")
    metas = [json.loads((d / "meta" / f"{i}.json").read_text()) for i in ids_in(d, a.split)]
    metas = [m for m in metas if all((d / v / f"{m['id']}.wav").exists() for v in ("real", *variants))]
    count: dict[str, int] = {}
    kept = []
    for m in metas:
        count[m["speaker"]] = count.get(m["speaker"], 0) + 1
        if count[m["speaker"]] <= a.per:
            kept.append(m)
    speakers, sexes, sentences = [m["speaker"] for m in kept], [m["sex"] for m in kept], [m["text"] for m in kept]
    verifier = Verifier(pick_device())
    vectors: dict[str, NDArray[np.float32]] = {}
    with ThreadPoolExecutor(a.jobs) as pool:
        for v in ("real", *variants):
            clips = pool.map(lambda m, v=v: load(d / v / f"{m['id']}.wav", SR)[0], kept)
            vectors[v] = np.stack([verifier.vector(x) for x in clips])
    out: dict[str, Any] = {"model": MODEL, "speakers": len(set(speakers)), "clips": len(kept), "variants": {}}
    print(f"{len(kept)} recordings of {len(set(speakers))} speakers ({sum(s == 'F' for s in {n: x for n, x in zip(speakers, sexes)}.values())} women)")
    for v in ("real", *variants):
        m = measure(vectors[v], vectors["real"], speakers, sexes, sentences)
        out["variants"][v] = m
        print(f"  {v:8s} own {m['own']:.3f}   other speakers of the sex {m['other']:.3f}   nearest is the speaker {100 * m['top1']:.1f} %   among the sex {100 * m['top1sex']:.1f} %")
    (d / f"{a.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
