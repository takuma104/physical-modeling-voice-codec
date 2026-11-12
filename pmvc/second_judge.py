"""A second judge for a results file: Whisper large-v3-turbo.

`pmvc.copysynth` optimises against one recogniser (wav2vec2) and `pmvc.evaluate` judges with
another (Whisper small.en). A gain that is real should show for a listener neither of them
resembles much: a far larger Whisper, which reads CHIMERA's renders at less than half the error
rate of the small one.

usage: python -m pmvc.second_judge [dir] [--name results]

Adds `heard2` and `errors2` to every variant of every recording in <dir>/<name>.json, and
`wer2` (errors capped at the sentence's length, with a bootstrap 95 % interval) to its summary.
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import torch

from .audio import load
from .evaluate import edits, wer_interval, words
from .paths import ROOT

MODEL: str = "openai/whisper-large-v3-turbo"


class BigWhisper:
    def __init__(self) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, logging

        logging.set_verbosity_error()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.float32 if self.device == "cpu" else torch.float16
            self.processor = WhisperProcessor.from_pretrained(MODEL)
            self.model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=self.dtype).to(self.device).eval()

    @torch.no_grad()
    def transcribe(self, wav: Path, language: str = "en") -> str:
        x, _ = load(wav, 16000)
        feats = self.processor(x, sampling_rate=16000, return_tensors="pt").input_features.to(self.device, self.dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.model.generate(feats, language=language, task="transcribe", num_beams=1, do_sample=False, max_new_tokens=96)
        return str(self.processor.batch_decode(out, skip_special_tokens=True)[0]).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "transplant")
    ap.add_argument("--name", type=str, default="results")
    a = ap.parse_args()
    d: Path = a.dir
    path = d / f"{a.name}.json"
    results: dict[str, Any] = json.loads(path.read_text())
    judge = BigWhisper()
    for row in results["utterances"]:
        ref = words(row["text"])
        for v, m in row["variants"].items():
            m["heard2"] = judge.transcribe(d / v / f"{row['id']}.wav")
            m["errors2"] = edits(ref, words(m["heard2"]))
    for group, pick in (("all", "FM"), ("female", "F"), ("male", "M")):
        sel = [r for r in results["utterances"] if r["sex"] in pick]
        for v in sel[0]["variants"] if sel else []:
            pairs = [(min(r["variants"][v]["errors2"], len(words(r["text"]))), len(words(r["text"]))) for r in sel]
            results["summary"][group][v]["wer2"] = [round(x, 2) for x in wer_interval(pairs)]
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    for v, s in results["summary"]["all"].items():
        if isinstance(s, dict) and "wer2" in s:
            small = f"{s['werCapped'][0]:5.1f} %" if "werCapped" in s else "    –"
            print(f"  {v:13s} small.en {small}   large-v3-turbo {s['wer2'][0]:5.1f} % [{s['wer2'][1]:.1f}, {s['wer2'][2]:.1f}]")


if __name__ == "__main__":
    main()
