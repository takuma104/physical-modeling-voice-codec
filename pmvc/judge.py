"""Judge renders of recordings in any language Whisper reads and the sentences' text is known in.

usage: python -m pmvc.judge [dir] [--variants sim,enc,encn] [--split all] [--ids ...] [--name results]
                            [--again] [--jobs N]

`pmvc.evaluate` is tied to TSP (its alignment, its other speakers, an English-only Whisper).
This scores what `pmvc.recordings` brought in, and TSP's recordings the same way, so that the
numbers can be set side by side. For the recording and for every variant of it:
  rate       Whisper small's errors against the sentence (small.en for English), greedy, as a
             share of the sentence's units, at most one error per unit, with a bootstrap 95 %
             interval over recordings. The units are words in English. Japanese is written
             without spaces and one word can be spelt several ways, so there both the sentence
             and what Whisper wrote are turned into phonemes (Open JTalk's reading) first.
  rate2      the same by Whisper large-v3-turbo
For every variant, against the recording:
  phoneRate  a judge that knows no text and no single language: a phone recogniser
             (wav2vec2 XLSR-53, fine-tuned to write IPA for many languages) writes down the
             recording and the render, and this is the share of the recording's phones the
             render got wrong
  distance   `pmvc.listener`'s feature distance over the recording's speech, for variants with
             a track that says how far the render runs behind
  voicing    the share of the recording's speech frames where a pitch tracker calls the
             recording and the render both voiced or both unvoiced; `f0Corr`, the correlation
             of their log pitch where both are voiced
Whisper leans on what a sentence is likely to say; the phone recogniser cannot. Neither was
optimised against. Two more columns make the numbers comparable:
  other      where another speaker in <dir> read the same sentence, the phoneRate of that
             recording against this one: what the phone recogniser makes of a different person
  letterRate, letterRate2   for a language scored by word, the same errors counted in letters,
             a unit about as fine as a phoneme

Writes <dir>/<name>.json: the rows, their summary by speaker and by language, and under
`sounds`, for each language whose units are phonemes, how often each phoneme of the sentences
was lost (replaced or left out in the best alignment with what large-v3-turbo heard).
`--again` redoes the summary of a file that is already there, judging nothing.
"""

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from faster_whisper import WhisperModel

from .audio import FRAME_RATE, load, pitch_track
from .encoder import pick_device, split_of
from .evaluate import edits, wer_interval, words
from .listener import FEATURE_RATE, LISTEN_SR, Listener, distance, to16k
from .paths import ROOT, WHISPER_CACHE
from .second_judge import BigWhisper

PHONES: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
# languages scored by phoneme, not by word
BY_PHONEME: tuple[str, ...] = ("ja",)


def units(text: str, lang: str) -> list[str]:
    """What errors are counted in: words, or for Japanese the phonemes of Open JTalk's reading
    (a devoiced vowel is the vowel, pauses are left out)."""
    if lang not in BY_PHONEME:
        return words(text)
    import pyopenjtalk

    return [p.lower() if p in "AIUEO" else p for p in str(pyopenjtalk.g2p(text)).split() if p != "pau"]


def letters(text: str) -> list[str]:
    return list("".join(words(text)))


def lost(ref: list[str], hyp: list[str]) -> list[bool]:
    """For each unit of `ref`, whether the best alignment with `hyp` replaces it or leaves it out."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    out = [True] * n
    i, j = n, m
    while i > 0 and j > 0:
        if d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            out[i - 1] = ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


class PhoneReader:
    """The phone recogniser: IPA symbols for a clip, with no text and no dictionary behind them."""

    def __init__(self, device: str) -> None:
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2ForCTC, logging

        logging.set_verbosity_error()
        self.device = device
        self.model = Wav2Vec2ForCTC.from_pretrained(PHONES).to(device).eval()  # type: ignore[arg-type]
        vocab: dict[str, int] = json.loads(Path(hf_hub_download(PHONES, "vocab.json")).read_text())
        self.symbols = {i: s for s, i in vocab.items() if not s.startswith("<")}

    @torch.no_grad()
    def read(self, x16k: np.ndarray[Any, Any]) -> list[str]:
        x = torch.as_tensor(x16k, dtype=torch.float32, device=self.device)[None]
        ids = self.model((x - x.mean()) / (x.std() + 1e-7)).logits[0].argmax(dim=-1).tolist()
        return [self.symbols[i] for k, i in enumerate(ids) if i in self.symbols and (k == 0 or i != ids[k - 1])]


_SMALL: dict[str, WhisperModel] = {}


def _small(job: tuple[Path, str]) -> str:
    """Whisper small's transcript, greedy: small.en for English, as `pmvc.evaluate` judges."""
    wav, lang = job
    name = "small.en" if lang == "en" else "small"
    if name not in _SMALL:
        _SMALL[name] = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=2, download_root=str(WHISPER_CACHE))
    x, _ = load(wav, 16000)
    segs, _ = _SMALL[name].transcribe(x.astype(np.float32), language=lang, beam_size=1, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
    return " ".join(s.text.strip() for s in segs)


def pitch_agreement(job: tuple[Path, str, str, str]) -> dict[str, float]:
    """`voicing` and `f0Corr` of a variant's render against the recording, over its speech."""
    d, uid, v, sex = job
    meta = json.loads((d / "meta" / f"{uid}.json").read_text())
    shift = round(json.loads((d / v / f"{uid}.tracks.json").read_text())["shift"] * FRAME_RATE)
    f_ref = np.load(d / "pitch" / f"{uid}.npy")
    f_syn = pitch_track(*load(d / v / f"{uid}.wav"), sex)
    r0, r1 = round(meta["span"][0] * FRAME_RATE), min(round(meta["span"][1] * FRAME_RATE), len(f_ref), len(f_syn) - shift)
    fr, fs = f_ref[r0:r1], f_syn[r0 + shift : r1 + shift]
    both = (fr > 0) & (fs > 0)
    out = {"voicing": float(((fr > 0) == (fs > 0)).mean())}
    if both.sum() > 10:
        out["f0Corr"] = float(np.corrcoef(np.log2(fr[both]), np.log2(fs[both]))[0, 1])
    return out


def compare(d: Path, rows: list[dict[str, Any]], names: list[str], jobs: int) -> None:
    """Adds to the rows what no judge is needed for: the pitch tracker's agreement, another
    speaker's recording of the sentence against this one by the phone recogniser's two
    transcripts, and the errors in letters."""
    todo = [(d, r["id"], v, r["sex"]) for r in rows for v in names if "distance" in r["variants"][v] and "voicing" not in r["variants"][v]]
    with ThreadPoolExecutor(jobs) as pool:
        for (_, uid, v, _), m in zip(todo, pool.map(pitch_agreement, todo)):
            next(r for r in rows if r["id"] == uid)["variants"][v].update(m)
    same: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        same.setdefault((r["lang"], r["text"]), []).append(r)
    for r in rows:
        said = r["variants"]["real"]["phones"].split()
        others = [o for o in same[(r["lang"], r["text"])] if o["speaker"] != r["speaker"]]
        if others:
            r["other"] = {"id": others[0]["id"], "phoneErrors": edits(said, others[0]["variants"]["real"]["phones"].split()), "phoneUnits": len(said)}
        if r["lang"] not in BY_PHONEME:
            ref = letters(r["text"])
            for v in names:
                m = r["variants"][v]
                m["letters"], m["letterErrors"], m["letterErrors2"] = len(ref), edits(ref, letters(m["heard"])), edits(ref, letters(m["heard2"]))


def summarise(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        for g in (r["speaker"], r["lang"]):
            groups.setdefault(g, []).append(r)
    summary: dict[str, Any] = {}
    for g, sel in groups.items():
        summary[g] = {"n": len(sel), "lang": sel[0]["lang"]}
        for v in names:
            ms = [r["variants"][v] for r in sel]
            s: dict[str, Any] = {}
            for key, errs, of in (("rate", "errors", "units"), ("rate2", "errors2", "units"), ("phoneRate", "phoneErrors", "phoneUnits"),
                                  ("letterRate", "letterErrors", "letters"), ("letterRate2", "letterErrors2", "letters")):  # fmt: skip
                if all(errs in m for m in ms):
                    s[key] = [round(x, 2) for x in wer_interval([(min(m[errs], m[of]), m[of]) for m in ms])]
            for key in ("distance", "voicing", "f0Corr"):
                if all("distance" in m for m in ms) and any(key in m for m in ms):
                    s[key] = round(float(np.mean([m[key] for m in ms if key in m])), 3)
            summary[g][v] = s
        others = [r["other"] for r in sel if "other" in r]
        if others:
            summary[g]["other"] = {"n": len(others), "phoneRate": [round(x, 2) for x in wer_interval([(min(o["phoneErrors"], o["phoneUnits"]), o["phoneUnits"]) for o in others])]}
    return summary


def sounds(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    """By language and variant: {phoneme: [times lost, times in the sentences]}."""
    out: dict[str, Any] = {}
    for r in rows:
        if r["lang"] in BY_PHONEME:
            ref = units(r["text"], r["lang"])
            for v in names:
                table = out.setdefault(r["lang"], {}).setdefault(v, {})
                for p, gone in zip(ref, lost(ref, units(r["variants"][v]["heard2"], r["lang"]))):
                    cell = table.setdefault(p, [0, 0])
                    cell[0] += int(gone)
                    cell[1] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "ja")
    ap.add_argument("--variants", type=str, default="sim,enc,encn")
    ap.add_argument("--split", choices=["test", "valid", "train", "all"], default="all")
    ap.add_argument("--ids", type=str, default="", help="only these recordings")
    ap.add_argument("--name", type=str, default="results")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--again", action="store_true", help="redo the summary of <name>.json, judging nothing")
    a = ap.parse_args()
    d: Path = a.dir
    if a.again:
        judged: list[dict[str, Any]] = json.loads((d / f"{a.name}.json").read_text())["utterances"]
        finish(d, a.name, judged, list(judged[0]["variants"]), a.jobs)
        return
    variants: list[str] = a.variants.split(",")
    names = ["real", *variants]
    metas = [json.loads(p.read_text()) for p in sorted((d / "meta").glob("*.json")) if a.split in ("all", split_of(p.stem))]
    metas = [m for m in metas if not a.ids or m["id"] in a.ids.split(",")]
    metas = [m for m in metas if all((d / v / f"{m['id']}.wav").exists() for v in names)]
    rows: list[dict[str, Any]] = [{"id": m["id"], "speaker": m["speaker"], "sex": m["sex"], "lang": m.get("lang", "en"), "text": m["text"], "variants": {v: {} for v in names}} for m in metas]

    jobs = [(d / v / f"{r['id']}.wav", r["lang"]) for r in rows for v in names]
    cells = [r["variants"][v] for r in rows for v in names]
    refs = [units(r["text"], r["lang"]) for r in rows for _ in names]
    with ProcessPoolExecutor(a.jobs) as pool:
        for cell, ref, (_, lang), heard in zip(cells, refs, jobs, pool.map(_small, jobs, chunksize=4)):
            cell["heard"], cell["errors"], cell["units"] = heard, edits(ref, units(heard, lang)), len(ref)

    device = pick_device()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        big, reader, listener = BigWhisper(), PhoneReader(device), Listener(device)
    for r, meta in zip(rows, metas):
        ref = units(r["text"], r["lang"])
        real, _ = load(d / "real" / f"{r['id']}.wav", LISTEN_SR)
        said = reader.read(real)
        feats = listener.hear(real).feats[0]
        a0, a1 = int(meta["span"][0] * FEATURE_RATE), int(meta["span"][1] * FEATURE_RATE)
        for v in names:
            cell, wav = r["variants"][v], d / v / f"{r['id']}.wav"
            cell["heard2"] = big.transcribe(wav, r["lang"])
            cell["errors2"] = edits(ref, units(cell["heard2"], r["lang"]))
            if v == "real":
                cell["phones"] = " ".join(said)
                continue
            x, _ = load(wav, LISTEN_SR)
            cell["phones"] = " ".join(written := reader.read(x))
            cell["phoneErrors"], cell["phoneUnits"] = edits(said, written), len(said)
            track = d / v / f"{r['id']}.tracks.json"
            if track.exists() and "shift" in (t := json.loads(track.read_text())):
                x48, _ = load(wav)
                behind = listener.hear(to16k(x48[round(t["shift"] * 48000) :])).feats[0]
                cell["distance"] = float(distance(feats[a0:a1], behind[a0:a1]).mean())

    finish(d, a.name, rows, names, a.jobs)


def finish(d: Path, name: str, rows: list[dict[str, Any]], names: list[str], jobs: int) -> None:
    compare(d, rows, names, jobs)
    summary = summarise(rows, names)
    (d / f"{name}.json").write_text(json.dumps({"summary": summary, "sounds": sounds(rows, names), "utterances": rows}, ensure_ascii=False, indent=1))
    for g, s in summary.items():
        print(f"\n{g} ({s['n']} recordings, {'phonemes' if s['lang'] in BY_PHONEME else 'words'})")
        for v in [*names, "other"]:
            if v in s:
                m = s[v]
                cols = [f"{label} {m[key][0]:5.1f} % [{m[key][1]:.1f}, {m[key][2]:.1f}]" for key, label in (("rate", "small"), ("rate2", "large-v3-turbo"), ("letterRate2", "letters"), ("phoneRate", "phones")) if key in m]
                rest = "".join(f"   {key} {m[key]:.3f}" for key in ("distance", "voicing", "f0Corr") if key in m)
                print(f"  {v:8s} " + "   ".join(cols) + rest)


if __name__ == "__main__":
    main()
