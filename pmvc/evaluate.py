"""Score a directory of renders against the recordings they copy.

usage: python -m pmvc.evaluate [dir] [--variants rule,dur,durf0] [--name results] [--no-asr]

`dir` (default out/transplant) holds real/<id>.wav, meta/<id>.json, f0/<id>.npy and one
directory of <id>.wav per variant, as `pmvc.transplant` and `pmvc.perturb` write them. Scores
the recordings that every variant has, writes <dir>/<name>.json and prints the summary.

Measures, each against the real recording of the same sentence:
  wer        Whisper's word error rate (small.en, greedy: the settings of CHIMERA's own speech
             judge, tools/asr/speech_wer.py), with a bootstrap 95 % interval over recordings
  mcd        mel-cepstral distortion (dB) along the best time warp, over each signal's speech,
             each signal's mean cepstrum removed
  other      the same distortion for another real speaker reading the sentence (of the same sex
             where the database has one): what "as close as a different person" is on this scale
For variants on the recording's timeline (all but `rule`):
  mcdLinear  the distortion frame against frame, with no warp
  timing     how far the best warp strays from frame-against-frame (ms, mean)
  f0Corr     correlation of log f0 where both are voiced; f0Rmse in semitones
  voicing    share of speech frames where both agree on voiced/unvoiced
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from faster_whisper import WhisperModel

from .audio import FRAME_RATE, MCD_SCALE, MEL_SR, Array, cepstra, dtw, load, log_mel, pitch_track, speech_span
from .paths import ROOT, WHISPER_CACHE
from .tsp import Utterance, load_utterances


def words(text: str) -> list[str]:
    text = re.sub(r"[^a-z' ]+", " ", text.lower().replace("’", "'"))
    return [w.strip("'") for w in text.split() if w.strip("'")]


def edits(ref: list[str], hyp: list[str]) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)]


def load_asr() -> WhisperModel:
    return WhisperModel("small.en", device="cpu", compute_type="int8", download_root=str(WHISPER_CACHE))


def transcribe(model: WhisperModel, wav: Path) -> str:
    x, _ = load(wav, 16000)
    segs, _ = model.transcribe(
        x.astype(np.float32), language="en", beam_size=1, temperature=0.0,
        condition_on_previous_text=False, vad_filter=False,
    )  # fmt: skip
    return " ".join(s.text.strip() for s in segs)


def speech_cepstra(wav: Path) -> tuple[Array, tuple[int, int]]:
    """Mel cepstra of the whole file, less their mean over its speech, and the speech's span in
    frames. Without the mean (a voice's own long-term spectrum, and the microphone's) two real
    speakers saying one sentence are as far apart as a render is from either."""
    x, _ = load(wav, MEL_SR)
    mel = log_mel(x)
    c, (a, b) = cepstra(mel), speech_span(mel)
    return c - c[a:b].mean(axis=0), (a, b)


def mcd(a: tuple[Array, tuple[int, int]], b: tuple[Array, tuple[int, int]]) -> float:
    (ca, (a0, a1)), (cb, (b0, b1)) = a, b
    return MCD_SCALE * dtw(ca[a0:a1], cb[b0:b1])[0]


def other_speaker(utt: Utterance, utts: dict[str, Utterance]) -> Utterance | None:
    """Another speaker who read this sentence (ids are <speaker><list>_<n>), of the same sex if
    there is one."""
    same = [u for u in utts.values() if u.id[2:] == utt.id[2:] and u.speaker != utt.speaker and u.wav.exists()]
    same.sort(key=lambda u: (u.sex != utt.sex, u.id))
    return same[0] if same else None


def aligned_measures(d: Path, meta: dict[str, Any], variant: str, real: Array, syn: Array) -> dict[str, float]:
    """Measures for a render on the recording's timeline (`shift` seconds later)."""
    s = round(meta["shift"] * FRAME_RATE)
    r0, r1 = (round(t * FRAME_RATE) for t in meta["span"])
    r1 = min(r1, len(real), len(syn) - s)
    a, b = real[r0:r1], syn[r0 + s : r1 + s]
    _, ia, ib = dtw(a, b)
    out = {
        "mcdLinear": float(MCD_SCALE * np.sqrt(((a - b) ** 2).sum(axis=1)).mean()),
        "timing": float(np.abs(ia - ib).mean() * 1000.0 / FRAME_RATE),
    }
    f_real = np.load(d / "f0" / f"{meta['id']}.npy")
    x, sr = load(d / variant / f"{meta['id']}.wav")
    f_syn = pitch_track(x, sr, meta["sex"])
    n = min(r1, len(f_real), len(f_syn) - s)
    fr, fs = f_real[r0:n], f_syn[r0 + s : n + s]
    both = (fr > 0) & (fs > 0)
    out["voicing"] = float(((fr > 0) == (fs > 0)).mean())
    if both.sum() > 10:
        lr, ls = np.log2(fr[both]), np.log2(fs[both])
        out["f0Corr"] = float(np.corrcoef(lr, ls)[0, 1])
        out["f0Rmse"] = float(12.0 * np.sqrt(((lr - ls) ** 2).mean()))
    return out


def wer_interval(rows: list[tuple[int, int]], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Word error rate (%) over (errors, words) rows, with a bootstrap 95 % interval."""
    e, w = np.array(rows, dtype=np.float64).T
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e), size=(n, len(e)))
    boot = 100.0 * e[idx].sum(axis=1) / w[idx].sum(axis=1)
    return 100.0 * e.sum() / w.sum(), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "transplant")
    ap.add_argument("--variants", type=str, default="rule,dur,durf0")
    ap.add_argument("--name", type=str, default="results")
    ap.add_argument("--no-asr", action="store_true")
    a = ap.parse_args()
    d: Path = a.dir
    variants: list[str] = a.variants.split(",")
    metas = [json.loads(p.read_text()) for p in sorted((d / "meta").glob("*.json"))]
    metas = [m for m in metas if all((d / v / f"{m['id']}.wav").exists() for v in variants)]
    utts = load_utterances()
    model = None if a.no_asr else load_asr()

    rows: list[dict[str, Any]] = []
    for meta in metas:
        uid = meta["id"]
        ref = words(meta["text"])
        real = speech_cepstra(d / "real" / f"{uid}.wav")
        row: dict[str, Any] = {"id": uid, "sex": meta["sex"], "text": meta["text"], "variants": {}}
        for v in ["real", *variants]:
            wav = d / v / f"{uid}.wav"
            m: dict[str, Any] = {}
            if model is not None:
                m["heard"] = transcribe(model, wav)
                m["errors"], m["words"] = edits(ref, words(m["heard"])), len(ref)
            if v != "real":
                syn = speech_cepstra(wav)
                m["mcd"] = mcd(real, syn)
                if v != "rule":
                    m.update(aligned_measures(d, meta, v, real[0], syn[0]))
            row["variants"][v] = m
        other = other_speaker(utts[uid], utts)
        if other is not None:
            row["other"] = {"id": other.id, "sameSex": other.sex == meta["sex"], "mcd": mcd(real, speech_cepstra(other.wav))}
        rows.append(row)

    summary: dict[str, Any] = {}
    for group, pick in (("all", "FM"), ("female", "F"), ("male", "M")):
        sel = [r for r in rows if r["sex"] in pick]
        g: dict[str, Any] = {"n": len(sel)}
        for v in ["real", *variants]:
            ms = [r["variants"][v] for r in sel]
            s: dict[str, Any] = {}
            if model is not None:
                s["wer"] = [round(x, 2) for x in wer_interval([(m["errors"], m["words"]) for m in ms])]
            for key in ("mcd", "mcdLinear", "timing", "voicing", "f0Corr", "f0Rmse"):
                vals = [m[key] for m in ms if key in m]
                if vals:
                    s[key] = round(float(np.mean(vals)), 3)
            g[v] = s
        others = [r["other"]["mcd"] for r in sel if "other" in r]
        if others:
            g["other"] = {"mcd": round(float(np.mean(others)), 3), "n": len(others)}
        summary[group] = g
    (d / f"{a.name}.json").write_text(json.dumps({"summary": summary, "utterances": rows}, ensure_ascii=False, indent=1))

    for group, g in summary.items():
        print(f"\n{group} ({g['n']} recordings)")
        for v in ["real", *variants, "other"]:
            if v in g:
                s = g[v]
                wer = f"WER {s['wer'][0]:5.1f} % [{s['wer'][1]:.1f}, {s['wer'][2]:.1f}]  " if "wer" in s else ""
                rest = "  ".join(f"{k} {x}" for k, x in s.items() if k not in ("wer", "n"))
                print(f"  {v:8s} {wer}{rest}")


if __name__ == "__main__":
    main()
