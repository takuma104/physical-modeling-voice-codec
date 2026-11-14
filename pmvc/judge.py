"""Judge renders of recordings in any language Whisper reads and the sentences' text is known in.

usage: python -m pmvc.judge [dir] [--variants sim,enc,encn] [--split all] [--ids ...] [--name results]
                            [--again] [--add] [--jobs N]

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
  stray      the share of recordings whose render is voiced for 30 ms or more where the
             recording is silent, before or after its speech (60 ms either side of the speech
             are left out); `outside`, the render's loudest 10 ms there against its level over
             the speech (dB, the median over recordings)
For the recording and every variant:
  clicks     `pmvc.clicks`: short bursts in the high band, how many a second; `clickLevel`,
             their mean level against the speech's (dB), and `clickLoad`, their energy a second;
             `releases` and `releaseLevel`, those where the recording goes from silence to
             sound, as a stop's release does; `oddClicks`, those of the kinds recordings hardly
             have (as the tongue closes, or in the middle of sound)
  silenceSounds  `pmvc.clicks.in_silences`: how many times a second the clip sounds where the
             recording is silent inside its speech (a stop's closure, a pause), and
             `silenceLevel`, its level there against its speech's (dB)
  extra4k    `pmvc.clicks.extra`: how many sounds a second the clip has that the recording has
             not, between 4 and 16 kHz (`extra80`, `extra300`, `extra2k` for 80-300 Hz,
             0.3-2 kHz and 2-4 kHz), and `closureClicks`, those of them that are clicks as the
             recording goes silent or in its silence, with their mean level `closureClickLevel`
A render whose timing is its own (`pmvc.tts speak`: its track says `free`) cannot be held
against the recording frame by frame. It gets rate, rate2 and phoneRate; distance, voicing,
stray and what is counted against the recording are left out. So that its clicks can still be
compared, a results file with such a render has for every clip
  ownReleases, ownReleaseLevel, ownClicks, ownClickLevel, ownOddClicks, ownClickLoad
             `pmvc.clicks.measure` over the clip's own speech, the clip standing for its own
             recording: the terms a recording's clicks are counted on
Where nothing was recorded at all (<dir> has no `real`: sentences `pmvc.tts speak --sentences`
spoke), only the text is there to go by: rate and rate2, and those clicks.
Whisper leans on what a sentence is likely to say; the phone recogniser cannot. Neither was
optimised against. Two more columns make the numbers comparable:
  other      where another speaker in <dir> read the same sentence (one of the same sex, if
             there is one), the phoneRate of that recording against this one: what the phone
             recogniser makes of a different person
  letterRate, letterRate2   for a language scored by word, the same errors counted in letters,
             a unit about as fine as a phoneme

Writes <dir>/<name>.json: the rows, their summary by speaker, by language and by language and
sex (`ja-F`), and under
`sounds`, for each language whose units are phonemes, how often each phoneme of the sentences
was lost (replaced or left out in the best alignment with what large-v3-turbo heard).
`--again` redoes the summary of a file that is already there, judging nothing; `--add` judges
only the variants the file does not have yet (and drops the ones not asked for).
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

from .audio import FRAME_RATE, MEL_SR, load, pitch_track, speech_span
from .clicks import BANDS, extra, in_silences, measure
from .encoder import ids_in, pick_device
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


# frames next to the recording's speech that are not counted as outside it: the voice's own
# run-up and decay
EDGE: int = 6


def pitch_agreement(job: tuple[Path, str, str, str]) -> dict[str, float]:
    """`voicing` and `f0Corr` of a variant's render against the recording, over its speech; and
    what the render does where the recording is silent: `strayVoiced`, its voiced frames before
    and after the speech, and `outside`, its loudest 10 ms there against its level over the
    speech (dB)."""
    d, uid, v, sex = job
    meta = json.loads((d / "meta" / f"{uid}.json").read_text())
    shift = round(json.loads((d / v / f"{uid}.tracks.json").read_text())["shift"] * FRAME_RATE)
    f_ref = np.load(d / "pitch" / f"{uid}.npy")
    x, sr = load(d / v / f"{uid}.wav")
    f_syn = pitch_track(x, sr, sex)
    r0, r1 = round(meta["span"][0] * FRAME_RATE), min(round(meta["span"][1] * FRAME_RATE), len(f_ref), len(f_syn) - shift)
    fr, fs = f_ref[r0:r1], f_syn[r0 + shift : r1 + shift]
    both = (fr > 0) & (fs > 0)
    out = {"voicing": float(((fr > 0) == (fs > 0)).mean())}
    if both.sum() > 10:
        out["f0Corr"] = float(np.corrcoef(np.log2(fr[both]), np.log2(fs[both]))[0, 1])
    hop = sr // FRAME_RATE
    power = (x[: len(x) // hop * hop].reshape(-1, hop) ** 2).mean(axis=1) + 1e-12
    k0, k1 = max(0, r0 + shift - EDGE), r1 + shift + EDGE
    away = np.concatenate([power[:k0], power[k1:]])
    out["strayVoiced"] = float((f_syn[:k0] > 0).sum() + (f_syn[k1 : len(power)] > 0).sum())
    out["outside"] = float(10.0 * np.log10(away.max() / power[r0 + shift : r1 + shift].mean())) if len(away) else -99.0
    return out


def clicks_of(job: tuple[Path, str, str]) -> dict[str, float]:
    """`pmvc.clicks`' measures for a clip, over the recording's speech (a render runs its
    track's shift behind, or the meta's). A render with a timing of its own (`pmvc.tts`: its
    track says `free`) has only `own_clicks`."""
    d, uid, v = job
    meta = json.loads((d / "meta" / f"{uid}.json").read_text())
    path = d / v / f"{uid}.tracks.json"
    track = json.loads(path.read_text()) if v != "real" and path.exists() else {}
    if track.get("free"):
        return {}
    shift = 0.0 if v == "real" else track["shift"] if "shift" in track else meta["shift"]
    x, sr = load(d / v / f"{uid}.wav")
    y = x[round(shift * sr) :]
    span, recording = (meta["span"][0], meta["span"][1]), None if v == "real" else load(d / "real" / f"{uid}.wav")
    return {**measure(y, sr, span, recording), **in_silences(y, sr, span, recording), **extra(y, sr, span, recording)}


def own_clicks(job: tuple[Path, str, str]) -> dict[str, float]:
    """A clip's clicks counted as a recording's are: over its own speech, and what kind each
    is going by the clip itself (`ownClicks`, `ownReleases`, ...). `clicks_of` holds a render
    against its recording, which a render with a timing of its own cannot be; these can be
    compared across all of them, and with the recording's."""
    d, uid, v = job
    path = d / v / f"{uid}.tracks.json"
    shift = json.loads(path.read_text()).get("shift", 0.0) if v != "real" and path.exists() else 0.0
    x, sr = load(d / v / f"{uid}.wav")
    y = x[round(shift * sr) :]
    a, b = speech_span(load(d / v / f"{uid}.wav", MEL_SR)[0][round(shift * MEL_SR) :])
    m = measure(y, sr, (a / FRAME_RATE, min(b / FRAME_RATE, len(y) / sr)), None)
    return {f"own{k[0].upper()}{k[1:]}": value for k, value in m.items()}


def compare(d: Path, rows: list[dict[str, Any]], names: list[str], jobs: int) -> None:
    """Adds to the rows what no judge is needed for: the pitch tracker's agreement, another
    speaker's recording of the sentence against this one by the phone recogniser's two
    transcripts, and the errors in letters."""
    todo = [(d, r["id"], v, r["sex"]) for r in rows for v in names if "distance" in r["variants"][v] and "outside" not in r["variants"][v]]
    by_id = {r["id"]: r for r in rows}
    with ThreadPoolExecutor(jobs) as pool:
        for (_, uid, v, _), m in zip(todo, pool.map(pitch_agreement, todo)):
            by_id[uid]["variants"][v].update(m)
        clicking = [(d, r["id"], v) for r in rows for v in names if "closureClicks" not in (m := r["variants"][v]) and not m.get("free") and "real" in r["variants"]]
        for (_, uid, v), m in zip(clicking, pool.map(clicks_of, clicking)):
            by_id[uid]["variants"][v].update(m)
        # where some render has a timing of its own, every clip's clicks are also counted on its own terms
        if any(r["variants"][v].get("free") for r in rows for v in names):
            owning = [(d, r["id"], v) for r in rows for v in names if "ownClicks" not in r["variants"][v]]
            for (_, uid, v), m in zip(owning, pool.map(own_clicks, owning)):
                by_id[uid]["variants"][v].update(m)
    same: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        same.setdefault((r["lang"], r["text"]), []).append(r)
    for r in rows:
        if "real" not in r["variants"]:
            continue
        said = r["variants"]["real"]["phones"].split()
        # of the same sex, where there is one
        others = sorted((o for o in same[(r["lang"], r["text"])] if o["speaker"] != r["speaker"]), key=lambda o: o["sex"] != r["sex"])
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
        for g in (r["speaker"], r["lang"], f"{r['lang']}-{r['sex']}"):
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
            if all("outside" in m for m in ms):
                # the share of recordings whose render is voiced for 30 ms or more outside the speech
                s["stray"] = round(100.0 * float(np.mean([m["strayVoiced"] >= 3 for m in ms])), 1)
                s["outside"] = round(float(np.median([m["outside"] for m in ms])), 1)
            if all("oddClicks" in m for m in ms):
                for key in ("clicks", "releases", "oddClicks"):
                    s[key] = round(float(np.mean([m[key] for m in ms])), 2)
                s["clickLoad"] = round(float(np.mean([m["clickLoad"] for m in ms])), 1)
                for key in ("clickLevel", "releaseLevel"):
                    if any(key in m for m in ms):
                        s[key] = round(float(np.mean([m[key] for m in ms if key in m])), 1)
            if all("ownClicks" in m for m in ms):
                for key in ("ownClicks", "ownReleases", "ownOddClicks"):
                    s[key] = round(float(np.mean([m[key] for m in ms])), 2)
                s["ownClickLoad"] = round(float(np.mean([m["ownClickLoad"] for m in ms])), 1)
                for key in ("ownClickLevel", "ownReleaseLevel"):
                    if any(key in m for m in ms):
                        s[key] = round(float(np.mean([m[key] for m in ms if key in m])), 1)
            if all("silenceSounds" in m for m in ms):
                s["silenceSounds"] = round(float(np.mean([m["silenceSounds"] for m in ms])), 2)
                # over all the silence of the group's recordings, each second counting the same
                seconds = sum(m["silence"] for m in ms)
                if seconds > 0.0:
                    s["silenceLevel"] = round(10.0 * float(np.log10(sum(m["silence"] * 10.0 ** (m["silenceLevel"] / 10.0) for m in ms if "silenceLevel" in m) / seconds)), 1)
            if all("closureClicks" in m for m in ms):
                for key in ("closureClicks", *(f"extra{name}" for name in BANDS)):
                    s[key] = round(float(np.mean([m[key] for m in ms])), 2)
                if any("closureClickLevel" in m for m in ms):
                    s["closureClickLevel"] = round(float(np.mean([m["closureClickLevel"] for m in ms if "closureClickLevel" in m])), 1)
            # what `pmvc.natural` added, if it was run
            for key in ("quality", "noisiness", "coloration", "discontinuity", "loudness", "natural"):
                if all(key in m for m in ms):
                    s[key] = round(float(np.mean([m[key] for m in ms])), 2)
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
    ap.add_argument("--add", action="store_true", help="keep what <name>.json already has of these variants and judge the rest")
    a = ap.parse_args()
    d: Path = a.dir
    if a.again:
        judged: list[dict[str, Any]] = json.loads((d / f"{a.name}.json").read_text())["utterances"]
        finish(d, a.name, judged, list(judged[0]["variants"]), a.jobs)
        return
    variants: list[str] = a.variants.split(",")
    # sentences that nobody recorded (`pmvc.tts speak --sentences`): only the text to go by
    texts = not (d / "real").exists()
    names = variants if texts else ["real", *variants]
    metas = [json.loads((d / "meta" / f"{i}.json").read_text()) for i in ids_in(d, a.split)]
    metas = [m for m in metas if not a.ids or m["id"] in a.ids.split(",")]
    metas = [m for m in metas if all((d / v / f"{m['id']}.wav").exists() for v in names)]
    path = d / f"{a.name}.json"
    prior: dict[str, dict[str, Any]] = {r["id"]: r["variants"] for r in json.loads(path.read_text())["utterances"]} if a.add and path.exists() else {}
    rows: list[dict[str, Any]] = [
        {"id": m["id"], "speaker": m["speaker"], "sex": m["sex"], "lang": m.get("lang", "en"), "text": m["text"],
         "variants": {v: prior.get(m["id"], {}).get(v, {}) for v in names}}
        for m in metas
    ]  # fmt: skip

    todo = [(r, v) for r in rows for v in names if "heard" not in r["variants"][v]]
    jobs = [(d / v / f"{r['id']}.wav", r["lang"]) for r, v in todo]
    cells = [r["variants"][v] for r, v in todo]
    refs = [units(r["text"], r["lang"]) for r, _ in todo]
    with ProcessPoolExecutor(a.jobs) as pool:
        for cell, ref, (_, lang), heard in zip(cells, refs, jobs, pool.map(_small, jobs, chunksize=4)):
            cell["heard"], cell["errors"], cell["units"] = heard, edits(ref, units(heard, lang)), len(ref)

    device = pick_device()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        big = BigWhisper()
        reader, listener = (None, None) if texts else (PhoneReader(device), Listener(device))
    for r, meta in zip(rows, metas):
        ref = units(r["text"], r["lang"])
        said: list[str] = []
        if reader is not None and listener is not None:
            real, _ = load(d / "real" / f"{r['id']}.wav", LISTEN_SR)
            said = r["variants"]["real"]["phones"].split() if "phones" in r["variants"]["real"] else reader.read(real)
            feats = listener.hear(real).feats[0]
            a0, a1 = int(meta["span"][0] * FEATURE_RATE), int(meta["span"][1] * FEATURE_RATE)
        for v in names:
            cell, wav = r["variants"][v], d / v / f"{r['id']}.wav"
            if "heard2" in cell:
                continue
            cell["heard2"] = big.transcribe(wav, r["lang"])
            cell["errors2"] = edits(ref, units(cell["heard2"], r["lang"]))
            if v == "real":
                cell["phones"] = " ".join(said)
                continue
            track = d / v / f"{r['id']}.tracks.json"
            t = json.loads(track.read_text()) if track.exists() else {}
            if t.get("free"):
                # a timing of its own: nothing to hold against the recording frame by frame
                cell["free"] = True
            if reader is None or listener is None:
                continue
            x, _ = load(wav, LISTEN_SR)
            cell["phones"] = " ".join(written := reader.read(x))
            cell["phoneErrors"], cell["phoneUnits"] = edits(said, written), len(said)
            if "shift" in t and not t.get("free"):
                x48, _ = load(wav, 48000)
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
                rest += f"   stray {m['stray']:.0f} %" if "stray" in m else ""
                rest += f"   releases {m['releases']:.2f}/s at {m.get('releaseLevel', float('nan')):.0f} dB, odd clicks {m['oddClicks']:.2f}/s" if "oddClicks" in m else ""
                rest += f"   on its own: releases {m['ownReleases']:.2f}/s at {m.get('ownReleaseLevel', float('nan')):.1f} dB, click load {m['ownClickLoad']:.1f} dB" if "ownClicks" in m else ""
                rest += f"   in silences {m['silenceSounds']:.2f}/s at {m.get('silenceLevel', float('nan')):.0f} dB" if "silenceSounds" in m else ""
                rest += f"   extra above 4 kHz {m['extra4k']:.2f}/s, closure clicks {m['closureClicks']:.2f}/s" if "closureClicks" in m else ""
                rest += f"   NISQA {m['quality']:.2f}" if "quality" in m else ""
                print(f"  {v:8s} " + "   ".join(cols) + rest)


if __name__ == "__main__":
    main()
