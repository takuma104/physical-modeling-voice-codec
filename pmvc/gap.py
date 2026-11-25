"""What does a render lack of its recording? Parts swapped between the two.

A render of an encoder's track runs on its recording's clock and at its pitch, so the two can
be taken apart the same way and put together crosswise. A clip is split, frame by frame, into
its spectral envelope (what the tract and the source's tilt shape: formants, their widths, the
level) and what is left when the envelope is divided out (the fine structure: harmonics against
noise, the pulses' phases, their irregularity). A hybrid takes each part from one of the two,
everywhere or in a band, and the judges say how much nearer the speaker and how much more
natural it has become. No vocoder is involved: a hybrid is one clip's own short-time spectrum
with a gain on it, or the two clips' spectra joined at band edges.

usage: python -m pmvc.gap [dir] --variant encv2 --enrol out/jvsall --out DIR [--name gap] [--per 25]
                          [--bands 0-250,250-500,500-1000] [--gates] [--only NAME,...]
                          [--no-speaker] [--no-nisqa]

Hybrids (S is the render, R the recording; all at 24 kHz, the recordings' rate):
  R, S                 the two as they are (S cut to the recording's clock)
  S+envR               the render with the recording's envelope in every frame
  R+envS               the recording with the render's envelope in every frame
  S+envR:<band>        the render with the recording's envelope in one band only
  S+envR:static        the render with one gain curve for the whole clip (the mean of the frames')
  S+envR:coarse        the render with the broad shape of the recording's envelope in every frame
                       (tilt and level; what a cepstrum keeps under COARSE seconds)
  S+envR:fine          the render with the detail of the recording's envelope, and its own broad shape
  S|R:<band>           the render, with the recording itself in one band
  R|S:<band>           the recording, with the render itself in one band
  S|R:<band>@<stretch> the render, with the recording itself in one band, in the voiced stretches
                       of the speech only, in the unvoiced ones, or in the silences (`--gates`;
                       the recording's pitch track and `pmvc.align`'s pauses say which is which),
                       and in the unvoiced stretches' edges or their insides only
  R|S:<band>@<stretch> the other way round
`--bands` names other bands than BANDS (Hz, `lo-hi`).
Judges: two speaker-verification models (`pmvc.whose`; each hybrid against the speakers that
the recordings in `--enrol`'s training split stand for, the clip's own speaker among them) and
NISQA's two models (`pmvc.natural`). Writes the clips to <out>/<hybrid>/ and the table to
<dir>/<name>.json.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import istft, stft

from .audio import FRAME_RATE, load
from .encoder import LEAD, ids_in, pick_device, split_in
from .paths import ROOT

Array = NDArray[np.float64]

SR: int = 24000
FFT: int = 1024
WIN: int = 768
HOP: int = 96
# the envelope is what a cepstrum keeps under this share of the speaker's pitch period, and
# never more than CUT_MAX seconds (so that men's and women's envelopes are about as detailed)
CUT_SHARE: float = 0.5
CUT_MAX: float = 0.002
# the broad shape of an envelope: what a cepstrum keeps under this (seconds)
COARSE: float = 0.0004
ITERATIONS: int = 12
# a gain never moves a bin further than this (dB)
LIMIT: float = 30.0
BANDS: dict[str, tuple[float, float]] = {"0-1k": (0.0, 1000.0), "1-2.5k": (1000.0, 2500.0), "2.5-5k": (2500.0, 5000.0), "5-12k": (5000.0, 12000.0)}
# half the width of a band's edge (Hz)
EDGE: float = 100.0
# an unvoiced frame this near a voiced one (seconds) is at the edge of its stretch
EDGE_NEAR: float = 0.03
VERIFIERS: dict[str, str] = {"wavlm": "microsoft/wavlm-base-plus-sv", "unispeech": "microsoft/unispeech-sat-base-plus-sv"}


def lifter(n: int) -> Array:
    """A window on a cepstrum that keeps quefrencies under `n` samples (a half-cosine fall over the last third)."""
    w = np.zeros(FFT)
    k = np.arange(n)
    edge = np.clip((n - k) / max(1.0, n / 3), 0.0, 1.0)
    w[:n] = 0.5 - 0.5 * np.cos(np.pi * edge)
    w[FFT - n + 1 :] = w[1:n][::-1]
    return w


def smooth(log_spec: Array, n: int) -> Array:
    return np.fft.rfft(np.fft.irfft(log_spec, FFT, axis=1) * lifter(n), FFT, axis=1).real


def envelope(mag: Array, n: int) -> Array:
    """The log envelope of each frame of `mag` (frames, bins): the smooth curve through the
    peaks of its log spectrum (the "true envelope": smooth, lift what lies under the spectrum
    up to it, smooth again)."""
    log_spec = np.log(np.maximum(mag, mag.max() * 1e-6))
    a, v = log_spec, log_spec
    for _ in range(ITERATIONS):
        v = smooth(a, n)
        a = np.maximum(log_spec, v)
    return v


def spectrum(x: Array) -> NDArray[np.complex128]:
    return stft(x, SR, nperseg=WIN, noverlap=WIN - HOP, nfft=FFT, boundary="zeros", padded=True)[2].T


def signal(s: NDArray[np.complex128], n: int) -> Array:
    x = istft(s.T, SR, nperseg=WIN, noverlap=WIN - HOP, nfft=FFT, boundary=True)[1]
    return np.pad(x, (0, max(0, n - len(x))))[:n]


def band_mask(lo: float, hi: float) -> Array:
    hz = np.fft.rfftfreq(FFT, 1 / SR)
    up = np.clip((hz - (lo - EDGE)) / (2 * EDGE), 0.0, 1.0) if lo > 0 else np.ones_like(hz)
    down = np.clip(((hi + EDGE) - hz) / (2 * EDGE), 0.0, 1.0) if hi < SR / 2 else np.ones_like(hz)
    return up * down


def lag(x: Array, y: Array, most: float = 0.03) -> int:
    """Samples by which `y` runs behind `x`, by their levels in 5 ms windows (within `most` seconds)."""
    k = SR // 1000
    n = min(len(x), len(y)) // k * k
    ex, ey = (np.convolve(np.abs(z[:n]).reshape(-1, k).mean(axis=1), np.ones(5) / 5, "same") for z in (x, y))
    ex, ey = ex - ex.mean(), ey - ey.mean()
    reach = round(most * 1000)
    scores = [float(np.dot(ex[max(0, -j) : len(ex) - max(0, j)], ey[max(0, j) : len(ey) - max(0, -j)])) for j in range(-reach, reach + 1)]
    return (int(np.argmax(scores)) - reach) * k


def pair(d: Path, variant: str, uid: str) -> tuple[Array, Array, int]:
    """A recording and a variant's render of it, at SR, on the recording's clock and at its
    level (so that a band of one sits in the other as loud as its own), and the samples the
    render ran behind."""
    x = load(d / "real" / f"{uid}.wav", SR)[0]
    y = load(d / variant / f"{uid}.wav", SR)[0][round(LEAD * SR) :]
    y = np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]
    late = lag(x, y)
    return x, np.roll(y, -late) * np.sqrt((x**2).mean() / max((y**2).mean(), 1e-12)), late


def stretches(d: Path, uid: str, n: int, aligned: dict[str, Any]) -> dict[str, Array]:
    """For each of a clip's `n` spectrum frames, how far it is (0 to 1, with edges of 10 ms)
    inside a voiced stretch of the recording's speech, an unvoiced one, or a silence (the
    aligner's `sil` and `pau`); and of the unvoiced stretches, their `edge` (the frames within
    EDGE_NEAR seconds of a voiced one: where voicing starts and stops, a stop is released)
    and their `inside` (the rest)."""
    phones: list[str] = []
    for p, k in zip(aligned[uid]["p"], aligned[uid]["frames"]):
        phones += [p] * k
    phones = phones[round(LEAD * FRAME_RATE) :]  # the alignment is on the voice's clock
    f0 = np.load(d / "pitch" / f"{uid}.npy")
    j = np.minimum((np.arange(n) * HOP / SR * FRAME_RATE).round().astype(int), min(len(phones), len(f0)) - 1)
    silent = np.array([phones[i] in ("sil", "pau") for i in j])
    voiced = (f0[j] > 0) & ~silent
    unvoiced = ~voiced & ~silent
    k = round(EDGE_NEAR * SR / HOP)
    near = np.convolve(voiced.astype(float), np.ones(2 * k + 1), "same") > 0
    kinds = {"voiced": voiced, "unvoiced": unvoiced, "silence": silent, "edge": unvoiced & near, "inside": unvoiced & ~near}
    return {k: np.convolve(v.astype(float), np.ones(5) / 5, "same")[:, None] for k, v in kinds.items()}


def bands_of(spec: str) -> dict[str, tuple[float, float]]:
    """`0-250,250-500` as bands by those names."""
    return {b: (float(b.split("-")[0]), float(b.split("-")[1])) for b in spec.split(",")}


def hybrids(x: Array, y: Array, pitch: float, bands: dict[str, tuple[float, float]] = BANDS, gates: dict[str, Array] | None = None) -> dict[str, Array]:
    """Every hybrid of a recording `x` and its render `y` (same clock, same length). With
    `gates` (`stretches`), also each band swapped within one kind of stretch only."""
    n = len(x)
    sx, sy = spectrum(x), spectrum(y)
    cut = round(min(CUT_MAX, CUT_SHARE / pitch) * SR)
    ex, ey = envelope(np.abs(sx), cut), envelope(np.abs(sy), cut)
    limit = LIMIT / 20 * np.log(10)
    diff = np.clip(ex - ey, -limit, limit)
    out: dict[str, Array] = {"R": x, "S": y}
    out["S+envR"] = signal(sy * np.exp(diff), n)
    out["R+envS"] = signal(sx * np.exp(-diff), n)
    for name, (lo, hi) in bands.items():
        m = band_mask(lo, hi)
        out[f"S+envR:{name}"] = signal(sy * np.exp(diff * m), n)
        out[f"S|R:{name}"] = signal(sy * (1 - m) + sx * m, n)
        out[f"R|S:{name}"] = signal(sx * (1 - m) + sy * m, n)
        for kind, w in (gates or {}).items():
            w = np.pad(w, ((0, max(0, len(sx) - len(w))), (0, 0)))[: len(sx)]
            out[f"S|R:{name}@{kind}"] = signal(sy * (1 - m * w) + sx * (m * w), n)
            out[f"R|S:{name}@{kind}"] = signal(sx * (1 - m * w) + sy * (m * w), n)
    # one curve for the clip: the frames' mean, over the frames where both have sound
    level_x, level_y = np.abs(sx).sum(axis=1), np.abs(sy).sum(axis=1)
    both = (level_x > 0.03 * level_x.max()) & (level_y > 0.03 * level_y.max())
    out["S+envR:static"] = signal(sy * np.exp(diff[both].mean(axis=0)), n)
    broad = smooth(diff, round(COARSE * SR))
    out["S+envR:coarse"] = signal(sy * np.exp(broad), n)
    out["S+envR:fine"] = signal(sy * np.exp(diff - broad), n)
    return out


def identify(vectors: NDArray[np.float32], who: list[str], means: dict[str, NDArray[np.float32]], sex: dict[str, str]) -> dict[str, Any]:
    """`vectors` (one a clip, of speakers `who`) against every enrolled speaker's mean: `own`,
    `other` (the other speakers of the sex) and `top1`, over all clips and over each sex's."""
    names = sorted(means)
    table = np.stack([means[n] for n in names])
    table = table / np.linalg.norm(table, axis=1, keepdims=True)
    cos = vectors @ table.T
    rows = []
    for c, s in zip(cos, who):
        k = names.index(s)
        rivals = np.array([sex[n] == sex[s] and n != s for n in names])
        rows.append((sex[s], float(c[k]), float(c[rivals].mean()), float(int(c.argmax()) == k)))
    def over(part: list[tuple[str, float, float, float]]) -> dict[str, float]:
        return {k: float(np.mean([r[i + 1] for r in part])) for i, k in enumerate(("own", "other", "top1"))}
    return {**over(rows), "sexes": {x: over([r for r in rows if r[0] == x]) for x in sorted({r[0] for r in rows})}}


def enrolled(d: Path, verifier: Any, short: str, jobs: int) -> tuple[dict[str, NDArray[np.float32]], dict[str, str]]:
    """The speakers that the training recordings in `d` stand for: each one's mean vector by
    this verifier (kept in <d>/enrolled_<short>.npz, with the recordings' names, and made
    again when those change), and each one's sex."""
    metas = [json.loads(p.read_text()) for p in sorted((d / "meta").glob("*.json"))]
    metas = [m for m in metas if split_in(d, m["id"]) == "train" and (d / "real" / f"{m['id']}.wav").exists()]
    sex = {m["speaker"]: m["sex"] for m in metas}
    cache, ids = d / f"enrolled_{short}.npz", np.array([m["id"] for m in metas])
    if cache.exists():
        kept = np.load(cache)
        if np.array_equal(kept["ids"], ids):
            return {s: v for s, v in zip(kept["speakers"], kept["means"])}, sex
    with ThreadPoolExecutor(jobs) as pool:
        heard = list(pool.map(lambda m: load(d / "real" / f"{m['id']}.wav", 16000)[0], metas))
    by: dict[str, list[NDArray[np.float32]]] = {}
    for m, x in zip(metas, heard):
        by.setdefault(m["speaker"], []).append(verifier.vector(x))
    means = {s: np.mean(v, axis=0) for s, v in by.items()}
    np.savez(cache, ids=ids, speakers=np.array(list(means)), means=np.stack(list(means.values())))
    return means, sex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "jvs")
    ap.add_argument("--variant", type=str, default="encv2")
    ap.add_argument("--enrol", type=Path, default=ROOT / "out" / "jvsall", help="a directory whose training recordings stand for the speakers")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", type=str, default="gap")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--per", type=int, default=25, help="recordings of each speaker")
    ap.add_argument("--only", type=str, default="", help="hybrids to make and judge (comma-separated; all if empty)")
    ap.add_argument("--bands", type=bands_of, default=BANDS, help="bands to swap, as lo-hi in Hz, comma-separated")
    ap.add_argument("--gates", action="store_true", help="also swap each band within voiced stretches, unvoiced ones and silences only")
    ap.add_argument("--no-speaker", action="store_true")
    ap.add_argument("--no-nisqa", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    d: Path = a.dir
    metas = [json.loads((d / "meta" / f"{i}.json").read_text()) for i in ids_in(d, a.split)]
    aligned: dict[str, Any] = json.loads((d / "aligned.json").read_text()) if a.gates else {}
    metas = [m for m in metas if all((d / v / f"{m['id']}.wav").exists() for v in ("real", a.variant)) and (not a.gates or m["id"] in aligned)]
    count: dict[str, int] = {}
    kept = []
    for m in metas:
        count[m["speaker"]] = count.get(m["speaker"], 0) + 1
        if count[m["speaker"]] <= a.per:
            kept.append(m)
    pitches: dict[str, list[float]] = {}
    for m in kept:
        pitches.setdefault(m["speaker"], []).append(m["f0Median"])
    only = set(a.only.split(",")) if a.only else None

    def path_of(k: str, uid: str) -> Path:
        return a.out / k.replace(":", "_").replace("|", "_in_").replace("@", "_at_") / f"{uid}.wav"

    def make(m: dict[str, Any]) -> tuple[list[str], int]:
        x, y, late = pair(d, a.variant, m["id"])
        gates = stretches(d, m["id"], len(x) // HOP + 16, aligned) if a.gates else None
        made = hybrids(x, y, float(np.median(pitches[m["speaker"]])), a.bands, gates)
        names = [k for k in made if only is None or k in only]
        for k in names:
            p = path_of(k, m["id"])
            p.parent.mkdir(parents=True, exist_ok=True)
            peak = float(np.abs(made[k]).max())
            sf.write(str(p), made[k] * min(1.0, 0.98 / max(peak, 1e-9)), SR, subtype="PCM_16")
        return names, late

    with ThreadPoolExecutor(a.jobs) as pool:
        done = list(pool.map(make, kept))
    names = done[0][0]
    lags = np.array([late for _, late in done]) / SR * 1000
    print(f"{len(kept)} recordings of {len(pitches)} speakers; the render runs {np.median(lags):+.1f} ms behind (5th to 95th percentile {np.percentile(lags, 5):+.1f} to {np.percentile(lags, 95):+.1f})")
    path = {k: [path_of(k, m["id"]) for m in kept] for k in names}

    out: dict[str, Any] = {"variant": a.variant, "clips": len(kept), "lagMs": float(np.median(lags)), "hybrids": {k: {} for k in names}}
    who = [m["speaker"] for m in kept]
    for short, model in {} if a.no_speaker else VERIFIERS.items():
        from .whose import Verifier

        verifier = Verifier(pick_device(), model)
        means, sex = enrolled(a.enrol, verifier, short, a.jobs)
        out["enrolled"] = len(means)
        with ThreadPoolExecutor(a.jobs) as pool:
            for k in names:
                clips = pool.map(lambda p: load(p, 16000)[0], path[k])
                out["hybrids"][k][short] = identify(np.stack([verifier.vector(x) for x in clips]), who, means, sex)
        del verifier
    if not a.no_nisqa:
        from .natural import score

        for k in names:
            rows = score(path[k])
            for measure in ("quality", "natural"):
                out["hybrids"][k][measure] = {
                    "all": float(np.mean([r[measure] for r in rows])),
                    **{x: float(np.mean([r[measure] for r, m in zip(rows, kept) if m["sex"] == x])) for x in ("F", "M")},
                }
    (d / f"{a.name}.json").write_text(json.dumps(out, indent=1))
    print(f"{'':24s}" + ("" if a.no_speaker else f"{'WavLM own/other/top1 (F, M top1)':38s} {'UniSpeech-SAT':38s} ") + ("" if a.no_nisqa else "quality F/M   natural F/M"))
    for k in names:
        h = out["hybrids"][k]
        cells = [] if a.no_speaker else [f"{h[v]['own']:.3f} {h[v]['other']:.3f} {100 * h[v]['top1']:5.1f} ({100 * h[v]['sexes']['F']['top1']:5.1f}, {100 * h[v]['sexes']['M']['top1']:5.1f})".ljust(38) for v in VERIFIERS]
        nisqa = "" if a.no_nisqa else f"{h['quality']['F']:.2f} {h['quality']['M']:.2f}    {h['natural']['F']:.2f} {h['natural']['M']:.2f}"
        print(f"{k:24s}" + " ".join(cells) + (" " if cells else "") + nisqa)


if __name__ == "__main__":
    main()
