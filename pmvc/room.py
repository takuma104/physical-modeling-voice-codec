"""A dry render as a recording of it would be: a microphone's high-pass, a room's short tail, a noise floor.

The voice's output is the sound at the lips with no walls around it and no noise under it. No
recording is like that, and a judge of naturalness taught with recordings hears it: on JVS's
test sentences NISQA's naturalness model gives the release encoder's renders 4.13 (women) and
3.55 (men) against the recordings' 4.78 and 4.51, and with nothing changed in the voice a noise
floor 45 dB under the speech brings them to 4.51 and 3.97, a 0.15 s tail 18 dB under the
direct sound to 4.36 and 4.10 (at a price in NISQA's quality score, which hears noise as
noise). `recorded` is a fixed, modest recording condition for a render, so that a judge's
verdict on the voice can be told from its verdict on the condition (`pmvc.bench` scores renders
both dry and through this).

usage: python -m pmvc.room IN.wav OUT.wav [--tail 0.15] [--wet -18] [--floor -45] [--highpass 50]

It is a stand-in for a recording chain, not a model of one room: the tail is noise that decays
by 60 dB in `--tail` seconds, after 4 ms, with `--wet` dB of the direct sound's energy; the
floor is noise that falls off above 300 Hz, `--floor` dB under the clip's speech. Both are
drawn from a fixed seed, so a clip always comes out the same.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

from .audio import Array, load, speech_level

TAIL: float = 0.15
WET: float = -18.0
FLOOR: float = -45.0
HIGHPASS: float = 50.0
# the tail starts this long after the direct sound (seconds)
GAP: float = 0.004


def tail(sr: int, seconds: float = TAIL, wet: float = WET, seed: int = 0) -> Array:
    """A room's impulse response: the direct sound, then decaying noise with `wet` dB of its energy."""
    n = int(seconds * 1.5 * sr)
    h = np.random.default_rng(seed).standard_normal(n) * np.exp(-6.91 * np.arange(n) / (seconds * sr))
    h[: int(GAP * sr)] = 0.0
    h *= 10.0 ** (wet / 20.0) / np.sqrt((h**2).sum())
    h[0] = 1.0
    return h


def floor(n: int, sr: int, level: float, seed: int = 1) -> Array:
    """`n` samples of a noise floor at `level` (rms): mostly under 300 Hz, as a room's is."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    noise = sosfilt(butter(1, 300.0, "lowpass", fs=sr, output="sos"), rng.standard_normal(n)) + 0.1 * white
    return noise * level / np.sqrt((noise**2).mean())


def recorded(x: Array, sr: int, seconds: float | None = TAIL, wet: float = WET, floor_db: float | None = FLOOR, highpass: float | None = HIGHPASS, seed: int = 0) -> Array:
    """`x` (a dry clip) through the recording condition; a part given as None is left out.
    The clip keeps its length and its clock."""
    y = np.asarray(x, dtype=np.float64)
    speech = speech_level(y, sr)
    if highpass:
        y = sosfilt(butter(2, highpass, "highpass", fs=sr, output="sos"), y)
    if seconds:
        y = fftconvolve(y, tail(sr, seconds, wet, seed))[: len(y)]
    if floor_db is not None:
        y = y + floor(len(y), sr, speech * 10.0 ** (floor_db / 20.0), seed + 1)
    return y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--tail", type=float, default=TAIL, help="seconds for the tail to fall by 60 dB (0: no tail)")
    ap.add_argument("--wet", type=float, default=WET, help="the tail's energy against the direct sound's (dB)")
    ap.add_argument("--floor", type=float, default=FLOOR, help="the noise floor against the speech (dB)")
    ap.add_argument("--no-floor", action="store_true")
    ap.add_argument("--highpass", type=float, default=HIGHPASS, help="Hz (0: none)")
    a = ap.parse_args()
    x, sr = load(a.src)
    y = recorded(x, sr, a.tail or None, a.wet, None if a.no_floor else a.floor, a.highpass or None)
    sf.write(str(a.dst), y * min(1.0, 0.98 / max(float(np.abs(y).max()), 1e-9)), sr, subtype="PCM_16")


if __name__ == "__main__":
    main()
