"""Sounds a render makes that its recording does not: clicks, and sounds inside silences.

Clicks are short bursts in the high band that stand out of what comes before and after them.
A stop's release is one, and recordings have them: about 0.4 a second in the Japanese test
recordings, 27 dB under the level of the speech. Hardly any fall elsewhere. A render under
CHIMERA's own voice has the same. Under a voice that lets more air through (a longer open
phase, louder noise sources) it has louder releases and clicks of two more kinds: as the tongue
closes, where a speaker's closure sets in silently, and in the middle of voiced sound.

`measure` counts them in a signal's speech. A frame of the 2-9 kHz band's level (2.5 ms apart)
is a click if it is a local peak no more than 42 dB under the speech's level, at least 12 dB
over the loudest of the 5 to 20 ms before it, and at least 8 dB over the mean of the 7.5 to
20 ms after it. What the recording does 10 to 30 ms before and after says of which kind it is.

A recording is silent in a stop's closure and in a pause: 55 dB under its speech. A render is
some 10 dB less so, and not evenly: once or twice a second a soft thump stands out, 35 dB
under the speech, most of it below 1 kHz, where `measure` does not look. That, more than the
clicks, is what is heard as a wet smacking of the tongue. It is the air the tract's walls
displace as they move (see `pmvc.encoder`'s `--no-wall-air`). `in_silences` counts it.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt

from .audio import Array

# seconds between the level's frames
HOP: float = 0.0025
# the band clicks are looked for in, and the one the speech's level is taken over (Hz)
HIGH: tuple[float, float] = (2000.0, 9000.0)
WHOLE: tuple[float, float] = (80.0, 9000.0)
# a signal with no click at all has this load (dB)
NO_LOAD: float = -60.0
# the recording sounds, or is silent, where its level is this far under its speech's (dB)
SOUND: float = -25.0
SILENT: float = -30.0
# for `in_silences`: a silence is where the recording stays this far under its speech's level
# (dB); its first and last EDGE frames are left out, as a render's closure and release need not
# fall on the recording's to the frame; and a render sounds where it comes this near its own
# speech's level (dB)
QUIET: float = -45.0
EDGE: int = 6
HEARD: float = -40.0


def level(x: Array, sr: int, band: tuple[float, float]) -> Array:
    """The level (dB) of `x` in `band`, a frame every HOP seconds."""
    y = sosfiltfilt(butter(4, band, btype="band", fs=sr, output="sos"), x)
    n = int(sr * HOP)
    return 10.0 * np.log10((y[: len(y) // n * n].reshape(-1, n) ** 2).mean(axis=1) + 1e-12)


def measure(x: Array, sr: int, span: tuple[float, float], recording: tuple[Array, int] | None = None) -> dict[str, float]:
    """The clicks of `x` between `span`'s seconds. `recording` (samples and rate) is what `x`
    renders, on the same clock; without it `x` is the recording itself.
      clicks        how many a second
      clickLevel    their mean level against the speech's (dB; the 95th percentile of the whole
                    band's level stands for the speech's)
      clickLoad     their energy a second on that scale (dB): ten at -30 dB count as one at -20
      releases, releaseLevel   those where the recording goes from silence to sound, as a stop's
                    release does, and their mean level
      oddClicks     those where it goes from sound to silence (the tongue closing) or sounds on
                    both sides: the kinds recordings hardly have
    """
    k0, k1 = round(span[0] / HOP), round(span[1] / HOP)
    high, whole = level(x, sr, HIGH)[k0:k1], level(x, sr, WHOLE)[k0:k1]
    ref = float(np.percentile(whole, 95))
    heard = level(*recording, WHOLE)[k0:k1] if recording else whole
    heard = heard - float(np.percentile(heard, 95))
    found: list[float] = []
    releases: list[float] = []
    odd = 0
    for t in range(12, min(len(high), len(heard)) - 12):
        if high[t] - ref < -42.0 or high[t] < high[t - 1] or high[t] < high[t + 1]:
            continue
        if high[t] - high[t - 8 : t - 2].max() >= 12.0 and high[t] - high[t + 3 : t + 8].mean() >= 8.0:
            found.append(float(high[t] - ref))
            before, after = heard[t - 12 : t - 4].mean(), heard[t + 4 : t + 12].mean()
            if before < SILENT and after >= SOUND:
                releases.append(found[-1])
            elif before >= SOUND and (after < SILENT or after >= SOUND):
                odd += 1
    seconds = max(1e-3, len(high) * HOP)
    energy = sum(10.0 ** (c / 10.0) for c in found) / seconds
    out = {"clicks": len(found) / seconds, "clickLoad": max(NO_LOAD, 10.0 * float(np.log10(energy + 1e-12))), "releases": len(releases) / seconds, "oddClicks": odd / seconds}
    if found:
        out["clickLevel"] = float(np.mean(found))
    if releases:
        out["releaseLevel"] = float(np.mean(releases))
    return out



def stretches(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """The runs of true frames in `mask`, as (first, one past the last)."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def in_silences(x: Array, sr: int, span: tuple[float, float], recording: tuple[Array, int] | None = None) -> dict[str, float]:
    """What `x` does where the recording is silent inside its speech: in a stop's closure, in a
    pause. `recording` and `span` as for `measure`.
      silence        the seconds of silence looked at (those QUIET under the speech's level and
                     longer than twice EDGE, less EDGE at each end)
      silenceLevel   `x`'s level over them against its speech's (dB), if there are any
      silenceSounds  how many times a second of speech `x` comes within HEARD of its speech's
                     level in them
    """
    k0, k1 = round(span[0] / HOP), round(span[1] / HOP)
    own = level(x, sr, WHOLE)[k0:k1]
    own = own - float(np.percentile(own, 95))
    heard = level(*recording, WHOLE)[k0:k1] if recording else own
    heard = heard - float(np.percentile(heard, 95))
    n = min(len(own), len(heard))
    inside = np.zeros(n, dtype=bool)
    for a, b in stretches(heard[:n] < QUIET):
        # a silence the speech begins or ends with is no closure and no pause
        if b - a > 2 * EDGE and a > 0 and b < n:
            inside[a + EDGE : b - EDGE] = True
    out = {"silence": float(inside.sum() * HOP), "silenceSounds": len(stretches(inside & (own[:n] > HEARD))) / max(1e-3, n * HOP)}
    if inside.any():
        out["silenceLevel"] = 10.0 * float(np.log10(np.mean(10.0 ** (own[:n][inside] / 10.0))))
    return out
