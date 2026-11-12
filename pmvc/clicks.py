"""Clicks: short bursts in the high band that stand out of what comes before and after them.

A stop's release is one, and recordings have them: about 0.4 a second in the Japanese test
recordings, 27 dB under the level of the speech. Hardly any fall elsewhere. A render under
CHIMERA's own voice has the same. Under a voice that lets more air through (a longer open
phase, louder noise sources) it has louder releases and clicks of two more kinds: as the tongue
closes, where a speaker's closure sets in silently, and in the middle of voiced sound. Together
they are heard as a wet smacking of the tongue.

`measure` counts them in a signal's speech. A frame of the 2-9 kHz band's level (2.5 ms apart)
is a click if it is a local peak no more than 42 dB under the speech's level, at least 12 dB
over the loudest of the 5 to 20 ms before it, and at least 8 dB over the mean of the 7.5 to
20 ms after it. What the recording does 10 to 30 ms before and after says of which kind it is.
"""

import numpy as np
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
