"""Audio I/O and the measures the experiments share: pitch, mel cepstra, dynamic time warping."""

from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from numpy.typing import NDArray
from scipy.fft import dct
from scipy.signal import resample_poly

Array = NDArray[np.float64]

# frames per second of every track and feature in this package
FRAME_RATE: int = 100
# analysis rate for the spectral measures (Hz): speech up to 8 kHz
MEL_SR: int = 16000


def load(path: Path, sr: int | None = None) -> tuple[Array, int]:
    """Mono float samples, resampled to `sr` when given."""
    x, rate = sf.read(str(path), dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if sr is not None and sr != rate:
        g = int(np.gcd(sr, rate))
        x = resample_poly(x, sr // g, rate // g)
        rate = sr
    return x, rate


def pitch_track(x: Array, sr: int, sex: str) -> Array:
    """f0 (Hz) at `i / FRAME_RATE` seconds, 0 where unvoiced (Praat's autocorrelation method)."""
    floor, ceiling = (100.0, 500.0) if sex == "F" else (60.0, 350.0)
    pitch = parselmouth.Sound(x, sampling_frequency=sr).to_pitch_ac(
        time_step=1.0 / FRAME_RATE, pitch_floor=floor, pitch_ceiling=ceiling
    )
    n = int(len(x) / sr * FRAME_RATE) + 1
    f0 = np.array([pitch.get_value_at_time(i / FRAME_RATE) for i in range(n)])
    return np.nan_to_num(f0, nan=0.0)


def fill_pitch(f0: Array) -> Array:
    """A continuous contour through the voiced frames: log-linear across unvoiced stretches,
    held at the ends, and kept within an octave of the median (octave errors, creak)."""
    voiced = f0 > 0
    if not voiced.any():
        raise ValueError("no voiced frames")
    med = float(np.median(f0[voiced]))
    idx = np.flatnonzero(voiced)
    logf = np.interp(np.arange(len(f0)), idx, np.log(np.clip(f0[idx], med / 1.9, med * 1.9)))
    return np.exp(logf)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> Array:
    def mel(f: Array | float) -> Array:
        return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    edges = 700.0 * (10.0 ** (np.linspace(mel(fmin), mel(fmax), n_mels + 2) / 2595.0) - 1.0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        fb[m] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0.0, None)
    return fb


def log_mel(x: Array, sr: int = MEL_SR, n_mels: int = 40, floor_db: float = 50.0) -> Array:
    """Log mel amplitude spectrogram, (frames, n_mels): 25 ms Hann windows every 10 ms,
    60 Hz-8 kHz, floored `floor_db` below the loudest band (a recording's silences sit on its
    noise, a render's on nothing at all). Frame i is centred on `i / FRAME_RATE` seconds."""
    win, hop, n_fft = int(0.025 * sr), sr // FRAME_RATE, 512
    x = np.concatenate([np.zeros(win // 2), x, np.zeros(win)])
    n = 1 + (len(x) - win) // hop
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:n] * np.hanning(win)
    power = np.abs(np.fft.rfft(frames, n_fft, axis=1)) ** 2
    mel = power @ _mel_filterbank(sr, n_fft, n_mels, 60.0, sr / 2).T
    return 0.5 * np.log(mel + mel.max() * 10.0 ** (-floor_db / 10.0) + 1e-20)


def cepstra(mel: Array, n: int = 13) -> Array:
    """Mel cepstra c1..c(n-1) of a log mel spectrogram (c0, the level, is left out), scaled so
    that the log spectrum is c0 + Σ c_m·cos(...), the convention mel-cepstral distortion uses."""
    return dct(mel, type=2, norm="ortho", axis=1)[:, 1:n] * np.sqrt(2.0 / mel.shape[1])


def speech_span(mel: Array, floor_db: float = 40.0) -> tuple[int, int]:
    """First and last frame within `floor_db` of the loudest frame."""
    level = 10.0 / np.log(10.0) * np.log(np.exp(2.0 * mel).sum(axis=1))
    on = np.flatnonzero(level > level.max() - floor_db)
    return int(on[0]), int(on[-1]) + 1


def dtw(a: Array, b: Array) -> tuple[float, NDArray[np.int64], NDArray[np.int64]]:
    """Dynamic time warping of two feature sequences (rows are frames) under the Euclidean
    distance. Returns the mean distance along the best path and the path's indices."""
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        row, prev, c = acc[i], acc[i - 1], cost[i - 1]
        for j in range(1, m + 1):
            row[j] = c[j - 1] + min(prev[j], prev[j - 1], row[j - 1])
    i, j = n, m
    pi, pj = [i - 1], [j - 1]
    while i > 1 or j > 1:
        step = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
        i, j = (i - 1, j - 1) if step == 0 else (i - 1, j) if step == 1 else (i, j - 1)
        pi.append(i - 1)
        pj.append(j - 1)
    ia, ib = np.array(pi[::-1]), np.array(pj[::-1])
    return float(cost[ia, ib].mean()), ia, ib


# mel-cepstral distortion (dB) per unit of Euclidean distance between cepstra
MCD_SCALE: float = 10.0 / np.log(10.0) * np.sqrt(2.0)
