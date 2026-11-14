"""Text to speech on the codec's latent: a Japanese sentence in, a control track out, and the
voice plays it.

`pmvc.encoder` writes, for a recording, the control track that makes the voice say it. This
learns to write such tracks from the text alone, from one speaker's recordings: what it learns
from is the tracks an encoder wrote for them. Whatever it writes, the voice turns into the
sound of a vocal tract, so its mistakes are mistakes of articulation, not of a waveform.

usage: python -m pmvc.tts train [dir] [--name tts] [--tracks enci] [--standin standin_2c]
                                [--models out/transplant/models] [--closed 1.0] [--epochs 60]
                                [--limit N] [--voice FILE] [--no-wall-air] [--closed-area CM2]
       python -m pmvc.tts pauses [dir] [--name tts] [--tracks enci] [--limit N]
       python -m pmvc.tts speak [dir] [--name tts] [--models DIR] [--out tts] [--split test]
                                [--ids ...] [--timing own|recording] [--sentences FILE]
                                [--speaker NAME]
       python -m pmvc.tts say "こんにちは。" [--name tts] [--models DIR] [--out say.wav]
                              [--speaker NAME]

The model is of the plainest kind that needs no alignment at run time (FastSpeech 2's shape):
  - the sentence as `pmvc.japanese` reads it, phonemes with their places against the pitch
    accent, through layers of self-attention;
  - for each phoneme, how many frames it lasts;
  - each phoneme's state repeated for that many frames, through more such layers, and out
    comes the track: every channel the encoder writes, and the pitch contour (which the
    encoder takes from a pitch tracker and has not to learn; here it is written too).
What says how long the speaker held each phoneme is `pmvc.align`.

`train` learns from the recordings of <dir> that `pmvc.align write` aligned and that have a
track in <dir>/<tracks> (`pmvc.encoder encode --no-render`). Its losses:
  the track   the mean error against the encoder's track, each channel on its spread; the
              pitch in octaves on its spread, counted as much as all the other channels
  the voice   as for the encoders: the distance between what a `pmvc.standin` model says the
              listener would hear of the track written and what it hears in the recording,
              times `--closed`. A track that averages over ways to say a phoneme need not say
              it; this term asks for one that does. It needs the frames to be the recording's,
              which they are while the durations are (they are, in training).
  durations   the error of the logarithm of each phoneme's frames (the durations the model
              then gives are scaled to add up to the speaker's: see `pace`)
Every `--every` epochs the voice itself plays the tracks written for validation recordings
(each phoneme as long as the speaker held it) and `pmvc.listener` says how far the renders are
from the recordings; the weights kept are the ones closest. The voice's options are the ones
the encoder's tracks were made for; the model remembers them, and the speaker's sex and
median pitch, which is all the voice is told of the speaker.

A speaker pauses where the text has no comma (`pmvc.align` marks it: `sp`), and a model that
only pauses at commas runs long sentences together. When the training is done, the model's
states at the borders of accent phrases are taught to say whether the speaker pauses there
(`pauses` does this for a model that is already there), and a sentence the model times itself
gets a short pause where they say so.

The recordings may be of several speakers (their metas say whose each is; `pmvc.jvs --all`
lays out a hundred). The model then has a vector for each speaker, added to every phoneme
and to every frame, and writes each speaker's pitch against that speaker's own mean and
spread. Of a speaker the voice is still told only the sex and the median pitch: what else
makes the speakers differ is in the tracks, or it is lost. `speak` gives a recording's
sentence its own speaker's voice, or `--speaker`'s.

`speak` writes <dir>/<out>/<id>.tracks.json and <id>.wav for the sentences of a split's
recordings (their meta's text). With `--timing recording` each phoneme lasts as long as in the
recording (`pmvc.align write` must have been run on <dir>), so the render runs along the
recording as an encoder's does and can be judged against it frame by frame. With the model's
own timing (the default) it cannot, and its track says so (`free`): `pmvc.judge` then only
asks what was said and how it sounds. `--sentences` names a file with a sentence a line that
nobody recorded: they are laid out under <dir> (their text is all there is of them) and
spoken. `say` speaks one sentence into a file.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray
from torch import nn

from .align import aligned
from .audio import FRAME_RATE, Array, load
from .encoder import (
    LEAD,
    PAUSE,
    PAUSE_IN,
    PAUSE_OUT,
    RUN_UP,
    Decoder,
    ids_in,
    pick_device,
    render_track,
)
from .japanese import NONE, SIZES, Phone, border, read
from .listener import FEATURE_RATE, LISTEN_SR, Listener, distance, to16k
from .paths import ROOT
from .standin import SKIP_HEARD, load_standin

# the fewest frames the silence before a sentence lasts when the model times it: the voice's
# lead and the run-up of the breath
OPENING: int = round(LEAD * FRAME_RATE) + RUN_UP


@dataclass
class Sentence:
    """A recording to learn from: its phonemes, how many frames each takes, the encoder's
    track, and what the listener hears in the recording (with the span of its speech there)."""

    uid: str
    phones: list[Phone]
    frames: NDArray[np.int64]
    track: NDArray[np.float32]
    # who reads it: the name, the sex and the recording's median pitch from its meta, and the
    # speaker's place among the model's (`Voice.talkers`)
    speaker: str = ""
    sex: str = "M"
    f0: float = 0.0
    who: int = 0
    heard: NDArray[np.float16] | None = None
    span: tuple[int, int] = (0, 0)


def _track(path: Path) -> NDArray[np.float32]:
    return np.array(json.loads(path.read_text())["frames"], dtype=np.float32)


def sentences(d: Path, tracks: str, split: str, jobs: int) -> tuple[list[Sentence], list[str]]:
    """The split's recordings that are aligned and have a track, and the track's channels."""
    known = aligned(d)
    ids = [i for i in ids_in(d, split) if i in known and (d / tracks / f"{i}.tracks.json").exists()]
    with ProcessPoolExecutor(jobs) as pool:
        read_tracks = list(pool.map(_track, [d / tracks / f"{i}.tracks.json" for i in ids], chunksize=32))
    out: list[Sentence] = []
    for uid, track in zip(ids, read_tracks):
        phones, lasting = known[uid]
        # the aligner's frames and the track's are one clock; a frame's difference at the end goes to the last silence
        frames = np.array(lasting, dtype=np.int64)
        frames[-1] += len(track) - frames.sum()
        if frames[-1] >= 1:
            meta = json.loads((d / "meta" / f"{uid}.json").read_text())
            out.append(Sentence(uid, phones, frames, track, meta["speaker"], meta["sex"], meta["f0Median"]))
    keys: list[str] = json.loads((d / tracks / f"{ids[0]}.tracks.json").read_text())["keys"]
    return out, keys


def hear(d: Path, data: list[Sentence], listener: Listener, jobs: int) -> None:
    """What the listener hears in each recording, kept in <dir>/heard for the next time."""
    tag = f"{listener.name.split('/')[-1]}-{listener.layer}"
    (d / "heard" / tag).mkdir(parents=True, exist_ok=True)

    def one(s: Sentence) -> tuple[NDArray[np.float16] | None, NDArray[np.float32] | None]:
        """What was heard before, or else the recording to hear."""
        path = d / "heard" / tag / f"{s.uid}.npy"
        return (np.load(path), None) if path.exists() else (None, load(d / "real" / f"{s.uid}.wav", LISTEN_SR)[0].astype(np.float32))

    with ThreadPoolExecutor(jobs) as pool:
        for s, (heard, x) in zip(data, pool.map(one, data)):
            if heard is None:
                assert x is not None
                heard = listener.hear(x).feats[0].half().numpy()
                np.save(d / "heard" / tag / f"{s.uid}.npy", heard)
            meta = json.loads((d / "meta" / f"{s.uid}.json").read_text())
            s.heard, s.span = heard, (int(meta["span"][0] * FEATURE_RATE), int(meta["span"][1] * FEATURE_RATE))


def fewer(data: list[Sentence], limit: int) -> list[Sentence]:
    """So many of the recordings (all of them for 0): of each speaker a share, spread evenly
    over the speaker's recordings and starting at another place for each speaker, so that
    speakers who read the same sentences are heard reading different ones."""
    if not limit or limit >= len(data):
        return data
    by: dict[str, list[Sentence]] = {}
    for s in data:
        by.setdefault(s.speaker, []).append(s)
    out: list[Sentence] = []
    for k, theirs in enumerate(by.values()):
        n = max(1, round(limit * len(theirs) / len(data)))
        start = round(k * 0.618 * len(theirs))
        out += [theirs[(start + round(j * len(theirs) / n)) % len(theirs)] for j in range(min(n, len(theirs)))]
    return out


def turn(x: torch.Tensor) -> torch.Tensor:
    """Rotary positions for (batch, heads, frames, width): what two frames make of each other
    then goes by how far apart they are."""
    n, w = x.shape[-2], x.shape[-1]
    angle = torch.arange(n, device=x.device)[:, None] / (10000.0 ** (torch.arange(0, w, 2, device=x.device) / w))[None]
    cos, sin = angle.cos(), angle.sin()
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)


class Block(nn.Module):
    """Self-attention, then a convolution over neighbouring frames, each after a norm."""

    def __init__(self, width: int, heads: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.conv = nn.Sequential(nn.Conv1d(width, 4 * width, kernel, padding=kernel // 2), nn.GELU(), nn.Conv1d(4 * width, width, 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """`x` (batch, frames, width); `real` (batch, frames) is false on the padding."""
        b, n, w = x.shape
        q, k, v = (t.reshape(b, n, self.heads, w // self.heads).transpose(1, 2) for t in self.qkv(self.norm1(x)).chunk(3, dim=-1))
        h = nn.functional.scaled_dot_product_attention(turn(q), turn(k), v, attn_mask=real[:, None, None, :], dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.out(h.transpose(1, 2).reshape(b, n, w)))
        h = self.conv((self.norm2(x) * real[..., None]).transpose(1, 2)).transpose(1, 2)
        return x + self.drop(h)


def at_rate(x: torch.Tensor, total: torch.Tensor, step: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Frame states (batch, frames, width), of which each clip's first `total` are real, at a
    lower rate: each new frame is the mean over `step` of the old ones (`step` need not be
    whole). Returns the new states and which of them are real."""
    b, t, w = x.shape
    m = (total.float() / step).round().long().clamp(min=1)
    edges = torch.arange(int(m.max()) + 1, device=x.device)[None] * step
    e = torch.minimum(edges.expand(b, -1), total[:, None].float())
    # the sum of a clip's states up to a point between two frames: the whole frames before
    # it, and the part of the frame it falls in
    summed = torch.cat([x.new_zeros(b, 1, w), x.cumsum(dim=1)], dim=1)
    lo = e.floor().long().clamp(max=t - 1)
    upto = summed.gather(1, lo[..., None].expand(-1, -1, w)) + (e - lo)[..., None] * x.gather(1, lo[..., None].expand(-1, -1, w))
    out = (upto[:, 1:] - upto[:, :-1]) / (e[:, 1:] - e[:, :-1]).clamp(min=1e-6)[..., None]
    return out, torch.arange(out.shape[1], device=x.device)[None] < m[:, None]


class Pausing(nn.Module):
    """Whether the speaker pauses at an accent phrase's border where the text has no comma,
    read off the states of the phonemes either side of it."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        """The states (borders, width) of the phoneme before each border and of the one after
        it. Returns the logit that the speaker pauses there."""
        return self.net(torch.cat([before, after], dim=-1))[:, 0]


class Speaker(nn.Module):
    """Phonemes in, their durations and the track's channels out."""

    def __init__(self, phones: list[str], channels: int, width: int = 256, heads: int = 4, text_layers: int = 4, frame_layers: int = 6, kernel: int = 5, dropout: float = 0.1, pausing: bool = False, speakers: int = 1, rate: float = float(FRAME_RATE)) -> None:
        super().__init__()
        self.phones = phones
        # how many of the model's own frames (FRAME_RATE a second) one frame of what it writes
        # takes: 1 for a control track; more for a latent with fewer frames a second (`pmvc.control`)
        self.step = FRAME_RATE / rate
        self.pausing = Pausing(width) if pausing else None
        # a model of several speakers is told which one speaks: a vector for each, added to
        # every phoneme before the text is read and to every frame before the track is written
        self.who = nn.Embedding(speakers, width) if speakers > 1 else None
        self.embed = nn.ModuleList([nn.Embedding(n, width) for n in (len(phones), *SIZES)])
        self.text = nn.ModuleList([Block(width, heads, kernel, dropout) for _ in range(text_layers)])
        self.text_norm = nn.LayerNorm(width)
        self.lasting = nn.ModuleList([nn.Conv1d(width, width, 3, padding=1), nn.Conv1d(width, width, 3, padding=1)])
        self.lasting_norm = nn.ModuleList([nn.LayerNorm(width), nn.LayerNorm(width)])
        self.lasting_out = nn.Linear(width, 1)
        self.lasting_drop = nn.Dropout(0.3)
        # where a frame is in its phoneme: from its start, from its end, and how long the phoneme is
        self.place = nn.Linear(3, width)
        self.frame = nn.ModuleList([Block(width, heads, kernel, dropout) for _ in range(frame_layers)])
        self.frame_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, channels)

    def read(self, ids: torch.Tensor, real: torch.Tensor, who: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """`ids` (batch, phonemes, 5) as `Phone.ids` gives them, and `who` (batch,) speaks, for
        a model of several speakers. Returns each phoneme's state and the logarithm of its
        frames."""
        h = sum(e(ids[..., k]) for k, e in enumerate(self.embed))
        assert isinstance(h, torch.Tensor)
        if self.who is not None:
            assert who is not None, "a model of several speakers has to be told who speaks"
            h = h + self.who(who)[:, None]
        for block in self.text:
            h = block(h, real)
        h = self.text_norm(h)
        g = h
        for conv, norm in zip(self.lasting, self.lasting_norm):
            g = self.lasting_drop(norm(torch.relu(conv((g * real[..., None]).transpose(1, 2)).transpose(1, 2))))
        return h, self.lasting_out(g)[..., 0]

    def states(self, h: torch.Tensor, frames: torch.Tensor, who: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`h` (batch, phonemes, width) and each phoneme's `frames` (0 on the padding): each
        phoneme's state for as many frames as it lasts, with where each frame is in it and who
        speaks. Returns the frames' states, which are real, and how many each clip has."""
        ends = frames.cumsum(dim=1)
        total = ends[:, -1]
        at = torch.arange(int(total.max()), device=h.device)[None].expand(len(h), -1)
        which = torch.searchsorted(ends, at.contiguous(), right=True).clamp(max=h.shape[1] - 1)
        real = at < total[:, None]
        long = frames.gather(1, which).clamp(min=1).float()
        since = (at - (ends - frames).gather(1, which)).float()
        place = torch.stack([(since + 0.5) / long, since / 50.0, long.log() / 4.0], dim=-1)
        x = h.gather(1, which[..., None].expand(-1, -1, h.shape[2])) + self.place(place)
        if self.who is not None:
            assert who is not None
            x = x + self.who(who)[:, None]
        return x, real, total

    def finish(self, x: torch.Tensor, real: torch.Tensor, total: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The frames' states through the frame layers and out, at the rate of what is written.
        Returns the channels, (batch, frames, channels), normalised, and which frames are real."""
        for block in self.frame:
            x = block(x, real)
        x = self.frame_norm(x)
        if self.step != 1.0:
            x, real = at_rate(x, total, self.step)
        return self.out(x), real

    def write(self, h: torch.Tensor, frames: torch.Tensor, who: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """`h` (batch, phonemes, width) and each phoneme's `frames` (0 on the padding). Returns
        the channels, (batch, frames, channels), normalised, and which frames are real."""
        return self.finish(*self.states(h, frames, who))


@dataclass
class Talker:
    """A speaker as a model knows them: the name; the sex and the median pitch (Hz), which
    are all the voice is told of a speaker (its body, and what its source idles at and which
    settings it gets); and the mean and spread of the log2 pitch, which the model's pitch is
    written against."""

    name: str
    sex: str
    f0: float
    pitch: tuple[float, float]


@dataclass
class Voice:
    """What a trained model needs beside its weights to write a track the voice can play: the
    channels and their spread, the speakers, and how the voice is to play it."""

    keys: list[str]
    mean: Array
    std: Array
    lo: Array
    hi: Array
    learned: list[int]
    talkers: list[Talker]
    decoder: Decoder
    # what the model's durations are multiplied by: for phonemes, and for pauses inside a sentence
    pace: tuple[float, float] = (1.0, 1.0)

    def state(self) -> dict[str, Any]:
        return {"keys": self.keys, "mean": self.mean.tolist(), "std": self.std.tolist(), "lo": self.lo.tolist(), "hi": self.hi.tolist(), "learned": self.learned,
                "talkers": [{"name": t.name, "sex": t.sex, "f0": t.f0, "pitch": list(t.pitch)} for t in self.talkers], "pace": list(self.pace),
                "decoder": {"fitted": self.decoder.fitted, "wall_air": self.decoder.wall_air, "closed_area": self.decoder.closed_area}}  # fmt: skip

    @classmethod
    def of(cls, s: dict[str, Any]) -> "Voice":
        # a model of one speaker from before there were several says of the speaker what a talker does
        talkers = s.get("talkers") or [{"name": "", "sex": s["sex"], "f0": s["f0"], "pitch": s["pitch"]}]
        pace = s.get("pace", [1.0, 1.0])
        return cls(s["keys"], np.array(s["mean"]), np.array(s["std"]), np.array(s["lo"]), np.array(s["hi"]), s["learned"],
                   [Talker(t["name"], t["sex"], t["f0"], (t["pitch"][0], t["pitch"][1])) for t in talkers], Decoder(**s["decoder"]), (pace[0], pace[1]))  # fmt: skip

    def who(self, name: str) -> int:
        """A speaker's place among the model's; a model of one speaker has only that one."""
        if len(self.talkers) == 1:
            return 0
        names = [t.name for t in self.talkers]
        if name not in names:
            raise ValueError(f"the model has no speaker {name!r}: it has {', '.join(names[:5])}, ... ({len(names)})")
        return names.index(name)

    def frames(self, out: NDArray[np.float32], who: int = 0) -> Array:
        """The model's channels as a track's frames, in the voice's units."""
        f0 = self.keys.index("f0")
        pitch = self.talkers[who].pitch
        frames = np.tile(self.mean, (len(out), 1))
        frames[:, self.learned] = out[:, :-1] * self.std[self.learned] + self.mean[self.learned]
        frames[:, f0] = 2.0 ** (out[:, -1] * pitch[1] + pitch[0])
        return np.clip(frames, self.lo, self.hi)

    def track(self, frames: Array, free: bool, who: int = 0) -> dict[str, Any]:
        """The frames as a track for the speaker's voice."""
        t = self.talkers[who]
        body = {"female": 1.0 if t.sex == "F" else 0.0, "child": 0.0}
        track = self.decoder.settle({"rate": float(FRAME_RATE), "body": body, "source": {"f0": round(t.f0, 2)}, "keys": self.keys}, t.f0)
        return {**track, "shift": LEAD, **({"free": True} if free else {}), "frames": np.round(frames, 5).tolist()}


def collate(batch: list[Sentence], phones: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The phonemes, their frames, which are real, and who speaks."""
    n = max(len(s.phones) for s in batch)
    ids = torch.zeros(len(batch), n, 5, dtype=torch.long)
    frames = torch.zeros(len(batch), n, dtype=torch.long)
    for i, s in enumerate(batch):
        ids[i, : len(s.phones)] = torch.tensor([p.ids(phones) for p in s.phones])
        frames[i, : len(s.phones)] = torch.from_numpy(s.frames)
    return ids.to(device), frames.to(device), (frames > 0).to(device), torch.tensor([s.who for s in batch], device=device)


def chunks(data: list[Sentence], size: int, rng: np.random.Generator | None) -> list[list[Sentence]]:
    order = np.argsort([len(s.track) + (rng.uniform(-25, 25) if rng is not None else 0) for s in data], kind="stable")
    out = [[data[i] for i in order[k : k + size]] for k in range(0, len(order), size)]
    return [out[i] for i in rng.permutation(len(out))] if rng is not None else out


@torch.no_grad()
def write_track(model: Speaker, voice: Voice, phones: list[Phone], device: str, frames: NDArray[np.int64] | None = None, who: int = 0) -> tuple[Array, NDArray[np.int64]]:
    """The track for a sentence in the voice of the model's speaker `who`, and how many
    frames each phoneme got: `frames` if given (the speaker's own timing), or else the model's."""
    ids = torch.tensor([[p.ids(model.phones) for p in phones]], device=device)
    speaker = torch.tensor([who], device=device)
    h, lasting = model.read(ids, torch.ones(1, len(phones), dtype=torch.bool, device=device), speaker)
    if frames is None:
        pace = np.array([1.0 if k in (0, len(phones) - 1) else voice.pace[1] if p.silent else voice.pace[0] for k, p in enumerate(phones)])
        frames = np.maximum(1, np.round(np.exp(lasting[0].cpu().numpy()) * pace)).astype(np.int64)
        frames[0] = max(frames[0], OPENING)
    out, _ = model.write(h, torch.from_numpy(frames).to(device)[None], speaker)
    track = voice.frames(out[0].cpu().numpy(), who)
    if not voice.decoder.quiet_pauses:
        return track, frames
    # as the encoder's tracks have it: no breath before and after the sentence, nor in a pause
    breath = voice.keys.index("breath")
    ends = np.cumsum(frames)
    track[: max(0, ends[0] - RUN_UP), breath] = 0.0
    track[ends[-2] + RUN_UP :, breath] = 0.0
    for k in range(1, len(phones) - 1):
        if phones[k].silent and frames[k] >= PAUSE:
            track[ends[k - 1] + PAUSE_IN : ends[k] - PAUSE_OUT, breath] = 0.0
    return track, frames


def unpaused(phones: list[Phone]) -> tuple[list[Phone], list[int], list[bool]]:
    """A recording's phonemes as its text gives them (the speaker's own pauses taken out), the
    borders of accent phrases there (each as the index of the phoneme after it), and whether
    the speaker paused at each."""
    plain: list[Phone] = []
    paused: set[int] = set()
    for p in phones:
        if p.p == "sp":
            paused.add(len(plain))
        else:
            plain.append(p)
    at = [k for k in range(len(plain)) if border(plain, k)]
    return plain, at, [k in paused for k in at]


@torch.no_grad()
def border_states(model: Speaker, data: list[Sentence], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For every accent phrase's border in `data`: the model's states either side of it (read
    from the sentence without the speaker's own pauses) and whether the speaker paused."""
    model.eval()
    before, after, paused = [], [], []
    for s in data:
        plain, at, did = unpaused(s.phones)
        if at:
            ids = torch.tensor([[p.ids(model.phones) for p in plain]], device=device)
            h, _ = model.read(ids, torch.ones(1, len(plain), dtype=torch.bool, device=device), torch.tensor([s.who], device=device))
            k = torch.tensor(at, device=device)
            before.append(h[0, k - 1])
            after.append(h[0, k])
            paused += did
    return torch.cat(before), torch.cat(after), torch.tensor(paused, device=device, dtype=torch.float32)


def fit_pausing(model: Speaker, fit: list[Sentence], held: list[Sentence], device: str, epochs: int = 40, seed: int = 0) -> dict[str, float]:
    """Teaches `model.pausing` on the recordings the model learned from; the model itself
    stays as it is. Every tenth of those recordings is kept aside, and the weights kept are
    the ones that say best where the speaker pauses in them: from a few hundred pauses a
    network learns the borders it saw by heart, and then pauses anywhere. Returns what it does
    at the held-out recordings' borders."""
    torch.manual_seed(seed)
    pausing = Pausing(model.out.in_features).to(device)
    model.pausing = pausing
    aside = fit[9::10] if len(fit) >= 20 else fit
    before, after, paused = border_states(model, [s for k, s in enumerate(fit) if k % 10 != 9] if len(fit) >= 20 else fit, device)
    b0, a0, p0 = border_states(model, aside, device)
    opt = torch.optim.AdamW(pausing.parameters(), lr=3e-4, weight_decay=0.1)
    best, kept = float("inf"), {k: v.clone() for k, v in pausing.state_dict().items()}
    for _ in range(epochs):
        pausing.train()
        for part in torch.randperm(len(paused), device=device).split(512):
            loss = nn.functional.binary_cross_entropy_with_logits(pausing(before[part], after[part]), paused[part])
            opt.zero_grad()
            loss.backward()
            opt.step()
        pausing.eval()
        with torch.no_grad():
            miss = float(nn.functional.binary_cross_entropy_with_logits(pausing(b0, a0), p0))
        if miss < best:
            best, kept = miss, {k: v.clone() for k, v in pausing.state_dict().items()}
    pausing.load_state_dict(kept)
    with torch.no_grad():
        b, a, did = border_states(model, held, device)
        said = pausing(b, a) > 0.0
    return {"borders": float(len(did)), "paused": float(did.sum()), "said": float(said.sum()), "right": float((said & (did > 0.5)).sum()),
            "fit:borders": float(len(paused) + len(p0)), "fit:paused": float(paused.sum() + p0.sum())}  # fmt: skip


@torch.no_grad()
def phrased(model: Speaker, phones: list[Phone], device: str, who: int = 0) -> list[Phone]:
    """The sentence's phonemes with a short pause where the model says the speaker `who`
    would pause; as they are if the model was not taught to say."""
    at = [k for k in range(len(phones)) if border(phones, k)]
    if model.pausing is None or not at:
        return phones
    ids = torch.tensor([[p.ids(model.phones) for p in phones]], device=device)
    h, _ = model.read(ids, torch.ones(1, len(phones), dtype=torch.bool, device=device), torch.tensor([who], device=device))
    k = torch.tensor(at, device=device)
    put = {i for i, logit in zip(at, model.pausing(h[0, k - 1], h[0, k]).tolist()) if logit > 0.0}
    return [q for i, p in enumerate(phones) for q in ((Phone("sp", NONE, NONE, NONE, p.ask), p) if i in put else (p,))]


@torch.no_grad()
def pace(model: Speaker, data: list[Sentence], device: str, batch: int = 32) -> tuple[float, float]:
    """What the model's durations have to be multiplied by to add up to the speaker's over
    `data`: for the phonemes, and for the pauses inside a sentence. The model learns the
    logarithm of a duration, and the mean of logarithms is short of the logarithm of the mean:
    left as they are, its sentences run some 6 % faster than the speaker's."""
    model.eval()
    said, written = np.zeros(2), np.zeros(2)
    for part in chunks(data, batch, None):
        ids, _, real, who = collate(part, model.phones, device)
        _, lasting = model.read(ids, real, who)
        for i, s in enumerate(part):
            inner = np.arange(len(s.phones))[1:-1]
            pause = np.array([p.silent for p in s.phones])[inner]
            mine = np.exp(lasting[i, inner].cpu().numpy())
            for k, sel in enumerate((~pause, pause)):
                said[k] += s.frames[inner][sel].sum()
                written[k] += mine[sel].sum()
    return float(said[0] / max(1.0, written[0])), float(said[1] / max(1.0, written[1]))


class Check:
    """How close the voice comes to held-out recordings when the model writes their tracks
    from the text, each phoneme as long as the speaker held it: `pmvc.listener`'s distance
    over the recording's speech, as `pmvc.encoder`'s check measures an encoder."""

    def __init__(self, data: list[Sentence], voice: Voice, listener: Listener, device: str, jobs: int) -> None:
        self.data, self.voice, self.listener, self.device, self.jobs = data, voice, listener, device, jobs

    def __call__(self, model: Speaker) -> float:
        model.eval()
        tracks = [self.voice.track(write_track(model, self.voice, s.phones, self.device, s.frames, s.who)[0], False, s.who) for s in self.data]
        with ThreadPoolExecutor(self.jobs) as pool:
            renders = list(pool.map(render_track, tracks))
        out: list[float] = []
        for s, x in zip(self.data, renders):
            assert s.heard is not None
            heard = self.listener.hear(to16k(x[round(LEAD * 48000) :])).feats[0]
            a, b = s.span
            out.append(float(distance(torch.from_numpy(s.heard[a:b].astype(np.float32)), heard[a:b]).mean()))
        return float(np.mean(out))


def train(a: argparse.Namespace) -> None:
    d: Path = a.dir
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = pick_device()
    standin, heard_by = load_standin(a.models / f"{a.standin}.pt", device)
    standin.train()
    listener = Listener(device, *heard_by)
    fit, keys = sentences(d, a.tracks, "train", a.jobs)
    held, _ = sentences(d, a.tracks, "valid", a.jobs)
    # every phoneme of the corpus has a place, whether or not it is among the recordings learned from
    phones = sorted({p.p for s in fit for p in s.phones})
    everyone = fit
    fit = fewer(fit, a.limit)
    hear(d, fit, listener, a.jobs)
    hear(d, held, listener, a.jobs)

    # the speakers, and how the voice plays the tracks
    decoder = Decoder.of(a)
    decoder.quiet_pauses = True
    f0 = keys.index("f0")
    by: dict[str, list[Sentence]] = {}
    for s in everyone:
        by.setdefault(s.speaker, []).append(s)
    talkers: list[Talker] = []
    for name, theirs in sorted(by.items()):
        octaves = np.log2(np.concatenate([s.track[:, f0] for s in theirs]).astype(np.float64))
        talkers.append(Talker(name, theirs[0].sex, float(np.median([s.f0 for s in theirs])), (float(octaves.mean()), float(octaves.std()))))
    places = {t.name: k for k, t in enumerate(talkers)}
    held = [s for s in held if s.speaker in places and all(p.p in phones for p in s.phones)]
    for s in (*fit, *held):
        s.who = places[s.speaker]
    pooled = np.concatenate([s.track for s in fit[:: max(1, len(fit) // 2000)]]).astype(np.float64)
    mean, std = pooled.mean(axis=0), pooled.std(axis=0)
    learned = [c for c in range(len(keys)) if c != f0 and std[c] > 1e-6]
    voice = Voice(keys, mean, std, pooled.min(axis=0), pooled.max(axis=0), learned, talkers, decoder)
    assert (standin.inp.in_features == len(keys) + 2) == (decoder.fitted is not None), "the stand-in and --voice do not go together"
    # what the stand-in is told of each speaker: the body, and the share of the fitted voice
    told = torch.tensor([[1.0 if t.sex == "F" else 0.0, decoder.share(t.f0)][: 2 if decoder.fitted else 1] for t in talkers], device=device)
    # each speaker's log2 pitch: its mean and spread
    centre = torch.tensor([t.pitch[0] for t in talkers], device=device, dtype=torch.float32)
    spread = torch.tensor([t.pitch[1] for t in talkers], device=device, dtype=torch.float32)

    model = Speaker(phones, len(learned) + 1, a.width, text_layers=a.text_layers, frame_layers=a.frame_layers, speakers=len(talkers)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(chunks(fit, a.batch, None))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    women = sum(t.sex == "F" for t in talkers)
    print(f"{len(fit)} recordings to fit ({sum(len(s.track) for s in fit) / FRAME_RATE / 3600:.1f} hours), {len(held)} held out; {len(phones)} phonemes, {len(learned)} channels and pitch; "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M weights; {len(talkers)} speaker(s), {women} of them women, median pitch {min(t.f0 for t in talkers):.0f}-{max(t.f0 for t in talkers):.0f} Hz; {device}", flush=True)  # fmt: skip

    mu = torch.tensor(mean[learned], device=device, dtype=torch.float32)
    sd = torch.tensor(std[learned], device=device, dtype=torch.float32)
    # the channels written, the pitch last and in octaves: where they go in a track, and their limits
    index = torch.tensor([*learned, f0], device=device)
    lo = torch.tensor([*voice.lo[learned], np.log2(voice.lo[f0])], device=device, dtype=torch.float32)
    hi = torch.tensor([*voice.hi[learned], np.log2(voice.hi[f0])], device=device, dtype=torch.float32)
    rest = torch.tensor(mean, device=device, dtype=torch.float32)

    def losses(batch: list[Sentence]) -> dict[str, torch.Tensor]:
        ids, frames, real, who = collate(batch, phones, device)
        h, lasting = model.read(ids, real, who)
        out, there = model.write(h, frames, who)
        n = out.shape[1]
        middle, wide = centre[who][:, None], spread[who][:, None]
        target = torch.zeros(len(batch), n, len(keys), device=device)
        for i, s in enumerate(batch):
            target[i, : len(s.track)] = torch.from_numpy(s.track).to(device)
        count = there.sum()
        parts = {
            "track": (((out[..., :-1] - (target[..., learned] - mu) / sd).abs().mean(dim=-1)) * there).sum() / count,
            "pitch": ((out[..., -1] - (torch.log2(target[..., f0].clamp(min=1.0)) - middle) / wide).abs() * there).sum() / count,
            "durations": (((lasting - frames.clamp(min=1).float().log()) ** 2) * real).sum() / real.sum(),
        }
        if a.closed > 0.0:
            # the track written, as the stand-in reads it: in the voice's units, the pitch in
            # octaves, past a clip's end its first frame (the voice at rest), the speaker last
            written = torch.cat([out[..., :-1] * sd + mu, (out[..., -1] * wide + middle)[..., None]], dim=-1)
            track = rest.expand(len(batch), n, -1).index_copy(-1, index, torch.minimum(torch.maximum(written, lo), hi))
            track = torch.where(there[..., None], track, track[:, :1])
            said = standin(torch.cat([track, told[who][:, None].expand(-1, n, -1)], dim=-1))[:, SKIP_HEARD:]
            total = out.new_zeros(())
            for i, s in enumerate(batch):
                assert s.heard is not None
                a0, a1 = s.span[0], min(s.span[1], len(s.heard), said.shape[1])
                total = total + (1.0 - (said[i, a0:a1] * torch.from_numpy(s.heard[a0:a1].astype(np.float32)).to(device)).sum(dim=-1)).mean()
            parts["voice"] = total / len(batch)
        return parts

    def weighed(parts: dict[str, torch.Tensor]) -> torch.Tensor:
        return parts["track"] + parts["pitch"] + parts["durations"] + a.closed * parts.get("voice", parts["track"].new_zeros(()))

    check = Check(held[:: a.stride], voice, listener, device, a.jobs)
    out_dir = d / "models"
    out_dir.mkdir(exist_ok=True)
    log: list[dict[str, float]] = []
    best = float("inf")
    for epoch in range(a.epochs):
        model.train()
        sums: dict[str, float] = {}
        count = 0
        for batch in chunks(fit, a.batch, rng):
            parts = losses(batch)
            opt.zero_grad()
            weighed(parts).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            count += 1
        row: dict[str, float] = {"epoch": epoch, **{k: v / count for k, v in sums.items()}}
        model.eval()
        with torch.no_grad():
            parts_held = [losses(b) for b in chunks(held, a.batch, None)]
        row.update({f"held:{k}": float(np.mean([p[k].item() for p in parts_held])) for k in parts_held[0]})
        if epoch % a.every == a.every - 1 or epoch == a.epochs - 1:
            row["listener"] = check(model)
            if row["listener"] < best:
                best = row["listener"]
                voice.pace = pace(model, fit[:: max(1, len(fit) // 1000)], device)
                row["pace"], row["pace:pauses"] = voice.pace
                torch.save({"state": model.state_dict(), "phones": phones, "voice": voice.state(), "width": a.width, "text_layers": a.text_layers, "frame_layers": a.frame_layers,
                            "epoch": epoch, "args": {k: str(v) for k, v in vars(a).items()}}, out_dir / f"{a.name}.pt")  # fmt: skip
        log.append(row)
        print(f"epoch {epoch:3d}  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "epoch") + ("  *" if row.get("listener") == best else ""), flush=True)
        (out_dir / f"{a.name}.json").write_text(json.dumps({"best": best, "epochs": log}, indent=1))
    teach_pauses(out_dir / f"{a.name}.pt", fit, held, device)


def teach_pauses(path: Path, fit: list[Sentence], held: list[Sentence], device: str) -> None:
    """Adds to the model in `path` its say on where the speaker pauses without a comma."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model, voice = load_model(path, device)
    names = {t.name for t in voice.talkers}
    held = [s for s in held if len(names) == 1 or s.speaker in names]
    for s in (*fit, *held):
        s.who = voice.who(s.speaker)
    r = fit_pausing(model, fit, held, device)
    ck["state"], ck["pausing"] = model.state_dict(), True
    torch.save(ck, path)
    print(f"pauses without a comma: the speaker makes {r['fit:paused']:.0f} at {r['fit:borders']:.0f} borders of accent phrases in the recordings learned from; "
          f"held out, {r['paused']:.0f} at {r['borders']:.0f}, and the model says {r['said']:.0f}, {r['right']:.0f} of them where the speaker does", flush=True)  # fmt: skip


def pauses(a: argparse.Namespace) -> None:
    d: Path = a.dir
    fit, _ = sentences(d, a.tracks, "train", a.jobs)
    held, _ = sentences(d, a.tracks, "valid", a.jobs)
    teach_pauses(d / "models" / f"{a.name}.pt", fewer(fit, a.limit), held, pick_device())


def load_model(path: Path, device: str) -> tuple[Speaker, Voice]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    voice = Voice.of(ck["voice"])
    voice.decoder.quiet_pauses = True
    model = Speaker(ck["phones"], len(voice.learned) + 1, ck["width"], text_layers=ck["text_layers"], frame_layers=ck["frame_layers"], pausing=ck.get("pausing", False), speakers=len(voice.talkers))
    model.load_state_dict({k: v for k, v in ck["state"].items() if ck.get("pausing") or not k.startswith("pausing.")})
    return model.to(device).eval(), voice


def phonemes(model: Speaker, text: str) -> list[Phone]:
    """The sentence's phonemes, as the model knows them. A palatalised or labialised consonant
    the model never met (kw in クィ, say) is said as the consonant and the glide; any other
    stranger cannot be said."""
    out: list[Phone] = []
    for p in read(text):
        if p.p in model.phones:
            out.append(p)
        elif len(p.p) > 1 and p.p[-1] in "yw" and p.p[:-1] in model.phones:
            out += [Phone(p.p[:-1], p.a1, p.a2, p.a3, p.ask), Phone(p.p[-1], p.a1, p.a2, p.a3, p.ask)]
        else:
            raise ValueError(f"the model was not taught the phoneme {p.p} of {text!r}")
    return out


def speak(a: argparse.Namespace) -> None:
    d: Path = a.dir
    device = pick_device()
    model, voice = load_model((a.models or d / "models") / f"{a.name}.pt", device)
    if a.sentences:
        # sentences nobody recorded: all there is of each is its text
        (d / "meta").mkdir(parents=True, exist_ok=True)
        for k, text in enumerate(filter(None, (line.strip() for line in a.sentences.read_text(encoding="utf-8").splitlines())), 1):
            meta = {"id": f"{a.set}_{k:03d}", "speaker": a.set, "sex": voice.talkers[voice.who(a.speaker)].sex, "lang": "ja", "text": text, "split": "test"}
            (d / "meta" / f"{meta['id']}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    ids = a.ids.split(",") if a.ids else [i for i in ids_in(d, a.split) if json.loads((d / "meta" / f"{i}.json").read_text()).get("lang") == "ja"]
    timed = aligned(d) if a.timing == "recording" else {}
    (d / a.out).mkdir(exist_ok=True)
    tracks: list[dict[str, Any]] = []
    for uid in ids:
        meta = json.loads((d / "meta" / f"{uid}.json").read_text())
        # in the voice asked for, or else the recording's own speaker's
        who = voice.who(a.speaker or meta["speaker"])
        if timed:
            phones, frames = timed[uid]
            track, _ = write_track(model, voice, phones, device, np.array(frames, dtype=np.int64), who)
        else:
            track, _ = write_track(model, voice, phrased(model, phonemes(model, meta["text"]), device, who), device, None, who)
        tracks.append(voice.track(track, not timed, who))
        (d / a.out / f"{uid}.tracks.json").write_text(json.dumps(tracks[-1]))
    with ThreadPoolExecutor(a.jobs) as pool:
        for uid, x in zip(ids, pool.map(render_track, tracks)):
            sf.write(str(d / a.out / f"{uid}.wav"), x, 48000, subtype="PCM_16")
    print(f"{len(tracks)} sentences → {d / a.out}")


def say(a: argparse.Namespace) -> None:
    device = pick_device()
    model, voice = load_model(a.models / f"{a.name}.pt", device)
    who = voice.who(a.speaker)
    phones = phrased(model, phonemes(model, a.text), device, who)
    frames, lasting = write_track(model, voice, phones, device, None, who)
    track = voice.track(frames, True, who)
    sf.write(str(a.out), render_track(track), 48000, subtype="PCM_16")
    a.out.with_suffix(".tracks.json").write_text(json.dumps(track))
    print(" ".join(f"{p.p}:{n}" for p, n in zip(phones, lasting)))
    print(f"{len(frames) / FRAME_RATE:.2f} s → {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    u = sub.add_parser("pauses")
    s = sub.add_parser("speak")
    y = sub.add_parser("say")
    for p in (t, u, s):
        p.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "hfcm")
    y.add_argument("text", type=str)
    for p in (t, u, s, y):
        p.add_argument("--name", type=str, default="tts")
        p.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    for p in (t, u):
        p.add_argument("--tracks", type=str, default="enci", help="the variant whose tracks are learned from")
        p.add_argument("--limit", type=int, default=0, help="learn from only so many of the recordings")
    t.add_argument("--models", type=Path, default=ROOT / "out" / "transplant" / "models", help="where the stand-in is")
    t.add_argument("--standin", type=str, default="standin_2c")
    t.add_argument("--closed", type=float, default=1.0, help="the weight of the term through the stand-in; 0 for none")
    t.add_argument("--voice", type=Path, default=None, help="pmvc.calibrate's file, as for pmvc.encoder")
    t.add_argument("--no-wall-air", action="store_true")
    t.add_argument("--closed-area", type=float, default=None)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--every", type=int, default=5, help="epochs between the listener's checks")
    t.add_argument("--stride", type=int, default=2, help="the listener checks every so-many-th validation recording")
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--width", type=int, default=256)
    t.add_argument("--text-layers", type=int, default=4)
    t.add_argument("--frame-layers", type=int, default=6)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--seed", type=int, default=0)
    s.add_argument("--models", type=Path, default=None, help="where the model is, if not <dir>/models")
    s.add_argument("--out", type=str, default="tts")
    s.add_argument("--split", choices=["test", "valid", "train", "all"], default="test")
    s.add_argument("--ids", type=str, default="")
    s.add_argument("--timing", choices=["own", "recording"], default="own")
    s.add_argument("--sentences", type=Path, default=None, help="a file with a sentence a line, to lay out under <dir> and speak")
    s.add_argument("--set", type=str, default="say", help="a name for those sentences: their ids start with it")
    for p in (s, y):
        p.add_argument("--speaker", type=str, default="", help="which of the model's speakers speaks (a model of several): for sentences nobody recorded, or instead of a recording's own speaker")
    y.add_argument("--models", type=Path, default=ROOT / "out" / "hfcm" / "models")
    y.add_argument("--out", type=Path, default=Path("say.wav"))
    a = ap.parse_args()
    {"train": train, "pauses": pauses, "speak": speak, "say": say}[a.command](a)


if __name__ == "__main__":
    main()
