"""The encoder: a recording in, a control track out, in one pass, with no text and no search.

With the voice as the decoder this is a codec whose latent is the control track.

usage: python -m pmvc.encoder train [dir] [--name encoder] [--target simcopy] [--synthetic simcopy]
                                    [--no-real] [--standin NAME] [--closed 1.0] [--tune 0]
                                    [--also DIR,...] [--ssl MODEL] [--silences]
                                    [--select listener|loss|last] [--epochs 40] [--hidden 256]
                                    [the voice's options, as for encode]
       python -m pmvc.encoder encode [dir] [--name encoder] [--out enc] [--split test] [--ids ...]
                                     [--models DIR] [--voice FILE] [--no-wall-air] [--quiet-pauses]
                                     [--closed-area CM2] [--sinus CM3] [--no-render]
       python -m pmvc.encoder revoice [dir] --of sim --out sim2c [--split all]
                                      [--voice FILE] [--no-wall-air] [--closed-area CM2]
                                      [--sinus CM3] [--speakers FILE]
                                      [--bright X] [--share W] [--pitch SEMITONES]

The network hears the recording through wav2vec2 (a learned mix of its layers: the model
`pmvc.listener` scores with, or `--ssl`'s; its last `--tune` layers learn too, the rest is frozen), and is
told what a pitch tracker makes of the recording and how loud it is, frame by frame; two
recurrent layers read that in both directions and write every channel but pitch, 100 frames a
second. Pitch is not learned: the track's pitch channel is the tracker's contour.

Speakers are held out in pairs. In TSP a woman and a man read the same six lists of sentences
(FA and MA lists 1-6, ...), so leaving out FK, MK, FL and ML for the test and FJ and MJ for
validation leaves out their sentences too. A recording from another corpus is in the split its
meta names (`pmvc.jvs`), and `--also` adds such directories' recordings and renders to what is
learned from and to the validation; the listener's check then averages over the directories,
each counting the same.

`train` has three things to learn from, and the encoders this repository reports are these:
  renders     `--synthetic`: each variant's renders → the tracks they were rendered from, pairs
              that are exact, in the voice's own timbre. Alone (`--no-real --synthetic warp`,
              the rule's tracks) they already teach an encoder that hears recordings: "A".
  tracks      `--target`: the training speakers' recordings → the tracks that variant has for
              them. With `simcopy`, tracks `pmvc.copysynth` searched from A's: "B". An encoder
              taught this way averages over tracks that each sound right.
  the voice   `--standin`: for each recording, the distance between what a `pmvc.standin`
              model says the listener would hear of the track the encoder writes, and what
              the listener hears in the recording, times `--closed`. This closes the loop
              through (a model of) the voice, and any track that sounds right will do: "C".
              The tracks of `--target` then only anchor it, and A's own (`--target sim
              --synthetic sim`) serve as well as searched ones: "C′", with no search at all.
Writes <dir>/models/<name>.pt and <name>.json (the losses by epoch). Which epoch's weights are
kept is `--select`:
  listener  the epoch after which the voice itself, driven by the tracks the encoder writes for
            validation recordings, comes closest to those recordings for `pmvc.listener` (its
            feature distance, checked every `--every` epochs on every `--stride`-th recording).
            It asks for no track to compare with.
  loss      the epoch with the least error against the validation recordings' `--target`
            tracks. A fit on renders alone does not aim at those tracks, and its error against
            them is least long before its renders are closest.
  last      the last epoch.

`encode` writes <dir>/<out>/<id>.tracks.json and <id>.wav for a split's recordings. The voice
needs time before the first sound, so the recording is heard after `LEAD` seconds of silence and
the render runs that far behind it (the track's `shift`). Of the speaker the voice is told the
sex (its body) and the median pitch. A recording needs its real/<id>.wav, pitch/<id>.npy and
meta/<id>.json and nothing else, so `--split all` on a directory `pmvc.recordings` made, with
`--models` saying where the encoder is, encodes recordings from outside TSP.

`--no-wall-air`: the tracks tell the voice that its walls displace no air as they move. CHIMERA
lets that air flow into the tract, and a track rendered with no breath at all is not silent: it
thumps, below a few hundred hertz, some 40 dB under the speech (30 dB for a man's body) and
up to 20 dB under it where the tongue moves fast. A man's voice covers that band. A woman's
does not, and in a stop's closure or a pause nothing does: it is heard there as a wet smacking
of the tongue (`pmvc.clicks.in_silences` counts it).

`--quiet-pauses`: the track has no breath in the recording's pauses (its silences of 120 ms and
more inside the speech). An encoder taught with `--silences` writes none before and after the
speech, but in a pause it lets a little run, unsteadily, and the voice murmurs there and pops
as the breath comes back.

`--closed-area`: the area (cm²) under which the voice takes a constriction for closed. CHIMERA
stops the flow through a constriction within one sample once its area is under 2e-4 cm². As the
tongue closes for a stop under a full breath, about 1 cm³/s is still running then, and its stop
is a click some 25 dB under the speech between 4 and 16 kHz, where a recording's closure is
silent. With 1e-6 the flow has died away by itself before it is cut (`pmvc.clicks.extra`
counts what is left).

`--sinus`: the volume (cm³, for an adult man's body) of the sinus on the voice's nose. CHIMERA's
nose has one sinus of 14 cm³, whose resonance takes a nasal's sound away near 700 Hz; a person's
two maxillary sinuses hold some 30 cm³. In women's recordings a nasal is little but its first
harmonic (the second lies 21 dB under it); in their renders the second lies 10 dB under it.
With 24 cm³ the dip is near 500 Hz and the second harmonic comes within 4 dB of the
recordings'. The men's nasals fit as they are, and with a larger sinus a verification model
finds the men's renders a little further from their speakers; the vowels do not change.

`--voice` also takes a pair of voices (a file with `low` and `high`, as `voices/v1.json`), each
with its body, source, noise, glottis, airway and nose, with the channels it holds (`set`: the
epilarynx, which no encoder moves) and its habits of posture (`shift`: what it adds to a
channel in every frame). A speaker the meta calls a man gets the low one and a woman the high
one, whatever their pitch: a man's median pitch can be 196 Hz and a woman's 173 Hz, and a body
picked by the pitch is the wrong one for them. A track whose body lies between the two (a dial
may put it there) gets every number that far between, in a straight line. A track says the
share of the high voice it was settled for (`voice`); its frames have the held channels as the
voice holds them, and `revoice` gives the tracks of an encoder that was not taught through the
voice its habits too (an encoder taught with such tracks writes them itself). The file's
`bright` says what one unit of brightness adds to each voice's settings (how abruptly the
glottis closes, how narrow the epilarynx is held): `--bright` turns that dial, and `revoice`
has the two other hands a listener may turn, `--pitch` (the contour that many semitones
higher) and `--share` (the same share of the high voice for every track, whoever spoke).

These options say how the voice plays an encoder's tracks, and an encoder taught through
a stand-in for CHIMERA's own way is not taught for another. To teach one for it: `revoice`
renders the tracks of a variant again as the options say (the tracks themselves stay), which
gives `pmvc.standin` (with `--voiced`) the renders to learn that voice from and `train` its
`--synthetic` pairs; and `train` takes the same options, so that the stand-in is told each
speaker's share of the fitted voice and the listener's check drives the voice that way.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray
from torch import nn

from . import clicks
from .audio import FRAME_RATE, Array, fill_pitch, load, pitch_track
from .calibrate import AIRWAY, GLOTTIS, SOURCE, make_voice, mixed
from .listener import FEATURE_RATE, LAYER, LISTEN_SR, MODEL, Listener, distance, to16k
from .paths import ROOT
from .standin import SKIP_HEARD, load_standin
from .transplant import F0_ADVANCE, START_MIN

TEST: tuple[str, ...] = ("FK", "FL", "MK", "ML")
VALID: tuple[str, ...] = ("FJ", "MJ")
# silence the encoder hears before a recording, for the voice to get ready in (s)
LEAD: float = START_MIN
# what a track runs past its recording (s), as `pmvc.warp` leaves it
TAIL: float = 0.2
# samples of the listener's rate per frame
HOP: int = LISTEN_SR // FRAME_RATE
# the pitch tracker's voicing, its pitch against the speaker's median (octaves), the level
AUX: int = 3
# with `--silences`: frames next to a recording's speech where the breath may already (or still) run
RUN_UP: int = 8
# with `--quiet-pauses`: a pause is a silence of the recording this many frames long or longer;
# the breath stops PAUSE_IN frames into it and is back PAUSE_OUT frames before it ends (the
# voice's breath rises with a time constant of 25 ms; back sooner, the voice sets in before
# the speaker's does)
PAUSE: int = 12
PAUSE_IN: int = 2
PAUSE_OUT: int = 3
# and the frame by which the network's output for the lead has settled
SETTLED: int = 4
# with `encode --voice`: the speaker's median pitch (Hz) up to which the voice has CHIMERA's own
# settings, and from which it has the fitted ones. NISQA's score for the renders falls with the
# speaker's pitch whatever the sex, and `pmvc.pitchmix` finds the fitted settings to help from
# about 175 Hz up and to harm at 150 Hz and below.
VOICE_PITCH: tuple[float, float] = (150.0, 175.0)

Audio = NDArray[np.float32]


def split_of(uid: str) -> str:
    """A TSP recording's split, by its speaker."""
    return "test" if uid[:2] in TEST else "valid" if uid[:2] in VALID else "train"


def split_in(d: Path, uid: str) -> str:
    """A recording's split: the one its meta names, or else TSP's by speaker."""
    return str(json.loads((d / "meta" / f"{uid}.json").read_text()).get("split") or split_of(uid))


def ids_in(d: Path, split: str) -> list[str]:
    """The recordings of a split, or all of them."""
    ids = sorted(p.stem for p in (d / "meta").glob("*.json"))
    return [i for i in ids if split in ("all", split_in(d, i))]


def aux_features(audio: Audio, f0: Array, n: int) -> Audio:
    """(n, AUX) for `audio` (16 kHz) and its pitch track (0 = unvoiced), both from frame 0."""
    f = np.zeros(n)
    f[: min(n, len(f0))] = f0[:n]
    voiced = f > 0
    out = np.zeros((n, AUX), dtype=np.float32)
    out[:, 0] = voiced
    if voiced.any():
        out[voiced, 1] = np.log2(f[voiced] / np.median(f[voiced]))
    x = np.zeros(n * HOP)
    x[: min(len(x), len(audio))] = audio[: len(x)]
    rms = np.sqrt((x.reshape(n, HOP) ** 2).mean(axis=1))
    out[:, 2] = np.clip(20.0 * np.log10(rms / (rms.max() + 1e-9) + 1e-9), -60.0, 0.0) / 60.0 + 1.0
    return out


@dataclass
class Pair:
    """What the encoder hears and what it should write: `frames` is None when only encoding."""

    uid: str
    audio: Audio
    aux: Audio
    frames: Audio | None
    # for a recording whose track runs LEAD behind it: what the listener hears in the
    # recording (frames, 768) and the span of its speech in those frames
    heard: torch.Tensor | None = None
    span: tuple[int, int] = (0, 0)
    # the speaker's body (how far towards the female one), and the share of the fitted voice
    # the speaker gets, for the stand-in
    female: float = 0.0
    voice: float = 0.0


def hear_recording(d: Path, uid: str, lead: float) -> tuple[Audio, Audio, Array]:
    """A recording after `lead` seconds of silence: its samples (16 kHz), the encoder's side
    features and its pitch track, all on the voice's clock."""
    x, _ = load(d / "real" / f"{uid}.wav", LISTEN_SR)
    f0 = np.load(d / "pitch" / f"{uid}.npy")
    k = round(lead * FRAME_RATE)
    n = round((len(x) / LISTEN_SR + lead + TAIL) * FRAME_RATE)
    audio = np.zeros(n * HOP, dtype=np.float32)
    audio[k * HOP : k * HOP + len(x)] = x[: n * HOP - k * HOP]
    f0 = np.concatenate([np.zeros(k), f0])
    return audio, aux_features(audio, f0, n), f0


def real_pair(d: Path, uid: str, target: str, listener: Listener | None = None, silences: bool = False, voice: float = 0.0) -> Pair:
    """A recording and the track `target` has for it, which runs its own `shift` behind the
    recording, or the meta's if it does not say. With a listener, also what it hears in the
    recording (for the closed-loop term, which needs the track to run LEAD behind).

    `silences`: the recording's silences are taught too: the track to copy has no breath
    outside the recording's speech (but for RUN_UP frames either side of it). Without this
    nothing says what a track should do where the recording is silent but the track of
    `target`, and an encoder that has only heard renders (A) starts to breathe as soon as a
    recording's room noise begins. The closed loop still compares over the speech alone: over
    the whole recording it has the encoder imitate the room's noise with a faint breath."""
    meta = json.loads((d / "meta" / f"{uid}.json").read_text())
    track = json.loads((d / target / f"{uid}.tracks.json").read_text())
    frames = np.array(track["frames"], dtype=np.float32)
    lead = track.get("shift", meta["shift"])
    if silences:
        k0, k1 = (round((lead + t) * FRAME_RATE) for t in meta["span"])
        breath = track["keys"].index("breath")
        frames[: max(0, k0 - RUN_UP), breath] = 0.0
        frames[k1 + RUN_UP :, breath] = 0.0
    audio, aux, _ = hear_recording(d, uid, lead)
    n = min(len(frames), len(aux))
    pair = Pair(uid, audio[: n * HOP], aux[:n], frames[:n], female=float(track["body"]["female"]), voice=voice)
    if listener is not None and track.get("shift") == LEAD:
        x, _ = load(d / "real" / f"{uid}.wav", LISTEN_SR)
        pair.heard = listener.hear(x).feats[0].half()
        pair.span = (int(meta["span"][0] * FEATURE_RATE), int(meta["span"][1] * FEATURE_RATE))
    return pair


def synthetic_pair(d: Path, uid: str, variant: str, voice: float = 0.0) -> Pair:
    """A render and the track it was rendered from."""
    meta = json.loads((d / "meta" / f"{uid}.json").read_text())
    track = json.loads((d / variant / f"{uid}.tracks.json").read_text())
    frames = np.array(track["frames"], dtype=np.float32)
    x48, sr = load(d / variant / f"{uid}.wav")
    x, _ = load(d / variant / f"{uid}.wav", LISTEN_SR)
    n = min(len(frames), len(x) // HOP)
    audio = x[: n * HOP].astype(np.float32)
    return Pair(uid, audio, aux_features(audio, pitch_track(x48, sr, meta["sex"]), n), frames[:n], female=float(track["body"]["female"]), voice=voice)


class Encoder(nn.Module):
    """wav2vec2's layers mixed, then the side features, then recurrent layers both ways."""

    def __init__(self, channels: int, hidden: int = 256, layers: int = 2, dropout: float = 0.3, tune: int = 0, ssl: str = MODEL) -> None:
        """`ssl`: the model the recording is heard through, any wav2vec2 or HuBERT on the hub."""
        super().__init__()
        from transformers import AutoModel, Wav2Vec2Model, logging

        logging.set_verbosity_error()
        # the default by its own class, which leaves the recogniser's character head behind
        self.ssl = (Wav2Vec2Model if ssl == MODEL else AutoModel).from_pretrained(ssl).eval().requires_grad_(False)
        # wav2vec2's last `tune` layers learn too
        self.tune = tune
        for layer in self.ssl.encoder.layers[len(self.ssl.encoder.layers) - tune :]:
            layer.requires_grad_(True)
        self.mix = nn.Parameter(torch.zeros(self.ssl.config.num_hidden_layers + 1))
        self.proj = nn.Sequential(nn.Linear(self.ssl.config.hidden_size, hidden), nn.Dropout(dropout))
        self.rnn = nn.LSTM(hidden + AUX, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, channels))

    def learned_state(self) -> dict[str, torch.Tensor]:
        """The weights that are not wav2vec2's own."""
        first = len(self.ssl.encoder.layers) - self.tune
        kept = tuple(f"ssl.encoder.layers.{i}." for i in range(first, first + self.tune))
        return {k: v for k, v in self.state_dict().items() if not k.startswith("ssl.") or k.startswith(kept)}

    def train(self, mode: bool = True) -> "Encoder":
        super().train(mode)
        self.ssl.eval()
        return self

    def forward(self, audio: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """`audio` (batch, samples), each clip normalised over its own length and padded with
        zeros; `aux` (batch, frames, AUX). Returns (batch, frames, channels), normalised."""
        with torch.set_grad_enabled(self.tune > 0 and self.training):
            states = torch.stack(self.ssl(audio, output_hidden_states=True).hidden_states)
        h = (torch.softmax(self.mix, 0)[:, None, None, None] * states).sum(0)
        # wav2vec2's frames are 20 ms apart: each stands for two of the track's
        h = self.proj(h).repeat_interleave(2, dim=1)
        n = aux.shape[1]
        h = torch.cat([h, h[:, -1:].expand(-1, max(0, n - h.shape[1]), -1)], dim=1)[:, :n]
        out, _ = self.rnn(torch.cat([h, aux], dim=-1))
        return self.head(out)


def collate(pairs: list[Pair], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """A batch: audio, side features, targets (if any) and which frames are real."""
    n = max(len(p.aux) for p in pairs)
    audio = np.zeros((len(pairs), n * HOP), dtype=np.float32)
    aux = np.zeros((len(pairs), n, AUX), dtype=np.float32)
    mask = np.zeros((len(pairs), n), dtype=bool)
    for i, p in enumerate(pairs):
        audio[i, : len(p.audio)] = (p.audio - p.audio.mean()) / (p.audio.std() + 1e-7)
        aux[i, : len(p.aux)] = p.aux
        mask[i, : len(p.aux)] = True
    frames = None
    if pairs[0].frames is not None:
        target = np.zeros((len(pairs), n, pairs[0].frames.shape[1]), dtype=np.float32)
        for i, p in enumerate(pairs):
            assert p.frames is not None
            target[i, : len(p.frames)] = p.frames
        frames = torch.from_numpy(target).to(device)
    return torch.from_numpy(audio).to(device), torch.from_numpy(aux).to(device), frames, torch.from_numpy(mask).to(device)


def batches(pairs: list[Pair], size: int, rng: np.random.Generator | None) -> list[list[Pair]]:
    """Batches of clips of about one length (wav2vec2 hears the padding); with `rng`, different
    ones every time and in a random order."""
    lengths = np.array([len(p.aux) for p in pairs], dtype=np.float64)
    if rng is not None:
        lengths += rng.uniform(-25.0, 25.0, len(pairs))
    order = np.argsort(lengths, kind="stable")
    out = [[pairs[i] for i in order[k : k + size]] for k in range(0, len(order), size)]
    return [out[i] for i in rng.permutation(len(out))] if rng is not None else out


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def ids_with(d: Path, variant: str, split: str) -> list[str]:
    ids = sorted(p.name.split(".")[0] for p in (d / variant).glob("*.tracks.json"))
    return [i for i in ids if split_in(d, i) == split]


def train(a: argparse.Namespace) -> None:
    d: Path = a.dir
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = pick_device()
    dirs = [d, *(Path(p) for p in filter(None, a.also.split(",")))]
    standin, heard_by = load_standin(d / "models" / f"{a.standin}.pt", device) if a.standin else (None, (MODEL, LAYER))
    if standin is not None:
        standin.train()
    # the listener the stand-in is for: the closed-loop term and the check both go by it
    listener = Listener(device, *heard_by) if a.standin or a.select == "listener" else None
    # how the voice will play the tracks, and each recording's share of the fitted voice
    decoder = Decoder.of(a)
    share = {dd: decoder.shares(dd) for dd in dirs}

    def every(variant: str, split: str) -> list[tuple[Path, str]]:
        return [(dd, i) for dd in dirs for i in ids_with(dd, variant, split)]

    with ThreadPoolExecutor(a.jobs) as pool:
        # the listener's own features are computed one recording at a time, in this thread
        fit = [] if a.no_real else [real_pair(dd, i, a.target, listener if standin is not None else None, a.silences, share[dd][i]) for dd, i in every(a.target, "train")]
        held = list(pool.map(lambda j: real_pair(j[0], j[1], a.target, None, a.silences, share[j[0]][j[1]]), every(a.target, "valid")))
        n_real = len(fit)
        for v in filter(None, a.synthetic.split(",")):
            fit += list(pool.map(lambda j, v=v: synthetic_pair(j[0], j[1], v, share[j[0]][j[1]]), [j for j in every(v, "train") if (j[0] / v / f"{j[1]}.wav").exists()]))
    keys: list[str] = json.loads(next(dd / a.target / f"{fit[0].uid}.tracks.json" for dd in dirs if (dd / a.target / f"{fit[0].uid}.tracks.json").exists()).read_text())["keys"]
    f0 = keys.index("f0")
    # a stand-in for a voice that goes by the speaker's pitch reads the speaker's share of it too
    voiced = standin is not None and standin.inp.in_features == len(keys) + 2
    assert voiced == (standin is not None and decoder.by_pitch), "the stand-in and --voice do not go together: one is for a fitted voice, the other is not"
    # every channel but pitch, on its spread over the tracks to learn; a channel that never
    # moves is written as its constant
    pooled = np.concatenate([p.frames for p in fit[: n_real or None] if p.frames is not None])
    mean, std = pooled.mean(axis=0), pooled.std(axis=0)
    # (nor is a channel the voice holds: the target's frames have it as the voice will)
    learned = [c for c in range(len(keys)) if c != f0 and std[c] > 1e-6 and keys[c] not in decoder.held]
    stats = {"keys": keys, "mean": mean.tolist(), "std": std.tolist(), "learned": learned,
             "lo": pooled.min(axis=0).tolist(), "hi": pooled.max(axis=0).tolist()}  # fmt: skip
    mu = torch.tensor(mean[learned], device=device)
    sd = torch.tensor(std[learned], device=device)
    model = Encoder(len(learned), a.hidden, tune=a.tune, ssl=a.ssl).to(device)
    tuned = [p for p in model.ssl.parameters() if p.requires_grad]
    own = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("ssl.")]
    params = own + tuned
    groups = [{"params": own, "lr": a.lr}] + ([{"params": tuned, "lr": a.tune_lr}] if tuned else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = a.epochs * len(batches(fit, a.batch, None))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.05)
    print(f"{n_real} recordings and {len(fit) - n_real} renders to fit, {len(held)} held out; {len(learned)} channels; {sum(p.numel() for p in params) / 1e6:.1f} M weights; {device}", flush=True)

    lo = torch.tensor(np.array(stats["lo"])[learned], device=device, dtype=torch.float32)
    hi = torch.tensor(np.array(stats["hi"])[learned], device=device, dtype=torch.float32)
    closed: list[float] = []

    def loss_of(batch: list[Pair]) -> torch.Tensor:
        audio, aux, frames, mask = collate(batch, device)
        assert frames is not None
        out = model(audio, aux)
        err = (out - (frames[..., learned] - mu) / sd).abs().mean(dim=-1)
        loss = (err * mask).sum() / mask.sum()
        real = [i for i, p in enumerate(batch) if p.heard is not None]
        if standin is not None and real and model.training:
            # the track the encoder writes, as the stand-in reads it: the learned channels in
            # the voice's units, the others and the pitch as the target has them, the body last
            track = frames[real].clone()
            track[..., f0] = torch.log2(track[..., f0].clamp(min=1.0))
            track = track.index_copy(-1, torch.tensor(learned, device=device), torch.minimum(torch.maximum(out[real] * sd + mu, lo), hi))
            # past a clip's end, its first frame: the voice at rest
            track = torch.where(mask[real][..., None], track, track[:, :1])
            who = torch.tensor([[[batch[i].female, batch[i].voice][: 2 if voiced else 1]] for i in real], device=device)
            said = standin(torch.cat([track, who.expand(-1, track.shape[1], -1)], dim=-1))[:, SKIP_HEARD:]
            total = out.new_zeros(())
            for row, i in enumerate(real):
                p = batch[i]
                assert p.heard is not None
                a0, a1 = p.span[0], min(p.span[1], len(p.heard), said.shape[1])
                total = total + (1.0 - (said[row, a0:a1] * p.heard[a0:a1].to(device).float()).sum(dim=-1)).mean()
            closed.append(float(total) / len(real))
            loss = loss + a.closed * total / len(real)
        return loss

    def held_loss() -> float:
        model.eval()
        with torch.no_grad():
            parts = [(float(loss_of(b)), sum(len(p.aux) for p in b)) for b in batches(held, a.batch, None)]
        return sum(v * n for v, n in parts) / max(1, sum(n for _, n in parts))

    loop = ClosedLoop([(dd, i) for dd in dirs for i in ids_in(dd, "valid")[:: a.stride]], device, a.jobs, listener, decoder) if a.select == "listener" else None
    out = d / "models"
    out.mkdir(exist_ok=True)
    log: list[dict[str, float]] = []
    best = float("inf")
    for epoch in range(a.epochs):
        model.train()
        total, count = 0.0, 0
        for batch in batches(fit, a.batch, rng):
            loss = loss_of(batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            total, count = total + loss.item(), count + 1
        row = {"epoch": epoch, "fit": total / count, "held": held_loss() if held else float("nan")}
        if closed:
            row["standin"] = float(np.mean(closed))
            closed.clear()
        if loop is not None and (epoch % a.every == a.every - 1 or epoch == a.epochs - 1):
            by_dir = loop(model, stats)
            row["listener"] = float(np.mean(list(by_dir.values())))
            if len(by_dir) > 1:
                row.update({f"listener:{name}": v for name, v in by_dir.items()})
        log.append(row)
        score = row.get("listener") if loop is not None else row["held"] if a.select == "loss" and held else -float(epoch)
        if score is not None and score < best:
            best = score
            torch.save({"state": model.learned_state(), "stats": stats, "hidden": a.hidden, "tune": a.tune, "ssl": a.ssl, "epoch": epoch, "args": {k: str(v) for k, v in vars(a).items()}}, out / f"{a.name}.pt")
        if "listener" in row or (loop is None and (epoch % 5 == 0 or epoch == a.epochs - 1)):
            heard = (f"  stand-in {row['standin']:.4f}" if "standin" in row else "") + (f"  listener {row['listener']:.4f}" if "listener" in row else "")
            heard += "".join(f"  {k[9:]} {v:.4f}" for k, v in row.items() if k.startswith("listener:"))
            print(f"epoch {epoch:3d}  fit {row['fit']:.4f}  held {row['held']:.4f}{heard}{'  *' if score == best else ''}", flush=True)
    (out / f"{a.name}.json").write_text(json.dumps({"select": a.select, "best": best, "epochs": log}, indent=1))


def load_model(path: Path, device: str) -> tuple[Encoder, dict[str, Any]]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = Encoder(len(ck["stats"]["learned"]), ck["hidden"], tune=ck.get("tune", 0), ssl=ck.get("ssl", MODEL))
    model.load_state_dict(ck["state"], strict=False)
    # whether it was taught the silences: its tracks then have no breath in the lead either
    stats = {**ck["stats"], "silences": ck.get("args", {}).get("silences") == "True"}
    return model.to(device).eval(), stats


@torch.no_grad()
def write_track(model: Encoder, stats: dict[str, Any], audio: Audio, aux: Audio, f0: Array, device: str) -> Array:
    """The control track for what the encoder hears: (frames, channels) in the voice's units."""
    a, x, _, _ = collate([Pair("", audio, aux, None)], device)
    learned: list[int] = stats["learned"]
    mean, std = np.array(stats["mean"]), np.array(stats["std"])
    n = len(aux)
    frames = np.tile(mean, (n, 1))
    frames[:, learned] = model(a, x)[0].cpu().numpy() * std[learned] + mean[learned]
    frames = np.clip(frames, np.array(stats["lo"]), np.array(stats["hi"]))
    if stats.get("silences"):
        # the lead is silence this encoder put there itself. The network's first frames, with
        # nothing before them to go by, stray from what it then settles to, and the jump from
        # the voice's rest to them is heard as a faint puff: the track starts settled, and
        # without breath up to the run-up
        frames[:SETTLED] = frames[SETTLED]
        frames[: round(LEAD * FRAME_RATE) - RUN_UP, stats["keys"].index("breath")] = 0.0
    # the tracker's contour, given early by the voice's glide, as in `pmvc.warp`
    contour = fill_pitch(f0)
    frames[:, stats["keys"].index("f0")] = np.interp(np.arange(n) / FRAME_RATE + F0_ADVANCE, np.arange(len(contour)) / FRAME_RATE, contour)
    return frames


def pauses(d: Path, uid: str) -> list[tuple[int, int]]:
    """A recording's pauses, in its own frames (first, one past the last): where, inside its
    speech, it stays `pmvc.clicks.QUIET` under its speech's level for PAUSE frames or more."""
    meta = json.loads((d / "meta" / f"{uid}.json").read_text())
    x, sr = load(d / "real" / f"{uid}.wav")
    lv = clicks.level(x, sr, clicks.WHOLE)
    per = round(1.0 / (FRAME_RATE * clicks.HOP))
    k0, k1 = round(meta["span"][0] / clicks.HOP), round(meta["span"][1] / clicks.HOP)
    lv = lv - float(np.percentile(lv[k0:k1], 95))
    quiet = lv[: len(lv) // per * per].reshape(-1, per).max(axis=1) < clicks.QUIET
    return [(a, b) for a, b in clicks.stretches(quiet) if b - a >= PAUSE and a > k0 // per and b < k1 // per]


def voice_share(pitch: float) -> float:
    """How much of the fitted voice a speaker of this median pitch gets: none up to
    VOICE_PITCH[0], all of it from VOICE_PITCH[1] up, and a share between (`pmvc.pitchmix`)."""
    lo, hi = VOICE_PITCH
    return float(np.clip(np.log(pitch / lo) / np.log(hi / lo), 0.0, 1.0))


def speaker_pitches(d: Path) -> dict[str, float]:
    """For each recording in `d`, the median pitch of its speaker over all of the speaker's
    recordings there: one voice for all of a speaker's recordings."""
    metas = {p.stem: json.loads(p.read_text()) for p in (d / "meta").glob("*.json")}
    by_speaker: dict[str, list[float]] = {}
    for meta in metas.values():
        by_speaker.setdefault(meta["speaker"], []).append(meta["f0Median"])
    return {uid: float(np.median(by_speaker[meta["speaker"]])) for uid, meta in metas.items()}


@dataclass
class Decoder:
    """How the voice plays a track: `fitted`, the settings `pmvc.calibrate` fitted to a high
    voice, for speakers with a high one (`--voice`); `wall_air`, whether the walls of the tract
    displace air as they move (`--no-wall-air`); `closed_area`, the area (cm²) under which a
    constriction counts as closed (`--closed-area`); `quiet_pauses`, whether an encoder's
    track gets no breath in the recording's pauses (`--quiet-pauses`); `sinus`, the volume
    (cm³, for an adult man's body) of the sinus on the nose (`--sinus`); and `family`, in place
    of `fitted`, a pair of voices, the low one for men and the high one for women (`--voice`
    with a file that has `low` and `high`). As it stands, CHIMERA's own way."""

    fitted: dict[str, float] | None = None
    wall_air: bool = True
    closed_area: float | None = None
    quiet_pauses: bool = False
    sinus: float | None = None
    family: dict[str, Any] | None = None
    # a pair of voices' dial: how bright the voice is, 0 as the pair has it (`--bright`)
    bright: float = 0.0

    @classmethod
    def of(cls, a: argparse.Namespace) -> "Decoder":
        voice: dict[str, Any] = json.loads(a.voice.read_text()) if a.voice else {}
        family: dict[str, Any] | None = {k: voice[k] for k in ("low", "high", "bright") if k in voice} if "low" in voice else None
        return cls(voice.get("settings"), not a.no_wall_air, a.closed_area, getattr(a, "quiet_pauses", False), getattr(a, "sinus", None), family, getattr(a, "bright", 0.0))

    @property
    def by_pitch(self) -> bool:
        """Whether the voice differs between speakers (and a stand-in is told a speaker's share)."""
        return self.fitted is not None or self.family is not None

    @property
    def held(self) -> tuple[str, ...]:
        """The channels the voice holds, whatever a track's frames say."""
        return tuple(self.family["low"].get("set", {})) if self.family else ()

    def share(self, pitch: float, female: float = 0.0) -> float:
        """How much of the high voice a speaker gets: of a pair of voices, as far as the body
        is the female one; of a fitted voice, by the speaker's median pitch."""
        if self.family:
            return float(np.clip(female, 0.0, 1.0))
        return voice_share(pitch) if self.fitted else 0.0

    def shares(self, d: Path) -> dict[str, float]:
        """For each recording in `d`, its speaker's share of the high voice."""
        if self.family:
            return {p.stem: self.share(0.0, 1.0 if json.loads(p.read_text())["sex"] == "F" else 0.0) for p in (d / "meta").glob("*.json")}
        return {uid: self.share(pitch) for uid, pitch in speaker_pitches(d).items()}

    def of_family(self, part: str, w: float) -> dict[str, float]:
        """A part of the pair's voice (`body`, `source`, ...) a share `w` of the way to the high one."""
        assert self.family is not None
        lo, hi = self.family["low"].get(part, {}), self.family["high"].get(part, {})
        out = {k: (1 - w) * lo[k] + w * hi[k] for k in lo}
        # the dial: what a unit of brightness adds to each voice's settings, all of which run from 0 to 1
        more = self.family.get("bright", {})
        for k in more.get("low", {}).get(part, {}) if self.bright else ():
            out[k] = float(np.clip(out[k] + self.bright * ((1 - w) * more["low"][part][k] + w * more["high"][part][k]), 0.0, 1.0))
        return {k: round(x, 5) for k, x in out.items()}

    def hold(self, frames: Array, keys: list[str], w: float, habits: bool = False) -> Array:
        """`frames` with the channels the voice holds as it holds them at share `w`. With
        `habits`, also with the voice's habits of posture (`shift`: what it adds to a channel in
        every frame). Those are for tracks written for another voice (`revoice`): an encoder
        taught with such tracks writes the habits itself."""
        for k, x in (self.of_family("set", w) if self.family else {}).items():
            frames[:, keys.index(k)] = x
        for k, x in (self.of_family("shift", w) if self.family and habits else {}).items():
            frames[:, keys.index(k)] += x
        return frames

    def settle(self, track: dict[str, Any], pitch: float) -> dict[str, Any]:
        """`track` with this voice's settings for its speaker: of a pair of voices by the
        track's body (or the share it was settled for before), of a fitted voice by the
        speaker's median pitch."""
        w = self.share(pitch, float(track.get("voice", track["body"].get("female", 0.0))))
        track = {**{k: v for k, v in track.items() if k not in ("noise", "glottis", "airway", "nose", "voice")}, "source": dict(track["source"])}
        if self.family:
            track["body"] = self.of_family("body", w)
            track["source"].update(self.of_family("source", w))
            track["noise"] = list(self.of_family("noise", w).values())
            track.update({part: self.of_family(part, w) for part in ("glottis", "airway", "nose")})
            track["voice"] = round(w, 4)
            if "frames" in track and (self.held or self.family["low"].get("shift")):
                track["frames"] = np.round(self.hold(np.array(track["frames"], dtype=np.float64), track["keys"], w, habits=True), 5).tolist()
        elif self.fitted and w > 0.0:
            s = mixed(self.fitted, w)
            track["source"].update({k: round(s[k], 5) for k in SOURCE})
            track["noise"] = [round(s[k], 5) for k in ("noiseGlottis", "noiseConstriction", "noiseTeeth")]
            track["glottis"] = {k: round(s[k], 5) for k in GLOTTIS}
            track["airway"] = {k: round(s[k], 5) for k in AIRWAY}
        if not self.wall_air:
            track["airway"] = {**track.get("airway", {}), "volumeSource": 0}
        if self.closed_area is not None:
            track["airway"] = {**track.get("airway", {}), "closedArea": self.closed_area}
        if self.sinus is not None:
            track["nose"] = {"sinusVolume": self.sinus}
        return track

    def track(self, d: Path, uid: str, frames: Array, keys: list[str], pitch: float | None = None) -> dict[str, Any]:
        """An encoder's `frames` as a track for the voice of a recording's speaker. All the
        voice is told of the speaker is in the recording's meta: the sex picks the body (and of
        a pair of voices, the voice), and the source idles at the median pitch (as
        `pmvc.transplant` sets them: the renders are the same to the sample). `pitch` is the
        speaker's median pitch over all of the speaker's recordings, which picks a fitted
        voice's share; without it the recording's own counts."""
        meta = json.loads((d / "meta" / f"{uid}.json").read_text())
        if self.quiet_pauses:
            lead = round(LEAD * FRAME_RATE)
            for first, last in pauses(d, uid):
                frames[lead + first + PAUSE_IN : lead + last - PAUSE_OUT, keys.index("breath")] = 0.0
        body = {"female": 1.0 if meta["sex"] == "F" else 0.0, "child": 0.0}
        frames = self.hold(frames, keys, body["female"])
        track = self.settle({"rate": float(FRAME_RATE), "body": body, "source": {"f0": round(meta["f0Median"], 2)}, "keys": keys}, pitch or meta["f0Median"])
        return {**track, "shift": LEAD, "frames": np.round(frames, 5).tolist()}


def render_track(track: dict[str, Any]) -> Array:
    v = make_voice(track, None)
    frames = np.array(track["frames"])
    v.set_track(frames, 0)
    return np.asarray(v.render((len(frames) - 1) * (48000 // FRAME_RATE)))


class ClosedLoop:
    """How close the voice comes to recordings when the encoder drives it: `pmvc.listener`'s
    feature distance between each recording and the render of the track written for it, over
    the recording's speech. It needs no track to compare with and no text."""

    def __init__(self, items: list[tuple[Path, str]], device: str, jobs: int, listener: Listener | None = None, decoder: Decoder | None = None) -> None:
        """`items`: (directory, id) of the recordings to check on. `decoder`: how the voice
        plays the tracks, if not CHIMERA's own way."""
        self.items, self.device, self.jobs = items, device, jobs
        self.decoder = decoder or Decoder()
        pitches = {d: speaker_pitches(d) for d in {d for d, _ in items}}
        self.pitches = [pitches[d][uid] for d, uid in items]
        self.listener = listener or Listener(device)
        self.inputs = [hear_recording(d, uid, LEAD) for d, uid in items]
        self.feats: list[torch.Tensor] = []
        self.spans: list[tuple[int, int]] = []
        for d, uid in items:
            x, _ = load(d / "real" / f"{uid}.wav", LISTEN_SR)
            span = json.loads((d / "meta" / f"{uid}.json").read_text())["span"]
            self.feats.append(self.listener.hear(x).feats[0])
            self.spans.append((int(span[0] * FEATURE_RATE), int(span[1] * FEATURE_RATE)))

    def __call__(self, model: "Encoder", stats: dict[str, Any]) -> dict[str, float]:
        """The mean distance over each directory's recordings, by the directory's name."""
        model.eval()
        tracks = [self.decoder.track(d, uid, write_track(model, stats, *inp, self.device), stats["keys"], pitch) for (d, uid), inp, pitch in zip(self.items, self.inputs, self.pitches)]
        with ThreadPoolExecutor(self.jobs) as pool:
            renders = list(pool.map(render_track, tracks))
        out: dict[str, list[float]] = {}
        for (d, _), x, feats, (a, b) in zip(self.items, renders, self.feats, self.spans):
            heard = self.listener.hear(to16k(x[round(LEAD * 48000) :])).feats[0]
            out.setdefault(d.name, []).append(float(distance(feats[a:b], heard[a:b]).mean()))
        return {name: float(np.mean(v)) for name, v in out.items()}


def encode(a: argparse.Namespace) -> None:
    d: Path = a.dir
    device = pick_device()
    model, stats = load_model((a.models or d / "models") / f"{a.name}.pt", device)
    ids = a.ids.split(",") if a.ids else ids_in(d, a.split)
    (d / a.out).mkdir(exist_ok=True)
    decoder = Decoder.of(a)
    pitches = speaker_pitches(d)

    def written(job: tuple[str, Array]) -> dict[str, Any]:
        uid, frames = job
        track = decoder.track(d, uid, frames, stats["keys"], pitches[uid])
        (d / a.out / f"{uid}.tracks.json").write_text(json.dumps(track))
        return track

    # a few hundred recordings at a time: a corpus's tracks do not all fit in memory as lists
    with ThreadPoolExecutor(a.jobs) as pool:
        for k in range(0, len(ids), 256):
            part = ids[k : k + 256]
            frames = [write_track(model, stats, *heard, device) for heard in pool.map(lambda uid: hear_recording(d, uid, LEAD), part)]
            tracks = list(pool.map(written, zip(part, frames)))
            if not a.no_render:
                for uid, x in zip(part, pool.map(render_track, tracks)):
                    sf.write(str(d / a.out / f"{uid}.wav"), x, 48000, subtype="PCM_16")
    print(f"{len(ids)} recordings → {d / a.out}")


def revoice(a: argparse.Namespace) -> None:
    """The tracks of one variant rendered again as the voice's options say: the same frames,
    under another voice."""
    d: Path = a.dir
    decoder = Decoder.of(a)
    pitches = speaker_pitches(d)
    ids = [i for i in sorted(p.name.split(".")[0] for p in (d / a.of).glob("*.tracks.json")) if a.split == "all" or split_in(d, i) == a.split]
    (d / a.out).mkdir(exist_ok=True)
    tracks = [json.loads((d / a.of / f"{uid}.tracks.json").read_text()) for uid in ids]
    for t in tracks:
        if a.share is not None:
            t["voice"] = a.share
        if a.pitch:
            # the whole contour that many semitones higher, and the pitch the source idles at
            up, f0 = 2.0 ** (a.pitch / 12.0), t["keys"].index("f0")
            t["frames"] = [[*row[:f0], row[f0] * up, *row[f0 + 1 :]] for row in t["frames"]]
            t["source"] = {**t["source"], "f0": round(t["source"]["f0"] * up, 2)}
    tracks = [decoder.settle(t, pitches[uid]) for t, uid in zip(tracks, ids)]
    if a.speakers:
        # each speaker's own body and glottis, as `pmvc.speaker` fitted them
        from .speaker import settled

        fitted = json.loads(a.speakers.read_text())["speakers"]
        who = [json.loads((d / "meta" / f"{uid}.json").read_text())["speaker"] for uid in ids]
        tracks = [settled(t, fitted[name]["settings"]) if name in fitted else t for t, name in zip(tracks, who)]
    with ThreadPoolExecutor(a.jobs) as pool:
        for uid, track, x in zip(ids, tracks, pool.map(render_track, tracks)):
            (d / a.out / f"{uid}.tracks.json").write_text(json.dumps(track))
            sf.write(str(d / a.out / f"{uid}.wav"), x, 48000, subtype="PCM_16")
    print(f"{len(tracks)} tracks of {a.of} → {d / a.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("train", "encode", "revoice"):
        p = sub.add_parser(name)
        p.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "transplant")
        p.add_argument("--name", type=str, default="encoder")
        p.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
        # how the voice plays the tracks
        p.add_argument("--voice", type=Path, default=None, help="pmvc.calibrate's file: speakers with a high voice (by their median pitch over the directory) are rendered under its settings")
        p.add_argument("--no-wall-air", action="store_true", help="the voice's walls displace no air as they move: no thumps where the recording is silent")
        p.add_argument("--closed-area", type=float, default=None, help="the area (cm²) under which the voice takes a constriction for closed, if not CHIMERA's 2e-4: 1e-6 keeps the tongue from clicking as it closes")
        p.add_argument("--sinus", type=float, default=None, help="the volume (cm³, for an adult man's body) of the sinus on the voice's nose, if not CHIMERA's 14: a larger one takes more of a nasal's sound above its lowest resonance")
        p.add_argument("--bright", type=float, default=0.0, help="with a pair of voices: how much brighter (or, negative, duller) than the pair has it, in the units of the voice file's `bright`")
        if name != "revoice":
            p.add_argument("--quiet-pauses", action="store_true", help="no breath in the recording's pauses (its silences of 120 ms and more inside the speech)")
    t = sub.choices["train"]
    t.add_argument("--target", type=str, default="simcopy")
    t.add_argument("--synthetic", type=str, default="simcopy", help="variants whose render → track pairs are added")
    t.add_argument("--no-real", action="store_true", help="fit on the renders alone")
    t.add_argument("--also", type=str, default="", help="more directories of recordings and renders, comma-separated")
    t.add_argument("--select", choices=["listener", "loss", "last"], default="listener", help="which epoch's weights to keep")
    t.add_argument("--every", type=int, default=5, help="epochs between the listener's checks")
    t.add_argument("--stride", type=int, default=2, help="the listener checks every so-many-th validation recording")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--standin", type=str, default="", help="a `pmvc.standin` model to close the loop through")
    t.add_argument("--closed", type=float, default=1.0, help="the weight of the closed-loop term")
    t.add_argument("--tune", type=int, default=0, help="how many of wav2vec2's last layers learn too")
    t.add_argument("--tune-lr", type=float, default=2e-5)
    t.add_argument("--ssl", type=str, default=MODEL, help="the model the encoder hears through, if not the listener's own")
    t.add_argument("--silences", action="store_true", help="teach the recordings' silences too: the track to copy has no breath outside the speech")
    t.add_argument("--seed", type=int, default=0)
    e = sub.choices["encode"]
    e.add_argument("--out", type=str, default="enc")
    e.add_argument("--split", choices=["test", "valid", "train", "all"], default="test")
    e.add_argument("--ids", type=str, default="")
    e.add_argument("--models", type=Path, default=None, help="where the encoder is, if not <dir>/models")
    e.add_argument("--no-render", action="store_true", help="write the tracks only: a corpus's, to learn from (`pmvc.tts`)")
    r = sub.choices["revoice"]
    r.add_argument("--speakers", type=Path, default=None, help="pmvc.speaker's file: each speaker's recordings are rendered under the settings fitted to the speaker")
    r.add_argument("--share", type=float, default=None, help="with a pair of voices: every track gets this share of the high voice (0 the low voice, 1 the high one), whoever spoke")
    r.add_argument("--pitch", type=float, default=0.0, help="the pitch contour that many semitones higher")
    r.add_argument("--of", type=str, required=True, help="the variant whose tracks are rendered again")
    r.add_argument("--out", type=str, required=True)
    r.add_argument("--split", choices=["test", "valid", "train", "all"], default="all")
    a = ap.parse_args()
    {"train": train, "encode": encode, "revoice": revoice}[a.command](a)


if __name__ == "__main__":
    main()
