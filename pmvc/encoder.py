"""The encoder: a recording in, a control track out, in one pass, with no text and no search.

With the voice as the decoder this is a codec whose latent is the control track.

usage: python -m pmvc.encoder train [dir] [--name encoder] [--target simcopy] [--synthetic simcopy]
                                    [--no-real] [--standin NAME] [--closed 1.0] [--tune 0]
                                    [--also DIR,...] [--ssl MODEL] [--silences]
                                    [--select listener|loss|last] [--epochs 40] [--hidden 256]
       python -m pmvc.encoder encode [dir] [--name encoder] [--out enc] [--split test] [--ids ...]
                                     [--models DIR] [--voice FILE]

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
    # the speaker's body, for the stand-in
    female: bool = False


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


def real_pair(d: Path, uid: str, target: str, listener: Listener | None = None, silences: bool = False) -> Pair:
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
    pair = Pair(uid, audio[: n * HOP], aux[:n], frames[:n], female=meta["sex"] == "F")
    if listener is not None and track.get("shift") == LEAD:
        x, _ = load(d / "real" / f"{uid}.wav", LISTEN_SR)
        pair.heard = listener.hear(x).feats[0].half()
        pair.span = (int(meta["span"][0] * FEATURE_RATE), int(meta["span"][1] * FEATURE_RATE))
    return pair


def synthetic_pair(d: Path, uid: str, variant: str) -> Pair:
    """A render and the track it was rendered from."""
    meta = json.loads((d / "meta" / f"{uid}.json").read_text())
    frames = np.array(json.loads((d / variant / f"{uid}.tracks.json").read_text())["frames"], dtype=np.float32)
    x48, sr = load(d / variant / f"{uid}.wav")
    x, _ = load(d / variant / f"{uid}.wav", LISTEN_SR)
    n = min(len(frames), len(x) // HOP)
    audio = x[: n * HOP].astype(np.float32)
    return Pair(uid, audio, aux_features(audio, pitch_track(x48, sr, meta["sex"]), n), frames[:n], female=meta["sex"] == "F")


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

    def every(variant: str, split: str) -> list[tuple[Path, str]]:
        return [(dd, i) for dd in dirs for i in ids_with(dd, variant, split)]

    with ThreadPoolExecutor(a.jobs) as pool:
        # the listener's own features are computed one recording at a time, in this thread
        fit = [] if a.no_real else [real_pair(dd, i, a.target, listener if standin is not None else None, a.silences) for dd, i in every(a.target, "train")]
        held = list(pool.map(lambda j: real_pair(j[0], j[1], a.target, None, a.silences), every(a.target, "valid")))
        n_real = len(fit)
        for v in filter(None, a.synthetic.split(",")):
            fit += list(pool.map(lambda j, v=v: synthetic_pair(j[0], j[1], v), [j for j in every(v, "train") if (j[0] / v / f"{j[1]}.wav").exists()]))
    keys: list[str] = json.loads(next(dd / a.target / f"{fit[0].uid}.tracks.json" for dd in dirs if (dd / a.target / f"{fit[0].uid}.tracks.json").exists()).read_text())["keys"]
    f0 = keys.index("f0")
    # every channel but pitch, on its spread over the tracks to learn; a channel that never
    # moves is written as its constant
    pooled = np.concatenate([p.frames for p in fit[: n_real or None] if p.frames is not None])
    mean, std = pooled.mean(axis=0), pooled.std(axis=0)
    learned = [c for c in range(len(keys)) if c != f0 and std[c] > 1e-6]
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
            female = torch.tensor([[[1.0 if batch[i].female else 0.0]] for i in real], device=device)
            said = standin(torch.cat([track, female.expand(-1, track.shape[1], -1)], dim=-1))[:, SKIP_HEARD:]
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

    loop = ClosedLoop([(dd, i) for dd in dirs for i in ids_in(dd, "valid")[:: a.stride]], device, a.jobs, listener) if a.select == "listener" else None
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


def voice_track(d: Path, uid: str, frames: Array, keys: list[str], fitted: dict[str, float] | None = None, pitch: float | None = None) -> dict[str, Any]:
    """`frames` as a track for the voice of a recording's speaker. All the voice is told of the
    speaker is in the recording's meta: the sex picks the body, and the source idles at the
    median pitch (as `pmvc.transplant` sets them: the renders are the same to the sample).

    `fitted`: settings `pmvc.calibrate` fitted to a high voice, for speakers with a high one.
    CHIMERA's own serve up to a median pitch of VOICE_PITCH[0], the fitted ones from
    VOICE_PITCH[1] up, and a mix of the two between (`pmvc.pitchmix`). `pitch` is the speaker's
    median pitch, so that all of a speaker's recordings get one voice; without it the
    recording's own counts."""
    meta = json.loads((d / "meta" / f"{uid}.json").read_text())
    body = {"female": 1.0 if meta["sex"] == "F" else 0.0, "child": 0.0}
    track: dict[str, Any] = {"rate": float(FRAME_RATE), "body": body, "source": {"f0": round(meta["f0Median"], 2)}, "keys": keys}
    lo, hi = VOICE_PITCH
    w = float(np.clip(np.log((pitch or meta["f0Median"]) / lo) / np.log(hi / lo), 0.0, 1.0))
    if fitted and w > 0.0:
        s = mixed(fitted, w)
        track["source"].update({k: round(s[k], 5) for k in SOURCE})
        track["noise"] = [round(s[k], 5) for k in ("noiseGlottis", "noiseConstriction", "noiseTeeth")]
        track["glottis"] = {k: round(s[k], 5) for k in GLOTTIS}
        track["airway"] = {k: round(s[k], 5) for k in AIRWAY}
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

    def __init__(self, items: list[tuple[Path, str]], device: str, jobs: int, listener: Listener | None = None) -> None:
        """`items`: (directory, id) of the recordings to check on."""
        self.items, self.device, self.jobs = items, device, jobs
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
        tracks = [voice_track(d, uid, write_track(model, stats, *inp, self.device), stats["keys"]) for (d, uid), inp in zip(self.items, self.inputs)]
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
    fitted: dict[str, float] | None = json.loads(a.voice.read_text())["settings"] if a.voice else None
    # each speaker's median pitch, over all of the speaker's recordings in the directory
    by_speaker: dict[str, list[float]] = {}
    speaker_of: dict[str, str] = {}
    if fitted:
        for p in (d / "meta").glob("*.json"):
            meta = json.loads(p.read_text())
            by_speaker.setdefault(meta["speaker"], []).append(meta["f0Median"])
            speaker_of[meta["id"]] = meta["speaker"]
    tracks: list[dict[str, Any]] = []
    for uid in ids:
        audio, aux, f0 = hear_recording(d, uid, LEAD)
        pitch = float(np.median(by_speaker[speaker_of[uid]])) if fitted else None
        tracks.append(voice_track(d, uid, write_track(model, stats, audio, aux, f0, device), stats["keys"], fitted, pitch))
        (d / a.out / f"{uid}.tracks.json").write_text(json.dumps(tracks[-1]))
    with ThreadPoolExecutor(a.jobs) as pool:
        for uid, x in zip(ids, pool.map(render_track, tracks)):
            sf.write(str(d / a.out / f"{uid}.wav"), x, 48000, subtype="PCM_16")
    print(f"{len(tracks)} recordings → {d / a.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("train", "encode"):
        p = sub.add_parser(name)
        p.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "transplant")
        p.add_argument("--name", type=str, default="encoder")
        p.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
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
    e.add_argument("--voice", type=Path, default=None, help="pmvc.calibrate's file: speakers with a high voice (by their median pitch over the directory) are rendered under its settings")
    a = ap.parse_args()
    train(a) if a.command == "train" else encode(a)


if __name__ == "__main__":
    main()
