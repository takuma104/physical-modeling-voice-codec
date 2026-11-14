"""The control for `pmvc.tts`: the same text-to-speech model, writing the latent of a neural
decoder instead of the voice's control track.

`pmvc.tts` finds that a model writing CHIMERA's control track needs little data and that its
mistakes sound like a person's. Is that the track's doing, or would any latent do? Here the
same text input, the same durations, the same layers and the same number of recordings go to
three latents that a neural network decodes:
  mel      a log mel spectrogram (100 bands, 93.75 frames a second) for the UnivNet vocoder:
           what a model of this shape was made for (FastSpeech 2)
  encodec  EnCodec's latent before it is quantized (128 values, 75 frames a second)
  mimi     Mimi's latent as its decoder takes it: the vector its 32 codebooks stand for
           (512 values, 12.5 frames a second). The encoder's output before quantization is
           in another space, and the decoder makes nothing of it.
All three come with `transformers`, run at 24 kHz and are used as they are.

usage: python -m pmvc.control copy [dir] --latent mel [--out copy_mel] [--split test] [--ids ...]
       python -m pmvc.control train [dir] --latent mel [--name ctl_mel] [--limit N] [--epochs 60]
                                    [--no-pitch]
       python -m pmvc.control speak [dir] [--name ctl_mel] [--models DIR] [--out ctl_mel]
                                    [--split test] [--ids ...] [--timing own|recording]
                                    [--sentences FILE]

`copy` is what the decoder can do at best: a recording into the latent and back.

`train` learns from the recordings `pmvc.tts train` learns from (those of <dir> that
`pmvc.align write` aligned), each heard as the encoder hears it (LEAD of silence, the
recording, TAIL), so that the latent's frames are on the alignment's clock. The model is
`pmvc.tts`'s, its frame layers at 100 frames a second as there; what they write is averaged
down to the latent's rate before the last layer. Two things differ, both in the control's
favour or neutral:
  pitch    the control track has a pitch channel, and `pmvc.tts` writes it like any other. A
           spectrogram has none, and FastSpeech 2 does not leave the pitch to it: a predictor
           says each frame's pitch, and the frame layers are told (the recording's in training,
           the predictor's own afterwards). The control has that predictor (`--no-pitch`: not).
  losses   the error against the recording's latent, each value on its spread; the pitch's;
           the durations'. There is no stand-in for the decoder to go through: the latent is
           the recording's own. (`pmvc.tts train --closed 0` is the model without its
           stand-in, for a like-for-like comparison.)
The weights kept are chosen as `pmvc.tts` chooses them: every `--every` epochs the validation
sentences are written with the speaker's durations and decoded, and `pmvc.listener` says how
far they are from the recordings.

`speak` is `pmvc.tts speak` for such a model: <dir>/<out>/<id>.wav at 24 kHz, and beside it a
note for `pmvc.judge` saying whether the timing is the model's own.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray
from scipy.signal import resample_poly
from torch import nn

from .align import aligned
from .audio import FRAME_RATE, fill_pitch, load
from .encoder import LEAD, ids_in, pick_device
from .japanese import Phone
from .listener import Listener, distance
from .paths import ROOT
from .standin import load_standin
from .tts import (
    OPENING,
    Sentence,
    Speaker,
    chunks,
    collate,
    fewer,
    fit_pausing,
    hear,
    pace,
    phonemes,
    phrased,
)

SR: int = 24000
Audio = NDArray[np.float32]


class Latent:
    """A neural decoder and the latent it takes: `rate` frames a second of `dims` values."""

    name: str
    rate: float
    dims: int

    def __init__(self, device: str) -> None:
        self.device = device

    def encode(self, x: Audio) -> NDArray[np.float32]:
        """A clip (24 kHz) as its latent, (frames, dims)."""
        raise NotImplementedError

    def decode(self, z: NDArray[np.float32]) -> Audio:
        """A latent as sound (24 kHz)."""
        raise NotImplementedError


class Mel(Latent):
    name, rate, dims = "mel", SR / 256, 100

    def __init__(self, device: str) -> None:
        from transformers import UnivNetFeatureExtractor, UnivNetModel, logging

        logging.set_verbosity_error()
        self.device = device
        self.features = UnivNetFeatureExtractor.from_pretrained("dg845/univnet-dev")
        self.model = UnivNetModel.from_pretrained("dg845/univnet-dev").to(device).eval()  # type: ignore[arg-type]

    def encode(self, x: Audio) -> NDArray[np.float32]:
        return np.asarray(self.features(x, sampling_rate=SR, pad_end=False, return_tensors="np")["input_features"][0], dtype=np.float32)

    @torch.no_grad()
    def decode(self, z: NDArray[np.float32]) -> Audio:
        # the vocoder's noise, the same every time
        noise = torch.randn(1, len(z), self.model.config.model_in_channels, generator=torch.Generator().manual_seed(0))
        y = self.model(input_features=torch.from_numpy(z)[None].to(self.device), noise_sequence=noise.to(self.device)).waveforms
        return y[0].float().cpu().numpy()


class Encodec(Latent):
    name, rate, dims = "encodec", SR / 320, 128

    def __init__(self, device: str) -> None:
        from transformers import EncodecModel, logging

        logging.set_verbosity_error()
        self.device = device
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device).eval()  # type: ignore[arg-type]

    @torch.no_grad()
    def encode(self, x: Audio) -> NDArray[np.float32]:
        return self.model.encoder(torch.from_numpy(x)[None, None].to(self.device))[0].T.float().cpu().numpy()

    @torch.no_grad()
    def decode(self, z: NDArray[np.float32]) -> Audio:
        return self.model.decoder(torch.from_numpy(z).T[None].to(self.device))[0, 0].float().cpu().numpy()


class Mimi(Latent):
    name, rate, dims = "mimi", 12.5, 512

    def __init__(self, device: str) -> None:
        from transformers import MimiModel, logging

        logging.set_verbosity_error()
        self.device = device
        self.model = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()  # type: ignore[arg-type]

    @torch.no_grad()
    def encode(self, x: Audio) -> NDArray[np.float32]:
        # what Mimi's decoder takes is not its encoder's output but the quantizer's: the
        # encoder's, coded with all 32 codebooks and turned back into a vector
        codes = self.model.encode(torch.from_numpy(x)[None, None].to(self.device)).audio_codes
        return self.model.quantizer.decode(codes)[0].T.float().cpu().numpy()

    @torch.no_grad()
    def decode(self, z: NDArray[np.float32]) -> Audio:
        return self.model.decoder(torch.from_numpy(z).T[None].to(self.device))[0].reshape(-1).float().cpu().numpy()


LATENTS: dict[str, type[Latent]] = {"mel": Mel, "encodec": Encodec, "mimi": Mimi}


def frames_of(n: int, rate: float) -> int:
    """How many frames of a latent `n` of the model's own frames make, as `tts.at_rate` counts."""
    return max(1, round(n / (FRAME_RATE / rate)))


def heard(d: Path, uid: str, n: int) -> Audio:
    """A recording as the encoder hears it, at 24 kHz: after LEAD of silence, `n` frames long."""
    x, _ = load(d / "real" / f"{uid}.wav", SR)
    out = np.zeros(n * (SR // FRAME_RATE), dtype=np.float32)
    k = round(LEAD * SR)
    out[k : k + len(x)] = x[: len(out) - k]
    return out


def sentences(d: Path, split: str) -> list[Sentence]:
    """The split's aligned recordings: their phonemes and how many frames each takes."""
    known = aligned(d)
    out: list[Sentence] = []
    for uid in ids_in(d, split):
        if uid in known:
            meta = json.loads((d / "meta" / f"{uid}.json").read_text())
            phones, lasting = known[uid]
            out.append(Sentence(uid, phones, np.array(lasting, dtype=np.int64), np.zeros((0, 0), dtype=np.float32), meta["speaker"], meta["sex"], meta["f0Median"]))
    return out


def learn_latents(d: Path, data: list[Sentence], latent: Latent, jobs: int) -> list[NDArray[np.float32]]:
    """Each recording's latent (kept in <dir>/latent for the next time) as its `track`, cut
    or held to the frames the model writes for it; and each recording's pitch contour at the
    model's own rate, in octaves."""
    (d / "latent" / latent.name).mkdir(parents=True, exist_ok=True)

    def one(s: Sentence) -> tuple[NDArray[np.float16] | None, Audio | None]:
        path = d / "latent" / latent.name / f"{s.uid}.npy"
        return (np.load(path), None) if path.exists() else (None, heard(d, s.uid, int(s.frames.sum())))

    def contour(s: Sentence) -> NDArray[np.float32]:
        n = int(s.frames.sum())
        f0 = np.concatenate([np.zeros(round(LEAD * FRAME_RATE)), np.load(d / "pitch" / f"{s.uid}.npy"), np.zeros(n)])[:n]
        return np.log2(fill_pitch(f0)).astype(np.float32)

    with ThreadPoolExecutor(jobs) as pool:
        for s, (z, x) in zip(data, pool.map(one, data)):
            if z is None:
                assert x is not None
                z = latent.encode(x).astype(np.float16)
                np.save(d / "latent" / latent.name / f"{s.uid}.npy", z)
            m = frames_of(int(s.frames.sum()), latent.rate)
            mine = z[:m].astype(np.float32)
            s.track = np.concatenate([mine, np.repeat(mine[-1:], m - len(mine), axis=0)]) if len(mine) < m else mine
        return list(pool.map(contour, data))


class Pitched(Speaker):
    """`pmvc.tts`'s model with FastSpeech 2's pitch predictor: each frame's pitch is said
    from its state before the frame layers, and the frame layers are told the pitch."""

    def __init__(self, phones: list[str], channels: int, rate: float, width: int = 256, text_layers: int = 4, frame_layers: int = 6, pausing: bool = False, pitch: bool = True) -> None:
        super().__init__(phones, channels, width, text_layers=text_layers, frame_layers=frame_layers, pausing=pausing, rate=rate)
        self.pitch = pitch
        if pitch:
            self.tone = nn.ModuleList([nn.Conv1d(width, width, 3, padding=1), nn.Conv1d(width, width, 3, padding=1)])
            self.tone_norm = nn.ModuleList([nn.LayerNorm(width), nn.LayerNorm(width)])
            self.tone_out = nn.Linear(width, 1)
            self.tone_drop = nn.Dropout(0.3)
            self.tone_in = nn.Linear(1, width)

    def write_with(self, h: torch.Tensor, frames: torch.Tensor, pitch: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """As `Speaker.write`, with the frames told `pitch` (batch, frames; normalised) or,
        without it, the model's own. Returns the latent and which of its frames are real, and
        the pitch the model says with which of the model's own frames are real."""
        x, real, total = self.states(h, frames)
        said = x.new_zeros(x.shape[:2])
        if self.pitch:
            g = x
            for conv, norm in zip(self.tone, self.tone_norm):
                g = self.tone_drop(norm(torch.relu(conv((g * real[..., None]).transpose(1, 2)).transpose(1, 2))))
            said = self.tone_out(g)[..., 0]
            x = x + self.tone_in((said if pitch is None else pitch)[..., None])
        out, there = self.finish(x, real, total)
        return out, there, said, real


@torch.no_grad()
def write_latent(model: Pitched, norm: dict[str, Any], phones: list[Phone], device: str, frames: NDArray[np.int64] | None = None) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """The latent for a sentence, and how many of the model's frames each phoneme got:
    `frames` if given (the speaker's own timing), or else the model's."""
    ids = torch.tensor([[p.ids(model.phones) for p in phones]], device=device)
    h, lasting = model.read(ids, torch.ones(1, len(phones), dtype=torch.bool, device=device))
    if frames is None:
        scale = np.array([1.0 if k in (0, len(phones) - 1) else norm["pace"][1] if p.silent else norm["pace"][0] for k, p in enumerate(phones)])
        frames = np.maximum(1, np.round(np.exp(lasting[0].cpu().numpy()) * scale)).astype(np.int64)
        frames[0] = max(frames[0], OPENING)
    out, _, _, _ = model.write_with(h, torch.from_numpy(frames).to(device)[None])
    return out[0].cpu().numpy() * np.array(norm["std"], dtype=np.float32) + np.array(norm["mean"], dtype=np.float32), frames


def to16k(x: Audio) -> Audio:
    return resample_poly(x, 2, 3).astype(np.float32)


def train(a: argparse.Namespace) -> None:
    d: Path = a.dir
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = pick_device()
    latent = LATENTS[a.latent](device)
    # the listener `pmvc.tts`'s check goes by: the one its stand-in is for
    _, heard_by = load_standin(a.models / f"{a.standin}.pt", "cpu")
    listener = Listener(device, *heard_by)
    fit = sentences(d, "train")
    held = sentences(d, "valid")
    phones = sorted({p.p for s in fit for p in s.phones})
    held = [s for s in held if all(p.p in phones for p in s.phones)]
    fit = fewer(fit, a.limit)
    pitches = dict(zip((s.uid for s in (*fit, *held)), learn_latents(d, [*fit, *held], latent, a.jobs)))
    hear(d, held, listener, a.jobs)

    pooled = np.concatenate([s.track for s in fit[:: max(1, len(fit) // 2000)]]).astype(np.float64)
    octaves = np.concatenate([pitches[s.uid] for s in fit[:: max(1, len(fit) // 2000)]]).astype(np.float64)
    norm: dict[str, Any] = {"mean": pooled.mean(axis=0).tolist(), "std": np.maximum(pooled.std(axis=0), 1e-5).tolist(), "pitch": [float(octaves.mean()), float(octaves.std())], "pace": [1.0, 1.0]}
    mu = torch.tensor(norm["mean"], device=device, dtype=torch.float32)
    sd = torch.tensor(norm["std"], device=device, dtype=torch.float32)

    model = Pitched(phones, latent.dims, latent.rate, a.width, a.text_layers, a.frame_layers, pitch=not a.no_pitch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(chunks(fit, a.batch, None))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    print(f"{len(fit)} recordings to fit ({sum(int(s.frames.sum()) for s in fit) / FRAME_RATE / 3600:.1f} hours), {len(held)} held out; {latent.name}: {latent.dims} values, {latent.rate:g} frames a second; "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M weights; {device}", flush=True)  # fmt: skip

    def losses(batch: list[Sentence]) -> dict[str, torch.Tensor]:
        ids, frames, real, _ = collate(batch, phones, device)
        h, lasting = model.read(ids, real)
        n = int(frames.sum(dim=1).max())
        told = torch.zeros(len(batch), n, device=device)
        for i, s in enumerate(batch):
            told[i, : len(pitches[s.uid])] = (torch.from_numpy(pitches[s.uid]).to(device) - norm["pitch"][0]) / norm["pitch"][1]
        out, there, said, own = model.write_with(h, frames, told)
        target = torch.zeros(len(batch), out.shape[1], latent.dims, device=device)
        for i, s in enumerate(batch):
            target[i, : len(s.track)] = torch.from_numpy(s.track).to(device)
        parts = {
            "latent": (((out - (target - mu) / sd).abs().mean(dim=-1)) * there).sum() / there.sum(),
            "durations": (((lasting - frames.clamp(min=1).float().log()) ** 2) * real).sum() / real.sum(),
        }
        if model.pitch:
            parts["pitch"] = ((said - told).abs() * own).sum() / own.sum()
        return parts

    def check() -> float:
        """How far from the recordings the validation sentences are, written with the
        speaker's durations and decoded: `pmvc.listener`'s distance over the speech."""
        model.eval()
        out: list[float] = []
        for s in held[:: a.stride]:
            assert s.heard is not None
            z, _ = write_latent(model, norm, s.phones, device, s.frames)
            x = latent.decode(z)
            feats = listener.hear(to16k(x[round(LEAD * SR) :])).feats[0]
            b0, b1 = s.span
            out.append(float(distance(torch.from_numpy(s.heard[b0:b1].astype(np.float32)), feats[b0:b1]).mean()))
        return float(np.mean(out))

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
            sum(parts.values()).backward()  # type: ignore[union-attr]
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
            row["listener"] = check()
            if row["listener"] < best:
                best = row["listener"]
                norm["pace"] = list(pace(model, fit[:: max(1, len(fit) // 1000)], device))
                torch.save({"state": model.state_dict(), "phones": phones, "latent": latent.name, "norm": norm, "width": a.width, "text_layers": a.text_layers, "frame_layers": a.frame_layers,
                            "pitch": model.pitch, "epoch": epoch, "args": {k: str(v) for k, v in vars(a).items()}}, out_dir / f"{a.name}.pt")  # fmt: skip
        log.append(row)
        print(f"epoch {epoch:3d}  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "epoch") + ("  *" if row.get("listener") == best else ""), flush=True)
        (out_dir / f"{a.name}.json").write_text(json.dumps({"best": best, "epochs": log}, indent=1))
    # where the speaker pauses without a comma, as `pmvc.tts` teaches it
    ck = torch.load(out_dir / f"{a.name}.pt", map_location="cpu", weights_only=False)
    model, _, _ = load_model(out_dir / f"{a.name}.pt", device)
    fit_pausing(model, fit, held, device)
    ck["state"], ck["pausing"] = model.state_dict(), True
    torch.save(ck, out_dir / f"{a.name}.pt")


def load_model(path: Path, device: str) -> tuple[Pitched, dict[str, Any], str]:
    """The model, what its latent is measured against, and the latent's name."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    latent = LATENTS[ck["latent"]]
    model = Pitched(ck["phones"], latent.dims, latent.rate, ck["width"], ck["text_layers"], ck["frame_layers"], pausing=ck.get("pausing", False), pitch=ck["pitch"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), ck["norm"], ck["latent"]


def note(path: Path, **what: Any) -> None:
    """Beside a clip, what `pmvc.judge` has to know of it: how far it runs behind the
    recording, or that its timing is its own (`free`)."""
    path.with_suffix(".tracks.json").write_text(json.dumps(what))


def speak(a: argparse.Namespace) -> None:
    d: Path = a.dir
    device = pick_device()
    model, norm, name = load_model((a.models or d / "models") / f"{a.name}.pt", device)
    latent = LATENTS[name](device)
    if a.sentences:
        (d / "meta").mkdir(parents=True, exist_ok=True)
        for k, text in enumerate(filter(None, (line.strip() for line in a.sentences.read_text(encoding="utf-8").splitlines())), 1):
            path = d / "meta" / f"{a.set}_{k:03d}.json"
            if not path.exists():
                path.write_text(json.dumps({"id": path.stem, "speaker": a.set, "sex": "M", "lang": "ja", "text": text, "split": "test"}, ensure_ascii=False, indent=1))
    ids = a.ids.split(",") if a.ids else [i for i in ids_in(d, a.split) if json.loads((d / "meta" / f"{i}.json").read_text()).get("lang") == "ja"]
    timed = aligned(d) if a.timing == "recording" else {}
    (d / a.out).mkdir(exist_ok=True)
    for uid in ids:
        if timed:
            phones, frames = timed[uid]
            z, _ = write_latent(model, norm, phones, device, np.array(frames, dtype=np.int64))
        else:
            z, _ = write_latent(model, norm, phrased(model, phonemes(model, json.loads((d / "meta" / f"{uid}.json").read_text())["text"]), device), device)
        sf.write(str(d / a.out / f"{uid}.wav"), latent.decode(z), SR, subtype="PCM_16")
        # with the speaker's timing the clip runs LEAD behind the recording, as an encoder's render does
        note(d / a.out / f"{uid}.wav", **({"shift": LEAD} if timed else {"shift": 0.0, "free": True}))
    print(f"{len(ids)} sentences → {d / a.out}")


def copy(a: argparse.Namespace) -> None:
    d: Path = a.dir
    latent = LATENTS[a.latent](pick_device())
    ids = a.ids.split(",") if a.ids else ids_in(d, a.split)
    out = a.out or f"copy_{a.latent}"
    (d / out).mkdir(exist_ok=True)
    for uid in ids:
        x, _ = load(d / "real" / f"{uid}.wav", SR)
        sf.write(str(d / out / f"{uid}.wav"), latent.decode(latent.encode(x.astype(np.float32))), SR, subtype="PCM_16")
        note(d / out / f"{uid}.wav", shift=0.0)
    print(f"{len(ids)} recordings → {d / out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c, t, s = sub.add_parser("copy"), sub.add_parser("train"), sub.add_parser("speak")
    for p in (c, t, s):
        p.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "hfcm")
        p.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    for p in (c, t):
        p.add_argument("--latent", choices=sorted(LATENTS), required=True)
    for p in (c, s):
        p.add_argument("--split", choices=["test", "valid", "train", "all"], default="test")
        p.add_argument("--ids", type=str, default="")
    c.add_argument("--out", type=str, default="")
    t.add_argument("--name", type=str, default="")
    t.add_argument("--models", type=Path, default=ROOT / "out" / "transplant" / "models", help="where the stand-in is that names the listener")
    t.add_argument("--standin", type=str, default="standin_2c")
    t.add_argument("--no-pitch", action="store_true", help="no pitch predictor: the latent has to carry the pitch alone")
    t.add_argument("--limit", type=int, default=0, help="learn from only so many of the recordings")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--every", type=int, default=5)
    t.add_argument("--stride", type=int, default=2)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--width", type=int, default=256)
    t.add_argument("--text-layers", type=int, default=4)
    t.add_argument("--frame-layers", type=int, default=6)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--seed", type=int, default=0)
    s.add_argument("--name", type=str, required=True)
    s.add_argument("--models", type=Path, default=None, help="where the model is, if not <dir>/models")
    s.add_argument("--out", type=str, default="")
    s.add_argument("--timing", choices=["own", "recording"], default="own")
    s.add_argument("--sentences", type=Path, default=None)
    s.add_argument("--set", type=str, default="say")
    a = ap.parse_args()
    if a.command == "train":
        a.name = a.name or f"ctl_{a.latent}"
    if a.command == "speak":
        a.out = a.out or a.name
    {"copy": copy, "train": train, "speak": speak}[a.command](a)


if __name__ == "__main__":
    main()
