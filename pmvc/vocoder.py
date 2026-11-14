"""A stronger control for `pmvc.control`'s mel latent: its vocoder, tuned on what the
text-to-speech model writes.

A vocoder has learned from the spectrograms of recordings. What a model writes is not one of
those: where the model is unsure it writes the average of what could be there, and the vocoder
makes a dull, buzzing sound of it. The usual cure (Tacotron 2; HiFi-GAN's last table) is to
tune the vocoder on the model's own spectrograms of the recordings, written with each
recording's durations and pitch so that they line up with its sound. `pmvc.control write`
makes those, each recording written by a model that did not learn from it.

usage: python -m pmvc.vocoder tune [dir] --model ctlmel200 [--name ctlmel200v] [--steps 5000]
                                   [--every 250] [--critics out/univnet/univ_c32.pt]
                                   [--written ctlmel200] [--quality N]

The vocoder is trained on as it was trained: against its discriminators, with UnivNet's loss
(Jang et al., 2021): the spectral error at three resolutions (2.5 times) and the critics'
verdict. `transformers`' UnivNet (`dg845/univnet-dev`) is the generator alone. The
discriminators it was trained against for a million steps (three that see the wave's
spectrogram at a resolution, five that see the wave folded at a period) are in the checkpoint
its authors released, UnivNet-c32 of github.com/maum-ai/univnet (a Google Drive file; only its
tensors are read): `--critics`. Without them a vocoder tuned on the spectral error alone
sounds worse than before, and new critics taught for ten thousand steps wreck it: both were
tried.

The recordings are those the model <dir>/models/<model>.pt learned from, and no others: the
input what `pmvc.control write` wrote for a stretch of a recording, the target the stretch.
Every `--every` steps the validation recordings say how it is going:
  spectral  the spectral error on what the model wrote for them with their own pitch
  listener  `pmvc.listener`'s distance to them of what the model wrote with its own pitch
  quality   NISQA's opinion of the first `--quality` of those (0: not asked)
The vocoder kept is the one with the highest `quality`, or without it the lowest `spectral`
(the vocoder as it came among them, so that tuning cannot make it worse by that measure):
<dir>/models/<name>.pt, for `pmvc.control speak --vocoder <name>`.
"""

import argparse
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .control import SR, Mel, heard, sentences, to16k
from .encoder import LEAD, pick_device
from .listener import Listener, distance
from .paths import ROOT
from .standin import load_standin
from .tts import Sentence, fewer, hear

HOP: int = 256
STRETCH: int = 64  # frames of a spectrogram in a training stretch: 0.68 s
RESOLUTIONS: tuple[tuple[int, int, int], ...] = ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))
PERIODS: tuple[int, ...] = (2, 3, 5, 7, 11)
SLOPE: float = 0.2


def magnitude(x: torch.Tensor, fft: int, hop: int, win: int, hann: bool = True) -> torch.Tensor:
    """The magnitude spectrogram of waves (batch, samples): (batch, bins, frames). Without
    `hann` the window is flat, as UnivNet's discriminators were given it."""
    pad = (fft - hop) // 2
    window = torch.hann_window(win, device=x.device) if hann else torch.ones(win, device=x.device)
    s = torch.stft(F.pad(x, (pad, pad), mode="reflect"), fft, hop, win, window, center=False, return_complex=True)
    return torch.sqrt(s.real**2 + s.imag**2 + 1e-7)


def spectral(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """UnivNet's (and Parallel WaveGAN's) spectral error: at each resolution how far the
    magnitudes are apart, as a share of the real ones and in their logarithms."""
    total = fake.new_zeros(())
    for fft, hop, win in RESOLUTIONS:
        a, b = magnitude(fake, fft, hop, win), magnitude(real, fft, hop, win)
        total = total + torch.linalg.norm(b - a) / torch.linalg.norm(b) + (b.log() - a.log()).abs().mean()
    return total / len(RESOLUTIONS)


class Periodic(nn.Module):
    """A critic of the wave folded every `period` samples (HiFi-GAN's)."""

    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period
        widths = (1, 64, 128, 256, 512)
        self.convs = nn.ModuleList([weight_norm(nn.Conv2d(a, b, (5, 1), (3, 1), padding=(2, 0))) for a, b in pairwise(widths)])
        self.convs.append(weight_norm(nn.Conv2d(512, 1024, (5, 1), 1, padding=(2, 0))))
        self.last = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.period:
            x = F.pad(x, (0, self.period - x.shape[-1] % self.period), mode="reflect")
        h = x.view(len(x), 1, -1, self.period)
        for conv in self.convs:
            h = F.leaky_relu(conv(h), SLOPE)
        return self.last(h).flatten(1)


class Resolved(nn.Module):
    """A critic of the wave's magnitude spectrogram at one resolution (UnivNet's)."""

    def __init__(self, fft: int, hop: int, win: int) -> None:
        super().__init__()
        self.resolution = (fft, hop, win)
        self.convs = nn.ModuleList(
            [weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4)))]
            + [weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))) for _ in range(3)]
            + [weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1)))]
        )
        self.last = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = magnitude(x, *self.resolution, hann=False)[:, None]
        for conv in self.convs:
            h = F.leaky_relu(conv(h), SLOPE)
        return self.last(h).flatten(1)


class Critics(nn.Module):
    """UnivNet's discriminators."""

    def __init__(self) -> None:
        super().__init__()
        self.all = nn.ModuleList([*(Resolved(*r) for r in RESOLUTIONS), *(Periodic(p) for p in PERIODS)])

    def take(self, path: Path) -> None:
        """Take the discriminators of UnivNet's own checkpoint (maum-ai/univnet's names for
        them: MRD and MPD, `conv_post`, a weight as its length `weight_g` and direction
        `weight_v`). Only tensors are read from the file."""
        theirs: dict[str, torch.Tensor] = torch.load(path, map_location="cpu", weights_only=True)["model_d"]
        leaves = {"bias": "bias", "weight_g": "parametrizations.weight.original0", "weight_v": "parametrizations.weight.original1"}
        mine: dict[str, torch.Tensor] = {}
        for name, value in theirs.items():
            kind, _, k, *layer, leaf = name.split(".")
            i = int(k) + (0 if kind == "MRD" else len(RESOLUTIONS))
            mine[f"all.{i}.{'.'.join(layer).replace('conv_post', 'last')}.{leaves[leaf]}"] = value
        self.load_state_dict(mine)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Each critic's verdicts on waves (batch, samples): 1 a recording, 0 not."""
        return [critic(x) for critic in self.all]


def told_apart(critics: Critics, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """The critics' loss: how far they are from telling the two apart."""
    return torch.stack([((1 - r) ** 2).mean() + (f**2).mean() for r, f in zip(critics(real), critics(fake))]).mean()


def fooled(critics: Critics, fake: torch.Tensor) -> torch.Tensor:
    """The vocoder's loss from the critics: how far they are from taking its wave for a recording."""
    return torch.stack([((1 - f) ** 2).mean() for f in critics(fake)]).mean()


def sing(mel: Mel, z: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    """The vocoder's waves (batch, samples) for spectrograms (batch, frames, bands), with
    gradients; the noise it starts from is the same every time if `seed` is given."""
    gen = None if seed is None else torch.Generator().manual_seed(seed)
    noise = torch.randn(len(z), z.shape[1], mel.model.config.model_in_channels, generator=gen).to(z.device)
    return mel.model(input_features=z, noise_sequence=noise).waveforms


class Stretches:
    """The recordings a model learned from and what was written for them, in memory: random
    stretches of the two, lined up. The stretches are of the recording itself: the silence
    that `control.heard` puts before and after it is all zeros, which no microphone hears, and
    a vocoder made to copy that learns little else."""

    def __init__(self, d: Path, written: Path, data: list[Sentence], jobs: int, rng: np.random.Generator) -> None:
        def one(s: Sentence) -> tuple[NDArray[np.float16], NDArray[np.int16], int, int]:
            z = np.load(written / f"{s.uid}.npy")
            wave = heard(d, s.uid, int(s.frames.sum()))
            x = (np.concatenate([wave, np.zeros(max(0, len(z) * HOP - len(wave)), dtype=np.float32)])[: len(z) * HOP] * 32767).astype(np.int16)
            sounding = np.flatnonzero(x)
            first, last = (int(sounding[0]) // HOP, int(sounding[-1]) // HOP + 1) if len(sounding) else (0, len(z))
            return z, x, first, max(first, min(last, len(z)) - STRETCH)

        with ThreadPoolExecutor(jobs) as pool:
            self.pairs = [p for p in pool.map(one, data) if len(p[0]) >= STRETCH]
        self.rng = rng

    def hours(self) -> float:
        return sum(len(x) for _, x, _, _ in self.pairs) / SR / 3600

    def batch(self, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        zs, xs = torch.zeros(n, STRETCH, self.pairs[0][0].shape[1]), torch.zeros(n, STRETCH * HOP)
        for i in range(n):
            z, x, first, last = self.pairs[self.rng.integers(len(self.pairs))]
            k = min(int(self.rng.integers(first, last + 1)), len(z) - STRETCH)
            zs[i] = torch.from_numpy(z[k : k + STRETCH].astype(np.float32))
            xs[i] = torch.from_numpy(x[k * HOP : (k + STRETCH) * HOP].astype(np.float32) / 32767)
        return zs.to(device), xs.to(device)


def tune(a: argparse.Namespace) -> None:
    d: Path = a.dir
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = pick_device()
    mel = Mel(device)
    judges = Critics()
    judges.take(a.critics)
    judges.to(device)
    written = d / "written" / (a.written or a.model)
    limit = int(torch.load(d / "models" / f"{a.model}.pt", map_location="cpu", weights_only=False)["args"]["limit"])
    fit = fewer(sentences(d, "train"), limit)
    held = [s for s in sentences(d, "valid") if (written / f"{s.uid}.own.npy").exists()][:: a.stride]
    _, heard_by = load_standin(a.models / f"{a.standin}.pt", "cpu")
    listener = Listener(device, *heard_by)
    hear(d, held, listener, a.jobs)
    data = Stretches(d, written, fit, a.jobs, rng)
    theirs = [(z, x) for z, x, _, _ in Stretches(d, written, held, a.jobs, rng).pairs]
    own = [torch.from_numpy(np.load(written / f"{s.uid}.own.npy").astype(np.float32)) for s in held]
    opt_v = torch.optim.AdamW(mel.model.parameters(), lr=a.lr, betas=(0.5, 0.9))
    opt_c = torch.optim.AdamW(judges.parameters(), lr=a.critics_lr, betas=(0.5, 0.9))
    print(f"{len(data.pairs)} recordings to tune on ({data.hours() * 60:.1f} minutes), {len(held)} held out; {device}", flush=True)

    @torch.no_grad()
    def check() -> dict[str, float]:
        mel.model.eval()
        row: dict[str, float] = {}
        row["spectral"] = float(np.mean([spectral(sing(mel, torch.from_numpy(z.astype(np.float32))[None].to(device), 0), torch.from_numpy(x.astype(np.float32) / 32767)[None].to(device)).item() for z, x in theirs]))
        clips = [sing(mel, z[None].to(device), 0)[0].cpu().numpy() for z in own]
        far: list[float] = []
        for s, x in zip(held, clips):
            assert s.heard is not None
            b0, b1 = s.span
            feats = listener.hear(to16k(x[round(LEAD * SR) :])).feats[0]
            far.append(float(distance(torch.from_numpy(s.heard[b0:b1].astype(np.float32)), feats[b0:b1]).mean()))
        row["listener"] = float(np.mean(far))
        if a.quality:
            from .natural import score

            with tempfile.TemporaryDirectory() as tmp:
                paths = [Path(tmp) / f"{k}.wav" for k in range(min(a.quality, len(clips)))]
                for path, x in zip(paths, clips):
                    sf.write(str(path), x[round(LEAD * SR) :], SR, subtype="PCM_16")
                row["quality"] = float(np.mean([r["quality"] for r in score(paths, natural=False)]))
        mel.model.train()
        return row

    def better(row: dict[str, float], best: dict[str, float]) -> bool:
        return row["quality"] > best["quality"] if a.quality else row["spectral"] < best["spectral"]

    def keep(step: int, to: str = "") -> None:
        torch.save({"state": mel.model.state_dict(), "step": step, "model": a.model, "args": {k: str(v) for k, v in vars(a).items()}}, out / f"{to or name}.pt")

    out = d / "models"
    name = a.name or f"{a.model}v"
    # the vocoder as it came is the one to beat
    best: dict[str, float] = {"step": 0, **check()}
    log: list[dict[str, float]] = [best]
    keep(0)
    print("step      0  " + "  ".join(f"{k} {v:.4f}" for k, v in best.items() if k != "step"), flush=True)
    mel.model.train()
    sums: dict[str, float] = {}
    for step in range(1, a.steps + 1):
        z, real = data.batch(a.batch, device)
        fake = sing(mel, z)
        apart = told_apart(judges, real, fake.detach())
        opt_c.zero_grad()
        apart.backward()
        opt_c.step()
        error = spectral(fake, real)
        fool = fooled(judges, fake)
        opt_v.zero_grad()
        (a.spectral * error + a.verdict * fool).backward()
        opt_v.step()
        for k, v in (("critics", apart), ("error", error), ("fooled", fool)):
            sums[k] = sums.get(k, 0.0) + v.item()
        if step % a.every == 0 or step == a.steps:
            row = {"step": step, **{k: v / (a.every if step % a.every == 0 else step % a.every) for k, v in sums.items()}, **check()}
            sums = {}
            if better(row, best):
                best = row
                keep(step)
            if a.last:
                keep(step, f"{name}.last")
            log.append(row)
            print(f"step {step:6d}  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "step") + ("  *" if row is best else ""), flush=True)
            (out / f"{name}.json").write_text(json.dumps({"best": best, "steps": log}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tune")
    t.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "hfcm")
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    t.add_argument("--model", type=str, required=True, help="the text-to-speech model whose `written` spectrograms to tune on")
    t.add_argument("--name", type=str, default="")
    t.add_argument("--written", type=str, default="", help="what `pmvc.control write` wrote, if not under the model's name")
    t.add_argument("--critics", type=Path, default=ROOT / "out" / "univnet" / "univ_c32.pt", help="UnivNet's own checkpoint, for its discriminators")
    t.add_argument("--steps", type=int, default=5000)
    t.add_argument("--every", type=int, default=250)
    t.add_argument("--stride", type=int, default=2, help="every so many of the validation recordings")
    t.add_argument("--quality", type=int, default=0, help="ask NISQA of so many validation clips, and keep the vocoder it likes best")
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--spectral", type=float, default=2.5, help="the weight of the spectral error in the vocoder's loss")
    t.add_argument("--verdict", type=float, default=1.0, help="the weight of the critics' verdict in it")
    t.add_argument("--last", action="store_true", help="keep the vocoder of the last check too, as <name>.last.pt")
    t.add_argument("--critics-lr", type=float, default=1e-4)
    t.add_argument("--models", type=Path, default=ROOT / "out" / "transplant" / "models", help="where the stand-in is that names the listener")
    t.add_argument("--standin", type=str, default="standin_2c")
    a = ap.parse_args()
    {"tune": tune}[a.command](a)


if __name__ == "__main__":
    main()
