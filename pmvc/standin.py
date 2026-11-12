"""A stand-in for the voice that gradients pass through: a control track in, what the listener
would hear of its render out.

An encoder taught to copy searched tracks (`pmvc.encoder`) averages over them: two tracks that
both sound right need not average to one that does, and its renders stay further from the
recordings than the search's. The remedy is as old as the problem (Jordan and Rumelhart's
distal teacher): teach the encoder through a model of what its output does, so that any track
that sounds right is as good as any other. CHIMERA cannot be differentiated, but its renders
are unlimited and exact, so a network can learn the track → listener's features map from them
and stand in for it.

usage: python -m pmvc.standin [dir] [--name standin] [--variants warp,sim,simcopy] [--epochs 30]

Fits on the training speakers' renders of `--variants` (their tracks and what `pmvc.listener`
hears of them, from `LEAD` seconds in, as the search hears a render against its recording) and
writes <dir>/models/<name>.pt; the validation speakers' renders say how far off it is.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from .audio import FRAME_RATE, Array, load
from .listener import LISTEN_SR, Listener
from .paths import ROOT

# track frames before the listener's first: a render is heard from this far in (`pmvc.encoder`'s LEAD)
SKIP: int = 20
# listener frames those are
SKIP_HEARD: int = SKIP // 2


def track_input(frames: Array, keys: list[str], female: float) -> NDArray[np.float32]:
    """A track as the stand-in reads it: pitch in octaves, and the body as a last channel."""
    x = np.array(frames, dtype=np.float64)
    f0 = keys.index("f0")
    x[:, f0] = np.log2(np.maximum(x[:, f0], 1.0))
    return np.concatenate([x, np.full((len(x), 1), female)], axis=1).astype(np.float32)


class Standin(nn.Module):
    def __init__(self, channels: int, hidden: int = 512, layers: int = 3, out: int = 768) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(channels))
        self.register_buffer("std", torch.ones(channels))
        self.inp = nn.Linear(channels, hidden)
        # no dropout: it is run in training mode when gradients pass through it (cuDNN's
        # recurrent layers allow no backward pass otherwise), and must be the same model then
        self.rnn = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.out = nn.Linear(4 * hidden, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` (batch, frames, channels) as `track_input` gives it. Returns unit features
        (batch, frames // 2, 768): the listener's frame k of the render is row SKIP_HEARD + k."""
        h, _ = self.rnn(torch.relu(self.inp((x - self.mean) / self.std)))
        n = h.shape[1] // 2
        h = h[:, : 2 * n].reshape(h.shape[0], n, -1)
        return nn.functional.normalize(self.out(h), dim=-1)


@dataclass
class Render:
    x: NDArray[np.float32]
    feats: torch.Tensor


def load_standin(path: Path, device: str) -> Standin:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = Standin(ck["channels"], ck["hidden"])
    model.load_state_dict(ck["state"])
    return model.to(device).requires_grad_(False)


def main() -> None:
    from .encoder import pick_device, split_of

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "transplant")
    ap.add_argument("--name", type=str, default="standin")
    ap.add_argument("--variants", type=str, default="warp,sim,simcopy")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    d: Path = a.dir
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = pick_device()
    listener = Listener(device)

    def read(job: tuple[str, str]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        v, uid = job
        track = json.loads((d / v / f"{uid}.tracks.json").read_text())
        x, _ = load(d / v / f"{uid}.wav", LISTEN_SR)
        return track_input(np.array(track["frames"]), track["keys"], float(track["body"]["female"])), x[SKIP * (LISTEN_SR // FRAME_RATE) :].astype(np.float32)

    sets: dict[str, list[Render]] = {"train": [], "valid": []}
    jobs = [(v, p.name.split(".")[0]) for v in a.variants.split(",") for p in sorted((d / v).glob("*.tracks.json")) if (d / v / p.name.replace(".tracks.json", ".wav")).exists()]
    jobs = [j for j in jobs if split_of(j[1]) in sets]
    with ThreadPoolExecutor(a.jobs) as pool:
        for (_, uid), (x, audio) in zip(jobs, pool.map(read, jobs)):
            sets[split_of(uid)].append(Render(x, listener.hear(audio).feats[0].half()))
    fit, held = sets["train"], sets["valid"]
    pooled = np.concatenate([r.x for r in fit])
    model = Standin(pooled.shape[1], a.hidden).to(device)
    model.mean.copy_(torch.from_numpy(pooled.mean(axis=0)))
    model.std.copy_(torch.from_numpy(np.where(pooled.std(axis=0) > 1e-6, pooled.std(axis=0), 1.0)))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * -(-len(fit) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    print(f"{len(fit)} renders to fit, {len(held)} held out; {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M weights; {device}", flush=True)

    def loss_of(batch: list[Render]) -> torch.Tensor:
        """The mean distance (1 − cosine) between what the stand-in says the listener hears and
        what it heard."""
        n = max(len(r.x) for r in batch)
        m = max(len(r.feats) for r in batch)
        x = torch.zeros(len(batch), n, batch[0].x.shape[1])
        y = torch.zeros(len(batch), m, batch[0].feats.shape[1])
        mask = torch.zeros(len(batch), m)
        for i, r in enumerate(batch):
            k = min(len(r.feats), len(r.x) // 2 - SKIP_HEARD)
            x[i, : len(r.x)] = torch.from_numpy(r.x)
            x[i, len(r.x) :] = torch.from_numpy(r.x[-1])
            y[i, :k] = r.feats[:k].float()
            mask[i, :k] = 1.0
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        pred = model(x)[:, SKIP_HEARD : SKIP_HEARD + m]
        pred = torch.cat([pred, pred.new_zeros(len(batch), m - pred.shape[1], pred.shape[2])], dim=1)
        return ((1.0 - (pred * y).sum(dim=-1)) * mask).sum() / mask.sum()

    def chunks(rs: list[Render], shuffle: bool) -> list[list[Render]]:
        order = np.argsort([len(r.x) + (rng.uniform(-25, 25) if shuffle else 0) for r in rs])
        out = [[rs[i] for i in order[k : k + a.batch]] for k in range(0, len(order), a.batch)]
        return [out[i] for i in rng.permutation(len(out))] if shuffle else out

    out = d / "models"
    out.mkdir(exist_ok=True)
    log: list[dict[str, float]] = []
    for epoch in range(a.epochs):
        model.train()
        total, count = 0.0, 0
        for batch in chunks(fit, True):
            loss = loss_of(batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, count = total + loss.item(), count + 1
        with torch.no_grad():
            heldout = float(np.mean([loss_of(b).item() for b in chunks(held, False)])) if held else float("nan")
        log.append({"epoch": epoch, "fit": total / count, "held": heldout})
        if epoch % 5 == 0 or epoch == a.epochs - 1:
            print(f"epoch {epoch:3d}  fit {total / count:.4f}  held {heldout:.4f}", flush=True)
    state: dict[str, Any] = {"state": model.state_dict(), "channels": pooled.shape[1], "hidden": a.hidden, "args": {k: str(v) for k, v in vars(a).items()}}
    torch.save(state, out / f"{a.name}.pt")
    (out / f"{a.name}.json").write_text(json.dumps({"epochs": log}, indent=1))


if __name__ == "__main__":
    main()
