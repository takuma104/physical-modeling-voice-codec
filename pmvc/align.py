"""Which frames of a recording each phoneme of its sentence takes.

A text-to-speech model that writes a control track phoneme by phoneme (`pmvc.tts`) has to be
told, for the recordings it learns from, how long the speaker held each phoneme. No aligner
for Japanese is at hand, and none is needed: with some thousands of sentences of one speaker
the alignment can be learned from the recordings and their text alone (Badlani et al., "One
TTS alignment to rule them all", 2021). A small network turns each phoneme (with its
neighbours) and each frame of the recording's mel spectrogram (with its neighbours) into a
point, and a frame belongs to the phonemes the nearer their points are. It is taught that
every recording is its sentence's phonemes in order, each for at least one frame: the
likelihood of all such paths together is what CTC computes. The best single path is then the
alignment.

usage: python -m pmvc.align train [dir] [--name aligner] [--epochs 12]
       python -m pmvc.align write [dir] [--name aligner] [--models DIR] [--split all] [--ids ...]
                                  [--no-pauses]
       python -m pmvc.align prune [dir] [--run 4] [--longest 80] [--pause 150]

`train` learns from every recording of <dir> with Japanese text (the alignment of a test
recording is not what is tested) and writes <dir>/models/<name>.pt. `write` writes
<dir>/aligned.json: for each recording its phonemes (`pmvc.japanese`) and how many frames each
takes, on the clock of an encoder's track for the recording: 100 frames a second, from
`pmvc.encoder`'s LEAD of silence before the recording to its TAIL after it.

A speaker pauses where the text has no comma: this one about once in two sentences, for a
quarter of a second. The sentence's phonemes have no pause there, and the silence goes to a
neighbour (a /d/ of 390 ms). So `write` aligns twice. Where a recording pauses
(`pmvc.encoder.pauses`: 120 ms and more of silence inside its speech) and the first alignment
has no pause, a short pause (`sp`) is put at the borders of accent phrases there, and the
second alignment says how long it is; one that then takes under SHORT frames was no pause and
is taken out again. (A silence inside an accent phrase is a stop's closure, and stays the
stop's.) With `--no-pauses` the phonemes stay as the text gives them.

`prune` is for recordings whose text a recogniser wrote (Emilia-YODAS), or that volunteers
recorded (Common Voice): where the text has a word the speaker did not say, the alignment gives
a run of phonemes a frame each and a neighbour the rest. It takes out of <dir>/aligned.json the
recordings with `--run` or more phonemes in a row of one frame each, those where a phoneme (but
the sentence's last, which has the silence after it too) takes more than `--longest` frames or
a pause more than `--pause`, and keeps what `write` wrote as aligned_all.json. (A single phoneme
of one frame is no sign of anything: of two equal vowels in a row, as in おいしい, one gets it.)
With people talking, or six hundred volunteers' rooms, twelve epochs do not find the alignment;
forty do (`train --epochs 40 --hinted 12 --sharpen 16`).
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from .audio import FRAME_RATE, log_mel
from .encoder import LEAD, hear_recording, ids_in, pauses, pick_device
from .japanese import NONE, Phone, border, read
from .paths import ROOT

MELS: int = 80
# what a frame's score for "no phoneme" is before the scores are normalised: CTC wants such a
# class, and this keeps it from being used
BLANK: float = -1.0
# how sharply the distance between a frame's point and a phoneme's counts
SHARP: float = 0.0005
# the spread of the first epochs' hint that a sentence's phonemes lie along its speech in order
HINT: float = 0.2
# a short pause put in where the recording pauses stays if the alignment gives it this many frames
SHORT: int = 8


@dataclass
class Item:
    uid: str
    phones: list[Phone]
    # (frames, MELS), the frames of the recording's speech, and those of its pauses
    mel: NDArray[np.float16]
    span: tuple[int, int]
    quiet: list[tuple[int, int]]


def item(job: tuple[Path, str]) -> Item:
    d, uid = job
    meta = json.loads((d / "meta" / f"{uid}.json").read_text())
    audio, aux, _ = hear_recording(d, uid, LEAD)
    mel = log_mel(audio.astype(np.float64), n_mels=MELS)[: len(aux)]
    k = round(LEAD * FRAME_RATE)
    span = (k + round(meta["span"][0] * FRAME_RATE), k + round(meta["span"][1] * FRAME_RATE))
    return Item(uid, read(meta["text"]), mel.astype(np.float16), span, [(k + a, k + b) for a, b in pauses(d, uid)])


def items(d: Path, ids: list[str], jobs: int) -> list[Item]:
    with ProcessPoolExecutor(jobs) as pool:
        return list(pool.map(item, [(d, uid) for uid in ids], chunksize=16))


class Aligner(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, phones: list[str], width: int = 80) -> None:
        super().__init__()
        self.phones = phones
        self.register_buffer("mean", torch.zeros(MELS))
        self.register_buffer("std", torch.ones(MELS))
        self.embed = nn.Embedding(len(phones), 256)
        self.text = nn.Sequential(nn.Conv1d(256, 512, 3, padding=1), nn.ReLU(), nn.Conv1d(512, width, 1))
        self.sound = nn.Sequential(nn.Conv1d(MELS, 2 * MELS, 3, padding=1), nn.ReLU(), nn.Conv1d(2 * MELS, MELS, 1), nn.ReLU(), nn.Conv1d(MELS, width, 1))

    def forward(self, tokens: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        """`tokens` (batch, phonemes) and `mel` (batch, frames, MELS). Returns each frame's
        score for each phoneme, (batch, frames, phonemes): not yet normalised."""
        keys = self.text(self.embed(tokens).transpose(1, 2)).transpose(1, 2)
        queries = self.sound(((mel - self.mean) / self.std).transpose(1, 2)).transpose(1, 2)
        return -SHARP * torch.cdist(queries, keys) ** 2


def collate(batch: list[Item], phones: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokens, mel, and how many of each are real."""
    n, t = max(len(b.phones) for b in batch), max(len(b.mel) for b in batch)
    tokens = torch.zeros(len(batch), n, dtype=torch.long)
    mel = torch.zeros(len(batch), t, MELS)
    for i, b in enumerate(batch):
        # a short pause is a pause to the aligner
        tokens[i, : len(b.phones)] = torch.tensor([phones.index("pau" if p.p == "sp" else p.p) for p in b.phones])
        mel[i, : len(b.mel)] = torch.from_numpy(b.mel.astype(np.float32))
    return tokens.to(device), mel.to(device), torch.tensor([len(b.phones) for b in batch], device=device), torch.tensor([len(b.mel) for b in batch], device=device)


def masked(scores: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """The scores with a sentence's padding out of the running."""
    pad = torch.arange(scores.shape[2], device=scores.device)[None] >= n[:, None]
    return scores.masked_fill(pad[:, None, :], -1e9)


def all_paths(scores: torch.Tensor, n: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """The negative log-likelihood, per phoneme, that the frames are the phonemes in order."""
    x = nn.functional.pad(scores, (1, 0), value=BLANK)
    logp = masked(x, n + 1).log_softmax(dim=-1).transpose(0, 1)
    targets = torch.arange(1, scores.shape[2] + 1, device=scores.device).expand(scores.shape[0], -1)
    return nn.functional.ctc_loss(logp, targets, t, n, blank=0, reduction="mean", zero_infinity=True)


def best_path(logp: NDArray[np.float32], n: NDArray[np.int64], t: NDArray[np.int64]) -> NDArray[np.int64]:
    """The likeliest path for each of a batch: for every frame the index of its phoneme,
    (batch, frames), in order from the first phoneme to the last, none left out. `logp` is
    (batch, frames, phonemes); `n` and `t` say how many phonemes and frames are real."""
    b, frames, m = logp.shape
    low = -1e18
    q = np.full((b, m), low)
    q[:, 0] = logp[:, 0, 0]
    stepped = np.zeros((b, frames, m), dtype=bool)
    for k in range(1, frames):
        moved = np.concatenate([np.full((b, 1), low), q[:, :-1]], axis=1)
        stepped[:, k] = moved > q
        q = np.where(stepped[:, k], moved, q) + logp[:, k]
    path = np.zeros((b, frames), dtype=np.int64)
    at = n - 1
    rows = np.arange(b)
    for k in range(frames - 1, -1, -1):
        live = k < t
        path[:, k] = np.where(live, at, 0)
        at = np.where(live & stepped[rows, k, at], at - 1, at)
    return path


def hint(n: torch.Tensor, t: torch.Tensor, spans: list[tuple[int, int]], phonemes: int, frames: int) -> torch.Tensor:
    """A log prior, (batch, frames, phonemes): the phonemes lie along the recording's speech in
    order, the first and the last taking the silence either side."""
    device = n.device
    a = torch.tensor([s[0] for s in spans], device=device)[:, None]
    b = torch.tensor([s[1] for s in spans], device=device)[:, None]
    u = ((torch.arange(frames, device=device)[None] - a) / (b - a).clamp(min=1)).clamp(0.0, 1.0)
    v = (torch.arange(phonemes, device=device)[None] + 0.5) / n[:, None]
    return -((u[:, :, None] - v[:, None, :]) ** 2) / (2.0 * HINT**2)


def durations(model: Aligner, batch: list[Item], device: str) -> list[list[int]]:
    """How many frames each phoneme of each recording takes."""
    tokens, mel, n, t = collate(batch, model.phones, device)
    with torch.no_grad():
        logp = masked(model(tokens, mel), n).log_softmax(dim=-1).cpu().numpy()
    path = best_path(logp, n.cpu().numpy(), t.cpu().numpy())
    return [np.bincount(path[i, : len(b.mel)], minlength=len(b.phones)).tolist() for i, b in enumerate(batch)]


def chunks(data: list[Item], size: int, rng: np.random.Generator | None) -> list[list[Item]]:
    order = np.argsort([len(x.mel) + (rng.uniform(-25, 25) if rng is not None else 0) for x in data], kind="stable")
    out = [[data[i] for i in order[k : k + size]] for k in range(0, len(order), size)]
    return [out[i] for i in rng.permutation(len(out))] if rng is not None else out


def train(a: argparse.Namespace) -> None:
    d: Path = a.dir
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = pick_device()
    ids = [i for i in ids_in(d, "all") if json.loads((d / "meta" / f"{i}.json").read_text()).get("lang") == "ja"]
    data = items(d, ids, a.jobs)
    phones = sorted({p.p for x in data for p in x.phones})
    model = Aligner(phones).to(device)
    pooled = np.concatenate([x.mel[:: 8].astype(np.float32) for x in data])
    model.mean.copy_(torch.from_numpy(pooled.mean(axis=0)))
    model.std.copy_(torch.from_numpy(pooled.std(axis=0)))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6)
    steps = a.epochs * len(chunks(data, a.batch, None))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    print(f"{len(data)} recordings, {len(phones)} phonemes, {sum(len(x.mel) for x in data) / FRAME_RATE / 3600:.1f} hours; {device}", flush=True)
    for epoch in range(a.epochs):
        # the hint fades over the first epochs; from `--sharpen` on, the paths are also pulled
        # towards the best one (frames that sit between two phonemes make up their mind)
        lean = max(0.0, 1.0 - epoch / max(1, a.hinted))
        total, hard, count = 0.0, 0.0, 0
        for batch in chunks(data, a.batch, rng):
            tokens, mel, n, t = collate(batch, phones, device)
            scores = model(tokens, mel)
            if lean > 0.0:
                scores = scores + lean * hint(n, t, [x.span for x in batch], scores.shape[2], scores.shape[1])
            loss = all_paths(scores, n, t)
            total += loss.item()
            if epoch >= a.sharpen:
                logp = masked(scores, n).log_softmax(dim=-1)
                path = torch.from_numpy(best_path(logp.detach().cpu().numpy(), n.cpu().numpy(), t.cpu().numpy())).to(device)
                real = torch.arange(scores.shape[1], device=device)[None] < t[:, None]
                on_path = -(logp.gather(2, path[:, :, None])[:, :, 0] * real).sum() / real.sum()
                hard += on_path.item()
                loss = loss + on_path
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            count += 1
        print(f"epoch {epoch:2d}  all paths {total / count:.4f}" + (f"  on the best {hard / count:.4f}" if epoch >= a.sharpen else ""), flush=True)
    (d / "models").mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "phones": phones, "args": {k: str(v) for k, v in vars(a).items()}}, d / "models" / f"{a.name}.pt")


def load_aligner(path: Path, device: str) -> Aligner:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = Aligner(ck["phones"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval()


def with_pauses(x: Item, frames: list[int]) -> list[Phone] | None:
    """The recording's phonemes with a short pause at each accent phrase's border where the
    recording pauses and the alignment (`frames`) has no pause; nothing if there is none."""
    ends = np.cumsum(frames)
    starts = ends - np.array(frames)
    at: set[int] = set()
    for a, b in x.quiet:
        over = [k for k in range(len(x.phones)) if min(b, ends[k]) > max(a, starts[k])]
        if not over or sum(min(b, ends[k]) - max(a, starts[k]) for k in over if x.phones[k].silent) >= 0.5 * (b - a):
            continue
        at.update(k for k in range(over[0], min(len(x.phones), over[-1] + 2)) if border(x.phones, k))
    if not at:
        return None
    return [q for k, p in enumerate(x.phones) for q in ((Phone("sp", NONE, NONE, NONE, p.ask), p) if k in at else (p,))]


def write(a: argparse.Namespace) -> None:
    d: Path = a.dir
    device = pick_device()
    model = load_aligner((a.models or d / "models") / f"{a.name}.pt", device)
    ids = a.ids.split(",") if a.ids else [i for i in ids_in(d, a.split) if json.loads((d / "meta" / f"{i}.json").read_text()).get("lang") == "ja"]
    data = items(d, ids, a.jobs)
    known = [x for x in data if all(p.p in model.phones for p in x.phones)]
    if len(known) < len(data):
        print(f"{len(data) - len(known)} recordings have a phoneme the aligner was not taught and are left out")
    lasting: dict[str, list[int]] = {}
    for batch in chunks(known, a.batch, None):
        lasting.update({x.uid: frames for x, frames in zip(batch, durations(model, batch, device))})
    put, kept = 0, 0
    if not a.no_pauses:
        again: list[Item] = []
        for x in known:
            phones = with_pauses(x, lasting[x.uid])
            if phones is not None:
                put += sum(p.p == "sp" for p in phones)
                x.phones = phones
                again.append(x)
        for batch in chunks(again, a.batch, None):
            for x, frames in zip(batch, durations(model, batch, device)):
                # a short pause that got next to nothing was none: its frames go to the phoneme after it
                for k in [k for k, p in enumerate(x.phones) if p.p == "sp" and frames[k] < SHORT][::-1]:
                    frames[k + 1] += frames[k]
                    del frames[k], x.phones[k]
                kept += sum(p.p == "sp" for p in x.phones)
                lasting[x.uid] = frames
    path = d / "aligned.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    for x in known:
        out[x.uid] = {"p": [p.p for p in x.phones], "a1": [p.a1 for p in x.phones], "a2": [p.a2 for p in x.phones], "a3": [p.a3 for p in x.phones], "ask": x.phones[0].ask, "frames": lasting[x.uid]}
    path.write_text(json.dumps(out, ensure_ascii=False))
    # what the alignment says of the speaker, to check it by
    by: dict[str, list[int]] = {}
    for x in known:
        for p, n in zip(out[x.uid]["p"][1:-1], out[x.uid]["frames"][1:-1]):
            by.setdefault(p, []).append(n)
    print(f"{len(known)} recordings → {path}" + (f"; {put} short pauses put in where a recording pauses, {kept} kept" if put else ""))
    print("  median frames by phoneme: " + "  ".join(f"{p} {np.median(v):.0f}" for p, v in sorted(by.items(), key=lambda kv: -len(kv[1]))))


def aligned(d: Path) -> dict[str, tuple[list[Phone], list[int]]]:
    """What `write` wrote for <dir>: each recording's phonemes and their frames."""
    out: dict[str, tuple[list[Phone], list[int]]] = {}
    for uid, r in json.loads((d / "aligned.json").read_text()).items():
        out[uid] = ([Phone(p, a1, a2, a3, r["ask"]) for p, a1, a2, a3 in zip(r["p"], r["a1"], r["a2"], r["a3"])], r["frames"])
    return out


def prune(a: argparse.Namespace) -> None:
    d: Path = a.dir
    whole = d / "aligned_all.json"
    if not whole.exists():
        whole.write_text((d / "aligned.json").read_text())
    rows: dict[str, Any] = json.loads(whole.read_text())
    kept: dict[str, Any] = {}
    for uid, r in rows.items():
        run = longest_run = 0
        for f in r["frames"][1:-1]:
            run = run + 1 if f <= 1 else 0
            longest_run = max(longest_run, run)
        # (the sentence's last phoneme has the silence after it too, as `write` leaves it)
        longest = max((f for p, f in zip(r["p"][:-2], r["frames"][:-2]) if p not in ("sil", "pau", "sp")), default=0)
        pause = max((f for p, f in zip(r["p"], r["frames"]) if p in ("pau", "sp")), default=0)
        if longest_run < a.run and longest <= a.longest and pause <= a.pause:
            kept[uid] = r
    (d / "aligned.json").write_text(json.dumps(kept, ensure_ascii=False))
    print(f"{len(kept)} of {len(rows)} recordings kept → {d / 'aligned.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    u = sub.add_parser("prune")
    u.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "hfcm")
    u.add_argument("--run", type=int, default=4, help="a recording with so many phonemes in a row of one frame each is taken out")
    u.add_argument("--longest", type=int, default=80, help="the frames a phoneme may take")
    u.add_argument("--pause", type=int, default=150, help="the frames a pause may take")
    for name in ("train", "write"):
        p = sub.add_parser(name)
        p.add_argument("dir", type=Path, nargs="?", default=ROOT / "out" / "hfcm")
        p.add_argument("--name", type=str, default="aligner")
        p.add_argument("--batch", type=int, default=32)
        p.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    t = sub.choices["train"]
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--hinted", type=int, default=3, help="epochs over which the hint that the phonemes lie along the speech fades")
    t.add_argument("--sharpen", type=int, default=4, help="the epoch from which the paths are pulled towards the best one")
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)
    w = sub.choices["write"]
    w.add_argument("--models", type=Path, default=None, help="where the aligner is, if not <dir>/models")
    w.add_argument("--split", choices=["test", "valid", "train", "all"], default="all")
    w.add_argument("--ids", type=str, default="")
    w.add_argument("--no-pauses", action="store_true", help="no short pauses where the recording pauses and the text has none")
    a = ap.parse_args()
    {"train": train, "write": write, "prune": prune}[a.command](a)


if __name__ == "__main__":
    main()
