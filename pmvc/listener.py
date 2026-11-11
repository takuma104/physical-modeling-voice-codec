"""A listener to optimise against: wav2vec2-base-960h, a speech recogniser trained on read
English, used two ways.

- Its hidden features (layer 9) are phonetic and nearly speaker-independent: along the best
  time warp, a second real speaker reading the same sentence is 0.16 from the first (1 − cosine),
  CHIMERA's rule render 0.39. Mel cepstra could not tell those two apart. So the distance between
  a render's features and the recording's, frame against frame, says how far the render is from
  what the speaker said, when.
- Its output is characters, so the CTC likelihood of the sentence's text says whether the
  render can be read at all.

Whisper stays the judge (`pmvc.evaluate`) and is never optimised against.
"""

import re
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.signal import resample_poly

# frames per second of the listener's features, and its input rate (Hz)
FEATURE_RATE: int = 50
LISTEN_SR: int = 16000
LAYER: int = 9
MODEL: str = "facebook/wav2vec2-base-960h"

Audio = NDArray[np.float32]


def to16k(x48: NDArray[np.floating]) -> Audio:
    """48 kHz samples at the listener's rate."""
    return resample_poly(x48, 1, 3).astype(np.float32)


@dataclass
class Heard:
    """What the listener makes of a batch of equally long clips."""

    # unit-length layer-9 features, (clips, frames, 768)
    feats: torch.Tensor
    # log posteriors over characters, (clips, frames, 32)
    logp: torch.Tensor


class Listener:
    def __init__(self, device: str | None = None) -> None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, logging

        logging.set_verbosity_error()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
            self.model = Wav2Vec2ForCTC.from_pretrained(MODEL).to(self.device).eval()
            self.vocab: dict[str, int] = Wav2Vec2Processor.from_pretrained(MODEL).tokenizer.get_vocab()

    def chars(self, text: str) -> list[int]:
        """The sentence as the recogniser's character ids."""
        t = " ".join(re.sub(r"[^A-Z' ]+", " ", text.upper().replace("’", "'")).split())
        return [self.vocab[c if c != " " else "|"] for c in t]

    @torch.no_grad()
    def hear(self, clips: NDArray[np.floating]) -> Heard:
        """`clips`: (n, samples) at 16 kHz, or one clip. Each is normalised on its own."""
        x = torch.as_tensor(np.atleast_2d(clips), dtype=torch.float32, device=self.device)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-7)
        out = self.model(x, output_hidden_states=True)
        feats = torch.nn.functional.normalize(out.hidden_states[LAYER].float(), dim=-1)
        return Heard(feats.cpu(), torch.log_softmax(out.logits.float(), dim=-1).cpu())

    def nll(self, logp: torch.Tensor, chars: list[int]) -> float:
        """CTC negative log-likelihood of the text per character, for one clip's (frames, 32)."""
        loss = torch.nn.functional.ctc_loss(
            logp[:, None, :], torch.tensor([chars]), torch.tensor([logp.shape[0]]), torch.tensor([len(chars)]),
            blank=0, reduction="sum", zero_infinity=True,
        )  # fmt: skip
        return float(loss) / len(chars)

    def read(self, logp: torch.Tensor) -> str:
        """The greedy transcript of one clip."""
        inv = {v: k for k, v in self.vocab.items()}
        out: list[str] = []
        prev = -1
        for i in logp.argmax(dim=-1).tolist():
            if i != prev and i != 0:
                out.append(inv[i])
            prev = i
        return "".join(out).replace("|", " ").lower()


def distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 − cosine between two feature sequences frame against frame, (..., frames)."""
    n = min(a.shape[-2], b.shape[-2])
    return 1.0 - (a[..., :n, :] * b[..., :n, :]).sum(dim=-1)
