"""Other opinions on how natural a clip sounds: two more predictors of a listener's score.

NISQA's naturalness model (`pmvc.natural`) is one opinion, and on this voice a peculiar one:
its verdict on a render follows the band under 250 Hz and rises with a noise floor
(`pmvc.gap`, `pmvc.room`). These two were taught with other data and in other ways, and are
asked the same question:
  squim   torchaudio's SQUIM_SUBJECTIVE (Kumar et al., 2023): a mean opinion score for a clip
          given some clean speech to hold it against (any speech; here recordings of two
          speakers that are in no test)
  utmos   UTMOSv2 (Baba et al., 2024; third-party/UTMOSv2, as it is): the system that won
          most tracks of the VoiceMOS Challenge 2024, a spectrogram network beside a
          self-supervised one. It draws a random stretch of a clip each time, so each clip is
          scored REPEATS times and the scores averaged
Neither has a say in any fit or training here.
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .audio import load
from .paths import ROOT

UTMOS: Path = ROOT / "third-party" / "UTMOSv2"
REPEATS: int = 3
SR: int = 16000


@torch.no_grad()
def squim(wavs: list[Path], references: list[Path]) -> list[float]:
    """SQUIM's opinion score for each clip, averaged over the references it is held against."""
    import torchaudio

    model = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model().eval()
    refs = [torch.as_tensor(load(p, SR)[0], dtype=torch.float32)[None] for p in references]
    out: list[float] = []
    for p in wavs:
        x = torch.as_tensor(load(p, SR)[0], dtype=torch.float32)[None]
        out.append(float(np.mean([model(x, r).item() for r in refs])))
    return out


def utmos(wavs: list[Path], repeats: int = REPEATS) -> list[float]:
    """UTMOSv2's opinion score for each clip."""
    sys.path.insert(0, str(UTMOS))
    import utmosv2  # type: ignore[import-not-found]

    # it prints its checkpoint's loading report and a progress bar
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = utmosv2.create_model(pretrained=True)
        with tempfile.TemporaryDirectory() as tmp:
            for i, p in enumerate(wavs):
                (Path(tmp) / f"{i:06d}.wav").symlink_to(p.resolve())
            # its spectrogram network is some ten times slower on a CPU
            device = "mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
            rows = model.predict(input_dir=tmp, device=device, num_repetitions=repeats, num_workers=0, verbose=False)
    by = {Path(r["file_path"]).stem: float(r["predicted_mos"]) for r in rows}
    return [by[f"{i:06d}"] for i in range(len(wavs))]
