"""The two synthesiser binaries: CHIMERA's own CLI and this repository's `pmvc-synth` driver."""

import json
import subprocess
from pathlib import Path
from typing import Any

from .paths import CHIMERA_BIN, SYNTH_BIN


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr.strip() or r.stdout.strip()}")


def render_text(text: str, body: str, wav: Path) -> None:
    """CHIMERA as shipped: its G2P, its duration rules, its intonation (`chimera render text`)."""
    _run([str(CHIMERA_BIN), "render", "text", str(wav.with_suffix("")), text, body])


def render_spec(spec: dict[str, Any], wav: Path, tracks: bool = False) -> dict[str, Any]:
    """Plan and render `spec` (see crates/pmvc-synth). Writes `<wav>.spec.json` and
    `<wav>.info.json` beside the WAV, `<wav>.tracks.json` when asked, and returns the info."""
    stem = wav.with_suffix("")
    spec_path, info_path = stem.with_suffix(".spec.json"), stem.with_suffix(".info.json")
    spec_path.write_text(json.dumps(spec, ensure_ascii=False))
    cmd = [str(SYNTH_BIN), "render", str(spec_path), str(wav), "--info", str(info_path)]
    if tracks:
        cmd += ["--tracks", str(stem.with_suffix(".tracks.json"))]
    _run(cmd)
    return json.loads(info_path.read_text())


def plan_spec(spec: dict[str, Any], stem: Path) -> dict[str, Any]:
    """Plan `spec` without rendering: the phones' times as the planner sets them. Writes
    `<stem>.plan-spec.json` and `<stem>.plan.json` and returns the latter's content."""
    spec_path, info_path = stem.with_suffix(".plan-spec.json"), stem.with_suffix(".plan.json")
    spec_path.write_text(json.dumps(spec, ensure_ascii=False))
    _run([str(SYNTH_BIN), "plan", str(spec_path), str(info_path)])
    return json.loads(info_path.read_text())


def drive(tracks: dict[str, Any], wav: Path) -> None:
    """Play a control track back (`pmvc-synth drive`). Writes `<wav>.tracks.json` beside the WAV."""
    path = wav.with_suffix("").with_suffix(".tracks.json")
    path.write_text(json.dumps(tracks))
    _run([str(SYNTH_BIN), "drive", str(path), str(wav)])
