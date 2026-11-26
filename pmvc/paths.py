"""Where things live."""

import os
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
CHIMERA: Path = ROOT / "third-party" / "chimera-vocal-synth"
# `cargo build --release -p chimera-cli` in the submodule
CHIMERA_BIN: Path = CHIMERA / "target" / "release" / "chimera"
# `cargo build --release` at the repository root
SYNTH_BIN: Path = ROOT / "target" / "release" / "pmvc-synth"
# the TSP speech database, fetched by third-party/chimera-vocal-synth/tools/tsp/fetch.sh
TSP_DIR: Path = Path(os.environ.get("TSP_DIR", "~/datasets/tsp")).expanduser()
WHISPER_CACHE: Path = Path("~/.cache/chimera-asr/models").expanduser()


def engine_is_fresh(module_file: str) -> None:
    """Stop if the built `pmvc_engine` is older than the Rust it is made of. uv rebuilds it when
    its own crate's files change, and takes no notice of the wrapper's (`crates/pmvc-voice`) or
    the engine's (the submodule), though `crates/pmvc-engine/pyproject.toml` names them: a
    measurement after an edit there was twice one of the build before."""
    sources = [p for d in (ROOT / "crates" / "pmvc-engine" / "src", ROOT / "crates" / "pmvc-voice" / "src", CHIMERA / "crates" / "chimera" / "src") if d.exists() for p in d.rglob("*.rs")]
    if sources and max(p.stat().st_mtime for p in sources) > Path(module_file).stat().st_mtime:
        raise RuntimeError("pmvc_engine is older than its Rust sources: run `uv sync --reinstall-package pmvc-engine`")
