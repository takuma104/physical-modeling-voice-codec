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
