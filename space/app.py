"""The Space: fetch the models, then serve the page and the control tracks (`pmvc.serve`).

The models are read from `models/<lang>.json` beside this file if they are there, and otherwise
fetched from the model repository (PMVC_MODELS, by default takuma104/pmvc-tts; a private one
needs a token in HF_TOKEN). PMVC_LANGS names the languages to offer, the first the one the page
begins with when the browser's language is neither.
"""

import os
import sys
from pathlib import Path

HERE: Path = Path(__file__).resolve().parent
REPO: str = os.environ.get("PMVC_MODELS", "takuma104/pmvc-tts")
LANGS: list[str] = [x for x in os.environ.get("PMVC_LANGS", "ja,en").split(",") if x]


def model(lang: str) -> Path | None:
    local = HERE / "models" / f"{lang}.json"
    if local.exists():
        return local
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(REPO, f"{lang}.safetensors")
        return Path(hf_hub_download(REPO, f"{lang}.json"))
    except Exception as e:  # noqa: BLE001
        print(f"no model for {lang} in {REPO}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return None


def main() -> None:
    import torch

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    models = [m for m in map(model, LANGS) if m is not None]
    if not models:
        sys.exit("no model to serve")
    sys.argv = [
        "pmvc.serve", "--model", *map(str, models), "--host", "0.0.0.0", "--port", os.environ.get("PORT", "7860"),
        "--web", str(HERE / "web"), "--wasm", str(HERE / "pmvc_wasm.wasm"), "--voice", str(HERE / "voices" / "v1.json"),
    ]  # fmt: skip
    from pmvc.serve import main as serve

    serve()


if __name__ == "__main__":
    main()
