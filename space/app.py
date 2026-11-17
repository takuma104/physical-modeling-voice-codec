"""The Space: the page and the control tracks of `pmvc.serve`, from a Gradio server.

The Space is on ZeroGPU hardware, which runs Gradio applications only, and the page is not a
Gradio one. Gradio's Server mode (`gr.Server`) is a FastAPI application with Gradio's API
engine in it and no page of Gradio's own, so this is one:

  GET  /                the page (the files of web/), which names its language in the address:
                        /?ja, /?en
  GET  /pmvc_wasm.wasm, /api/models, /api/voice, POST /api/speak
                        what the page asks for, answered as `pmvc.serve` answers (the same code)
  the Gradio API `speak`    the same sentence for programs (`gradio_client`), which can say
                        where the model is to run

The models are small, and the Space's CPU writes a sentence in about a tenth of a second, so
that is where they run for the page. Through the Gradio API a sentence can be sent to the GPU
instead (`device="cuda"`), which is all ZeroGPU is asked for here, and which is taken from the
GPU time of whoever asks.

The models are read from `models/<lang>.json` beside this file if they are there, and otherwise
fetched from the model repository (PMVC_MODELS, by default takuma104/pmvc-tts; a private one
needs a token in HF_TOKEN). PMVC_LANGS names the languages to offer, the first the one the page
begins with when the browser's language is neither.

To run it where it is not a Space: `PORT=7861 python app.py`, with `gradio` installed beside
what `requirements.txt` names.
"""

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import spaces  # type: ignore[import-untyped]  # on ZeroGPU this has to be imported before torch
except ImportError:  # anywhere else
    spaces = None

import gradio as gr
import torch
from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

HERE: Path = Path(__file__).resolve().parent
# the engine's Python module (it reads English text), unpacked from its wheel by tools/space.sh:
# a Gradio Space installs what requirements.txt names before the repository's files are there,
# so a wheel in the repository cannot be named in it. Where the module is installed, that one
# is used.
sys.path.append(str(HERE / "engine"))

from pmvc.serve import PACE, Model, load, spoken, which

REPO: str = os.environ.get("PMVC_MODELS", "takuma104/pmvc-tts")
LANGS: list[str] = [x for x in os.environ.get("PMVC_LANGS", "ja,en").split(",") if x]
ZERO_GPU: bool = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "t", "true")
HAS_GPU: bool = ZERO_GPU or torch.cuda.is_available()
GPU_SECONDS: int = 10  # the ZeroGPU time reserved for a sentence
FRESH: dict[str, str] = {"Cache-Control": "no-cache"}  # a browser asks again before it uses a file it has


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


torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
# The models are loaded here, on the CPU, before the first request. On ZeroGPU the GPU is only
# there inside a worker that is forked from this process for a `spaces.GPU` call: the worker
# moves its own copy of a model to the GPU, and this process keeps its models where they are.
MODELS: dict[str, Model] = load([m for m in map(model, LANGS) if m is not None], "", "", HERE / "voices" / "v1.json", PACE, "cpu")
if not MODELS:
    sys.exit("no model to serve")

gpu: Callable[[Callable[..., Any]], Callable[..., Any]] = spaces.GPU(duration=GPU_SECONDS) if spaces is not None else (lambda f: f)


@gpu
def written_on_gpu(asked: dict[str, Any]) -> tuple[dict[str, Any], int]:
    return spoken(MODELS, asked, "cuda")


server = gr.Server(title="PMVC-TTS", docs_url=None, redoc_url=None, openapi_url=None)


def file(path: Path, kind: str | None = None) -> Callable[[], FileResponse]:
    return lambda: FileResponse(path, media_type=kind, headers=FRESH)


# The page's files, each by its own name: these routes come before Gradio's own, and one that
# took every name would hide Gradio's.
server.add_api_route("/", file(HERE / "web" / "index.html"), methods=["GET", "HEAD"])
for f in sorted((HERE / "web").iterdir()):
    if f.is_file():
        server.add_api_route(f"/{f.name}", file(f), methods=["GET", "HEAD"])
server.add_api_route("/pmvc_wasm.wasm", file(HERE / "pmvc_wasm.wasm", "application/wasm"), methods=["GET", "HEAD"])


@server.get("/api/models")
def models() -> JSONResponse:
    return JSONResponse({"langs": list(MODELS)}, headers=FRESH)


@server.get("/api/voice")
def voice(lang: str = "") -> JSONResponse:
    try:
        return JSONResponse(which(MODELS, lang).about(), headers=FRESH)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, 400, headers=FRESH)


@server.post("/api/speak")
async def speak(request: Request) -> JSONResponse:
    try:
        asked = json.loads((await request.body())[: 1 << 16])
    except ValueError as e:
        return JSONResponse({"error": str(e)}, 400, headers=FRESH)
    answer, status = await run_in_threadpool(spoken, MODELS, asked, "cpu")
    return JSONResponse(answer, status, headers=FRESH)


@server.api(name="speak")
def track(lang: str, text: str, share: float = 0.0, pitch: float = 0.0, rate: float = 0.0, spread: float = 0.0, device: str = "cpu") -> dict[str, Any]:
    """A sentence's control track, as POST /api/speak answers: `keys` (the channels), `rate`
    (frames a second), `frames`, `pitch` and `share` (the voice the frames were written for)
    and `phones` ([phoneme, frames] pairs). `share` is how far the voice is towards the high
    one (0 to 1), `pitch` its median in Hz, `rate` the pace in phonemes a second and `spread`
    how far the pitch moves, in octaves: 0 for what the model's speakers have at the median.
    `device` is where the model runs: "cpu", or "cuda" for ZeroGPU."""
    if device not in ("cpu", "cuda"):
        raise gr.Error(f"no device {device!r}: cpu or cuda")
    if device == "cuda" and not HAS_GPU:
        raise gr.Error("there is no GPU here")
    asked = {"lang": lang, "text": text, "share": share, "pitch": pitch, "rate": rate, "spread": spread}
    answer, status = written_on_gpu(asked) if device == "cuda" else spoken(MODELS, asked, "cpu")
    if status != 200:
        raise gr.Error(str(answer.get("error", "")))
    return answer


if __name__ == "__main__":
    print(f"PMVC-TTS: {', '.join(MODELS)}; {'ZeroGPU' if ZERO_GPU else 'GPU' if HAS_GPU else 'no GPU'}", flush=True)
    # where and on which port is Gradio's to say on a Space (GRADIO_SERVER_NAME, GRADIO_SERVER_PORT)
    server.launch(
        server_name=os.environ.get("HOST") or None,
        server_port=int(os.environ["PORT"]) if os.environ.get("PORT") else None,
        ssr_mode=False,
        show_error=True,
    )
