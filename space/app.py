"""The Space: the page and the control tracks of `pmvc.serve`, from a Gradio application.

The Space is on ZeroGPU hardware, which runs Gradio applications only, and the page is not a
Gradio one. So this is a FastAPI application with two things in it:

  /         the page, served as `pmvc.serve` serves it (the same answers, from the same code):
            the server writes a sentence's control track and the browser makes the sound
  /gradio   a plain Gradio form beside it: the same sentence with the sound made here, on the
            server, for a browser that cannot run the page and for programs (`gradio_client`)

The models are small, and the Space's CPU writes a sentence in about a tenth of a second, so
that is where they run. The form can send a sentence to the GPU instead, which is all ZeroGPU
is asked for here, and which costs the visitor who asks some of their GPU time.

The models are read from `models/<lang>.json` beside this file if they are there, and otherwise
fetched from the model repository (PMVC_MODELS, by default takuma104/pmvc-tts; a private one
needs a token in HF_TOKEN). PMVC_LANGS names the languages to offer, the first the one the page
begins with when the browser's language is neither.

To run it where it is not a Space: `python app.py` (PORT, by default 7860), with `gradio`
installed beside what `requirements.txt` names.
"""

import json
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import spaces  # type: ignore[import-untyped]  # on ZeroGPU this has to be imported before torch
except ImportError:  # anywhere else
    spaces = None

import gradio as gr
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from numpy.typing import NDArray

HERE: Path = Path(__file__).resolve().parent
# the engine's Python module, unpacked from its wheel by tools/space.sh: a Gradio Space installs
# what requirements.txt names before the repository's files are there, so a wheel in the
# repository cannot be named in it. Where the module is installed, that one is used.
sys.path.append(str(HERE / "engine"))

from pmvc.encoder import render_track
from pmvc.serve import LONGEST, PACE, Model, load, spoken, which

REPO: str = os.environ.get("PMVC_MODELS", "takuma104/pmvc-tts")
LANGS: list[str] = [x for x in os.environ.get("PMVC_LANGS", "ja,en").split(",") if x]
ZERO_GPU: bool = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "t", "true")
HAS_GPU: bool = ZERO_GPU or torch.cuda.is_available()
GPU_LABEL: str = "ZeroGPU" if ZERO_GPU else "GPU"
GPU_SECONDS: int = 10  # the ZeroGPU time reserved for a sentence
FRESH: dict[str, str] = {"Cache-Control": "no-cache"}
NAMES: dict[str, str] = {"ja": "日本語", "en": "English"}
FIRST: dict[str, str] = {"ja": "こんにちは。今日はいい天気ですね。", "en": "The quick brown fox jumps over the lazy dog."}
REFUSED: dict[str, str] = {
    "empty": "読める文字がありません。 / There is nothing to read in the text.",
    "phoneme": "このモデルが習っていない音が含まれています。 / The text has a sound the model was not taught.",
    "server": "サーバでこの文を書けませんでした。 / The server could not write this sentence.",
}


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
    return spoken(MODELS, asked, True, "cuda")


def sound(lang: str, text: str, pitch: float, rate: float, device: str) -> tuple[tuple[int, NDArray[np.int16]], str]:
    """The form's sentence: its track from the model, and the track's sound from the engine."""
    if device == "cuda" and not HAS_GPU:
        raise gr.Error("この環境に GPU はありません。 / There is no GPU here.")
    # the body follows the pitch, as on the page (web/voice.js, `shareFor`)
    share = min(1.0, max(0.0, math.log(pitch / 150.0) / math.log(220.0 / 150.0)))
    asked = {"lang": lang, "text": text, "share": share, "pitch": pitch, "rate": rate, "spread": 0.0}
    started = time.perf_counter()
    track, status = written_on_gpu(asked) if device == "cuda" else spoken(MODELS, asked, True, "cpu")
    wrote = time.perf_counter() - started
    if status != 200:
        raise gr.Error(REFUSED.get(track.get("code", ""), str(track.get("error", ""))))
    started = time.perf_counter()
    x = render_track(track)
    made = time.perf_counter() - started
    samples = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    how = f"{len(x) / 48000:.1f} s · {len(track['frames'])} frames × {len(track['keys'])} channels · model {wrote:.2f} s ({GPU_LABEL if device == 'cuda' else 'CPU'}) · voice {made:.2f} s (CPU)"
    return (48000, samples), how


with gr.Blocks(title="PMVC-TTS") as form:
    gr.Markdown(
        """
# PMVC-TTS: the plain form

[← 断面図とつまみのある画面に戻る / back to the page with the vocal tract and the dials](../)

ここでは音をサーバで作ります。モデルが制御列（調音器官の動かし方、21 チャネル × 毎秒 100 フレーム）を書き、
[CHIMERA](https://github.com/jamesondh/chimera-vocal-synth)（Jameson Hodge さんの声道の物理モデル、MIT ライセンス、改変なし）
がそれを音にします。

Here the sound is made on the server: the model writes a control track (how to move the
articulators, 21 channels, 100 frames a second) and CHIMERA, Jameson Hodge's physical model of
the vocal tract (MIT licence, unmodified), turns it into sound.
"""
    )
    first = next(iter(MODELS))
    lang = gr.Radio(label="言語 / Language", choices=[(NAMES.get(k, k), k) for k in MODELS], value=first)
    text = gr.Textbox(label="テキスト / Text", value=FIRST.get(first, ""), lines=3, max_length=LONGEST)
    pitch = gr.Slider(label="声の高さ / Pitch (Hz)", minimum=80, maximum=320, step=1, value=125)
    rate = gr.Slider(label="速さ（毎秒の音素数） / Pace (phonemes a second)", minimum=6, maximum=16, step=0.2, value=round(MODELS[first].pace, 1))
    device = gr.Radio(
        label="モデルを動かす場所 / Where the model runs",
        choices=[("CPU", "cpu")] + ([(GPU_LABEL, "cuda")] if HAS_GPU else []),
        value="cpu",
        info="CPU で足ります（1 文 0.1 秒ほど）。ZeroGPU を選ぶと、あなたの Hugging Face アカウントの GPU 利用枠を使います。 / The CPU is enough (about 0.1 s a sentence). ZeroGPU uses your Hugging Face account's GPU quota."
        if ZERO_GPU
        else None,
    )
    button = gr.Button("話す / Speak", variant="primary")
    heard = gr.Audio(label="音 / Sound", interactive=False, autoplay=True)
    told = gr.Markdown()
    lang.change(lambda k: (FIRST.get(k, ""), round(MODELS[k].pace, 1)), lang, [text, rate], api_visibility="private")
    button.click(sound, [lang, text, pitch, rate, device], [heard, told], api_name="speak")


class Page(StaticFiles):
    """The page's files: a browser asks again before it uses one it has."""

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers.update(FRESH)
        return response


app = FastAPI(title="PMVC-TTS", docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/api/models")
def models() -> JSONResponse:
    return JSONResponse({"langs": list(MODELS), "plain": "gradio/"}, headers=FRESH)


@app.get("/api/voice")
def voice(lang: str = "") -> JSONResponse:
    try:
        return JSONResponse(which(MODELS, lang).about(), headers=FRESH)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, 400, headers=FRESH)


@app.post("/api/speak")
async def speak(request: Request) -> JSONResponse:
    try:
        asked = json.loads((await request.body())[: 1 << 16])
    except ValueError as e:
        return JSONResponse({"error": str(e)}, 400, headers=FRESH)
    answer, status = await run_in_threadpool(spoken, MODELS, asked, False, "cpu")
    return JSONResponse(answer, status, headers=FRESH)


@app.api_route("/pmvc_wasm.wasm", methods=["GET", "HEAD"])
def engine() -> FileResponse:
    return FileResponse(HERE / "pmvc_wasm.wasm", media_type="application/wasm", headers=FRESH)


@app.get("/gradio")
def plain() -> RedirectResponse:
    return RedirectResponse("gradio/")


app = gr.mount_gradio_app(app, form.queue(max_size=30), path="/gradio", ssr_mode=False, show_error=True)
app.mount("/", Page(directory=HERE / "web", html=True))

if __name__ == "__main__":
    print(f"PMVC-TTS: {', '.join(MODELS)}; {GPU_LABEL if HAS_GPU else 'no GPU'}", flush=True)
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "7860")))
