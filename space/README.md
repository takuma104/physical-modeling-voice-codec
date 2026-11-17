---
title: PMVC-TTS
emoji: 🗣️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Text-to-speech through a physical model of the vocal tract
models:
  - takuma104/pmvc-tts
---

# PMVC-TTS

Text-to-speech in Japanese and English whose sound is not made by a neural network.

A small network writes a **control track**: 21 channels, 100 frames a second, that say how to
move the tongue, the jaw, the lips, the soft palate and the vocal folds. The sound is made in
your browser by [CHIMERA](https://github.com/jamesondh/chimera-vocal-synth), Jameson Hodge's
physical simulation of the airway (MIT licence), used unmodified and compiled to WebAssembly.
This server sends the control track of a sentence, some tens of kilobytes of numbers, and
nothing of a waveform. Because the voice is simulated where you listen, its pitch and its
brightness can be turned while it speaks.

- Code and technical report: <https://github.com/takuma104/physical-modeling-voice-codec>
- Model weights (CC BY 4.0): <https://huggingface.co/takuma104/pmvc-tts>
- The voice itself, with far more to play with (singing, the articulators by hand, animals):
  <https://github.com/jamesondh/chimera-vocal-synth>

## The address

The page comes up in the browser's language, Japanese or English. To send someone to one of
them, name it in the address:

- `https://huggingface.co/spaces/takuma104/pmvc-tts?ja` 日本語
- `https://huggingface.co/spaces/takuma104/pmvc-tts?en` English

## What is in this Space

Everything here is assembled from the repository above by `tools/space.sh` and is not edited
by hand.

| | |
|---|---|
| `app.py` | the application: the page, its control tracks, and the same call as a Gradio API |
| `pmvc/` | the server's answers (`pmvc.serve`) and the model (`pmvc.tts`) |
| `web/` | the page |
| `pmvc_wasm.wasm` | the CHIMERA engine driven by a control track, for the browser |
| `engine/` | the same engine as a Python module (CHIMERA's English front end is in it) |
| `models/` | the models, as published at <https://huggingface.co/takuma104/pmvc-tts> |
| `third-party/chimera-vocal-synth/` | CHIMERA's licence, and the CMU Pronouncing Dictionary with its own |

The Space is on ZeroGPU hardware, which runs Gradio applications only, and the page is not a
Gradio one. So `app.py` is Gradio's Server mode (`gr.Server`): a FastAPI application that
serves the page and answers it, with Gradio's API engine beside it. Programs can ask for a
sentence's control track through that:

```python
from gradio_client import Client

client = Client("takuma104/pmvc-tts")
track = client.predict(lang="ja", text="こんにちは。", pitch=125.0, rate=10.2, api_name="/speak")
# track["frames"]: a row for each frame (10 ms) of track["keys"], the 21 channels
```

The models are small: the Space's CPU writes a sentence in about a tenth of a second, so that
is where they run. Through the API a sentence can be sent to the GPU instead (`device="cuda"`),
which is taken from the caller's ZeroGPU quota.

## Licences

This Space's own code is MIT. CHIMERA is MIT (© Jameson Hodge). The CMU Pronouncing Dictionary
is BSD (© Carnegie Mellon University). Japanese text is read by
[pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) (Open JTalk, modified BSD).
The models were taught with Emilia-YODAS (CC BY 4.0), Mozilla Common Voice (CC0) and LibriTTS-R
(CC BY 4.0) and are CC BY 4.0; see the model card.
