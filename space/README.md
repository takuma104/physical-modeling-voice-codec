---
title: PMVC-TTS
emoji: 🗣️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
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

## What is in this Space

Everything here is assembled from the repository above by `tools/space.sh` and is not edited
by hand.

| | |
|---|---|
| `app.py` | fetches the models and starts the server |
| `pmvc/` | the server (`pmvc.serve`) and the model (`pmvc.tts`) |
| `web/` | the page |
| `pmvc_wasm.wasm` | the CHIMERA engine driven by a control track, for the browser |
| `wheels/` | the same engine as a Python module (it reads English text) |
| `third-party/chimera-vocal-synth/` | CHIMERA's licence, and the CMU Pronouncing Dictionary with its own |

## Licences

This Space's own code is MIT. CHIMERA is MIT (© Jameson Hodge). The CMU Pronouncing Dictionary
is BSD (© Carnegie Mellon University). Japanese text is read by
[pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) (Open JTalk, modified BSD).
The models were taught with Emilia-YODAS (CC BY 4.0), Mozilla Common Voice (CC0) and LibriTTS-R
(CC BY 4.0) and are CC BY 4.0; see the model card.
