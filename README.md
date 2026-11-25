# physical-modeling-voice-codec

[日本語](README_ja.md)

A neural speech codec whose decoder is a physical model of the voice.
The latent is a set of control parameters for articulation and the voice source, and the decoder is a physical simulation of the vocal tract. The hypothesis this repository tests: with such a decoder, a latent that is somewhat wrong still comes out as a sound a human voice could make.

- **Demo** (nothing to install): <https://huggingface.co/spaces/takuma104/pmvc-tts?en> (日本語: [`?ja`](https://huggingface.co/spaces/takuma104/pmvc-tts?ja))
- **Model weights** (CC BY 4.0): <https://huggingface.co/takuma104/pmvc-tts>
- **Technical report** (draft): [English](docs/report/physical_modeling_voice_codec_report_en.pdf) · [日本語](docs/report/physical_modeling_voice_codec_report_ja.pdf)
- **Running the experiments**: [docs/experiments.md](docs/experiments.md)

The first idea memo and its survey of earlier work are in [docs/plans/idea.md](docs/plans/idea.md) (in Japanese). Section 9 of the report sets this work against prior work.

## Quick start: PMVC-TTS, the text-to-speech demo

A model that writes a control track from text (21 channels × 100 Hz), and a page that turns the track into sound inside the browser. It speaks Japanese and English.
The server returns only the control track; the sound is made in the browser by the CHIMERA engine (WebAssembly). The pitch and the brightness of the voice can be changed while it speaks.

Needs Rust (with the `wasm32-unknown-unknown` target), [uv](https://docs.astral.sh/uv/) and a browser.

```sh
git clone --recursive https://github.com/takuma104/physical-modeling-voice-codec && cd physical-modeling-voice-codec
rustup target add wasm32-unknown-unknown
uv sync                                                                    # the Python environment; this also builds pmvc-engine (Rust)
cargo build -p pmvc-wasm --release --target wasm32-unknown-unknown         # the engine for the browser (667 kB)
uvx --from huggingface_hub hf download takuma104/pmvc-tts --local-dir models
uv run python -m pmvc.serve --model models/ja.json models/en.json          # → http://127.0.0.1:7860/
```

The page comes up in the browser's language. `?ja` or `?en` in the address names the language (`http://127.0.0.1:7860/?en`).

The page is remade after CHIMERA's Speak screen and is not CHIMERA itself. The simulation of the vocal tract is [CHIMERA](https://github.com/jamesondh/chimera-vocal-synth) by Jameson Hodge (MIT), used unmodified. To make the voice sing, or to move the tongue and lips by hand, go to the original.

## What is here

| Path | What it is |
|---|---|
| `third-party/chimera-vocal-synth` | CHIMERA, the physical-modeling voice synthesizer (a submodule, used unmodified) |
| `third-party/arti-6` | A model that estimates six-dimensional articulatory features from speech and synthesizes from them (a submodule) |
| `third-party/NISQA` | A model that predicts perceived quality from the audio alone (a submodule, used unmodified; its weights are CC BY-NC-SA 4.0) |
| `crates/pmvc-synth` | A CLI that drives CHIMERA with JSON: `render` takes the phonemes' durations and the F0 from outside, `drive` plays a control track as it is |
| `crates/pmvc-engine` | An extension module (PyO3) that calls CHIMERA from Python. It has `Voice.fork()`, which branches a voice together with its physical state; `Voice.set_glottis()` and `Voice.set_airway()`, which set how the glottis and a constriction close; `Voice.set_nose()`, which sets the shape of the nasal cavity (areas, length, the sinus's opening and volume); a way to give the body as four numbers (size, the pharynx's share, the tongue's radius, the lips' width) instead of a mix of the three presets; `Voice.probe`, which reads the internal state (areas of the glottis and of constrictions, pressures, flows); and `outline()`, which returns the outlines of the tract's section in a posture (figure 1 of the report is drawn with it) |
| `crates/pmvc-voice` | What plays a control track on CHIMERA. `pmvc-engine` and `pmvc-wasm` share it, so the same control track gives the same samples in Python and in the browser |
| `crates/pmvc-wasm` | The engine for the browser (a C ABI with no imports; it runs inside an AudioWorklet). It also returns the geometry the cross-section is drawn from |
| `web/` | The demo's page (static files, nothing to build), served by `pmvc.serve` |
| `space/` | `app.py` and the rest of the Hugging Face Space (Gradio SDK, CPU inference on ZeroGPU hardware), assembled by `tools/space.sh` |
| `voices/v1.json` | The two voices the models were taught with (a low one and a high one), and what the brightness dial does |
| `pmvc/` | The experiments' scripts (Python). The codec's encoder is `pmvc.encoder`, text-to-speech is `pmvc.tts`, and the control with neural decoders is `pmvc.control` and `pmvc.vocoder` |
| `docs/` | The technical report (`docs/report/`, English and Japanese), how to run the experiments (`docs/experiments.md`), and the idea memo and notes on related papers (in Japanese) |

## Experiments

How to run the report's experiments again (setup, the data they need, the commands) is in [docs/experiments.md](docs/experiments.md).
