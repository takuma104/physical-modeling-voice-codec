"""The demo's server: text in, a control track out; the browser's simulated voice makes the sound.

usage: python -m pmvc.serve [--model out/cvja/models/ttscv.pt] [--port 7860] [--web web]
                            [--wasm target/wasm32-unknown-unknown/release/pmvc_wasm.wasm]
                            [--voice voices/v1.json] [--male NAME] [--female NAME]

What crosses the wire is the codec's latent: for a sentence, the 21 channels of a control track,
100 frames a second, and nothing of a waveform. The page (`web/`) plays it through the CHIMERA
engine compiled to WebAssembly (`crates/pmvc-wasm`, the code `pmvc_engine` renders with), so
its dials can turn while the voice speaks: the pitch and the brightness are the page's own to
change, and the body, the pace and the pitch's spread are asked of the model again.

  GET  /                the page (the files of `--web`)
  GET  /pmvc_wasm.wasm  the engine
  GET  /api/voice       the pair of voices the model was taught through (`voices/v1.json`'s
                        `low`, `high`, `bright`, `dull`), how the voice is to play a track
                        (`airway`), and what the model's speakers have at the median: `pitch`
                        (Hz), `rate` (phonemes a second) and `spread` (octaves), each for the low
                        voice and the high one
  POST /api/speak       {"text", "share", "pitch", "rate", "spread"} → {"keys", "rate",
                        "frames", "pitch" (the median the frames were written for), "share",
                        "phones" ([phoneme, frames] pairs)}

A model that is told the voice (`pmvc.tts train --by voice`) speaks in the voice asked for. One
with a vector for each speaker can only speak as one of them: `--male` and `--female` name who
(the first of each sex, if not given), and the page's pitch then moves that speaker's contour.
"""

import argparse
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .encoder import pick_device
from .paths import ROOT
from .tts import Speaker, Voice, conditions, load_model, phonemes, phrased, write_track


class Model:
    """A text-to-speech model and the lock on it: one sentence at a time."""

    def __init__(self, path: Path, male: str, female: str, dial: Path) -> None:
        self.device = pick_device()
        # the dial is the page's and the voice file's, not the model's: a model taught before a
        # hand of the dial was there can be played with it
        self.dial: dict[str, Any] = {k: v for k, v in json.loads(dial.read_text()).items() if k in ("bright", "dull")} if dial.exists() else {}
        self.model: Speaker
        self.voice: Voice
        self.model, self.voice = load_model(path, self.device)
        self.lock = threading.Lock()
        self.own = len(self.voice.talkers)
        by_sex = {sex: [t for t in self.voice.talkers if t.sex == sex] for sex in "MF"}
        self.named = {"M": male or (by_sex["M"] or self.voice.talkers)[0].name, "F": female or (by_sex["F"] or self.voice.talkers)[0].name}
        self.median = {
            name: [float(np.median([of(t) for t in by_sex[sex] or self.voice.talkers])) for sex in "MF"]
            for name, of in (("pitch", lambda t: t.f0), ("rate", lambda t: t.rate), ("spread", lambda t: t.pitch[1]))
        }

    def about(self) -> dict[str, Any]:
        d = self.voice.decoder
        airway = {**({} if d.wall_air else {"volumeSource": 0}), **({} if d.closed_area is None else {"closedArea": d.closed_area})}
        return {**(d.family or {}), **self.dial, "airway": airway, "told": self.model.tell is not None, **self.median}

    def speak(self, q: dict[str, Any]) -> dict[str, Any]:
        text = str(q.get("text", "")).strip()[:400]
        if not text:
            raise ValueError("no text")
        share = float(np.clip(float(q.get("share", 0.0)), 0.0, 1.0))
        pitch, rate, spread = (float(q.get(k, 0.0) or 0.0) for k in ("pitch", "rate", "spread"))
        with self.lock, torch.no_grad():
            if self.model.tell is not None:
                # a voice for this sentence: a row more in the table, taken out again after
                who = self.voice.dial(share, pitch, rate, spread)
                self.model.table = torch.cat([self.model.table[: self.own], conditions([self.voice.talkers[who]]).to(self.model.table.device)])
            else:
                who = self.voice.who(self.named["F" if share >= 0.5 else "M"])
            try:
                phones = phrased(self.model, phonemes(self.model, text), self.device, who)
                frames, lasting = write_track(self.model, self.voice, phones, self.device, None, who, rate)
                talker = self.voice.talkers[who]
                frames = self.voice.decoder.hold(frames, self.voice.keys, talker.high)
                return {
                    "keys": self.voice.keys, "rate": 100, "frames": np.round(frames, 4).tolist(), "pitch": talker.f0, "share": talker.high,
                    "phones": [[p.p, int(n)] for p, n in zip(phones, lasting)],
                }  # fmt: skip
            finally:
                del self.voice.talkers[self.own :]


class Handler(SimpleHTTPRequestHandler):
    model: Model
    wasm: Path

    def reply(self, body: bytes, kind: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/pmvc_wasm.wasm":
            self.reply(self.wasm.read_bytes(), "application/wasm")
        elif self.path == "/api/voice":
            self.reply(json.dumps(self.model.about()).encode(), "application/json")
        else:
            super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/speak":
            self.reply(b"{}", "application/json", 404)
            return
        try:
            asked = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 1 << 16)))
            self.reply(json.dumps(self.model.speak(asked)).encode(), "application/json")
        except (ValueError, KeyError) as e:
            self.reply(json.dumps({"error": str(e)}).encode(), "application/json", 400)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=ROOT / "out" / "cvja" / "models" / "ttscv.pt")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--web", type=Path, default=ROOT / "web")
    ap.add_argument("--wasm", type=Path, default=ROOT / "target" / "wasm32-unknown-unknown" / "release" / "pmvc_wasm.wasm")
    ap.add_argument("--voice", type=Path, default=ROOT / "voices" / "v1.json", help="the pair of voices' file, for what the brightness dial does")
    ap.add_argument("--male", type=str, default="", help="for a model with a vector for each speaker: who speaks in the low voice")
    ap.add_argument("--female", type=str, default="", help="and who in the high one")
    a = ap.parse_args()
    Handler.model = Model(a.model, a.male, a.female, a.voice)
    Handler.wasm = a.wasm
    server = ThreadingHTTPServer((a.host, a.port), partial(Handler, directory=str(a.web)))
    print(f"http://{a.host}:{a.port}/  ({a.model.name}; {'told the voice' if Handler.model.model.tell is not None else 'one of its speakers'})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
