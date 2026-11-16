"""The demo's server: text in, a control track out; the browser's simulated voice makes the sound.

usage: python -m pmvc.serve [--model MODEL.pt [MODEL.pt ...]] [--port 7860] [--host 127.0.0.1]
                            [--web web] [--wasm target/wasm32-unknown-unknown/release/pmvc_wasm.wasm]
                            [--voice voices/v1.json] [--pace ja=10.2,...] [--male NAME] [--female NAME]

What crosses the wire is the codec's latent: for a sentence, the 21 channels of a control track,
100 frames a second, and nothing of a waveform. The page (`web/`) plays it through the CHIMERA
engine compiled to WebAssembly (`crates/pmvc-wasm`, the code `pmvc_engine` renders with), so
its dials can turn while the voice speaks: the pitch and the brightness are the page's own to
change, and the body, the pace and the pitch's spread are asked of the model again.

A model speaks one language (the one its recordings were in: `pmvc.tts train`), and the server
has a model for each language it is given; the page offers the languages there are.

  GET  /                the page (the files of `--web`)
  GET  /pmvc_wasm.wasm  the engine
  GET  /api/models      {"langs": the languages there are a model for, the first the default}
  GET  /api/voice?lang= the pair of voices the language's model was taught through
                        (`voices/v1.json`'s `low`, `high`, `bright`, `dull`), how the voice is to
                        play a track (`airway`), what the model's speakers have at the median:
                        `pitch` (Hz), `rate` (phonemes a second) and `spread` (octaves), each for
                        the low voice and the high one, the `pace` to begin with, and the
                        `keys` of a track's columns
  POST /api/speak       {"lang", "text", "share", "pitch", "rate", "spread"} → {"keys", "rate",
                        "frames", "pitch" (the median the frames were written for), "share",
                        "phones" ([phoneme, frames] pairs)}, or {"error", "code"}: `empty`
                        (nothing to read), `phoneme` (a sound the model was not taught) or
                        `server`

A model that is told the voice (`pmvc.tts train --by voice`) speaks in the voice asked for. One
with a vector for each speaker can only speak as one of them: `--male` and `--female` name who
(the first of each sex, if not given), and the page's pitch then moves that speaker's contour.

The pace to begin with is the median of the model's speakers, or what `--pace` says for the
language. People talking and volunteers reading off a screen are a fifth faster than a studio's
readers, and it is the studio's pace (JVS: 10.2 phonemes a second) that sounds unhurried in
Japanese, so that is where the Japanese dial starts.
"""

import argparse
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import torch

from .encoder import pick_device
from .paths import ROOT
from .tts import Speaker, Voice, conditions, load_model, phonemes, phrased, write_track

# the pace a language's dial starts at (phonemes a second), where it is not the model's median
PACE: dict[str, float] = {"ja": 10.2}
# the longest text read at once (characters)
LONGEST: int = 300


class Refused(ValueError):
    """A sentence the server will not write, and why, in a word the page has its own words for."""

    def __init__(self, code: str, text: str) -> None:
        super().__init__(text)
        self.code = code


class Model:
    """A text-to-speech model and the lock on it: one sentence at a time."""

    def __init__(self, path: Path, male: str, female: str, dial: Path, pace: dict[str, float]) -> None:
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
        self.lang: str = self.model.lang
        self.pace = pace.get(self.lang, float(np.mean(self.median["rate"])))

    def about(self) -> dict[str, Any]:
        d = self.voice.decoder
        airway = {**({} if d.wall_air else {"volumeSource": 0}), **({} if d.closed_area is None else {"closedArea": d.closed_area})}
        return {**(d.family or {}), **self.dial, "airway": airway, "told": self.model.tell is not None, **self.median, "pace": self.pace, "lang": self.lang, "keys": self.voice.keys}

    def speak(self, q: dict[str, Any]) -> dict[str, Any]:
        text = str(q.get("text", "")).strip()[:LONGEST]
        if not text:
            raise Refused("empty", "no text")
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
                try:
                    read = phonemes(self.model, text)
                except ValueError as e:
                    raise Refused("phoneme", str(e)) from e
                if not any(not p.silent for p in read):
                    raise Refused("empty", "nothing in the text that the language's front end can read")
                phones = phrased(self.model, read, self.device, who)
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
    models: dict[str, Model]
    wasm: Path

    def reply(self, body: bytes, kind: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def said(self, answer: Any, status: int = 200) -> None:
        self.reply(json.dumps(answer).encode(), "application/json", status)

    def model(self, lang: str) -> Model:
        if lang and lang not in self.models:
            raise ValueError(f"no model for the language {lang!r}")
        return self.models[lang or next(iter(self.models))]

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        try:
            if url.path == "/pmvc_wasm.wasm":
                self.reply(self.wasm.read_bytes(), "application/wasm")
            elif url.path == "/api/models":
                self.said({"langs": list(self.models)})
            elif url.path == "/api/voice":
                self.said(self.model("".join(parse_qs(url.query).get("lang", [""]))).about())
            else:
                super().do_GET()
        except ValueError as e:
            self.said({"error": str(e)}, 400)

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/speak":
            self.said({}, 404)
            return
        try:
            asked = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 1 << 16)))
            self.said(self.model(str(asked.get("lang", ""))).speak(asked))
        except Refused as e:
            self.said({"error": str(e), "code": e.code}, 400)
        except (ValueError, KeyError, TypeError) as e:
            self.said({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            # whatever else goes wrong with a sentence is the server's to log, and the page's to be told of
            print(f"speak: {type(e).__name__}: {e}", flush=True)
            self.said({"error": "the server could not write this sentence", "code": "server"}, 500)

    def end_headers(self) -> None:
        # the page's files change with the server: a browser asks again before it uses one it has
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, nargs="+", default=[ROOT / "out" / "cvja" / "models" / "ttscv2.pt"], help="a model for each language; the first is the page's to begin with")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--web", type=Path, default=ROOT / "web")
    ap.add_argument("--wasm", type=Path, default=ROOT / "target" / "wasm32-unknown-unknown" / "release" / "pmvc_wasm.wasm")
    ap.add_argument("--voice", type=Path, default=ROOT / "voices" / "v1.json", help="the pair of voices' file, for what the brightness dial does")
    ap.add_argument("--pace", type=str, default="", help="the pace a language's dial starts at, in phonemes a second: ja=10.2,en=12 (otherwise PACE's, or the median of the model's speakers)")
    ap.add_argument("--male", type=str, default="", help="for a model with a vector for each speaker: who speaks in the low voice")
    ap.add_argument("--female", type=str, default="", help="and who in the high one")
    a = ap.parse_args()
    pace = {**PACE, **{k: float(v) for k, v in (x.split("=") for x in filter(None, a.pace.split(",")))}}
    Handler.models = {}
    for path in a.model:
        m = Model(path, a.male, a.female, a.voice, pace)
        assert m.lang not in Handler.models, f"two models for {m.lang}"
        Handler.models[m.lang] = m
    Handler.wasm = a.wasm
    server = ThreadingHTTPServer((a.host, a.port), partial(Handler, directory=str(a.web)))
    told = ", ".join(f"{lang}: {m.model.__class__.__name__} of {path.name}, {'told the voice' if m.model.tell is not None else 'one of its speakers'}" for (lang, m), path in zip(Handler.models.items(), a.model))
    print(f"http://{a.host}:{a.port}/  ({told})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
