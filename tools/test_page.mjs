// The page's own path, outside a browser: a sentence asked of a running `pmvc.serve`, the
// header and frames web/voice.js makes of the answer, played by web/worklet.js's class.
//
// usage: node tools/test_page.mjs http://127.0.0.1:7860 pmvc_wasm.wasm out.f32 [share pitch bright]
//
// Writes the samples (f32) and, beside them, <out>.json: the track as it was played, for the
// Python module to render the same.
import { readFileSync, writeFileSync } from "node:fs";
import { framesOf, glottisPatch, headerOf } from "../web/voice.js";

const [base, wasmPath, outPath, share = "0", pitch = "125", bright = "0"] = process.argv.slice(2);
let Processor = null;
const sent = [];
globalThis.AudioWorkletProcessor = class {
  constructor() {
    this.port = { postMessage: (m) => sent.push(m), onmessage: null };
  }
};
globalThis.registerProcessor = (_name, cls) => (Processor = cls);
await import(new URL("../web/worklet.js", import.meta.url));

const about = await (await fetch(`${base}/api/voice`)).json();
const asked = { text: "こんにちは。声道の物理モデルで話しています。", share: Number(share), pitch: Number(pitch), rate: 10, spread: 0 };
const track = await (await fetch(`${base}/api/speak`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(asked) })).json();
const header = headerOf(about, track, Number(bright), Number(pitch));
const flat = framesOf(about, track, Number(bright), Number(pitch));

const p = new Processor();
p.port.onmessage({ data: { type: "wasm", bytes: readFileSync(wasmPath) } });
p.port.onmessage({ data: { type: "track", header: new TextEncoder().encode(JSON.stringify(header)), frames: flat.slice(), cols: track.keys.length, rate: track.rate } });
// a dial turned to where it is: nothing may change
p.port.onmessage({ data: { type: "patch", kind: 1, bytes: new TextEncoder().encode(JSON.stringify(glottisPatch(about, track.share, Number(bright)))) } });
p.port.onmessage({ data: { type: "frames", frames: flat.slice() } });
const errors = sent.filter((m) => m.type === "error");
if (errors.length) throw new Error(errors[0].text);
const total = (track.frames.length - 1) * 480;
const got = new Float32Array(total);
for (let at = 0; at < total; at += 128) {
  const out = [[new Float32Array(128), new Float32Array(128)]];
  p.process([], out);
  got.set(out[0][0].subarray(0, Math.min(128, total - at)), at);
}
writeFileSync(outPath, Buffer.from(got.buffer));
const cols = track.keys.length;
const rows = Array.from({ length: track.frames.length }, (_, i) => Array.from(flat.subarray(i * cols, (i + 1) * cols)));
writeFileSync(`${outPath}.json`, JSON.stringify({ ...header, frames: rows }));
console.log(`${track.frames.length} frames, share ${track.share}, ended: ${sent.some((m) => m.type === "ended")}, peak ${got.reduce((m, x) => Math.max(m, Math.abs(x)), 0).toFixed(3)}`);
