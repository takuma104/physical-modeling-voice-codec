// web/worklet.js outside a browser: the worklet's class with stand-ins for what the audio
// thread gives it, fed a track and asked for quanta, against tools/play_wasm.mjs's samples.
//
// usage: node tools/test_worklet.mjs pmvc_wasm.wasm track.tracks.json expected.f32
import { readFileSync } from "node:fs";

const [wasmPath, trackPath, expectedPath] = process.argv.slice(2);
let Processor = null;
const sent = [];
globalThis.AudioWorkletProcessor = class {
  constructor() {
    this.port = { postMessage: (m) => sent.push(m), onmessage: null };
  }
};
globalThis.registerProcessor = (_name, cls) => (Processor = cls);
await import(new URL("../web/worklet.js", import.meta.url));

const p = new Processor();
p.port.onmessage({ data: { type: "wasm", bytes: readFileSync(wasmPath) } });
const { frames, ...header } = JSON.parse(readFileSync(trackPath, "utf8"));
const cols = header.keys.length;
const flat = new Float64Array(frames.length * cols);
frames.forEach((row, i) => flat.set(row, i * cols));
p.port.onmessage({ data: { type: "track", header: new TextEncoder().encode(JSON.stringify(header)), frames: flat, cols, rate: header.rate } });
const errors = sent.filter((m) => m.type === "error");
if (errors.length) throw new Error(errors[0].text);

const expected = new Float32Array(readFileSync(expectedPath).buffer.slice(0));
const got = new Float32Array(expected.length);
let at = 0;
while (!sent.some((m) => m.type === "ended") && at < expected.length + 4800) {
  const out = [[new Float32Array(128), new Float32Array(128)]];
  p.process([], out);
  got.set(out[0][0].subarray(0, Math.min(128, expected.length - at)), Math.min(at, expected.length));
  at += 128;
}
let worst = 0;
for (let i = 0; i < expected.length; i++) worst = Math.max(worst, Math.abs(expected[i] - got[i]));
console.log(`${expected.length} samples; ended: ${sent.some((m) => m.type === "ended")}; progress messages: ${sent.filter((m) => m.type === "at").length}; max |difference| from play_wasm: ${worst}`);
