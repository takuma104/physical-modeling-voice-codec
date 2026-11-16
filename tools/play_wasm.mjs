// A control track through the browser's player (crates/pmvc-wasm), outside a browser: for
// checking that it sounds as the Python module's does.
//
// usage: node tools/play_wasm.mjs pmvc_wasm.wasm track.tracks.json out.f32
//
// Writes the samples (48 kHz, 32-bit floats, little-endian) the track renders to.
import { readFileSync, writeFileSync } from "node:fs";

const [wasmPath, trackPath, outPath] = process.argv.slice(2);
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const w = instance.exports;
const track = JSON.parse(readFileSync(trackPath, "utf8"));
const { frames, ...header } = track;

function put(bytes) {
  const p = w.pmvc_alloc(bytes.length);
  new Uint8Array(w.memory.buffer, p, bytes.length).set(bytes);
  return p;
}
function said() {
  return new TextDecoder().decode(new Uint8Array(w.memory.buffer, w.pmvc_error_ptr(), w.pmvc_error_len()));
}

const head = new TextEncoder().encode(JSON.stringify(header));
const hp = put(head);
const player = w.player_new(hp, head.length);
w.pmvc_free(hp, head.length);
if (!player) throw new Error(said());

const cols = header.keys.length;
const flat = new Float64Array(frames.length * cols);
frames.forEach((row, i) => flat.set(row, i * cols));
const fp = w.pmvc_alloc(flat.byteLength);
new Float64Array(w.memory.buffer, fp, flat.length).set(flat);
if (!w.player_set_track(player, fp, frames.length, cols, 0)) throw new Error(said());
w.pmvc_free(fp, flat.byteLength);

const perFrame = Math.round(w.pmvc_sample_rate() / header.rate);
const total = (frames.length - 1) * perFrame;
const block = 128 * 64;
const op = w.pmvc_alloc(block * 4);
const out = new Float32Array(total);
const t0 = performance.now();
for (let at = 0; at < total; at += block) {
  const n = Math.min(block, total - at);
  w.player_render(player, op, n);
  out.set(new Float32Array(w.memory.buffer, op, n), at);
}
const seconds = (performance.now() - t0) / 1000;
w.player_free(player);
writeFileSync(outPath, Buffer.from(out.buffer));
console.log(`${total} samples (${(total / w.pmvc_sample_rate()).toFixed(2)} s) in ${seconds.toFixed(2)} s: ${(total / w.pmvc_sample_rate() / seconds).toFixed(1)} times real time`);
