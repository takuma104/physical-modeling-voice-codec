// The page: asks the server for a sentence's control track, hands it to the voice in the audio
// thread (worklet.js), and turns the voice's dials (voice.js says what each position means).

import { clamp, framesOf, glottisPatch, headerOf, shareFor } from "./voice.js";

const $ = (id) => document.getElementById(id);
const el = { text: $("text"), speak: $("speak"), stop: $("stop"), status: $("status"), view: $("view"), phones: $("phones") };
const dial = Object.fromEntries(["pitch", "bright", "share", "rate", "spread"].map((k) => [k, $(k)]));
const shown = Object.fromEntries(Object.keys(dial).map((k) => [k, $(`${k}-out`)]));

let about = null;     // the model's voice, from the server
let audio = null;     // the AudioContext, made at the first click
let node = null;      // the voice
let track = null;     // what is playing: {keys, frames (rows), pitch, share, phones}
let playing = false;
let at = 0;           // the frame now sounding

const bytes = (x) => new TextEncoder().encode(JSON.stringify(x));

async function start() {
  audio = new AudioContext({ sampleRate: 48000 });
  await audio.audioWorklet.addModule("worklet.js");
  node = new AudioWorkletNode(audio, "pmvc-player", { numberOfInputs: 0, outputChannelCount: [2] });
  node.connect(audio.destination);
  const ready = new Promise((ok) => {
    node.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === "ready") ok();
      else if (m.type === "at") at = m.frame;
      else if (m.type === "ended") stopped();
      else if (m.type === "error") { el.status.textContent = `声のエラー: ${m.text}`; stopped(); }
    };
  });
  const wasm = await (await fetch("pmvc_wasm.wasm")).arrayBuffer();
  node.port.postMessage({ type: "wasm", bytes: wasm }, [wasm]);
  await ready;
}

function stopped() {
  playing = false;
  el.stop.disabled = true;
  el.speak.disabled = false;
}

async function speak() {
  el.speak.disabled = true;
  el.status.textContent = "制御列を書いています…";
  try {
    if (!audio) await start();
    await audio.resume();
    const pitch = Number(dial.pitch.value);
    const share = Number(dial.share.value);
    const asked = { text: el.text.value, share, pitch, rate: Number(dial.rate.value), spread: Number(dial.spread.value) * ((1 - share) * about.spread[0] + share * about.spread[1]) };
    const r = await fetch("api/speak", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(asked) });
    const got = await r.json();
    if (!r.ok) throw new Error(got.error || r.statusText);
    track = got;
    at = 0;
    const bright = Number(dial.bright.value);
    const flat = framesOf(about, track, bright, pitch);
    node.port.postMessage({ type: "track", header: bytes(headerOf(about, track, bright, pitch)), frames: flat, cols: track.keys.length, rate: track.rate }, [flat.buffer]);
    playing = true;
    el.stop.disabled = false;
    el.status.textContent = `${(track.frames.length / track.rate).toFixed(1)} 秒、${track.frames.length} フレーム × ${track.keys.length} チャネル`;
    el.phones.textContent = track.phones.map(([p]) => p).join(" ");
    draw();
  } catch (err) {
    el.status.textContent = `できませんでした: ${err.message}`;
    stopped();
  }
}

// A dial that the voice can follow while it speaks.
function turned() {
  if (!playing || !track) return;
  const bright = Number(dial.bright.value);
  node.port.postMessage({ type: "patch", kind: 1, bytes: bytes(glottisPatch(about, track.share, bright)) });
  const flat = framesOf(about, track, bright, Number(dial.pitch.value));
  node.port.postMessage({ type: "frames", frames: flat }, [flat.buffer]);
}

function show() {
  shown.pitch.textContent = `${dial.pitch.value} Hz`;
  shown.bright.textContent = Number(dial.bright.value) > 0 ? `+${dial.bright.value}` : dial.bright.value;
  shown.share.textContent = Number(dial.share.value).toFixed(2);
  shown.rate.textContent = `${Number(dial.rate.value).toFixed(1)} 音素/秒`;
  shown.spread.textContent = `× ${Number(dial.spread.value).toFixed(2)}`;
}

// A few channels of the track over time, and where the voice is.
const DRAWN = [["f0", "声の高さ"], ["jaw", "あご"], ["tbY", "舌の高さ"], ["ttY", "舌先の高さ"], ["lipAp", "唇の開き"], ["velum", "鼻への通路"], ["breath", "息"]];
function draw() {
  const c = el.view;
  const g = c.getContext("2d");
  const css = getComputedStyle(document.documentElement);
  g.clearRect(0, 0, c.width, c.height);
  if (!track) return;
  const n = track.frames.length;
  const rowH = c.height / DRAWN.length;
  g.font = "13px system-ui, sans-serif";
  DRAWN.forEach(([key, label], r) => {
    const k = track.keys.indexOf(key);
    if (k < 0) return;
    const xs = track.frames.map((row) => row[k]);
    const voiced = key === "f0" ? xs.filter((x) => x > 1) : xs;
    const lo = Math.min(...voiced), hi = Math.max(...voiced);
    g.strokeStyle = css.getPropertyValue("--line");
    g.strokeRect(0, r * rowH, c.width, rowH);
    g.beginPath();
    g.strokeStyle = css.getPropertyValue("--accent");
    g.lineWidth = 1.5;
    xs.forEach((x, i) => {
      const y = (r + 0.9) * rowH - ((clamp(x, lo, hi) - lo) / (hi - lo || 1)) * rowH * 0.8;
      if (i === 0) g.moveTo(0, y); else g.lineTo((i / (n - 1)) * c.width, y);
    });
    g.stroke();
    g.fillStyle = css.getPropertyValue("--soft");
    g.fillText(label, 6, r * rowH + 15);
  });
  g.fillStyle = css.getPropertyValue("--mark");
  g.fillRect((at / Math.max(1, n - 1)) * c.width - 1, 0, 2, c.height);
  if (playing) requestAnimationFrame(draw);
}

el.speak.addEventListener("click", speak);
el.stop.addEventListener("click", () => node && node.port.postMessage({ type: "stop" }));
for (const k of Object.keys(dial)) {
  dial[k].addEventListener("input", () => {
    if (k === "pitch" && $("follow").checked) dial.share.value = shareFor(Number(dial.pitch.value));
    if (k === "share") $("follow").checked = false;
    show();
    if (k === "pitch" || k === "bright") turned();
  });
}
$("follow").addEventListener("change", () => {
  if ($("follow").checked) dial.share.value = shareFor(Number(dial.pitch.value));
  show();
});

about = await (await fetch("api/voice")).json();
dial.rate.value = ((about.rate[0] + about.rate[1]) / 2).toFixed(1);
show();
