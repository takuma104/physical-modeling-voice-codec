// The page: asks the server for a sentence's control track, hands it to the voice in the audio
// thread (worklet.js), turns the voice's dials (voice.js says what each position means), and
// draws what the voice is doing (draw.js).

import { clamp, framesOf, glottisPatch, headerOf, partOf, shareFor } from "./voice.js";
import { Body, Score, drawTract, drawTube } from "./draw.js";
import { CHANNELS, EXAMPLES, LANGS, WORDS } from "./words.js";

const $ = (id) => document.getElementById(id);
const el = { text: $("text"), speak: $("speak"), stop: $("stop"), status: $("status"), phones: $("phones"), examples: $("examples"), langs: $("langs"), tract: $("tract"), tube: $("tube"), follow: $("follow"), names: $("names") };
const dial = Object.fromEntries(["pitch", "bright", "share", "rate", "spread"].map((k) => [k, $(k)]));
const shown = Object.fromEntries(Object.keys(dial).map((k) => [k, $(`${k}-out`)]));
const read = Object.fromEntries(["sound", "voice", "jaw", "lips", "velum"].map((k) => [k, $(`r-${k}`)]));
const score = new Score($("score"));
const QUIET = ["sil", "pau", "sp"];

let lang = "ja";
let words = WORDS.ja;
const abouts = {};    // what the server says of each language's model
let about = null;     // the one of the language now
let wasm = null;      // the engine's bytes
let body = null;      // the engine's geometry, in this thread
let audio = null;     // the AudioContext, made at the first click
let node = null;      // the voice
let track = null;     // the last track: {id, keys, frames (rows), rate, pitch, share, phones, starts, flat, cols}
let playing = false;
let heard = { frame: 0, time: 0, voicing: 0 };   // the audio thread's last word of where it is
let serial = 0;
let now = -1;         // the phoneme marked in the transcription
let named = false;    // whether the parts' names are written on the section

const bytes = (x) => new TextEncoder().encode(JSON.stringify(x));
const put = (e, text) => { if (e.textContent !== text) e.textContent = text; };
const rows = () => CHANNELS.filter(([key]) => about.keys.includes(key)).map(([key, ja, en]) => [key, lang === "ja" ? ja : en]);

async function start() {
  audio = new AudioContext({ sampleRate: 48000 });
  await audio.audioWorklet.addModule("worklet.js");
  node = new AudioWorkletNode(audio, "pmvc-player", { numberOfInputs: 0, outputChannelCount: [2] });
  node.connect(audio.destination);
  const ready = new Promise((ok) => {
    node.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === "ready") ok();
      else if (m.type === "at") heard = { frame: m.frame, time: performance.now(), voicing: m.voicing };
      else if (m.type === "ended") stopped();
      else if (m.type === "error") { el.status.textContent = `${words["voice-error"]}: ${m.text}`; stopped(); }
    };
  });
  const copy = wasm.slice(0);
  node.port.postMessage({ type: "wasm", bytes: copy }, [copy]);
  await ready;
}

function stopped() {
  playing = false;
  el.stop.disabled = true;
  mark(-1);
  rest();
}

async function speak() {
  el.speak.disabled = true;
  el.speak.textContent = words.preparing;
  try {
    if (!audio) await start();
    await audio.resume();
    const pitch = Number(dial.pitch.value);
    const share = Number(dial.share.value);
    const asked = { lang, text: el.text.value, share, pitch, rate: Number(dial.rate.value), spread: Number(dial.spread.value) * ((1 - share) * about.spread[0] + share * about.spread[1]) };
    const r = await fetch("api/speak", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(asked) });
    const raw = await r.text();
    const got = JSON.parse(raw);
    if (!r.ok) throw new Error(words.errors[got.code] || got.error || r.statusText);
    let at = 0;
    track = { ...got, id: ++serial, cols: got.keys.length, starts: got.phones.map(([, n]) => (at += n) - n) };
    const bright = Number(dial.bright.value);
    const header = headerOf(about, track, bright, pitch);
    const flat = framesOf(about, track, bright, pitch);
    track.flat = flat.slice();
    body.of({ body: header.body, keys: track.keys });
    node.port.postMessage({ type: "track", header: bytes(header), frames: flat, cols: track.cols, rate: track.rate }, [flat.buffer]);
    heard = { frame: 0, time: performance.now(), voicing: 0 };
    el.status.textContent = words.status((track.frames.length / track.rate).toFixed(1), track.frames.length, track.cols, Math.round(raw.length / 1000));
    el.phones.replaceChildren(...track.phones.flatMap(([p], i) => {
      const s = document.createElement("span");
      s.textContent = p;
      if (QUIET.includes(p)) s.className = "quiet";
      return i ? [" ", s] : [s];
    }));
    now = -1;
    el.stop.disabled = false;
    if (!playing) {
      playing = true;
      requestAnimationFrame(tick);
    }
  } catch (err) {
    el.status.textContent = `${words.failed}: ${err.message}`;
    if (playing) node.port.postMessage({ type: "stop" });
  } finally {
    el.speak.disabled = false;
    el.speak.textContent = words.speak;
  }
}

// A dial that the voice can follow while it speaks.
function turned() {
  if (!playing || !track) return;
  const bright = Number(dial.bright.value);
  node.port.postMessage({ type: "patch", kind: 1, bytes: bytes(glottisPatch(about, track.share, bright)) });
  const flat = framesOf(about, track, bright, Number(dial.pitch.value));
  track.flat = flat.slice();
  node.port.postMessage({ type: "frames", frames: flat }, [flat.buffer]);
}

function show() {
  put(shown.pitch, `${dial.pitch.value} Hz`);
  put(shown.bright, Number(dial.bright.value) > 0 ? `+${dial.bright.value}` : dial.bright.value);
  put(shown.share, Number(dial.share.value).toFixed(2));
  put(shown.rate, `${Number(dial.rate.value).toFixed(1)} ${words["rate-unit"]}`);
  put(shown.spread, `× ${Number(dial.spread.value).toFixed(2)}`);
}

// The phoneme now sounding, in the transcription.
function mark(k) {
  if (k === now) return;
  const spans = el.phones.querySelectorAll("span");
  if (now >= 0 && spans[now]) spans[now].classList.remove("now");
  if (k >= 0 && spans[k]) spans[k].classList.add("now");
  now = k;
}

function readouts(f, row, sound, voicing) {
  const k = (name) => track.keys.indexOf(name);
  put(read.sound, sound || "–");
  put(read.voice, row ? (voicing > 0.3 ? `${words.voiced} ${Math.round(row[k("f0")])} Hz` : words.voiceless) : "–");
  put(read.jaw, row ? `${row[k("jaw")].toFixed(1)} cm` : "–");
  put(read.lips, f ? `${f.lipArea.toFixed(1)} cm²` : "–");
  put(read.velum, f ? (f.velumPort > 0.02 ? words.open : words.shut) : "–");
}

// The body at rest, as the body dial has it.
function rest() {
  if (!body || !about) return;
  body.of({ body: partOf(about, "body", Number(dial.share.value)), keys: about.keys });
  const f = body.frame(new Float64Array(about.keys.length).fill(NaN));
  drawTract(el.tract, f, null, named ? words.parts : null);
  drawTube(el.tube, f, words);
  score.draw(track, rows(), -1);
  readouts(f, null, "", 0);
}

// A picture a display frame: the body where the track has it now.
function tick() {
  if (!playing || !track) return;
  const n = track.frames.length, c = track.cols;
  const late = (audio.baseLatency || 0) + (audio.outputLatency || 0);
  const ahead = Math.min(4, ((performance.now() - heard.time) * track.rate) / 1000);
  const pos = clamp(heard.frame + ahead - late * track.rate, 0, n - 1);
  const i = Math.floor(pos), j = Math.min(n - 1, i + 1), u = pos - i;
  const row = new Float64Array(c);
  for (let k = 0; k < c; k++) row[k] = track.flat[i * c + k] * (1 - u) + track.flat[j * c + k] * u;
  const f = body.frame(row);
  let p = 0;
  while (p + 1 < track.starts.length && track.starts[p + 1] <= pos) p++;
  const name = track.phones[p][0];
  const sound = QUIET.includes(name) ? "" : name;
  drawTract(el.tract, f, { voicing: heard.voicing, sound }, named ? words.parts : null);
  drawTube(el.tube, f, words);
  score.draw(track, rows(), pos);
  readouts(f, row, sound, heard.voicing);
  mark(p);
  requestAnimationFrame(tick);
}

// The page's words, in the language that is spoken.
function say() {
  document.documentElement.lang = lang;
  for (const e of document.querySelectorAll("[data-t]")) {
    const w = words[e.dataset.t];
    if (typeof w !== "string") continue;
    if (w.includes("<")) e.innerHTML = w; else e.textContent = w;
  }
  el.examples.setAttribute("aria-label", words.examples);
  el.examples.replaceChildren(...EXAMPLES[lang].map((text) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = text;
    b.title = text;
    b.addEventListener("click", () => { el.text.value = text; });
    return b;
  }));
}

async function setLang(to) {
  if (playing) node.port.postMessage({ type: "stop" });
  about = abouts[to] || (abouts[to] = await (await fetch(`api/voice?lang=${to}`)).json());
  const offered = Object.values(EXAMPLES).flat();
  lang = to;
  words = WORDS[to] || WORDS.en;
  track = null;
  now = -1;
  for (const b of el.langs.children) b.setAttribute("aria-pressed", String(b.dataset.lang === to));
  say();
  if (!el.text.value.trim() || offered.includes(el.text.value)) el.text.value = EXAMPLES[to][0];
  el.status.textContent = "";
  dial.rate.value = about.pace.toFixed(1);
  show();
  rest();
}

// The address says the language the page was switched to, so that it can be passed on as it is.
function address(to) {
  const rest = new URLSearchParams(location.search);
  for (const k of [...Object.keys(LANGS), "lang"]) rest.delete(k);
  try {
    history.replaceState(null, "", `?${to}${rest.size ? `&${rest}` : ""}${location.hash}`);
  } catch {
    // a page that may not rewrite its address still speaks
  }
}

el.speak.addEventListener("click", speak);
el.stop.addEventListener("click", () => node && node.port.postMessage({ type: "stop" }));
el.text.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !el.speak.disabled) speak();
});
for (const k of Object.keys(dial)) {
  dial[k].addEventListener("input", () => {
    if (k === "pitch" && el.follow.checked) dial.share.value = shareFor(Number(dial.pitch.value));
    if (k === "share") el.follow.checked = false;
    show();
    if (k === "pitch" || k === "bright") turned();
    if (!playing && (k === "share" || k === "pitch")) rest();
  });
}
el.follow.addEventListener("change", () => {
  if (el.follow.checked) dial.share.value = shareFor(Number(dial.pitch.value));
  show();
  if (!playing) rest();
});
el.names.addEventListener("click", () => {
  named = !named;
  el.names.setAttribute("aria-pressed", String(named));
  if (!playing) rest();
});
window.addEventListener("resize", () => { if (!playing) rest(); });

wasm = await (await fetch("pmvc_wasm.wasm")).arrayBuffer();
body = await Body.load(wasm.slice(0));
const { langs } = await (await fetch("api/models")).json();
el.langs.replaceChildren(...langs.map((l) => {
  const b = document.createElement("button");
  b.type = "button";
  b.dataset.lang = l;
  b.textContent = LANGS[l] || l;
  b.addEventListener("click", async () => { await setLang(l); address(l); });
  return b;
}));
el.langs.hidden = langs.length < 2;
// The language the address asks for (?ja, ?en; ?lang=ja too), and otherwise the browser's.
const asked = new URLSearchParams(location.search);
const first = langs.find((l) => asked.has(l)) || asked.get("lang") || ((navigator.language || "").startsWith("ja") ? "ja" : "en");
await setLang(langs.includes(first) ? first : langs[0]);
