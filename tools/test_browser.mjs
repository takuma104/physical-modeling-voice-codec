// The page in a real browser: headless Chrome is started with a profile of its own, opens the
// page of a running `pmvc.serve`, clicks its button and turns its dials while the voice speaks,
// and listens to what comes out of the audio thread.
//
// usage: node tools/test_browser.mjs http://127.0.0.1:7860/ shots/ [--width 1100] [--text "..."]
//                                    [--lang en] [--dark]
//
// Needs nothing installed but Chrome and Node 22 or later (the DevTools protocol over Node's own
// WebSocket). The sound is muted, and tapped inside the page: where the voice's node connects to
// the output, an AnalyserNode is connected too, and its samples say how loud the voice is and
// at what pitch. Writes screenshots (PNG) into the directory and prints what it found; exits
// with 1 if the page logged an error, stayed silent, or did not follow a dial.
//
// The address may name the language (?ja, ?en): the page has to come up in it. For a page that
// asks who is calling (a private Hugging Face Space), a token in the environment's BEARER is
// sent with the requests to the page's own origin, and to no other.
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const args = process.argv.slice(2);
const flag = (name, otherwise) => {
  const i = args.indexOf(`--${name}`);
  return i < 0 ? otherwise : args.splice(i, 2)[1];
};
const dark = args.includes("--dark") && args.splice(args.indexOf("--dark"), 1).length > 0;
const width = Number(flag("width", "1100"));
const text = flag("text", "");
const lang = flag("lang", "");
const [url, shots] = args;
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
mkdirSync(shots, { recursive: true });
const profile = mkdtempSync(join(tmpdir(), "pmvc-chrome-"));
const chrome = spawn(CHROME, ["--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "--mute-audio", "--no-first-run", "--no-default-browser-check", `--window-size=${width},1400`, "about:blank"], { stdio: ["ignore", "ignore", "pipe"] });
const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
const problems = [];

try {
  const ws = await new Promise((ok, no) => {
    let seen = "";
    chrome.stderr.on("data", (d) => {
      seen += d;
      const m = seen.match(/DevTools listening on (ws:\/\/\S+)/);
      if (m) ok(m[1]);
    });
    setTimeout(() => no(new Error("Chrome did not start")), 20000);
  });
  const port = new URL(ws).port;
  const tab = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === "page");
  const sock = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((ok) => sock.addEventListener("open", ok));
  let next = 0;
  const waiting = new Map();
  const logged = [];
  sock.addEventListener("message", (e) => {
    const m = JSON.parse(e.data);
    if (m.id && waiting.has(m.id)) {
      const [ok, no] = waiting.get(m.id);
      waiting.delete(m.id);
      if (m.error) no(new Error(m.error.message)); else ok(m.result);
    } else if (m.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(m.params.type)) {
      logged.push(`console.${m.params.type}: ${m.params.args.map((a) => a.value ?? a.description).join(" ")}`);
    } else if (m.method === "Fetch.requestPaused") {
      const headers = Object.entries({ ...m.params.request.headers, Authorization: `Bearer ${process.env.BEARER}` }).map(([name, value]) => ({ name, value }));
      send("Fetch.continueRequest", { requestId: m.params.requestId, headers }).catch(() => {});
    } else if (m.method === "Runtime.exceptionThrown") {
      logged.push(`exception: ${m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text}`);
    } else if (m.method === "Log.entryAdded" && m.params.entry.level === "error") {
      logged.push(`log: ${m.params.entry.text} ${m.params.entry.url || ""}`);
    }
  });
  const send = (method, params = {}) => new Promise((ok, no) => {
    waiting.set(++next, [ok, no]);
    sock.send(JSON.stringify({ id: next, method, params }));
  });
  const run = async (expression) => {
    const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
    return r.result.value;
  };
  const shot = async (name) => {
    const size = await run("({w: document.documentElement.scrollWidth, h: document.documentElement.scrollHeight, inner: innerWidth})");
    const r = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: { x: 0, y: 0, width: size.inner, height: size.h, scale: 1 } });
    writeFileSync(join(shots, `${name}.png`), Buffer.from(r.data, "base64"));
    if (size.w > size.inner + 1) problems.push(`${name}: the page is ${size.w}px wide in a ${size.inner}px window`);
  };
  const until = async (expression, ms, what) => {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      if (await run(expression)) return;
      await sleep(100);
    }
    throw new Error(`waited ${ms} ms for ${what}`);
  };
  const click = async (selector) => {
    const at = await run(`(() => { const e = document.querySelector(${JSON.stringify(selector)}); e.scrollIntoView({block: "center"}); const r = e.getBoundingClientRect(); return {x: r.x + r.width / 2, y: r.y + r.height / 2}; })()`);
    for (const type of ["mousePressed", "mouseReleased"]) await send("Input.dispatchMouseEvent", { type, x: at.x, y: at.y, button: "left", clickCount: 1 });
  };
  const turn = (id, value) => run(`(() => { const e = document.getElementById(${JSON.stringify(id)}); e.value = ${JSON.stringify(String(value))}; e.dispatchEvent(new Event("input", {bubbles: true})); e.dispatchEvent(new Event("change", {bubbles: true})); return e.value; })()`);

  await send("Runtime.enable");
  await send("Log.enable");
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width, height: 1400, deviceScaleFactor: 1, mobile: width < 600 });
  if (dark) await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "dark" }] });
  // the tap: whatever connects to the output is heard by an analyser too
  await send("Page.addScriptToEvaluateOnNewDocument", {
    source: `(() => {
      const connect = AudioNode.prototype.connect;
      AudioNode.prototype.connect = function (to, ...rest) {
        if (to instanceof AudioDestinationNode && !window.__tap) {
          window.__tap = new AnalyserNode(this.context, { fftSize: 4096 });
          connect.call(this, window.__tap);
        }
        return connect.call(this, to, ...rest);
      };
      // loudness (RMS) and pitch (Hz, by autocorrelation between 60 and 500 Hz) of the last 85 ms
      window.__heard = () => {
        if (!window.__tap) return { rms: 0, f0: 0 };
        const x = new Float32Array(4096);
        window.__tap.getFloatTimeDomainData(x);
        const sr = window.__tap.context.sampleRate;
        let e = 0;
        for (const v of x) e += v * v;
        let best = 0, lag = 0;
        for (let k = Math.floor(sr / 500); k <= Math.floor(sr / 60); k++) {
          let s = 0;
          for (let i = 0; i + k < x.length; i++) s += x[i] * x[i + k];
          s /= (x.length - k);
          if (s > best) { best = s; lag = k; }
        }
        return { rms: Math.sqrt(e / x.length), f0: lag && best > 0.3 * e / x.length ? sr / lag : 0 };
      };
    })()`,
  });
  if (process.env.BEARER) await send("Fetch.enable", { patterns: [{ urlPattern: `${new URL(url).origin}/*` }] });
  await send("Page.navigate", { url });
  await until("document.readyState === 'complete' && !!document.querySelector('#speak') && document.querySelector('#pitch-out').textContent !== ''", 15000, "the page");
  const named = ["ja", "en"].find((l) => new URL(url).searchParams.has(l)) || new URL(url).searchParams.get("lang");
  const shownIn = await run("document.documentElement.lang");
  if (named && shownIn !== named) problems.push(`the address asks for ${named}, and the page is in ${shownIn}`);
  if (lang) {
    await click(`[data-lang="${lang}"]`);
    await until(`document.documentElement.lang === ${JSON.stringify(lang)}`, 5000, "the language to change");
    const address = await run("location.search");
    if (!new URLSearchParams(address).has(lang)) problems.push(`the page was switched to ${lang}, and its address says ${address}`);
    console.log("switched:", lang, "| address", address);
  }
  if (text) await run(`(() => { const e = document.getElementById("text"); e.value = ${JSON.stringify(text)}; e.dispatchEvent(new Event("input", {bubbles: true})); })()`);
  await shot("1-loaded");
  console.log("loaded:", await run("document.title"), "|", await run("document.querySelector('#speak').textContent.trim()"));

  // the median of the pitches heard over a while, and the loudest moment
  const listen = async (ms) => {
    const f0 = [];
    let peak = 0;
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      const h = await run("window.__heard()");
      peak = Math.max(peak, h.rms);
      if (h.f0 > 0 && h.rms > 0.01) f0.push(h.f0);
      await sleep(40);
    }
    f0.sort((a, b) => a - b);
    return { peak, f0: f0.length ? f0[f0.length >> 1] : 0, voiced: f0.length };
  };

  await turn("pitch", 120);
  await click("#speak");
  await until("!document.querySelector('#stop').disabled", 30000, "the voice to start");
  console.log("status:", await run("document.querySelector('#status').textContent"));
  const first = await listen(1200);
  await shot("2-speaking");
  await turn("pitch", 240);
  await sleep(150);
  const second = await listen(1200);
  await turn("bright", 1.5);
  const third = await listen(600);
  console.log(`heard: peak ${first.peak.toFixed(3)}, pitch ${first.f0.toFixed(0)} Hz with the dial at 120; ${second.f0.toFixed(0)} Hz after it was turned to 240 while speaking (peak ${second.peak.toFixed(3)}); brightness turned, peak ${third.peak.toFixed(3)}`);
  if (first.peak < 0.02) problems.push("the voice is silent");
  if (!(first.f0 > 0 && second.f0 > 1.5 * first.f0)) problems.push(`the pitch dial was not followed while speaking (${first.f0.toFixed(0)} → ${second.f0.toFixed(0)} Hz)`);
  await until("document.querySelector('#stop').disabled", 60000, "the sentence to end");
  await sleep(200);
  const after = await run("window.__heard().rms");
  if (after > 0.005) problems.push(`sound after the end (${after.toFixed(4)})`);
  // a second sentence, stopped by the button
  await turn("pitch", 220);
  await click("#speak");
  await until("!document.querySelector('#stop').disabled", 30000, "the voice to start again");
  const again = await listen(800);
  await click("#stop");
  await until("document.querySelector('#stop').disabled", 5000, "the stop button");
  console.log(`second sentence: peak ${again.peak.toFixed(3)}, pitch ${again.f0.toFixed(0)} Hz at 220, share ${await run("document.querySelector('#share').value")}; stopped by the button`);
  if (again.peak < 0.02) problems.push("the second sentence is silent");
  await shot("3-after");
  for (const line of logged) problems.push(line);
  sock.close();
} catch (err) {
  problems.push(String(err.message || err));
} finally {
  chrome.kill();
  await sleep(300);
  rmSync(profile, { recursive: true, force: true });
}
console.log(problems.length ? `PROBLEMS:\n  ${problems.join("\n  ")}` : "no problems");
process.exit(problems.length ? 1 : 0);
