// The voice in the audio thread: a control track in, samples out, 128 at a time.
//
// The module is crates/pmvc-wasm (the CHIMERA engine driven by a track, the same code the Python
// module renders with). An AudioWorklet has no fetch and no TextEncoder, so the page sends
// everything as bytes: the wasm, a track's header (UTF-8 JSON) and its frames (f64, row after row).
//
// messages in:   {type: "wasm", bytes}                       once
//                {type: "track", header, frames, cols}       play this track from its start
//                {type: "frames", frames}                    the same track with other values from here on (a dial turned)
//                {type: "patch", kind, bytes}                settings of the voice, as JSON (1 the glottis, 3 the nose, ...)
//                {type: "stop"}
// messages out:  {type: "ready"}, {type: "ended"}, {type: "error", text}, and fifty times a second
//                {type: "at", frame, of, voicing, glottis}: the frame now sounding (with its fraction), how
//                much the folds vibrate (0 to 1) and how open the glottis is (cm²), from the simulation

class PmvcPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.w = null;        // the module's exports
    this.player = 0;      // a pointer, or 0
    this.out = 0;         // room for a quantum of samples in the module's memory
    this.cols = 0;
    this.rows = 0;
    this.perFrame = 480;
    this.sample = 0;      // samples rendered of the track
    this.total = 0;
    this.told = 0;        // samples since the page was last told where we are
    this.port.onmessage = (e) => this.take(e.data);
  }

  said() {
    const w = this.w;
    const bytes = new Uint8Array(w.memory.buffer, w.pmvc_error_ptr(), w.pmvc_error_len());
    return String.fromCharCode(...bytes);
  }

  put(bytes) {
    const p = this.w.pmvc_alloc(bytes.byteLength);
    new Uint8Array(this.w.memory.buffer, p, bytes.byteLength).set(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
    return p;
  }

  release() {
    if (this.player) this.w.player_free(this.player);
    this.player = 0;
  }

  take(m) {
    try {
      if (m.type === "wasm") {
        this.w = new WebAssembly.Instance(new WebAssembly.Module(m.bytes), {}).exports;
        this.out = this.w.pmvc_alloc(128 * 4);
        this.probe = this.w.pmvc_alloc(9 * 8);
        this.port.postMessage({ type: "ready", rate: this.w.pmvc_sample_rate() });
      } else if (m.type === "track") {
        this.release();
        const hp = this.put(m.header);
        const player = this.w.player_new(hp, m.header.byteLength);
        this.w.pmvc_free(hp, m.header.byteLength);
        if (!player) throw new Error(this.said());
        this.player = player;
        this.cols = m.cols;
        this.rows = m.frames.length / m.cols;
        this.perFrame = Math.round(this.w.pmvc_sample_rate() / m.rate);
        this.follow(m.frames, 0);
        this.sample = 0;
        this.total = (this.rows - 1) * this.perFrame;
      } else if (m.type === "frames") {
        // from the frame now playing on: the frames before it have been played
        if (this.player) this.follow(m.frames, Math.min(this.rows - 1, Math.floor(this.sample / this.perFrame) + 1));
      } else if (m.type === "patch") {
        if (!this.player) return;
        const p = this.put(m.bytes);
        const ok = this.w.player_patch(this.player, m.kind, p, m.bytes.byteLength);
        this.w.pmvc_free(p, m.bytes.byteLength);
        if (!ok) throw new Error(this.said());
      } else if (m.type === "stop") {
        this.release();
        this.port.postMessage({ type: "ended" });
      }
    } catch (err) {
      this.port.postMessage({ type: "error", text: String(err && err.message ? err.message : err) });
    }
  }

  follow(frames, first) {
    const p = this.put(frames);
    const ok = this.w.player_set_track(this.player, p, this.rows, this.cols, first);
    this.w.pmvc_free(p, frames.byteLength);
    if (!ok) throw new Error(this.said());
  }

  process(_inputs, outputs) {
    const left = outputs[0][0];
    if (!this.player || this.sample >= this.total) return true;
    const n = Math.min(left.length, this.total - this.sample);
    this.w.player_render(this.player, this.out, n);
    // the memory may have grown since the last quantum: a fresh view every time
    left.set(new Float32Array(this.w.memory.buffer, this.out, n));
    for (let c = 1; c < outputs[0].length; c++) outputs[0][c].set(left);
    this.sample += n;
    this.told += n;
    if (this.sample >= this.total) {
      this.release();
      this.port.postMessage({ type: "ended" });
    } else if (this.told >= 960) {
      this.told = 0;
      this.w.player_probe(this.player, this.probe);
      const [glottis, voicing] = new Float64Array(this.w.memory.buffer, this.probe, 2);
      this.port.postMessage({ type: "at", frame: this.sample / this.perFrame, of: this.rows, voicing, glottis });
    }
    return true;
  }
}

registerProcessor("pmvc-player", PmvcPlayer);
