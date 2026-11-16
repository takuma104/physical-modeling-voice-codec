// What the page draws: the body in the posture of the frame now sounding (the engine's own
// geometry, from a second instance of the module in this thread), the tube that posture makes
// from the glottis to the lips, and a track's channels over time.

const enc = new TextEncoder();
const dec = new TextDecoder();
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// A canvas at the size it is shown, sharp on a dense screen: its context (in CSS pixels) and size.
export function fit(canvas) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { g, w, h, dpr };
}

// The body of a track (crates/pmvc-wasm's `tract_*`): the outlines for a row of the track.
export class Body {
  static async load(bytes) {
    const { instance } = await WebAssembly.instantiate(bytes, {});
    return new Body(instance.exports);
  }

  constructor(w) {
    this.w = w;
    this.tract = 0;
    this.cols = 0;
    this.row = 0;
  }

  said() {
    return dec.decode(new Uint8Array(this.w.memory.buffer, this.w.pmvc_error_ptr(), this.w.pmvc_error_len()));
  }

  // `header`: a track's `body` and `keys`
  of(header) {
    const bytes = enc.encode(JSON.stringify(header));
    const p = this.w.pmvc_alloc(bytes.byteLength);
    new Uint8Array(this.w.memory.buffer, p, bytes.byteLength).set(bytes);
    const tract = this.w.tract_new(p, bytes.byteLength);
    this.w.pmvc_free(p, bytes.byteLength);
    if (!tract) throw new Error(this.said());
    if (this.tract) this.w.tract_free(this.tract);
    if (this.row) this.w.pmvc_free(this.row, this.cols * 8);
    this.tract = tract;
    this.cols = header.keys.length;
    this.row = this.w.pmvc_alloc(this.cols * 8);
  }

  // The outlines (cm, the glottis at the origin) for a row of values; a value that is not a number leaves its articulator at rest.
  frame(values) {
    new Float64Array(this.w.memory.buffer, this.row, this.cols).set(values);
    if (!this.w.tract_frame(this.tract, this.row, this.cols)) throw new Error(this.said());
    return JSON.parse(dec.decode(new Uint8Array(this.w.memory.buffer, this.w.pmvc_result_ptr(), this.w.pmvc_result_len())));
  }
}

// The part of the head that is drawn (cm, the glottis near the bottom): the same for every body, so that a smaller one looks smaller.
const BOX = { x0: -2.9, x1: 9.5, y0: 0.0, y1: 12.4 };

// The midsagittal section, facing right, as a textbook draws it: flesh, and the air's way
// through it. The walls, the tongue, the soft palate, the teeth and the lips are the engine's
// (what it computes its tube from); the skin of the face is only a frame for them, hung on
// the lips. `live`: {voicing (0 to 1), sound (a phoneme, or "")}.
export function drawTract(canvas, f, live) {
  const { g, w, h } = fit(canvas);
  g.clearRect(0, 0, w, h);
  if (!f) return;
  const s = Math.min(w / (BOX.x1 - BOX.x0), h / (BOX.y1 - BOX.y0));
  const ox = (w - s * (BOX.x1 - BOX.x0)) / 2, oy = (h - s * (BOX.y1 - BOX.y0)) / 2;
  const X = (x) => ox + (x - BOX.x0) * s, Y = (y) => h - oy - (y - BOX.y0) * s;
  const to = (p, first) => (first ? g.moveTo(X(p.x), Y(p.y)) : g.lineTo(X(p.x), Y(p.y)));
  const path = (pts, close = false) => {
    g.beginPath();
    pts.forEach((p, i) => to(p, i === 0));
    if (close) g.closePath();
  };
  const ink = css("--ink"), soft = css("--soft");
  const k = f.tongueRadius / 2.25;   // the body's size beside the adult male's
  const gl = f.glottis;
  const far = { left: BOX.x0 - 1, bottom: BOX.y0 - 1, top: BOX.y1 + 1 };
  g.save();
  g.beginPath();
  g.roundRect(ox, oy, s * (BOX.x1 - BOX.x0), s * (BOX.y1 - BOX.y0), 10);
  g.clip();
  g.lineJoin = "round";
  g.lineCap = "round";

  // the air below the folds, and above them between the two walls
  g.fillStyle = css("--air");
  g.fillRect(X(gl.x - 0.62 * k), Y(gl.y), s * 1.3 * k, s * (gl.y - far.bottom));
  const grid = f.grid.filter((c) => Number.isFinite(c.outer.x) && Number.isFinite(c.inner.x));
  const outer = f.outer.filter((p) => p.y > gl.y);
  path([{ x: gl.x - 0.62 * k, y: gl.y }, ...outer, ...[...f.inner].reverse(), { x: gl.x + 0.68 * k, y: gl.y }], true);
  g.fill();

  // the flesh behind and above the way: the back of the throat, the palate, the nose and the upper lip
  const up = f.upperLip[f.upperLip.length - 1], low = f.lowerLip[f.lowerLip.length - 1];
  g.beginPath();
  to({ x: gl.x - 0.62 * k, y: far.bottom }, true);
  to({ x: gl.x - 0.62 * k, y: gl.y });
  outer.forEach((p) => to(p));
  to(up);
  g.quadraticCurveTo(X(up.x + 0.15 * k), Y(up.y + 0.45 * k), X(up.x + 0.95 * k), Y(up.y + 0.5 * k));
  g.quadraticCurveTo(X(up.x + 2.35 * k), Y(up.y + 0.75 * k), X(up.x + 2.0 * k), Y(up.y + 1.55 * k));
  g.quadraticCurveTo(X(up.x + 1.5 * k), Y(up.y + 2.6 * k), X(up.x + 1.15 * k), Y(far.top));
  to({ x: far.left, y: far.top });
  to({ x: far.left, y: far.bottom });
  g.closePath();
  g.fillStyle = css("--flesh");
  g.fill();
  g.strokeStyle = ink;
  g.lineWidth = 1.5;
  g.stroke();
  // the flesh below and in front of it: the floor of the mouth, the chin and the neck
  g.beginPath();
  to({ x: gl.x + 0.68 * k, y: far.bottom }, true);
  to({ x: gl.x + 0.68 * k, y: gl.y });
  f.inner.forEach((p) => to(p));
  to(low);
  g.quadraticCurveTo(X(low.x - 0.25 * k), Y(low.y - 0.55 * k), X(low.x + 0.35 * k), Y(low.y - 1.0 * k));
  g.quadraticCurveTo(X(low.x + 0.75 * k), Y(low.y - 2.1 * k), X(low.x - 0.55 * k), Y(low.y - 2.35 * k));
  g.quadraticCurveTo(X(low.x - 3.0 * k), Y(low.y - 2.6 * k), X(2.9 * k), Y(Math.min(low.y - 3.6 * k, 1.9 * k)));
  g.quadraticCurveTo(X(2.45 * k), Y(0.9 * k), X(2.4 * k), Y(far.bottom));
  g.closePath();
  g.fill();
  g.stroke();

  // the nose's hollow, which the soft palate opens and shuts
  if (f.nasalCavity.length > 2) {
    path(f.nasalCavity, true);
    g.fillStyle = css(f.velumPort > 0.02 ? "--air" : "--hollow");
    g.fill();
    g.lineWidth = 1;
    g.strokeStyle = soft;
    g.stroke();
  }
  // the tongue
  path(f.tongue, true);
  g.fillStyle = css("--tongue");
  g.fill();
  // the walls of the way
  g.strokeStyle = ink;
  g.lineWidth = 2;
  path(outer);
  g.stroke();
  path(f.inner);
  g.stroke();
  path(f.epiglottis);
  g.stroke();
  // the soft palate
  path(f.velum, true);
  g.fillStyle = css("--tongue");
  g.fill();
  g.stroke();
  // the teeth and the lips
  g.lineWidth = 1.25;
  for (const teeth of [f.upperTeeth, f.lowerTeeth]) {
    path(teeth, true);
    g.fillStyle = css("--tooth");
    g.fill();
    g.stroke();
  }
  g.lineWidth = 1.5;
  for (const lip of [f.upperLip, f.lowerLip]) {
    path(lip, true);
    g.fillStyle = css("--lip");
    g.fill();
    g.stroke();
  }
  // the vocal folds: two wedges across the way, bright when they vibrate
  const voicing = Math.max(0, Math.min(1, live?.voicing ?? 0));
  g.fillStyle = voicing > 0.3 ? css("--mark") : soft;
  for (const side of [-1, 1]) {
    const wall = gl.x + (side < 0 ? -0.62 : 0.68) * k;
    path([{ x: wall, y: gl.y - 0.16 * k }, { x: gl.x + side * 0.06 * k, y: gl.y + 0.1 * k }, { x: wall, y: gl.y + 0.36 * k }], true);
    g.fill();
  }
  g.restore();
  // the sound being made, in front of the mouth
  if (live?.sound) {
    g.fillStyle = css("--mark");
    g.font = `600 ${Math.round(Math.max(18, s * 1.05))}px ${css("--mono")}`;
    g.textAlign = "left";
    g.textBaseline = "middle";
    const lips = Math.max(...f.upperLip.map((p) => p.x), ...f.lowerLip.map((p) => p.x));
    g.fillText(live.sound, Math.min(w - s * 1.9, X(lips + 0.55 * k)), Y((up.y + low.y) / 2 - 0.9 * k));
  }
}

// The tube the posture makes: its radius along the way from the glottis to the lips.
export function drawTube(canvas, f, words) {
  const { g, w, h } = fit(canvas);
  g.clearRect(0, 0, w, h);
  if (!f) return;
  const pad = 8, top = 16;
  const grid = f.grid.filter((c) => Number.isFinite(c.area));
  const X = (sAt) => pad + (sAt / 18.5) * (w - 2 * pad);
  const mid = top + (h - top) / 2;
  const R = (area) => Math.min((h - top) / 2 - 2, (Math.sqrt(Math.max(0, area) / Math.PI) / 2.2) * ((h - top) / 2 - 2));
  g.beginPath();
  grid.forEach((c, i) => (i ? g.lineTo(X(c.s), mid - R(c.area)) : g.moveTo(X(c.s), mid - R(c.area))));
  [...grid].reverse().forEach((c) => g.lineTo(X(c.s), mid + R(c.area)));
  g.closePath();
  g.fillStyle = css("--air");
  g.fill();
  g.strokeStyle = css("--ink");
  g.lineWidth = 1.5;
  g.lineJoin = "round";
  g.stroke();
  // where the nose branches off
  g.fillStyle = f.velumPort > 0.02 ? css("--mark") : css("--line");
  g.fillRect(X(f.pharynxLength) - 1, top, 2, h - top);
  g.fillStyle = css("--soft");
  g.font = `11px ${css("--sans")}`;
  g.textBaseline = "top";
  g.textAlign = "left";
  g.fillText(words["tube-from"], pad, 0);
  g.textAlign = "right";
  g.fillText(words["tube-to"], X(grid.length ? grid[grid.length - 1].s : 17), 0);
  g.textAlign = "center";
  g.fillText(words["r-velum"], X(f.pharynxLength), 0);
}

// A track's channels over time: drawn once into a picture of its own, over which the playhead moves.
export class Score {
  constructor(canvas) {
    this.canvas = canvas;
    this.picture = null;
    this.key = "";
  }

  // `rows`: [key, label] in the order drawn; `track`: {keys, frames, phones}
  paint(track, rows, size) {
    const { w, h, dpr } = size;
    const picture = document.createElement("canvas");
    picture.width = Math.round(w * dpr);
    picture.height = Math.round(h * dpr);
    const g = picture.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const gutter = this.gutter(w), head = 18;
    const n = track.frames.length;
    const rowH = (h - head) / rows.length;
    const X = (i) => gutter + (i / Math.max(1, n - 1)) * (w - gutter - 1);
    const sans = css("--sans"), mono = css("--mono"), soft = css("--soft"), line = css("--line");
    // the phonemes: a faint line where each begins, its name where there is room
    let at = 0;
    g.font = `11px ${mono}`;
    g.textBaseline = "middle";
    g.textAlign = "center";
    let last = -100;
    for (const [name, frames] of track.phones) {
      g.fillStyle = line;
      g.fillRect(Math.round(X(at)), head, 1, h - head);
      const x = X(at + frames / 2);
      if (x - last > 13 && !["sil", "pau", "sp"].includes(name)) {
        g.fillStyle = soft;
        g.fillText(name, x, head / 2);
        last = x;
      }
      at += frames;
    }
    g.textAlign = "left";
    rows.forEach(([key, label], r) => {
      const k = track.keys.indexOf(key);
      const y0 = head + r * rowH;
      g.fillStyle = r % 2 ? css("--band") : "transparent";
      g.fillRect(0, y0, w, rowH);
      g.fillStyle = soft;
      g.font = `11px ${sans}`;
      g.fillText(label, 4, y0 + rowH / 2, gutter - 8);
      if (k < 0) return;
      const xs = track.frames.map((row) => row[k]);
      const lo = Math.min(...xs), hi = Math.max(...xs);
      g.beginPath();
      xs.forEach((x, i) => {
        const y = y0 + rowH - 2.5 - (hi > lo ? (x - lo) / (hi - lo) : 0.5) * (rowH - 5);
        if (i) g.lineTo(X(i), y); else g.moveTo(X(i), y);
      });
      g.strokeStyle = css("--accent");
      g.lineWidth = 1.25;
      g.lineJoin = "round";
      g.stroke();
    });
    this.picture = picture;
    this.X = X;
  }

  gutter(w) {
    return w < 520 ? 84 : 112;
  }

  // `position`: the frame now sounding, or a negative number for none
  draw(track, rows, position) {
    const size = fit(this.canvas);
    const { g, w, h } = size;
    g.clearRect(0, 0, w, h);
    if (!track) {
      this.picture = null;
      this.key = "";
      return;
    }
    const key = `${track.id}/${rows.map((r) => r[1]).join()}/${size.w}x${size.h}@${size.dpr}`;
    if (key !== this.key) {
      this.paint(track, rows, size);
      this.key = key;
    }
    g.drawImage(this.picture, 0, 0, w, h);
    if (position >= 0) {
      g.fillStyle = css("--mark");
      g.fillRect(this.X(Math.min(position, track.frames.length - 1)) - 1, 0, 2, h);
    }
  }
}
