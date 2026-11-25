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
const BOX = { x0: -2.9, x1: 9.5, y0: -0.75, y1: 12.05 };
// The back wall of the throat where it goes on up behind the soft palate and turns forward into
// the nose's roof (cm in the adult male; CHIMERA's fixed landmarks, `HumanVocalTract::new`).
// The engine's outer wall leaves it for the soft palate's tip, where its airway's sections end.
const BACK = [[-1.38, 7.5], [-1.33, 8.9], [-1.05, 10.1], [-0.45, 10.9], [0.4, 11.3], [1.6, 11.32]];
// how thick the bone of the hard palate is drawn, and how high the hard palate's back end stands (cm in the adult male)
const BONE = 0.36, PALATE_END = 9.75;

// A smooth line through points: a cubic through each, by how far along the points it is.
function through(p, per = 6) {
  const t = [0];
  for (let i = 1; i < p.length; i++) t.push(t[i - 1] + Math.hypot(p[i].x - p[i - 1].x, p[i].y - p[i - 1].y));
  const slope = (i, c) => {
    const a = Math.max(0, i - 1), b = Math.min(p.length - 1, i + 1);
    return (p[b][c] - p[a][c]) / Math.max(1e-9, t[b] - t[a]);
  };
  const out = [];
  for (let i = 0; i + 1 < p.length; i++) {
    const h = t[i + 1] - t[i];
    for (let j = 0; j < per; j++) {
      const u = j / per, u2 = u * u, u3 = u2 * u;
      const at = (c) => (2 * u3 - 3 * u2 + 1) * p[i][c] + (u3 - 2 * u2 + u) * h * slope(i, c) + (-2 * u3 + 3 * u2) * p[i + 1][c] + (u3 - u2) * h * slope(i + 1, c);
      out.push({ x: at("x"), y: at("y") });
    }
  }
  out.push(p[p.length - 1]);
  return out;
}

// The midsagittal section, facing right, drawn as a duct and not as a body: the air is what has
// a colour, what is cut through is the card's own, and what moves (the tongue, the soft palate,
// the lips) has one step of the paper's on it. The walls, the tongue, the soft palate, the teeth
// and the lips are the engine's (what it computes its tube from). The nose is drawn as the
// report's figure draws it (docs/report/make_figures.py): the throat's back wall goes on up into
// the nose's roof, and the nose's floor is the soft palate and the bone of the hard one, so that
// the way into the nose is seen to open and shut behind the soft palate. The skin of the face is
// only a frame, hung on the lips, and faint. `live`: {voicing (0 to 1), sound (a phoneme, or "")}.
// `names`: the parts' names to write on it ({nose, velum, tongue, lips, folds}), or nothing.
export function drawTract(canvas, f, live, names) {
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
  const line = (colour, width) => {
    g.strokeStyle = colour;
    g.lineWidth = width;
    g.stroke();
  };
  const wall = css("--wall"), faint = css("--faint");
  const k = f.tongueRadius / 2.25;   // the body's size beside the adult male's, for the face
  const gl = f.glottis;
  const far = { left: BOX.x0 - 1, bottom: BOX.y0 - 1, top: BOX.y1 + 1 };
  g.save();
  g.beginPath();
  g.roundRect(ox, oy, s * (BOX.x1 - BOX.x0), s * (BOX.y1 - BOX.y0), 10);
  g.clip();
  g.lineJoin = "round";
  g.lineCap = "round";

  // The engine's outer wall in its parts: the back of the throat up to the height of the soft
  // palate's tip, the soft palate's under side (tip to root), the hard palate, and the gum, the
  // incisors and the upper lip. Its soft palate is the upper side (root to tip), then the under side.
  const outer = f.outer.filter((p) => p.y > gl.y);
  const half = f.velum.length / 2;
  const tip = f.velum[half];
  const at = outer.reduce((best, p, i) => (Math.hypot(p.x - tip.x, p.y - tip.y) < Math.hypot(outer[best].x - tip.x, outer[best].y - tip.y) ? i : best), 0);
  const throat = outer.slice(0, at);
  const size = f.nasalCavity[0].y / PALATE_END;   // the body's size, exactly: the nose's roof starts at the hard palate's back end
  let front = at + half - 1;
  while (front + 1 < outer.length && outer[front + 1].y > 8.5 * size) front++;
  const hard = outer.slice(at + half - 1, front + 1);
  const bone = hard.map((p) => ({ x: p.x, y: p.y + BONE * size }));
  const last = throat[throat.length - 1];
  const roof = f.nasalCavity.filter((p) => p.x > 2.9 * size);
  const ceiling = through([last, ...BACK.map(([x, y]) => ({ x: x * size, y: y * size })).filter((p) => p.y > last.y + 0.2 * size), ...roof]);
  const back = [{ x: gl.x - 0.62 * k, y: far.bottom }, { x: gl.x - 0.62 * k, y: gl.y }, ...throat, ...ceiling.slice(1)];
  const floor = [{ x: gl.x + 0.68 * k, y: far.bottom }, { x: gl.x + 0.68 * k, y: gl.y }, ...f.inner];

  // the air: below the folds, above them between the two walls, and in the nose
  g.fillStyle = css("--air");
  g.fillRect(X(gl.x - 0.62 * k), Y(gl.y), s * 1.3 * k, s * (gl.y - far.bottom));
  path([{ x: gl.x - 0.62 * k, y: gl.y }, ...outer, ...[...f.inner].reverse(), { x: gl.x + 0.68 * k, y: gl.y }], true);
  g.fill();
  path([...ceiling, ...[...bone].reverse(), ...f.velum.slice(0, half + 1)], true);
  g.fill();
  // the soft palate; shut, its tip is pressed into the back wall, and what is drawn next covers what went in
  path(f.velum, true);
  g.fillStyle = css("--tongue");
  g.fill();
  line(wall, 1.5);

  // what is cut through: behind the throat and over the nose, and in front of the nose down to the
  // upper lip; then below and in front of the way, the floor of the mouth, the chin and the neck
  const up = f.upperLip[f.upperLip.length - 1], low = f.lowerLip[f.lowerLip.length - 1];
  const face = () => {
    g.quadraticCurveTo(X(up.x + 0.15 * k), Y(up.y + 0.45 * k), X(up.x + 0.95 * k), Y(up.y + 0.5 * k));
    g.quadraticCurveTo(X(up.x + 2.35 * k), Y(up.y + 0.75 * k), X(up.x + 2.0 * k), Y(up.y + 1.55 * k));
    g.quadraticCurveTo(X(up.x + 1.5 * k), Y(up.y + 2.6 * k), X(up.x + 1.15 * k), Y(far.top));
  };
  const chin = () => {
    g.quadraticCurveTo(X(low.x - 0.25 * k), Y(low.y - 0.55 * k), X(low.x + 0.35 * k), Y(low.y - 1.0 * k));
    g.quadraticCurveTo(X(low.x + 0.75 * k), Y(low.y - 2.1 * k), X(low.x - 0.55 * k), Y(low.y - 2.35 * k));
    g.quadraticCurveTo(X(low.x - 3.0 * k), Y(low.y - 2.6 * k), X(2.9 * k), Y(Math.min(low.y - 3.6 * k, 1.9 * k)));
    g.quadraticCurveTo(X(2.45 * k), Y(0.9 * k), X(2.4 * k), Y(far.bottom));
  };
  g.fillStyle = css("--flesh");
  path([...back, bone[bone.length - 1], ...outer.slice(front), up]);
  face();
  to({ x: far.left, y: far.top });
  to({ x: far.left, y: far.bottom });
  g.closePath();
  g.fill();
  path([...floor, low]);
  chin();
  g.closePath();
  g.fill();
  // the skin of the face and of the neck
  path([up]);
  face();
  line(faint, 1.25);
  path([low]);
  chin();
  line(faint, 1.25);

  // the tongue; its colour lets go towards the straight side that closes its outline, which is no edge of anything
  const root = f.tongue[0], blade = f.tongue[f.tongue.length - 1];
  const mid = { x: (root.x + blade.x) / 2, y: (root.y + blade.y) / 2 };
  const across = { x: root.y - blade.y, y: blade.x - root.x };
  const inward = ((f.tongueCenter.x - mid.x) * across.x + (f.tongueCenter.y - mid.y) * across.y < 0 ? -1 : 1) * f.tongueRadius / Math.hypot(across.x, across.y);
  const shade = g.createLinearGradient(X(mid.x), Y(mid.y), X(mid.x + across.x * inward), Y(mid.y + across.y * inward));
  shade.addColorStop(0, css("--flesh"));
  shade.addColorStop(1, css("--tongue"));
  path(f.tongue, true);
  g.fillStyle = shade;
  g.fill();
  // the walls of the way: the throat's back and the nose's roof, the roof of the mouth, the floor
  for (const side of [back, outer.slice(front), floor, f.epiglottis]) {
    path(side);
    line(wall, 1.5);
  }
  // the bone of the hard palate, the teeth and the lips
  for (const white of [[...hard, ...[...bone].reverse()], f.upperTeeth.slice(1, 4), f.lowerTeeth]) {
    path(white, true);
    g.fillStyle = css("--tooth");
    g.fill();
    line(css("--hard"), 1);
  }
  for (const lip of [f.upperLip, f.lowerLip]) {
    path(lip, true);
    g.fillStyle = css("--lip");
    g.fill();
    line(wall, 1.25);
  }
  // the vocal folds: two wedges across the way, bright when they vibrate
  const voicing = Math.max(0, Math.min(1, live?.voicing ?? 0));
  g.fillStyle = voicing > 0.3 ? css("--mark") : css("--soft");
  for (const side of [-1, 1]) {
    const edge = gl.x + (side < 0 ? -0.62 : 0.68) * k;
    path([{ x: edge, y: gl.y - 0.16 * k }, { x: gl.x + side * 0.06 * k, y: gl.y + 0.1 * k }, { x: edge, y: gl.y + 0.36 * k }], true);
    g.fill();
  }
  g.restore();
  // the parts' names, for who has not seen such a section before
  if (names) {
    const px = Math.max(10.5, s * 0.34);
    g.font = `${px}px ${css("--sans")}`;
    g.textBaseline = "middle";
    // a name at a place (cm), its lines one under another; with `from`, a line to it from its part
    // (to its side, or from below when the name is set on its middle)
    const name = (text, x, y, align = "center", from = null) => {
      const lines = text.split("\n");
      if (from) {
        g.beginPath();
        g.moveTo(X(from.x), Y(from.y));
        if (align === "center") g.lineTo(X(x), Y(y) + lines.length * px * 0.6 + 2); else g.lineTo(X(x) + (align === "right" ? 4 : -4), Y(y));
        line(faint, 1);
      }
      g.textAlign = align;
      g.fillStyle = css("--soft");
      lines.forEach((t, i) => g.fillText(t, X(x), Y(y) + (i - (lines.length - 1) / 2) * px * 1.2));
    };
    const along = Math.floor(half / 2), above = f.velum[along], below = f.velum[f.velum.length - 1 - along];
    // the nose's is over its roof: inside, a long name does not fit a small body
    name(names.nose, 1.3 * size, 11.32 * size + 0.36, "center", { x: 1.3 * size, y: 10.8 * size });
    name(names.tongue, f.tongueCenter.x + 0.2 * k, f.tongueCenter.y - 0.3 * k);
    name(names.velum, -1.5 * size, 9.0 * size, "right", { x: (above.x + below.x) / 2, y: (above.y + below.y) / 2 });
    name(names.folds, gl.x - 1.15 * k, gl.y + 0.1 * k, "right", { x: gl.x - 0.62 * k, y: gl.y + 0.1 * k });
    name(names.lips, low.x + 1.7 * k, low.y + 0.45 * k, "left", f.lowerLip[Math.floor(f.lowerLip.length * 0.7)]);
  }
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
  g.strokeStyle = css("--wall");
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
    // a canvas that is not laid out yet (the window is being resized) has nothing to draw into
    if (w < 2 || h < 2) return;
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
