// The voice's settings for a position of the dials, and a track's frames as the dials have them.
//
// The voice is a pair (voices/v1.json: `low` for a man, `high` for a woman): every setting
// between the two goes in a straight line with the share of the high one, and `bright` and
// `dull` say what a unit of the brightness dial adds or takes. This is pmvc.encoder's Decoder
// again, for the page; `about` is what the server's /api/voice gives.

export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const mix = (lo, hi, w) => Object.fromEntries(Object.keys(lo || {}).map((k) => [k, (1 - w) * lo[k] + w * hi[k]]));

// The skew of the glottal pulse as the engine makes it of the source's settings.
const skewOf = (about, share) => {
  const s = mix(about.low.source, about.high.source, share);
  return 0.56 + 0.16 * s.tension + 0.2 * (s.brightness - 0.25);
};

// A part of the voice (body, source, glottis, set, ...) at this share of the high voice and this brightness.
export function partOf(about, name, share, bright = 0) {
  const out = mix(about.low[name], about.high[name], share);
  const more = about[bright > 0 ? "bright" : "dull"] || about.bright;
  if (bright !== 0 && more && more.low[name]) {
    for (const k of Object.keys(more.low[name])) {
      // the skew is the engine's to make, until the dial moves it
      if (!(k in out) && name === "glottis" && k === "skew") out[k] = skewOf(about, share);
      out[k] = clamp(out[k] + bright * ((1 - share) * more.low[name][k] + share * more.high[name][k]), 0, 1);
    }
  }
  return out;
}

// The body that goes with a pitch, when the listener has not said: the low voice up to 150 Hz, the high one from 220 Hz.
export const shareFor = (pitch) => clamp(Math.log(pitch / 150) / Math.log(220 / 150), 0, 1);

// A track's header for the engine (the track without its frames).
export function headerOf(about, track, bright, pitch) {
  const w = track.share;
  return {
    rate: track.rate,
    body: partOf(about, "body", w),
    source: { f0: pitch, ...partOf(about, "source", w) },
    keys: track.keys,
    noise: Object.values(partOf(about, "noise", w)),
    glottis: partOf(about, "glottis", w, bright),
    airway: { ...partOf(about, "airway", w), ...about.airway },
    nose: partOf(about, "nose", w),
  };
}

// The track's frames, row after row: the pitch moved to `pitch`, the channels the voice holds as it holds them.
export function framesOf(about, track, bright, pitch) {
  const cols = track.keys.length;
  const f0 = track.keys.indexOf("f0");
  const up = pitch / track.pitch;
  const held = Object.entries(partOf(about, "set", track.share, bright)).map(([k, x]) => [track.keys.indexOf(k), x]);
  const flat = new Float64Array(track.frames.length * cols);
  track.frames.forEach((row, i) => {
    flat.set(row, i * cols);
    flat[i * cols + f0] = row[f0] * up;
    for (const [k, x] of held) flat[i * cols + k] = x;
  });
  return flat;
}

// What the brightness dial changes of the glottis while the voice speaks. Only that: the
// amplitude's scale in a header multiplies the amplitude, and would again with every patch.
export function glottisPatch(about, share, bright) {
  const g = partOf(about, "glottis", share, bright);
  return { closureSharpness: g.closureSharpness, skew: "skew" in g ? g.skew : skewOf(about, share) };
}
