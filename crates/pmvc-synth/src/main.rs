//! `pmvc-synth`: drives the CHIMERA voice (third-party/chimera-vocal-synth, unmodified) from JSON.
//!
//!   pmvc-synth render <spec.json> <out.wav> [--info info.json] [--tracks tracks.json] [--rate 100]
//!   pmvc-synth plan   <spec.json> <info.json>
//!   pmvc-synth drive  <tracks.json> <out.wav>
//!
//! `render` plans phones with CHIMERA's own gestural planner, but takes what a text front end
//! would otherwise decide from the caller: each phone's duration, each phrase's start time and,
//! optionally, the pitch contour. `--tracks` also writes what the voice did as a control track:
//! the articulator positions and the glottal, breath and pitch controls, `rate` times a second.
//! `plan` only plans: it writes the phones' times (the durations the planner's rules give the
//! phones that came without one) and renders nothing.
//!
//! `drive` plays such a control track back with no planner: every frame becomes one gesture
//! whose activation is a Hann window one frame wide on each side, so neighbouring frames
//! cross-fade to exactly one and the voice follows the track. That makes the track a complete
//! description of an utterance: a candidate codec latent, and something to perturb.

use chimera::anatomy::human::{ArticKey, BodyMix, N_ARTIC};
use chimera::control::articulators::ARTIC_SPECS;
use chimera::control::score::{ControlKey, Gesture, Targets};
use chimera::math::PI;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::phonetics::{Intonation, PlannedSegment, Segment};
use chimera::voice::human_voice::{HumanVoice, SourcePatch, oversample_for};
use serde::{Deserialize, Serialize};
use std::process::ExitCode;

/// The output rate of every render (as `chimera render`).
const FS: f64 = 48000.0;

/// The planner starts a phrase's first phone this long after the phrase's own start (its breath lead).
const PLAN_LEAD: f64 = 0.06;

/// The source controls a track carries after the articulators.
const SOURCE_KEYS: [&str; 4] = ["abduction", "closure", "breath", "f0"];

const USAGE: &str = "\
usage: pmvc-synth render <spec.json> <out.wav> [--info info.json] [--tracks tracks.json] [--rate 100]
       pmvc-synth plan   <spec.json> <info.json>
       pmvc-synth drive  <tracks.json> <out.wav>";

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Spec {
    /// a preset name or `{ "female": f, "child": c }`
    body: BodyMix,
    /// voice source settings; `f0` is the planner's baseline pitch
    #[serde(default)]
    source: SourcePatch,
    phrases: Vec<PhraseSpec>,
    /// pitch (Hz) over the whole render; replaces the planner's intonation
    #[serde(default)]
    f0_track: Option<Curve>,
    /// silence rendered after the last phrase (s)
    #[serde(default = "default_tail")]
    tail: f64,
}

fn default_tail() -> f64 {
    0.3
}

/// One breath group. `t0` is when its first phone starts on the voice clock (s).
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PhraseSpec {
    t0: f64,
    segments: Vec<Segment>,
    #[serde(default)]
    intonation: Option<Intonation>,
}

/// Values at `i / rate` seconds on the voice clock.
#[derive(Deserialize)]
struct Curve {
    rate: f64,
    values: Vec<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Info {
    segments: Vec<PlannedSegment>,
    /// end of the last phrase (s)
    end: f64,
    /// length of the render (s)
    duration: f64,
    peak: f64,
    non_finite: usize,
}

/// A control track: `frames[i]` holds the value of each of `keys` at `i / rate` seconds.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Tracks {
    rate: f64,
    body: BodyMix,
    source: SourcePatch,
    keys: Vec<String>,
    frames: Vec<Vec<f64>>,
    /// how many times faster than their natural rate the articulators follow the track (`drive`)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    speed: Option<f64>,
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &str) -> Result<T, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))
}

fn write_json<T: Serialize>(path: &str, x: &T) -> Result<(), String> {
    let text = serde_json::to_string(x).map_err(|e| format!("{path}: {e}"))?;
    std::fs::write(path, text).map_err(|e| format!("{path}: {e}"))
}

/// 16-bit PCM mono WAV.
fn write_wav(path: &str, x: &[f32]) -> Result<(), String> {
    let n = x.len() as u32;
    let mut b = Vec::with_capacity(44 + 2 * x.len());
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + 2 * n).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&1u16.to_le_bytes()); // mono
    b.extend_from_slice(&(FS as u32).to_le_bytes());
    b.extend_from_slice(&(FS as u32 * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&(2 * n).to_le_bytes());
    for &v in x {
        let s = if v.is_finite() { (v.clamp(-1.0, 1.0) * 32767.0).round() as i16 } else { 0 };
        b.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, b).map_err(|e| format!("{path}: {e}"))
}

fn new_voice(body: BodyMix, source: &SourcePatch) -> HumanVoice {
    let mut v = HumanVoice::new(FS, body.body(), oversample_for(FS));
    let mut src = SourcePatch::f0(chimera::engine::default_f0(body));
    src.merge(source);
    v.set_source(&src);
    v
}

/// One gesture per frame, active over a Hann window that reaches the frames on either side.
fn frame_gesture(t: f64, hop: f64, strength: f64, targets: Targets) -> Gesture {
    Gesture::new(t, t, hop, hop, strength, targets)
}

/// The source controls as the voice applies them at time `t` (see `HumanVoice::control`).
fn source_controls(v: &HumanVoice, t: f64) -> [f64; 4] {
    let half = |k: ControlKey| {
        let b = v.score.blend(k, t);
        if b.weight > 0.0 { b.value * (b.weight / 2.0).min(1.0) } else { 0.0 }
    };
    let br = v.score.blend(ControlKey::BREATH, t);
    let breath = if br.weight > 0.0 { br.value * br.weight.min(1.0) } else { 0.0 };
    let pf = v.score.blend(ControlKey::F0, t);
    let f0 = if pf.weight > 0.0 { (v.source.f0 * 0.02 + pf.value * pf.weight) / (0.02 + pf.weight) } else { v.source.f0 };
    [half(ControlKey::ABDUCTION), half(ControlKey::CLOSURE), breath, f0]
}

/// The gestures of every phrase of `spec` (with its pitch contour, if it has one, in place of
/// the planner's intonation), the phones as planned, and the end of the last phrase (s).
fn plan_spec(spec: &Spec, base_f0: f64) -> Result<(Vec<Gesture>, Vec<PlannedSegment>, f64), String> {
    let mut gestures: Vec<Gesture> = Vec::new();
    let mut segments: Vec<PlannedSegment> = Vec::new();
    let mut end: f64 = 0.0;
    for p in &spec.phrases {
        if p.t0 < PLAN_LEAD {
            return Err(format!("a phrase starts at {} s; the planner needs {PLAN_LEAD} s before it", p.t0));
        }
        let o = PlanOptions { body: spec.body, t0: p.t0 - PLAN_LEAD, f0: Some(base_f0), intonation: p.intonation, ..Default::default() };
        let plan = plan_utterance(&p.segments, &o);
        gestures.extend_from_slice(&plan.gestures);
        segments.extend(plan.segments);
        end = end.max(plan.end);
    }
    if let Some(c) = &spec.f0_track {
        gestures.retain(|g| g.label_str() != "pitch");
        let hop = 1.0 / c.rate;
        for (i, &hz) in c.values.iter().enumerate() {
            if hz.is_finite() && hz > 0.0 {
                gestures.push(frame_gesture(i as f64 * hop, hop, 8.0, Targets::default().with(ControlKey::F0, hz)).label("pitch"));
            }
        }
    }
    Ok((gestures, segments, end))
}

fn plan(args: &[String]) -> Result<(), String> {
    let [spec_path, info_path] = args else { return Err(USAGE.into()) };
    let spec: Spec = read_json(spec_path)?;
    chimera::init();
    let base_f0 = new_voice(spec.body, &spec.source).source.f0;
    let (_, segments, end) = plan_spec(&spec, base_f0)?;
    write_json(info_path, &Info { segments, end, duration: end + spec.tail, peak: 0.0, non_finite: 0 })
}

fn render(args: &[String]) -> Result<(), String> {
    let mut pos: Vec<&String> = Vec::new();
    let (mut info_out, mut tracks_out, mut rate) = (None, None, 100.0);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--info" => info_out = Some(it.next().ok_or("--info needs a file")?.clone()),
            "--tracks" => tracks_out = Some(it.next().ok_or("--tracks needs a file")?.clone()),
            "--rate" => rate = it.next().and_then(|r| r.parse().ok()).ok_or("--rate needs a number")?,
            _ => pos.push(a),
        }
    }
    let [spec_path, wav_path] = pos[..] else { return Err(USAGE.into()) };
    let spec: Spec = read_json(spec_path)?;
    chimera::init();
    let mut voice = new_voice(spec.body, &spec.source);
    let (gestures, segments, end) = plan_spec(&spec, voice.source.f0)?;
    voice.schedule(&gestures);
    if voice.score.dropped > 0 {
        return Err(format!("{} gestures did not fit the score", voice.score.dropped));
    }

    let total = ((end + spec.tail) * FS).round() as usize;
    let mut out = vec![0f32; total];
    let hop = (FS / rate).round() as usize;
    let mut frames: Vec<Vec<f64>> = Vec::new();
    let mut at = 0;
    loop {
        if tracks_out.is_some() {
            let mut f = voice.artic.x.to_array().to_vec();
            f.extend(source_controls(&voice, at as f64 / FS));
            frames.push(f);
        }
        if at >= total {
            break;
        }
        let to = (at + hop).min(total);
        for block in out[at..to].chunks_mut(128) {
            voice.render(block);
        }
        at = to;
    }
    write_wav(wav_path, &out)?;
    let peak = out.iter().fold(0f64, |p, &v| if v.is_finite() { p.max((v as f64).abs()) } else { p });
    let non_finite = out.iter().filter(|v| !v.is_finite()).count();
    if let Some(path) = info_out {
        write_json(&path, &Info { segments, end, duration: total as f64 / FS, peak, non_finite })?;
    }
    if let Some(path) = tracks_out {
        let keys = ArticKey::NAMES.iter().chain(SOURCE_KEYS.iter()).map(|s| s.to_string()).collect();
        write_json(&path, &Tracks { rate, body: spec.body, source: SourcePatch::from_source(&voice.source), keys, frames, speed: None })?;
    }
    println!("{wav_path}: {:.2} s, peak {peak:.3}, non-finite {non_finite}", total as f64 / FS);
    Ok(())
}

/// Linear interpolation of column `col` of the track at time `t` (held at the ends).
fn sample(frames: &[Vec<f64>], col: usize, rate: f64, t: f64) -> f64 {
    let x = (t * rate).clamp(0.0, (frames.len() - 1) as f64);
    let i = x.floor() as usize;
    let j = (i + 1).min(frames.len() - 1);
    let a = x - i as f64;
    frames[i][col] * (1.0 - a) + frames[j][col] * a
}

fn drive(args: &[String]) -> Result<(), String> {
    let [track_path, wav_path] = args else { return Err(USAGE.into()) };
    let tr: Tracks = read_json(track_path)?;
    if tr.frames.is_empty() || tr.frames.iter().any(|f| f.len() != tr.keys.len()) {
        return Err(format!("{track_path}: every frame needs one value per key"));
    }
    let keys: Vec<ControlKey> = tr.keys.iter().map(|k| ControlKey::from_name(k).ok_or(format!("unknown key {k:?}"))).collect::<Result<_, _>>()?;
    chimera::init();
    let mut voice = new_voice(tr.body, &tr.source);
    let speed = tr.speed.unwrap_or(6.0);
    let hop = 1.0 / tr.rate;
    // An articulator follows its target through a critically damped second-order system, which
    // trails a moving target by 2/ω. The track holds positions, so each target is read that much
    // ahead and the articulator arrives on time.
    let lag: Vec<f64> = keys
        .iter()
        .map(|k| match k {
            ControlKey::Artic(a) => 2.0 / (2.0 * PI * ARTIC_SPECS[a.index()].hz * speed),
            ControlKey::Source(_) => 0.0,
        })
        .collect();
    debug_assert!(keys.len() <= N_ARTIC + 5);
    let gestures: Vec<Gesture> = (0..tr.frames.len())
        .map(|i| {
            let t = i as f64 * hop;
            let mut tg = Targets::default();
            for (c, k) in keys.iter().enumerate() {
                let x = sample(&tr.frames, c, tr.rate, t + lag[c]);
                if x.is_finite() {
                    tg.set(*k, x);
                }
            }
            frame_gesture(t, hop, 100.0, tg).speed(speed)
        })
        .collect();
    voice.schedule(&gestures);
    if voice.score.dropped > 0 {
        return Err(format!("{} gestures did not fit the score", voice.score.dropped));
    }
    let total = ((tr.frames.len() - 1) as f64 * hop * FS).round() as usize;
    let mut out = vec![0f32; total];
    for block in out.chunks_mut(128) {
        voice.render(block);
    }
    write_wav(wav_path, &out)?;
    let peak = out.iter().fold(0f64, |p, &v| if v.is_finite() { p.max((v as f64).abs()) } else { p });
    let non_finite = out.iter().filter(|v| !v.is_finite()).count();
    println!("{wav_path}: {:.2} s, peak {peak:.3}, non-finite {non_finite}", total as f64 / FS);
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("render") => render(&args[1..]),
        Some("plan") => plan(&args[1..]),
        Some("drive") => drive(&args[1..]),
        _ => Err(USAGE.to_string()),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
