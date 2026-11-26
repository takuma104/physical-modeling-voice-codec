//! The CHIMERA voice (third-party/chimera-vocal-synth: from V2 on a fork of it, whose changes are
//! settings that default to the engine's own) driven by a control track:
//! what `pmvc_engine` (the Python module) and `pmvc-wasm` (the browser's player) share.
//!
//! A [`Player`] follows a control track the way `pmvc-synth drive` does (one gesture per frame,
//! active over a Hann window, the articulators' lag compensated) and renders samples on demand.
//! It can be cloned at any point: the clone carries the whole physical state (airway pressures
//! and flows, articulator positions and velocities, glottal phase), so a search can render a
//! track up to a frame once and try many continuations from there.

use chimera::airway::frequency::transfer_function;
use chimera::anatomy::human::{ArticKey, BodyMix, HumanArtic, HumanBody, HumanVocalTract, N_ARTIC, NEUTRAL};
use chimera::control::articulators::ARTIC_SPECS;
use chimera::control::score::{ControlKey, Gesture, Targets};
use chimera::math::PI;
use chimera::phonetics::english::EnglishG2P;
use chimera::voice::human_voice::{HumanVoice, SourcePatch, oversample_for};

/// The output rate (Hz), as every CHIMERA render.
pub const FS: f64 = 48000.0;

/// What went wrong, in words.
pub type Said<T> = Result<T, String>;

fn bad(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// A body from its description: a preset name or `{"female": f, "child": c}`, CHIMERA's own
/// continuous family, and with any of `scale`, `pharynx`, `tongueRadius` and `lipWidth` beside
/// them that number of the anatomy as given, in place of the family's (a tract longer than
/// the adult male's, a throat of another proportion). Also the family's member, for what
/// CHIMERA keys by it (the pitch a voice idles at).
pub fn body_of(s: &str) -> Said<(BodyMix, HumanBody)> {
    let v: serde_json::Value = serde_json::from_str(s).or_else(|_| serde_json::from_str(&format!("{s:?}"))).map_err(bad)?;
    let mut own = [None; 4];
    let mut family = v.clone();
    if let Some(o) = family.as_object_mut() {
        for (k, key) in ["scale", "pharynx", "tongueRadius", "lipWidth"].iter().enumerate() {
            if let Some(x) = o.remove(*key) {
                own[k] = Some(x.as_f64().ok_or_else(|| bad(format!("{key} is not a number")))?);
            }
        }
    }
    let mix: BodyMix = serde_json::from_value(family).map_err(bad)?;
    let b = mix.body();
    let body = HumanBody {
        name: b.name,
        scale: own[0].unwrap_or(b.scale),
        pharynx: own[1].unwrap_or(b.pharynx),
        tongue_radius: own[2].unwrap_or(b.tongue_radius),
        lip_width: own[3].unwrap_or(b.lip_width),
    };
    Ok((mix, body))
}

#[derive(Clone)]
pub struct Player {
    pub v: HumanVoice,
    keys: Vec<ControlKey>,
    /// how far ahead each key's target is read (s): an articulator trails its target by 2/ω
    lag: Vec<f64>,
    rate: f64,
    speed: f64,
    /// the piriform fossa's length (cm) and area (cm²) as set, for an adult man's body
    fossa: (f64, f64),
    /// the walls' second layer as set: its loss (1000 over its resistance) and its corner (Hz)
    layer: (f64, f64),
}

impl Player {
    /// `body`: a preset name or `{"female": f, "child": c}` as JSON, with `scale`, `pharynx`,
    /// `tongueRadius` or `lipWidth` beside them for an anatomy of its own (see `body_of`). `source`: a JSON object of
    /// voice source settings (`{"f0": 120, "tension": 0.5, ...}`). `keys`: the track's columns.
    /// `rate`: frames per second. `speed`: how many times faster than their natural rate the
    /// articulators follow the track.
    pub fn new(body: &str, source: &str, keys: &[String], rate: f64, speed: f64) -> Said<Self> {
        let (mix, body) = body_of(body)?;
        let patch: SourcePatch = serde_json::from_str(source).map_err(bad)?;
        chimera::init();
        let mut v = HumanVoice::new(FS, body, oversample_for(FS));
        let mut src = SourcePatch::f0(chimera::engine::default_f0(mix));
        src.merge(&patch);
        v.set_source(&src);
        let keys: Vec<ControlKey> = keys.iter().map(|k| ControlKey::from_name(k).ok_or_else(|| bad(format!("unknown key {k:?}")))).collect::<Said<_>>()?;
        let lag = keys
            .iter()
            .map(|k| match k {
                ControlKey::Artic(a) => 2.0 / (2.0 * PI * ARTIC_SPECS[a.index()].hz * speed),
                ControlKey::Source(_) => 0.0,
            })
            .collect();
        // the fossa as the engine builds it
        let (k, t) = (body.scale, v.airway.fossa);
        let fossa = (v.airway.net.tube_length[t.index] / k, v.airway.net.area[t.node(0)] / (k * k));
        Ok(Player { v, keys, lag, rate, speed, fossa, layer: (0.0, 8000.0) })
    }

    /// The voice clock (s).
    pub fn clock(&self) -> f64 {
        self.v.clock
    }

    /// Follow `frames` (`n` rows, one a frame, of the keys' values, row after row) from frame
    /// `first` on, in place of whatever was scheduled. Frame `first - 1` is rescheduled too: its
    /// window is still open.
    pub fn set_track(&mut self, frames: &[f64], n: usize, first: usize) -> Said<()> {
        let cols = if n == 0 { 0 } else { frames.len() / n };
        let a = |i: usize, c: usize| frames[i * cols + c];
        if cols != self.keys.len() || n == 0 || frames.len() != n * cols {
            return Err(bad(format!("the track needs {} columns and at least one frame", self.keys.len())));
        }
        let hop = 1.0 / self.rate;
        let sample = |c: usize, t: f64| {
            let x = (t * self.rate).clamp(0.0, (n - 1) as f64);
            let i = x.floor() as usize;
            let j = (i + 1).min(n - 1);
            let w = x - i as f64;
            a(i, c) * (1.0 - w) + a(j, c) * w
        };
        self.v.cancel_score();
        for i in first.saturating_sub(1)..n {
            let t = i as f64 * hop;
            let mut tg = Targets::default();
            for (c, k) in self.keys.iter().enumerate() {
                let x = sample(c, t + self.lag[c]);
                if x.is_finite() {
                    tg.set(*k, x);
                }
            }
            self.v.schedule(&[Gesture::new(t, t, hop, hop, 100.0, tg).speed(self.speed)]);
        }
        if self.v.score.dropped > 0 {
            return Err(bad("the track does not fit the gesture score"));
        }
        Ok(())
    }

    /// The next samples, as many as `out` holds.
    pub fn render(&mut self, out: &mut [f32]) {
        for block in out.chunks_mut(128) {
            self.v.render(block);
        }
    }

    /// Voice source settings (a JSON object, as in the constructor), applied now.
    pub fn set_source(&mut self, source: &str) -> Said<()> {
        let patch: SourcePatch = serde_json::from_str(source).map_err(bad)?;
        self.v.set_source(&patch);
        Ok(())
    }

    /// Glottis parameters set directly, over what the source's settings made of them: a JSON
    /// object with any of `openQuotient`, `skew`, `closureSharpness`, `amplitude` (cm²),
    /// `amplitudeScale` (times the present amplitude), `chink` (cm²), `jitter`, `shimmer`.
    /// The source's settings reach these only through `tension`, `brightness`, `breathiness` and
    /// `roughness`, each over a limited range; this is for a voice outside it (a softer one than
    /// tension 0 gives, say). They hold until the source is set again.
    pub fn set_glottis(&mut self, patch: &str) -> Said<()> {
        let v: serde_json::Value = serde_json::from_str(patch).map_err(bad)?;
        let obj = v.as_object().ok_or_else(|| bad("a JSON object is needed"))?;
        let g = &mut self.v.glottis.params;
        for (k, x) in obj {
            let x = x.as_f64().ok_or_else(|| bad(format!("{k}: a number is needed")))?;
            match k.as_str() {
                "openQuotient" => g.open_quotient = x,
                "skew" => g.skew = x,
                "closureSharpness" => g.closure_sharpness = x,
                "amplitude" => g.amplitude = x,
                "amplitudeScale" => g.amplitude *= x,
                "chink" => g.chink = x,
                "jitter" => g.jitter = x,
                "shimmer" => g.shimmer = x,
                _ => return Err(bad(format!("unknown glottis parameter {k:?}"))),
            }
        }
        Ok(())
    }

    /// The glottis parameters `set_glottis` takes, as they are now (a JSON object).
    pub fn glottis(&self) -> String {
        let g = &self.v.glottis.params;
        serde_json::json!({"openQuotient": g.open_quotient, "skew": g.skew, "closureSharpness": g.closure_sharpness, "amplitude": g.amplitude, "chink": g.chink, "jitter": g.jitter, "shimmer": g.shimmer}).to_string()
    }

    /// How fast a constriction of the tract may close and open, set directly: a JSON object with
    /// any of `maxAreaRate` (cm²/s, the opening rate, which sets how sharp a burst is),
    /// `areaRateGain` (1/s per cm² of area), `sealRate` (cm²/s, the closing rate near contact),
    /// `sealLo`, `sealHi` (cm², between which the closing rate eases from the one to the other)
    /// and `sealEase` (cm², below which it slows on to the seal). The engine's own values keep a
    /// closure from clicking at a man's pitch; a voice that lets more air through needs gentler ones.
    /// `volumeSource` (0 or 1) is the engine's switch for the air the tract's walls displace as
    /// they move: on, that air flows into the tract and is heard, without any breath, as soft
    /// thumps below a few hundred hertz. `closedArea` (cm²) is the area under which the engine
    /// takes a constriction for closed and stops the flow through it within one sample: at the
    /// engine's own 2e-4 the flow still running then (about 1 cm³/s under a full breath) stops as
    /// a step, a click that reaches 16 kHz. `noiseDraws` (the fork's; 0, the engine's way, or 3
    /// for a voice's three noise sources) is how many random numbers the noise draws a sample
    /// whatever the number of sources: at 0 the breath noise of everything after the first
    /// narrowing depends on the control step in which the narrowing's source joined, so two
    /// tracks that are all but the same render with different noise.
    pub fn set_airway(&mut self, patch: &str) -> Said<()> {
        let v: serde_json::Value = serde_json::from_str(patch).map_err(bad)?;
        let obj = v.as_object().ok_or_else(|| bad("a JSON object is needed"))?;
        let a = &mut self.v.airway;
        for (k, x) in obj {
            let x = x.as_f64().ok_or_else(|| bad(format!("{k}: a number is needed")))?;
            match k.as_str() {
                "maxAreaRate" => a.max_area_rate = x,
                "areaRateGain" => a.area_rate_gain = x,
                "sealRate" => a.seal_rate = x,
                "sealLo" => a.seal_lo = x,
                "sealHi" => a.seal_hi = x,
                "sealEase" => a.seal_ease = x,
                "volumeSource" => a.net.enable_volume_source = x != 0.0,
                "closedArea" => a.net.closed_area = x,
                "noiseDraws" => a.net.noise_draws = x as usize,
                _ => return Err(bad(format!("unknown airway parameter {k:?}"))),
            }
        }
        Ok(())
    }

    /// The tube above the glottis (ventricle, ventricular folds, vestibule), set directly: a JSON
    /// object with any of `length` (times the length the body gives it: 2.4 cm × scale ×
    /// pharynx), and `ventricle`, `tube`, `wide`, `narrow` (cm², for an adult man's body: the
    /// areas of the ventricle, of the tube as it leaves the ventricular folds, and of its top
    /// with the epilarynx channel at 0 and at 1). The engine's own are 1, 1.3, 1.2, 1.9 and
    /// 0.35. The tube's resonance is the voice's fourth formant; a longer tube brings it down.
    /// Only the acoustic length changes: the tract is drawn as before.
    pub fn set_larynx(&mut self, patch: &str) -> Said<()> {
        let v: serde_json::Value = serde_json::from_str(patch).map_err(bad)?;
        let obj = v.as_object().ok_or_else(|| bad("a JSON object is needed"))?;
        let t = &mut self.v.airway.tract.laryngeal_tube;
        for (k, x) in obj {
            let x = x.as_f64().ok_or_else(|| bad(format!("{k}: a number is needed")))?;
            if !(x > 0.0) {
                return Err(bad(format!("{k}: a positive number is needed")));
            }
            match k.as_str() {
                "length" if !(0.5..=2.5).contains(&x) => return Err(bad("length: between 0.5 and 2.5 times the body's")),
                "length" => t.length = x,
                "ventricle" => t.ventricle = x,
                "tube" => t.tube = x,
                "wide" => t.wide = x,
                "narrow" => t.narrow = x,
                _ => return Err(bad(format!("unknown larynx parameter {k:?}"))),
            }
        }
        Ok(())
    }

    /// The tube above the glottis as it is now (a JSON object, as `set_larynx` takes it).
    pub fn larynx(&self) -> String {
        let t = self.v.airway.tract.laryngeal_tube;
        serde_json::json!({"length": t.length, "ventricle": t.ventricle, "tube": t.tube, "wide": t.wide, "narrow": t.narrow}).to_string()
    }

    /// The nose's shape, set directly: a JSON object with any of `profile` (the nasal tube's area
    /// from the velum to the nostrils, as [place, cm²] pairs with the place from 0 to 1, joined by
    /// straight lines), `length` (cm), `sinus` (cm², the opening of the maxillary sinus into the
    /// nasal tube) and `sinusVolume` (cm³). All for an adult man's body: they are scaled to the voice's body as the
    /// engine scales its own (lengths by the body's scale, areas by its square). The engine builds
    /// the nose once, from constants, and no posture changes it.
    pub fn set_nose(&mut self, patch: &str) -> Said<()> {
        let v: serde_json::Value = serde_json::from_str(patch).map_err(bad)?;
        let obj = v.as_object().ok_or_else(|| bad("a JSON object is needed"))?;
        let a = &mut self.v.airway;
        let (k, t) = (a.body.scale, a.nasal);
        let number = |name: &str| -> Said<Option<f64>> { obj.get(name).map(|x| x.as_f64().ok_or_else(|| bad(format!("{name}: a number is needed")))).transpose() };
        let length = number("length")?.map(|x| x * k);
        if let Some(l) = length {
            if l / (t.n as f64) < a.dx_stable {
                return Err(bad(format!("length: under {:.2} cm the nasal tube's sections are shorter than a time step allows", a.dx_stable * t.n as f64 / k)));
            }
        }
        let areas: Vec<f64> = match obj.get("profile") {
            Some(p) => {
                let pts: Vec<(f64, f64)> = serde_json::from_value(p.clone()).map_err(bad)?;
                if pts.is_empty() {
                    return Err(bad("profile: at least one [place, area] pair is needed"));
                }
                (0..t.n)
                    .map(|i| {
                        let u = (i as f64 + 0.5) / t.n as f64;
                        let j = pts.iter().position(|q| q.0 >= u).unwrap_or(pts.len());
                        let area = if j == 0 {
                            pts[0].1
                        } else if j == pts.len() {
                            pts[j - 1].1
                        } else {
                            let (p0, p1) = (pts[j - 1], pts[j]);
                            p0.1 + (p1.1 - p0.1) * (u - p0.0) / (p1.0 - p0.0).max(1e-9)
                        };
                        area * k * k
                    })
                    .collect()
            }
            None => (0..t.n).map(|i| a.net.area[t.node(i)]).collect(),
        };
        if obj.contains_key("profile") || length.is_some() {
            a.net.set_tube_targets(t, &areas, length, None);
        }
        if let Some(x) = number("sinus")? {
            // the sinus's port is the second branch the engine adds after the velum's
            a.net.set_port_area(a.velum_port + 2, x * k * k);
        }
        if let Some(x) = number("sinusVolume")? {
            // the sinus is the node the engine adds after the lungs': one section, whose area times its length is the volume
            let node = a.lungs + 1;
            a.net.set_node_area_target(node, x * k * k * k / a.net.dx[node]);
        }
        Ok(())
    }

    /// The nose as it is now (a JSON object): the nasal tube's areas (cm², for an adult man's
    /// body, from the velum to the nostrils), its length (cm, likewise), the sinus's opening and volume.
    pub fn nose(&self) -> String {
        let a = &self.v.airway;
        let (k, t) = (a.body.scale, a.nasal);
        let areas: Vec<f64> = (0..t.n).map(|i| a.net.area[t.node(i)] / (k * k)).collect();
        let node = a.lungs + 1;
        serde_json::json!({"areas": areas, "length": a.net.tube_length[t.index] / k, "sinus": a.net.port_area[a.velum_port + 2] / (k * k), "sinusVolume": a.net.area[node] * a.net.dx[node] / (k * k * k)}).to_string()
    }

    /// The airway parameters `set_airway` takes, as they are now (a JSON object).
    pub fn airway(&self) -> String {
        let a = &self.v.airway;
        serde_json::json!({
            "maxAreaRate": a.max_area_rate, "areaRateGain": a.area_rate_gain, "sealRate": a.seal_rate, "sealLo": a.seal_lo, "sealHi": a.seal_hi, "sealEase": a.seal_ease,
            "volumeSource": if a.net.enable_volume_source { 1.0 } else { 0.0 },
            "closedArea": a.net.closed_area,
        })
        .to_string()
    }

    /// What the voice is doing at this sample, for diagnosis: the glottal area (cm²) and the
    /// voicing gate (0..1); the narrowest point of the oral tube (cm²) and where it is (0 at the
    /// pharynx, 1 at the lips); the narrowest point of the pharynx (cm²); the pressure just above
    /// the glottis and just behind the oral tube's narrowest point (the engine's units); the flow
    /// out of the lips and out of the nose (cm³/s).
    pub fn probe(&self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let a = &self.v.airway;
        let narrowest = |xs: &[f64]| xs.iter().copied().enumerate().fold((0usize, f64::INFINITY), |m, (i, x)| if x < m.1 { (i, x) } else { m });
        let (at, oral) = narrowest(&a.or_min);
        let (_, pharynx) = narrowest(&a.ph_min);
        let behind = if at > 0 { a.oral.node(at - 1) } else { a.pharynx.last_node() };
        (
            self.v.glottis.area,
            self.v.glottis.voicing,
            oral,
            at as f64 / (a.n_or.max(2) - 1) as f64,
            pharynx,
            a.net.p[a.pharynx.first_node()],
            a.net.p[behind],
            a.net.radiated_flow(a.rad_lips),
            a.net.radiated_flow(a.rad_nose),
        )
    }

    /// The tract's transfer function as it stands at this sample, by the engine's own
    /// frequency-domain model of its airway (the simulation's lumped elements, linearised; the
    /// glottis an ideal source of flow, which leaves the windpipe out): the level (dB, against an
    /// arbitrary reference) of the sound radiated from lips, nose and walls for a unit of flow at
    /// the glottis, at each of `freqs` (Hz). What the tract's shape alone does to a voice, with
    /// no source in it: where its resonances and its dips are.
    pub fn transfer(&self, freqs: &[f64]) -> Vec<f64> {
        let a = &self.v.airway;
        transfer_function(&a.net.linear_elements(), a.pharynx.first_node(), freqs, &[a.glottis_port]).pressure_db
    }

    /// The articulators where they stand at this sample: CHIMERA's seventeen, in its own order
    /// (`ArticKey::ALL`). What a track asks for is not quite where they are: they follow their
    /// targets with their own inertia, and a tongue that would enter the palate is held back.
    pub fn pose(&self) -> Vec<f64> {
        self.v.artic.x.to_array().to_vec()
    }

    /// The tract's transfer function (as `transfer` gives it) with the articulators standing as
    /// each of `poses` has them (rows of seventeen, as `pose` gives them): the levels at `freqs`,
    /// row after row. Nothing is rendered, so a form of the tract (`set_larynx`, `set_form`) can
    /// be tried on the postures of many frames in the time one of them takes to sound. The
    /// airway's targets are left at the last pose: this is for a voice that is not rendering.
    pub fn transfers(&mut self, poses: &[f64], freqs: &[f64]) -> Said<Vec<f64>> {
        if poses.len() % N_ARTIC != 0 {
            return Err(bad(format!("a pose needs {N_ARTIC} numbers")));
        }
        let mut out = Vec::with_capacity(poses.len() / N_ARTIC * freqs.len());
        for row in poses.chunks(N_ARTIC) {
            self.v.airway.apply(&HumanArtic::from_array(row), 0.0);
            out.extend(self.transfer(freqs));
        }
        Ok(out)
    }

    /// The tract's form where neither the body's four numbers nor a posture sets it: a JSON
    /// object with any of `fossaLength` (cm) and `fossaArea` (cm²), the piriform fossa's (the
    /// side branch above the tube over the glottis; for an adult man's body, scaled to the
    /// voice's as the engine scales its own: 2.0 and 0.9); `pharynxAlpha`, `pharynxBeta`,
    /// `oralAlpha`, `oralBeta`, `frontAlpha`, `frontBeta`, the area that a width d (cm) of the
    /// midsagittal outline stands for, α·d^β, in the pharynx, in the mouth and behind the teeth
    /// (the engine's: 2.0, 2.6 and 2.0, all to the power 1.4); `viscous`, what the viscous loss
    /// at the tubes' walls is multiplied by, and `viscousHz`, the frequency (Hz) it is computed
    /// for (the engine's loss does not grow with frequency: 700); `layerLoss` and `layerCorner`,
    /// a second layer at the tubes' walls (the fork's: a resistance of 1000/`layerLoss`
    /// dyn·s/cm³ per unit area against a stiffness, with no mass) whose loss grows with the
    /// square of the frequency up to `layerCorner` (Hz), so that it widens the upper
    /// resonances and leaves the lowest alone. At 1 and 8000 it adds about 70 Hz to the width
    /// of a resonance at 3 kHz in a tube of 3 cm², 10 Hz at 1 kHz and 200 Hz at 6 kHz. The
    /// engine has none (0). And `profile`, a tract of the speaker's own (the fork's
    /// `HumanAirway::set_area_profile`): a list of natural logarithms of what the tract's area
    /// is multiplied by at places equally far apart from the glottis to the lips, whatever the
    /// articulators do (empty: none, the engine's).
    pub fn set_form(&mut self, patch: &str) -> Said<()> {
        let v: serde_json::Value = serde_json::from_str(patch).map_err(bad)?;
        let obj = v.as_object().ok_or_else(|| bad("a JSON object is needed"))?;
        let a = &mut self.v.airway;
        let (k, t) = (a.body.scale, a.fossa);
        if let Some(p) = obj.get("profile") {
            let logs: Option<Vec<f64>> = p.as_array().and_then(|l| l.iter().map(|x| x.as_f64().filter(|x| x.abs() <= 3.0)).collect());
            a.set_area_profile(&logs.ok_or_else(|| bad("profile: a list of logarithms of area ratios is needed, none beyond 3"))?);
        }
        for (key, x) in obj.iter().filter(|(key, _)| key.as_str() != "profile") {
            let x = x.as_f64().ok_or_else(|| bad(format!("{key}: a number is needed")))?;
            if !(x > 0.0 || (key == "layerLoss" && x == 0.0)) {
                return Err(bad(format!("{key}: a positive number is needed")));
            }
            let law = &mut a.tract.area_law;
            match key.as_str() {
                "fossaLength" if x * k / (t.n as f64) < a.dx_stable => return Err(bad(format!("fossaLength: under {:.2} cm the fossa's sections are shorter than a time step allows", a.dx_stable * t.n as f64 / k))),
                "fossaLength" => self.fossa.0 = x,
                "fossaArea" => self.fossa.1 = x,
                "pharynxAlpha" => law.pharynx.0 = x,
                "pharynxBeta" => law.pharynx.1 = x,
                "oralAlpha" => law.oral.0 = x,
                "oralBeta" => law.oral.1 = x,
                "frontAlpha" => law.front.0 = x,
                "frontBeta" => law.front.1 = x,
                "viscous" => a.net.viscous_scale = x,
                "viscousHz" => a.net.viscous_omega = 2.0 * PI * x,
                "layerLoss" => self.layer.0 = x,
                "layerCorner" => self.layer.1 = x,
                _ => return Err(bad(format!("unknown form parameter {key:?}"))),
            }
        }
        // the layer's resistance, and the stiffness that puts its corner where it is asked for
        a.net.layer_r = if self.layer.0 > 0.0 { 1000.0 / self.layer.0 } else { 0.0 };
        a.net.layer_k = 2.0 * PI * self.layer.1 * a.net.layer_r;
        if obj.contains_key("fossaLength") || obj.contains_key("fossaArea") {
            a.net.set_tube_targets(t, &vec![self.fossa.1 * k * k; t.n], Some(self.fossa.0 * k), None);
        }
        Ok(())
    }

    /// The tract's form as it is now (a JSON object, as `set_form` takes it).
    pub fn form(&self) -> String {
        let a = &self.v.airway;
        let law = a.tract.area_law;
        serde_json::json!({
            "fossaLength": self.fossa.0, "fossaArea": self.fossa.1,
            "pharynxAlpha": law.pharynx.0, "pharynxBeta": law.pharynx.1, "oralAlpha": law.oral.0, "oralBeta": law.oral.1, "frontAlpha": law.front.0, "frontBeta": law.front.1,
            "viscous": a.net.viscous_scale, "viscousHz": a.net.viscous_omega / (2.0 * PI),
            "layerLoss": self.layer.0, "layerCorner": self.layer.1, "profile": a.area_profile(),
        })
        .to_string()
    }

    /// The gains of the turbulence noise at the glottis, at constrictions and at the teeth.
    pub fn noise_gains(&self) -> (f64, f64, f64) {
        let g = self.v.airway.noise_gain;
        (g.glottis, g.constriction, g.teeth)
    }

    pub fn set_noise_gains(&mut self, g: (f64, f64, f64)) {
        let n = &mut self.v.airway.noise_gain;
        (n.glottis, n.constriction, n.teeth) = g;
    }

    /// A voice for a control track, from the track without its frames (JSON): `body`, `source`,
    /// `keys`, `rate`, and where it has them `speed`, `noise` (what the three noise gains are
    /// multiplied by), `glottis`, `airway`, `nose`, `larynx` and `form`. As `pmvc.calibrate.make_voice` makes one.
    pub fn of_track(header: &str) -> Said<Self> {
        let t: serde_json::Value = serde_json::from_str(header).map_err(bad)?;
        let part = |name: &str| t.get(name).filter(|x| x.as_object().is_some_and(|o| !o.is_empty())).map(|x| x.to_string());
        let keys: Vec<String> = serde_json::from_value(t.get("keys").cloned().ok_or_else(|| bad("the track has no keys"))?).map_err(bad)?;
        let body = t.get("body").ok_or_else(|| bad("the track has no body"))?.to_string();
        let source = t.get("source").ok_or_else(|| bad("the track has no source"))?.to_string();
        let rate = t.get("rate").and_then(|x| x.as_f64()).unwrap_or(100.0);
        let speed = t.get("speed").and_then(|x| x.as_f64()).unwrap_or(6.0);
        let mut p = Player::new(&body, &source, &keys, rate, speed)?;
        if let Some(gains) = t.get("noise").and_then(|x| x.as_array()) {
            let g: Vec<f64> = gains.iter().filter_map(|x| x.as_f64()).collect();
            if g.len() != 3 {
                return Err(bad("noise: three numbers are needed"));
            }
            let own = p.noise_gains();
            p.set_noise_gains((own.0 * g[0], own.1 * g[1], own.2 * g[2]));
        }
        if let Some(x) = part("glottis") {
            p.set_glottis(&x)?;
        }
        if let Some(x) = part("airway") {
            p.set_airway(&x)?;
        }
        if let Some(x) = part("nose") {
            p.set_nose(&x)?;
        }
        if let Some(x) = part("larynx") {
            p.set_larynx(&x)?;
        }
        if let Some(x) = part("form") {
            p.set_form(&x)?;
        }
        Ok(p)
    }
}

/// The midsagittal outlines of a body in a posture, as CHIMERA computes them for its own
/// display and for the airway's areas: JSON, in cm with the glottis at the origin. `body` as
/// for a [`Player`]; `posture`: a JSON object of articulator values, the rest neutral.
pub fn outline(body: &str, posture: &str) -> Said<String> {
    let (_, body) = body_of(body)?;
    let values: std::collections::BTreeMap<String, f64> = serde_json::from_str(posture).map_err(bad)?;
    let mut a = NEUTRAL;
    for (name, x) in &values {
        a.set(ArticKey::from_name(name).ok_or_else(|| bad(format!("unknown articulator {name:?}")))?, *x);
    }
    serde_json::to_string(&HumanVocalTract::new(body).compute_new(&a)).map_err(bad)
}

/// The body of a control track in the postures of its frames, for a display: what [`outline`]
/// gives, for a row of the track at a time. The columns that are not articulators (the pitch,
/// the lungs' pressure, the voice's settings) do not move a body and are passed over.
pub struct Tract {
    tract: HumanVocalTract,
    keys: Vec<Option<ArticKey>>,
}

impl Tract {
    /// From the track without its frames (JSON), as a [`Player`] is: its `body` and `keys`.
    pub fn of_track(header: &str) -> Said<Self> {
        let t: serde_json::Value = serde_json::from_str(header).map_err(bad)?;
        let keys: Vec<String> = serde_json::from_value(t.get("keys").cloned().ok_or_else(|| bad("the track has no keys"))?).map_err(bad)?;
        let (_, body) = body_of(&t.get("body").ok_or_else(|| bad("the track has no body"))?.to_string())?;
        chimera::init();
        Ok(Tract { tract: HumanVocalTract::new(body), keys: keys.iter().map(|k| ArticKey::from_name(k)).collect() })
    }

    /// The midsagittal outlines with the articulators where a row of the track puts them
    /// (JSON, in cm with the glottis at the origin).
    pub fn frame(&mut self, row: &[f64]) -> Said<String> {
        if row.len() != self.keys.len() {
            return Err(bad(format!("the row needs {} columns", self.keys.len())));
        }
        let mut a = NEUTRAL;
        for (k, x) in self.keys.iter().zip(row) {
            if let (Some(k), true) = (k, x.is_finite()) {
                a.set(*k, *x);
            }
        }
        serde_json::to_string(&self.tract.compute_new(&a)).map_err(bad)
    }
}

/// CHIMERA's own English front end (its dictionary, CMUdict, with letter-to-sound rules for a
/// word that is not in it, numbers spelt out, and the reductions of running speech): text in,
/// the phrases' segments out. The engine's planner would turn them into gestures by rule; a
/// text-to-speech model (`pmvc.tts`) takes them as what it reads instead.
pub struct English {
    g: EnglishG2P,
}

impl English {
    /// `dict`: the dictionary's text, a line a word ("word\tPHONES": the file CHIMERA ships as
    /// `public/data/cmudict.txt.gz`, unzipped).
    pub fn new(dict: &str) -> Self {
        let mut g = EnglishG2P::new();
        g.load_dict(dict);
        English { g }
    }

    /// The text's phrases as JSON: for each its `segments` (`ipa`, and where they apply
    /// `stress` and `wordEnd`), its `intonation`, the `pause` after it (s) and its `words`.
    pub fn phrases(&self, text: &str) -> Said<String> {
        serde_json::to_string(&self.g.to_phrases(text)).map_err(bad)
    }
}
