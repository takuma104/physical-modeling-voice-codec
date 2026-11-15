//! `pmvc_engine`: the CHIMERA voice (third-party/chimera-vocal-synth, unmodified) for Python.
//!
//! A [`Voice`] follows a control track the way `pmvc-synth drive` does (one gesture per frame,
//! active over a Hann window, the articulators' lag compensated) and renders samples on demand.
//! It can be forked at any point: the fork carries the whole physical state (airway pressures
//! and flows, articulator positions and velocities, glottal phase), so a search can render a
//! track up to a frame once and try many continuations from there.

use chimera::anatomy::human::{ArticKey, BodyMix, HumanBody, HumanVocalTract, NEUTRAL};
use chimera::control::articulators::ARTIC_SPECS;
use chimera::control::score::{ControlKey, Gesture, Targets};
use chimera::math::PI;
use chimera::voice::human_voice::{HumanVoice, SourcePatch, oversample_for};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The output rate (Hz), as every CHIMERA render.
const FS: f64 = 48000.0;

fn bad(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A body from its description: a preset name or `{"female": f, "child": c}`, CHIMERA's own
/// continuous family, and with any of `scale`, `pharynx`, `tongueRadius` and `lipWidth` beside
/// them that number of the anatomy as given, in place of the family's (a tract longer than
/// the adult male's, a throat of another proportion). Also the family's member, for what
/// CHIMERA keys by it (the pitch a voice idles at).
fn body_of(s: &str) -> PyResult<(BodyMix, HumanBody)> {
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

#[pyclass]
struct Voice {
    v: HumanVoice,
    keys: Vec<ControlKey>,
    /// how far ahead each key's target is read (s): an articulator trails its target by 2/ω
    lag: Vec<f64>,
    rate: f64,
    speed: f64,
}

#[pymethods]
impl Voice {
    /// `body`: a preset name or `{"female": f, "child": c}` as JSON, with `scale`, `pharynx`,
    /// `tongueRadius` or `lipWidth` beside them for an anatomy of its own (see `body_of`). `source`: a JSON object of
    /// voice source settings (`{"f0": 120, "tension": 0.5, ...}`). `keys`: the track's columns.
    /// `rate`: frames per second. `speed`: how many times faster than their natural rate the
    /// articulators follow the track.
    #[new]
    #[pyo3(signature = (body, source, keys, rate=100.0, speed=6.0))]
    fn new(body: &str, source: &str, keys: Vec<String>, rate: f64, speed: f64) -> PyResult<Self> {
        let (mix, body) = body_of(body)?;
        let patch: SourcePatch = serde_json::from_str(source).map_err(bad)?;
        chimera::init();
        let mut v = HumanVoice::new(FS, body, oversample_for(FS));
        let mut src = SourcePatch::f0(chimera::engine::default_f0(mix));
        src.merge(&patch);
        v.set_source(&src);
        let keys: Vec<ControlKey> = keys.iter().map(|k| ControlKey::from_name(k).ok_or_else(|| bad(format!("unknown key {k:?}")))).collect::<PyResult<_>>()?;
        let lag = keys
            .iter()
            .map(|k| match k {
                ControlKey::Artic(a) => 2.0 / (2.0 * PI * ARTIC_SPECS[a.index()].hz * speed),
                ControlKey::Source(_) => 0.0,
            })
            .collect();
        Ok(Voice { v, keys, lag, rate, speed })
    }

    /// A copy with the same physical state, to continue differently.
    fn fork(&self) -> Voice {
        Voice { v: self.v.clone(), keys: self.keys.clone(), lag: self.lag.clone(), rate: self.rate, speed: self.speed }
    }

    /// The voice clock (s).
    #[getter]
    fn clock(&self) -> f64 {
        self.v.clock
    }

    /// Follow `frames` (rows are frames, columns the keys) from frame `first` on, in place of
    /// whatever was scheduled. Frame `first - 1` is rescheduled too: its window is still open.
    fn set_track(&mut self, frames: PyReadonlyArray2<f64>, first: usize) -> PyResult<()> {
        let a = frames.as_array();
        let (n, cols) = a.dim();
        if cols != self.keys.len() || n == 0 {
            return Err(bad(format!("the track needs {} columns and at least one frame", self.keys.len())));
        }
        let hop = 1.0 / self.rate;
        let sample = |c: usize, t: f64| {
            let x = (t * self.rate).clamp(0.0, (n - 1) as f64);
            let i = x.floor() as usize;
            let j = (i + 1).min(n - 1);
            let w = x - i as f64;
            a[[i, c]] * (1.0 - w) + a[[j, c]] * w
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

    /// The next `n` samples. Other Python threads run while it renders.
    fn render<'py>(&mut self, py: Python<'py>, n: usize) -> Bound<'py, PyArray1<f32>> {
        let mut out = vec![0f32; n];
        let v = &mut self.v;
        py.allow_threads(|| {
            for block in out.chunks_mut(128) {
                v.render(block);
            }
        });
        out.into_pyarray(py)
    }

    /// Voice source settings (a JSON object, as in the constructor), applied now.
    fn set_source(&mut self, source: &str) -> PyResult<()> {
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
    fn set_glottis(&mut self, patch: &str) -> PyResult<()> {
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
    #[getter]
    fn glottis(&self) -> String {
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
    /// a step, a click that reaches 16 kHz.
    fn set_airway(&mut self, patch: &str) -> PyResult<()> {
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
                _ => return Err(bad(format!("unknown airway parameter {k:?}"))),
            }
        }
        Ok(())
    }

    /// The nose's shape, set directly: a JSON object with any of `profile` (the nasal tube's area
    /// from the velum to the nostrils, as [place, cm²] pairs with the place from 0 to 1, joined by
    /// straight lines), `length` (cm), `sinus` (cm², the opening of the maxillary sinus into the
    /// nasal tube) and `sinusVolume` (cm³). All for an adult man's body: they are scaled to the voice's body as the
    /// engine scales its own (lengths by the body's scale, areas by its square). The engine builds
    /// the nose once, from constants, and no posture changes it.
    fn set_nose(&mut self, patch: &str) -> PyResult<()> {
        let v: serde_json::Value = serde_json::from_str(patch).map_err(bad)?;
        let obj = v.as_object().ok_or_else(|| bad("a JSON object is needed"))?;
        let a = &mut self.v.airway;
        let (k, t) = (a.body.scale, a.nasal);
        let number = |name: &str| -> PyResult<Option<f64>> { obj.get(name).map(|x| x.as_f64().ok_or_else(|| bad(format!("{name}: a number is needed")))).transpose() };
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
    #[getter]
    fn nose(&self) -> String {
        let a = &self.v.airway;
        let (k, t) = (a.body.scale, a.nasal);
        let areas: Vec<f64> = (0..t.n).map(|i| a.net.area[t.node(i)] / (k * k)).collect();
        let node = a.lungs + 1;
        serde_json::json!({"areas": areas, "length": a.net.tube_length[t.index] / k, "sinus": a.net.port_area[a.velum_port + 2] / (k * k), "sinusVolume": a.net.area[node] * a.net.dx[node] / (k * k * k)}).to_string()
    }

    /// The airway parameters `set_airway` takes, as they are now (a JSON object).
    #[getter]
    fn airway(&self) -> String {
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
    #[getter]
    fn probe(&self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
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

    /// The gains of the turbulence noise at the glottis, at constrictions and at the teeth.
    #[getter]
    fn noise_gains(&self) -> (f64, f64, f64) {
        let g = self.v.airway.noise_gain;
        (g.glottis, g.constriction, g.teeth)
    }

    #[setter]
    fn set_noise_gains(&mut self, g: (f64, f64, f64)) {
        let n = &mut self.v.airway.noise_gain;
        (n.glottis, n.constriction, n.teeth) = g;
    }
}

/// The midsagittal outlines of a body in a posture, as CHIMERA computes them for its own
/// display and for the airway's areas: JSON, in cm with the glottis at the origin. `body` as
/// for a [`Voice`]; `posture`: a JSON object of articulator values, the rest neutral.
#[pyfunction]
fn outline(body: &str, posture: &str) -> PyResult<String> {
    let (_, body) = body_of(body)?;
    let values: std::collections::BTreeMap<String, f64> = serde_json::from_str(posture).map_err(bad)?;
    let mut a = NEUTRAL;
    for (name, x) in &values {
        a.set(ArticKey::from_name(name).ok_or_else(|| bad(format!("unknown articulator {name:?}")))?, *x);
    }
    serde_json::to_string(&HumanVocalTract::new(body).compute_new(&a)).map_err(bad)
}

#[pymodule]
fn pmvc_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Voice>()?;
    m.add_function(wrap_pyfunction!(outline, m)?)?;
    m.add("SAMPLE_RATE", FS as u32)?;
    Ok(())
}
