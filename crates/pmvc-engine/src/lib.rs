//! `pmvc_engine`: the CHIMERA voice (third-party/chimera-vocal-synth, unmodified) for Python.
//!
//! A [`Voice`] follows a control track the way `pmvc-synth drive` does (one gesture per frame,
//! active over a Hann window, the articulators' lag compensated) and renders samples on demand.
//! It can be forked at any point: the fork carries the whole physical state (airway pressures
//! and flows, articulator positions and velocities, glottal phase), so a search can render a
//! track up to a frame once and try many continuations from there.

use chimera::anatomy::human::BodyMix;
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
    /// `body`: a preset name or `{"female": f, "child": c}` as JSON. `source`: a JSON object of
    /// voice source settings (`{"f0": 120, "tension": 0.5, ...}`). `keys`: the track's columns.
    /// `rate`: frames per second. `speed`: how many times faster than their natural rate the
    /// articulators follow the track.
    #[new]
    #[pyo3(signature = (body, source, keys, rate=100.0, speed=6.0))]
    fn new(body: &str, source: &str, keys: Vec<String>, rate: f64, speed: f64) -> PyResult<Self> {
        let body: BodyMix = serde_json::from_str(body).or_else(|_| serde_json::from_str(&format!("{body:?}"))).map_err(bad)?;
        let patch: SourcePatch = serde_json::from_str(source).map_err(bad)?;
        chimera::init();
        let mut v = HumanVoice::new(FS, body.body(), oversample_for(FS));
        let mut src = SourcePatch::f0(chimera::engine::default_f0(body));
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
                _ => return Err(bad(format!("unknown airway parameter {k:?}"))),
            }
        }
        Ok(())
    }

    /// The airway parameters `set_airway` takes, as they are now (a JSON object).
    #[getter]
    fn airway(&self) -> String {
        let a = &self.v.airway;
        serde_json::json!({"maxAreaRate": a.max_area_rate, "areaRateGain": a.area_rate_gain, "sealRate": a.seal_rate, "sealLo": a.seal_lo, "sealHi": a.seal_hi, "sealEase": a.seal_ease}).to_string()
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

#[pymodule]
fn pmvc_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Voice>()?;
    m.add("SAMPLE_RATE", FS as u32)?;
    Ok(())
}
