//! `pmvc_engine`: the CHIMERA voice (third-party/chimera-vocal-synth, unmodified) for Python.
//!
//! A [`Voice`] follows a control track the way `pmvc-synth drive` does (one gesture per frame,
//! active over a Hann window, the articulators' lag compensated) and renders samples on demand.
//! It can be forked at any point: the fork carries the whole physical state (airway pressures
//! and flows, articulator positions and velocities, glottal phase), so a search can render a
//! track up to a frame once and try many continuations from there. What it does is
//! `pmvc_voice::Player`'s, which the browser's player (`pmvc-wasm`) has too.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pmvc_voice::{FS, Player};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn bad(e: String) -> PyErr {
    PyValueError::new_err(e)
}

#[pyclass]
struct Voice {
    p: Player,
}

#[pymethods]
impl Voice {
    /// `body`: a preset name or `{"female": f, "child": c}` as JSON, with `scale`, `pharynx`,
    /// `tongueRadius` or `lipWidth` beside them for an anatomy of its own (see `pmvc_voice::body_of`). `source`: a JSON object of
    /// voice source settings (`{"f0": 120, "tension": 0.5, ...}`). `keys`: the track's columns.
    /// `rate`: frames per second. `speed`: how many times faster than their natural rate the
    /// articulators follow the track.
    #[new]
    #[pyo3(signature = (body, source, keys, rate=100.0, speed=6.0))]
    fn new(body: &str, source: &str, keys: Vec<String>, rate: f64, speed: f64) -> PyResult<Self> {
        Ok(Voice { p: Player::new(body, source, &keys, rate, speed).map_err(bad)? })
    }

    /// A copy with the same physical state, to continue differently.
    fn fork(&self) -> Voice {
        Voice { p: self.p.clone() }
    }

    /// The voice clock (s).
    #[getter]
    fn clock(&self) -> f64 {
        self.p.clock()
    }

    /// Follow `frames` (rows are frames, columns the keys) from frame `first` on, in place of
    /// whatever was scheduled. Frame `first - 1` is rescheduled too: its window is still open.
    fn set_track(&mut self, frames: PyReadonlyArray2<f64>, first: usize) -> PyResult<()> {
        let a = frames.as_array();
        let rows: Vec<f64> = a.iter().copied().collect();
        self.p.set_track(&rows, a.nrows(), first).map_err(bad)
    }

    /// The next `n` samples. Other Python threads run while it renders.
    fn render<'py>(&mut self, py: Python<'py>, n: usize) -> Bound<'py, PyArray1<f32>> {
        let mut out = vec![0f32; n];
        let p = &mut self.p;
        py.allow_threads(|| p.render(&mut out));
        out.into_pyarray(py)
    }

    /// Voice source settings (a JSON object, as in the constructor), applied now.
    fn set_source(&mut self, source: &str) -> PyResult<()> {
        self.p.set_source(source).map_err(bad)
    }

    /// Glottis parameters set directly, over what the source's settings made of them: a JSON
    /// object with any of `openQuotient`, `skew`, `closureSharpness`, `amplitude` (cm²),
    /// `amplitudeScale` (times the present amplitude), `chink` (cm²), `jitter`, `shimmer`.
    /// They hold until the source is set again (see `pmvc_voice::Player::set_glottis`).
    fn set_glottis(&mut self, patch: &str) -> PyResult<()> {
        self.p.set_glottis(patch).map_err(bad)
    }

    /// The glottis parameters `set_glottis` takes, as they are now (a JSON object).
    #[getter]
    fn glottis(&self) -> String {
        self.p.glottis()
    }

    /// How fast a constriction of the tract may close and open, whether the walls displace air
    /// and the area under which a constriction counts as closed, set directly: a JSON object
    /// (see `pmvc_voice::Player::set_airway`).
    fn set_airway(&mut self, patch: &str) -> PyResult<()> {
        self.p.set_airway(patch).map_err(bad)
    }

    /// The nose's shape, set directly: a JSON object with any of `profile`, `length`, `sinus`
    /// and `sinusVolume` (see `pmvc_voice::Player::set_nose`).
    fn set_nose(&mut self, patch: &str) -> PyResult<()> {
        self.p.set_nose(patch).map_err(bad)
    }

    /// The nose as it is now (a JSON object): the nasal tube's areas (cm², for an adult man's
    /// body, from the velum to the nostrils), its length (cm, likewise), the sinus's opening and volume.
    #[getter]
    fn nose(&self) -> String {
        self.p.nose()
    }

    /// The airway parameters `set_airway` takes, as they are now (a JSON object).
    #[getter]
    fn airway(&self) -> String {
        self.p.airway()
    }

    /// What the voice is doing at this sample, for diagnosis: the glottal area (cm²) and the
    /// voicing gate (0..1); the narrowest point of the oral tube (cm²) and where it is (0 at the
    /// pharynx, 1 at the lips); the narrowest point of the pharynx (cm²); the pressure just above
    /// the glottis and just behind the oral tube's narrowest point (the engine's units); the flow
    /// out of the lips and out of the nose (cm³/s).
    #[getter]
    fn probe(&self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        self.p.probe()
    }

    /// The gains of the turbulence noise at the glottis, at constrictions and at the teeth.
    #[getter]
    fn noise_gains(&self) -> (f64, f64, f64) {
        self.p.noise_gains()
    }

    #[setter]
    fn set_noise_gains(&mut self, g: (f64, f64, f64)) {
        self.p.set_noise_gains(g)
    }
}

/// CHIMERA's English front end: text in, the phrases' segments out (see `pmvc_voice::English`).
#[pyclass]
struct English {
    e: pmvc_voice::English,
}

#[pymethods]
impl English {
    /// `dict`: the dictionary's text ("word\tPHONES" lines).
    #[new]
    fn new(dict: &str) -> Self {
        English { e: pmvc_voice::English::new(dict) }
    }

    /// The text's phrases as JSON.
    fn phrases(&self, text: &str) -> PyResult<String> {
        self.e.phrases(text).map_err(bad)
    }
}

/// The midsagittal outlines of a body in a posture, as CHIMERA computes them for its own
/// display and for the airway's areas: JSON, in cm with the glottis at the origin. `body` as
/// for a [`Voice`]; `posture`: a JSON object of articulator values, the rest neutral.
#[pyfunction]
fn outline(body: &str, posture: &str) -> PyResult<String> {
    pmvc_voice::outline(body, posture).map_err(bad)
}

#[pymodule]
fn pmvc_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Voice>()?;
    m.add_class::<English>()?;
    m.add_function(wrap_pyfunction!(outline, m)?)?;
    m.add("SAMPLE_RATE", FS as u32)?;
    Ok(())
}
