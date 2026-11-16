//! The CHIMERA voice driven by a control track, for the browser: C-ABI exports with no imports,
//! as CHIMERA's own `chimera-wasm` has them, so that an AudioWorklet can hold the module (its
//! global scope has no TextDecoder and no fetch: everything crosses as numbers and as pointers
//! into this module's memory). What a player does is `pmvc_voice::Player`'s, which the Python
//! module has too: a track sounds the same in both.
//!
//! JavaScript writes its bytes (a track's header as UTF-8 JSON, frames as f64) into memory it
//! asks for with `pmvc_alloc`, and frees it with `pmvc_free`. A call that can fail returns 0
//! (or a null player) and leaves what went wrong, in words, at `pmvc_error_ptr` for
//! `pmvc_error_len` bytes. Views into the memory go stale when it grows: make them again after
//! every call that may allocate.

use pmvc_voice::Player;
use std::cell::RefCell;

thread_local! {
    static ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn failed(e: String) -> i32 {
    ERROR.with(|x| *x.borrow_mut() = e);
    0
}

/// # Safety
/// `p` is `n` bytes of UTF-8 that JavaScript wrote.
unsafe fn text<'a>(p: *const u8, n: usize) -> Result<&'a str, String> {
    std::str::from_utf8(unsafe { std::slice::from_raw_parts(p, n) }).map_err(|e| e.to_string())
}

#[unsafe(no_mangle)]
pub extern "C" fn pmvc_alloc(n: usize) -> *mut u8 {
    let mut v = Vec::<u64>::with_capacity(n.div_ceil(8).max(1));
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p as *mut u8
}

/// # Safety
/// `p` came from `pmvc_alloc(n)` and is not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pmvc_free(p: *mut u8, n: usize) {
    drop(unsafe { Vec::from_raw_parts(p as *mut u64, 0, n.div_ceil(8).max(1)) });
}

#[unsafe(no_mangle)]
pub extern "C" fn pmvc_error_ptr() -> *const u8 {
    ERROR.with(|x| x.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn pmvc_error_len() -> usize {
    ERROR.with(|x| x.borrow().len())
}

/// The output rate (Hz).
#[unsafe(no_mangle)]
pub extern "C" fn pmvc_sample_rate() -> f64 {
    pmvc_voice::FS
}

/// A voice for a control track, from the track without its frames (`n` bytes of JSON: `body`,
/// `source`, `keys`, `rate`, and what it has of `speed`, `noise`, `glottis`, `airway`, `nose`).
/// Null if the header will not do.
///
/// # Safety
/// `header` is `n` bytes of UTF-8 that JavaScript wrote.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn player_new(header: *const u8, n: usize) -> *mut Player {
    match unsafe { text(header, n) }.and_then(Player::of_track) {
        Ok(p) => Box::into_raw(Box::new(p)),
        Err(e) => {
            failed(e);
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `p` came from `player_new` and is not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn player_free(p: *mut Player) {
    if !p.is_null() {
        drop(unsafe { Box::from_raw(p) });
    }
}

/// Follow `frames` (`n` rows of `cols` f64, row after row) from frame `first` on, in place of
/// whatever was scheduled: the whole track at 0, or its rest from the frame now playing after a
/// listener has turned a dial. 1 if it will do.
///
/// # Safety
/// `p` is a live player and `frames` is `n * cols` f64 that JavaScript wrote.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn player_set_track(p: *mut Player, frames: *const f64, n: usize, cols: usize, first: usize) -> i32 {
    let rows = unsafe { std::slice::from_raw_parts(frames, n * cols) };
    match unsafe { &mut *p }.set_track(rows, n, first) {
        Ok(()) => 1,
        Err(e) => failed(e),
    }
}

/// The next `n` samples into `out`.
///
/// # Safety
/// `p` is a live player and `out` has room for `n` f32.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn player_render(p: *mut Player, out: *mut f32, n: usize) {
    unsafe { &mut *p }.render(unsafe { std::slice::from_raw_parts_mut(out, n) });
}

/// The voice's clock (s).
///
/// # Safety
/// `p` is a live player.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn player_clock(p: *const Player) -> f64 {
    unsafe { &*p }.clock()
}

/// Settings of the voice changed while it plays, as JSON: `kind` 0 the source, 1 the glottis,
/// 2 the airway, 3 the nose (what `pmvc_voice::Player`'s `set_…` take). 1 if they will do.
///
/// # Safety
/// `p` is a live player and `patch` is `n` bytes of UTF-8 that JavaScript wrote.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn player_patch(p: *mut Player, kind: u32, patch: *const u8, n: usize) -> i32 {
    let player = unsafe { &mut *p };
    let done = unsafe { text(patch, n) }.and_then(|s| match kind {
        0 => player.set_source(s),
        1 => player.set_glottis(s),
        2 => player.set_airway(s),
        3 => player.set_nose(s),
        _ => Err(format!("no such kind of settings: {kind}")),
    });
    match done {
        Ok(()) => 1,
        Err(e) => failed(e),
    }
}
