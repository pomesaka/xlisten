//! speech.cpp, through its C API (`include/speech.h` of v0.10.1, built by `mise run speech` into
//! `.cache/speech.cpp/`): NVIDIA's RE-USE (speech enhancement) and Nemotron 3 Diarization (who
//! speaks where, as the audio arrives). The one file that calls it, so the `unsafe` of it is here.
#![allow(unsafe_code, reason = "the bindings of speech.cpp's C API")]

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;
use std::ptr;

use anyhow::{Result, anyhow, bail};

#[repr(C)]
struct Model {
    _private: [u8; 0],
}
#[repr(C)]
struct Request {
    _private: [u8; 0],
}
#[repr(C)]
struct Diarization {
    _private: [u8; 0],
}

const SPEECH_OK: c_int = 0;

type AudioCallback =
    extern "C" fn(samples: *const f32, n_samples: usize, user_data: *mut c_void) -> c_int;

#[link(name = "speech")]
unsafe extern "C" {
    fn speech_last_error() -> *const c_char;
    fn speech_model_load(
        path: *const c_char,
        params: *const c_void,
        model: *mut *mut Model,
    ) -> c_int;
    fn speech_model_free(model: *mut Model);
    fn speech_request_new(model: *mut Model, request: *mut *mut Request) -> c_int;
    fn speech_request_free(request: *mut Request);
    fn speech_request_set_audio(
        request: *mut Request,
        samples: *const f32,
        n_samples: usize,
        sample_rate: c_int,
    ) -> c_int;
    fn speech_enhance(
        request: *mut Request,
        on_audio: AudioCallback,
        user_data: *mut c_void,
    ) -> c_int;
    fn speech_diarization_start(
        request: *const Request,
        sample_rate: c_int,
        diarization: *mut *mut Diarization,
    ) -> c_int;
    fn speech_diarization_free(diarization: *mut Diarization);
    fn speech_diarization_push(
        diarization: *mut Diarization,
        samples: *const f32,
        n_samples: usize,
    ) -> c_int;
    fn speech_diarization_segment_count(diarization: *const Diarization) -> usize;
    fn speech_diarization_segment(
        diarization: *const Diarization,
        index: usize,
        start: *mut f64,
        end: *mut f64,
        speaker: *mut c_int,
    ) -> c_int;
    fn speech_diarization_settled(diarization: *const Diarization) -> f64;
}

fn check(status: c_int, what: &str) -> Result<()> {
    if status == SPEECH_OK {
        return Ok(());
    }
    // SAFETY: speech_last_error() gives a string of this thread that lives until the next call.
    let message = unsafe {
        let p = speech_last_error();
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    bail!("speech.cpp: {what}: {status} {message}")
}

/// A model of speech.cpp, loaded on its default device (Metal).
pub struct SpeechModel {
    model: *mut Model,
}

// SAFETY: speech.cpp's models are used from one thread at a time here (each is owned by one
// worker), which its documentation allows; the pointer itself may move between threads.
unsafe impl Send for SpeechModel {}

impl SpeechModel {
    /// # Errors
    /// When the file cannot be loaded.
    pub fn load(path: &Path) -> Result<Self> {
        let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| anyhow!("{e}"))?;
        let mut model = ptr::null_mut();
        // SAFETY: `c` is a valid C string; NULL params take the defaults.
        check(
            unsafe { speech_model_load(c.as_ptr(), ptr::null(), &raw mut model) },
            &format!("loading {}", path.display()),
        )?;
        Ok(Self { model })
    }

    fn request(&self) -> Result<RequestGuard> {
        let mut request = ptr::null_mut();
        // SAFETY: the model is loaded and outlives the request (RequestGuard is dropped first).
        check(
            unsafe { speech_request_new(self.model, &raw mut request) },
            "a request",
        )?;
        Ok(RequestGuard(request))
    }

    /// RE-USE: the same audio with its noise and the echo of a room taken out, at the same rate.
    ///
    /// # Errors
    /// When speech.cpp fails.
    pub fn enhance(&self, samples: &[f32], rate: u32) -> Result<Vec<f32>> {
        extern "C" fn gather(samples: *const f32, n: usize, user: *mut c_void) -> c_int {
            // SAFETY: `user` is the `Vec` below, alive for the call; speech.cpp hands `n` samples.
            unsafe {
                let out = &mut *user.cast::<Vec<f32>>();
                out.extend_from_slice(std::slice::from_raw_parts(samples, n));
            }
            0
        }
        let request = self.request()?;
        let rate = c_int::try_from(rate)?;
        // SAFETY: the samples are copied by speech.cpp.
        check(
            unsafe { speech_request_set_audio(request.0, samples.as_ptr(), samples.len(), rate) },
            "the audio",
        )?;
        let mut out: Vec<f32> = Vec::with_capacity(samples.len());
        // SAFETY: `out` lives until speech_enhance returns, which calls `gather` only meanwhile.
        check(
            unsafe { speech_enhance(request.0, gather, (&raw mut out).cast()) },
            "enhancing",
        )?;
        Ok(out)
    }

    /// A diarization of audio at `rate` that arrives a piece at a time, in its `low_latency` mode.
    ///
    /// # Errors
    /// When speech.cpp fails.
    pub fn diarization(&self, rate: u32) -> Result<Diarizer> {
        let request = self.request()?;
        let mut d = ptr::null_mut();
        // SAFETY: the request is valid; speech.cpp copies its options.
        check(
            unsafe { speech_diarization_start(request.0, c_int::try_from(rate)?, &raw mut d) },
            "starting a diarization",
        )?;
        Ok(Diarizer { d, given: 0 })
    }
}

impl Drop for SpeechModel {
    fn drop(&mut self) {
        // SAFETY: every request and diarization of it is freed before (they borrow nothing, but the
        // owners drop them first: see `crate::listen`).
        unsafe { speech_model_free(self.model) }
    }
}

struct RequestGuard(*mut Request);

impl Drop for RequestGuard {
    fn drop(&mut self) {
        // SAFETY: the request is not running once its call returned.
        unsafe { speech_request_free(self.0) }
    }
}

/// A segment of one speaker, in seconds from the start of the audio pushed.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub speaker: i32,
}

pub struct Diarizer {
    d: *mut Diarization,
    given: usize,
}

// SAFETY: a diarization is used by one thread at a time (it sits behind a Mutex), as its
// documentation asks.
unsafe impl Send for Diarizer {}

impl Diarizer {
    /// Pushes the next samples and gives the segments that became final.
    ///
    /// # Errors
    /// When speech.cpp fails.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<Segment>> {
        // SAFETY: the samples are copied.
        check(
            unsafe { speech_diarization_push(self.d, samples.as_ptr(), samples.len()) },
            "diarizing",
        )?;
        // SAFETY: the diarization is valid.
        let count = unsafe { speech_diarization_segment_count(self.d) };
        let mut out = Vec::new();
        for i in self.given..count {
            let (mut start, mut end, mut speaker) = (0.0, 0.0, 0);
            // SAFETY: `i` is below the count; the pointers are to locals.
            check(
                unsafe {
                    speech_diarization_segment(
                        self.d,
                        i,
                        &raw mut start,
                        &raw mut end,
                        &raw mut speaker,
                    )
                },
                "a segment",
            )?;
            out.push(Segment {
                start,
                end,
                speaker,
            });
        }
        self.given = count;
        Ok(out)
    }

    /// The time, in seconds of the audio pushed, before which every segment has been given.
    #[must_use]
    pub fn settled(&self) -> f64 {
        // SAFETY: the diarization is valid.
        unsafe { speech_diarization_settled(self.d) }
    }
}

impl Drop for Diarizer {
    fn drop(&mut self) {
        // SAFETY: the diarization is not used after this.
        unsafe { speech_diarization_free(self.d) }
    }
}
