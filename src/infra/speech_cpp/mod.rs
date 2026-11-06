//! speech.cpp, through its C API (`include/speech.h` of v0.10.1, built by `mise run speech` into
//! `.cache/speech.cpp/`): NVIDIA's RE-USE (speech enhancement) and Nemotron 3 Diarization (who
//! speaks where, as the audio arrives). The one module that calls it, so the `unsafe` of it is here.
#![allow(unsafe_code, reason = "the bindings of speech.cpp's C API")]

pub mod speakers;

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};

use anyhow::{Context as _, Result, anyhow, bail};

use crate::app::ports::Enhancer;
use crate::domain::speaker::{SpeakerLabel, SpeakerTurn};

/// The files `mise run speech` fetches into `.cache/models`.
pub const RE_USE: &str = "NVIDIA-NemotronLabs-RE-USE-10M-F32.gguf";
pub const DIARIZATION: &str = "Nemotron-3-Diarization-99M-Q8_0.gguf";

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
    model: NonNull<Model>,
}

// SAFETY: a model is used from one thread at a time (each is owned by one worker, or sits behind
// a Mutex), which speech.cpp's documentation allows; the pointer itself may move between threads.
unsafe impl Send for SpeechModel {}

impl SpeechModel {
    /// One of the files `mise run speech` fetched (`RE_USE`, `DIARIZATION`).
    ///
    /// # Errors
    /// When it is not there, or cannot be loaded.
    pub fn fetched(file: &str) -> Result<Self> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(".cache/models")
            .join(file);
        if !path.exists() {
            bail!(
                "{} is not there: run `mise run speech` first",
                path.display()
            );
        }
        Self::load(&path)
    }

    fn load(path: &Path) -> Result<Self> {
        let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| anyhow!("{e}"))?;
        let mut model = ptr::null_mut();
        // SAFETY: `c` is a valid C string; NULL params take the defaults.
        check(
            unsafe { speech_model_load(c.as_ptr(), ptr::null(), &raw mut model) },
            &format!("loading {}", path.display()),
        )?;
        Ok(Self {
            model: NonNull::new(model).context("speech.cpp: a model of NULL")?,
        })
    }

    fn request(&self) -> Result<RequestGuard> {
        let mut request = ptr::null_mut();
        // SAFETY: the model is loaded and outlives the request (RequestGuard is dropped first).
        check(
            unsafe { speech_request_new(self.model.as_ptr(), &raw mut request) },
            "a request",
        )?;
        Ok(RequestGuard(
            NonNull::new(request).context("speech.cpp: a request of NULL")?,
        ))
    }

    /// A diarization of audio at `rate` that arrives a piece at a time, in its `low_latency` mode.
    /// It must be dropped before this model (`speakers::State` holds both, in that order).
    ///
    /// # Errors
    /// When speech.cpp fails.
    pub fn diarization(&self, rate: u32) -> Result<Diarizer> {
        let request = self.request()?;
        let mut d = ptr::null_mut();
        // SAFETY: the request is valid; speech.cpp copies its options.
        check(
            unsafe {
                speech_diarization_start(request.0.as_ptr(), c_int::try_from(rate)?, &raw mut d)
            },
            "starting a diarization",
        )?;
        Ok(Diarizer {
            d: NonNull::new(d).context("speech.cpp: a diarization of NULL")?,
            given: 0,
        })
    }
}

impl Enhancer for SpeechModel {
    /// RE-USE.
    fn enhance(&mut self, samples: &[f32], rate: u32) -> Result<Vec<f32>> {
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
            unsafe {
                speech_request_set_audio(request.0.as_ptr(), samples.as_ptr(), samples.len(), rate)
            },
            "the audio",
        )?;
        let mut out: Vec<f32> = Vec::with_capacity(samples.len());
        // SAFETY: `out` lives until speech_enhance returns, which calls `gather` only meanwhile.
        check(
            unsafe { speech_enhance(request.0.as_ptr(), gather, (&raw mut out).cast()) },
            "enhancing",
        )?;
        Ok(out)
    }
}

impl Drop for SpeechModel {
    fn drop(&mut self) {
        // SAFETY: every request and diarization of it was dropped before (requests within the call
        // that made them, a diarization by the field order of `speakers::State`).
        unsafe { speech_model_free(self.model.as_ptr()) }
    }
}

struct RequestGuard(NonNull<Request>);

impl Drop for RequestGuard {
    fn drop(&mut self) {
        // SAFETY: the request is not running once its call returned.
        unsafe { speech_request_free(self.0.as_ptr()) }
    }
}

pub struct Diarizer {
    d: NonNull<Diarization>,
    /// How many segments were given already.
    given: usize,
}

// SAFETY: a diarization is used by one thread at a time (it sits behind a Mutex), as its
// documentation asks.
unsafe impl Send for Diarizer {}

impl Diarizer {
    /// Pushes the next samples and gives the turns that became final.
    ///
    /// # Errors
    /// When speech.cpp fails.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<SpeakerTurn>> {
        let d = self.d.as_ptr();
        // SAFETY: the samples are copied.
        check(
            unsafe { speech_diarization_push(d, samples.as_ptr(), samples.len()) },
            "diarizing",
        )?;
        // SAFETY: the diarization is valid.
        let count = unsafe { speech_diarization_segment_count(d) };
        let mut turns = Vec::new();
        for i in self.given..count {
            let (mut start_s, mut end_s, mut speaker) = (0.0, 0.0, 0);
            // SAFETY: `i` is below the count; the pointers are to locals.
            check(
                unsafe {
                    speech_diarization_segment(
                        d,
                        i,
                        &raw mut start_s,
                        &raw mut end_s,
                        &raw mut speaker,
                    )
                },
                "a segment",
            )?;
            // The model tells 8 speakers apart at the most, each one a letter.
            if let Some(speaker) = SpeakerLabel::from_index(speaker) {
                turns.push(SpeakerTurn {
                    start_s,
                    end_s,
                    speaker,
                });
            }
        }
        self.given = count;
        Ok(turns)
    }

    /// The time, in seconds of the audio pushed, before which every turn has been given.
    #[must_use]
    pub fn settled(&self) -> f64 {
        // SAFETY: the diarization is valid.
        unsafe { speech_diarization_settled(self.d.as_ptr()) }
    }
}

impl Drop for Diarizer {
    fn drop(&mut self) {
        // SAFETY: the diarization is not used after this.
        unsafe { speech_diarization_free(self.d.as_ptr()) }
    }
}
