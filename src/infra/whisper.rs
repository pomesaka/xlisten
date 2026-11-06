//! `whisper.cpp`, through `whisper-rs` 0.16.0 (`whisper.cpp` 1.8.3, built from the source it carries),
//! as `xmoq-speech` of `XmoQ` uses it (`crates/speech/src/whisper.rs` and `vad.rs` there): a model loaded
//! once, and Silero, which says whether an utterance has a voice in it at all.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, anyhow};

use crate::app::ports::SpeechToText;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
    WhisperVadContext, WhisperVadContextParams,
};

/// Where `XmoQ`'s `mise run speech-model` puts the models (`crates/speech/scripts/fetch-model.sh`).
#[must_use]
pub fn model_dir() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map_or_else(
            || PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache"),
            PathBuf::from,
        )
        .join("xmoq/whisper")
}

/// A name of `XmoQ`'s cache (`large-v3-turbo-q5_0`), or the path of a `.bin`.
#[must_use]
pub fn model_path(name: &str) -> PathBuf {
    if Path::new(name).extension().is_some_and(|e| e == "bin") {
        PathBuf::from(name)
    } else {
        model_dir().join(format!("ggml-{name}.bin"))
    }
}

/// The share of a Silero window above which it is a voice: whisper.cpp's default
/// (`whisper_vad_default_params`, threshold 0.5).
const VOICE: f32 = 0.5;

pub struct Whisper {
    state: WhisperState,
    silero: WhisperVadContext,
}

impl Whisper {
    /// # Errors
    /// When a model cannot be read.
    pub fn open(model: &Path) -> Result<Self> {
        // whisper.cpp and ggml write their progress to stderr unless this is called.
        whisper_rs::install_logging_hooks();
        let context = WhisperContext::new_with_params(model, WhisperContextParameters::default())
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| {
                format!(
                    "loading {} (in XmoQ: mise run speech-model -- <name>)",
                    model.display()
                )
            })?;
        let state = context
            .create_state()
            .map_err(|e| anyhow!("the state of whisper.cpp: {e}"))?;
        let vad = model_dir().join("ggml-silero-v5.1.2.bin");
        let mut params = WhisperVadContextParams::default();
        // WHY on the processor: the model is small; XmoQ measured 0.12 ms for 32 ms of sound there.
        params.set_use_gpu(false);
        params.set_n_threads(1);
        let silero = WhisperVadContext::new(&vad.to_string_lossy(), params)
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| {
                format!(
                    "loading {} (in XmoQ: mise run speech-model -- silero-v5.1.2)",
                    vad.display()
                )
            })?;
        Ok(Self { state, silero })
    }
}

impl SpeechToText for Whisper {
    /// WHY Silero first: over music and noise, whisper writes words of its own
    /// (`ご視聴ありがとうございました`).
    fn read(&mut self, samples: &[f32]) -> Result<String> {
        self.silero
            .detect_speech(samples)
            .map_err(|e| anyhow!("Silero: {e}"))?;
        if !self.silero.probabilities().iter().any(|p| *p > VOICE) {
            return Ok(String::new());
        }
        // WHY greedy and not a beam: XmoQ's reason, a live line that comes late is worth less than
        // a word it got wrong.
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("ja"));
        // Each utterance is read on its own: the words before, given as context, are what whisper
        // repeats when it makes words up.
        params.set_no_context(true);
        params.set_single_segment(true);
        params.set_no_timestamps(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        self.state
            .full(params, samples)
            .map_err(|e| anyhow!("whisper.cpp: {e}"))?;
        let mut text = String::new();
        for segment in self.state.as_iter() {
            text.push_str(
                &segment
                    .to_str_lossy()
                    .map_err(|e| anyhow!("whisper.cpp: {e}"))?,
            );
        }
        Ok(text.trim().to_owned())
    }
}
