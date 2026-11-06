//! What the use cases need of the world outside; `crate::infra` implements each.

use anyhow::Result;

use crate::domain::event::Event;
use crate::domain::speaker::SpeakerLabel;
use crate::domain::time::Span;
use crate::domain::utterance::{AudioBlock, Utterance};

/// The rate `SpeechToText` reads, and RE-USE runs at (whisper's).
pub const SPEECH_RATE: u32 = 16_000;

/// whisper.cpp.
pub trait SpeechToText: Send {
    /// The words of `samples` (mono, `SPEECH_RATE`), trimmed; empty where there is no voice in them.
    ///
    /// # Errors
    /// When the model fails.
    fn read(&mut self, samples: &[f32]) -> Result<String>;
}

/// RE-USE.
pub trait Enhancer: Send {
    /// The same sound with its noise and the echo of a room taken out, at the same rate.
    ///
    /// # Errors
    /// When the model fails.
    fn enhance(&mut self, samples: &[f32], rate: u32) -> Result<Vec<f32>>;
}

/// The diarization, hearing the sound of the apps as it comes.
pub trait SpeakerFinder: Send + Sync {
    /// # Errors
    /// When the model fails.
    fn hear(&self, block: &AudioBlock) -> Result<()>;

    /// Whoever spoke the most within `span`; may wait for the diarization to catch up with it.
    fn speaker(&self, span: Span) -> Option<SpeakerLabel>;
}

/// Whether the sound kept is as it was heard, or after RE-USE.
#[derive(Clone, Copy)]
pub enum Stage {
    Heard,
    Enhanced,
}

/// `--keep-audio`.
pub trait AudioKeeper: Send {
    /// Keeps the sound of `utterance` at a `stage` (at `SPEECH_RATE`); a failure is only told.
    fn keep(&self, utterance: &Utterance, stage: Stage, samples: &[f32]);
}

/// Where the lines for the agent go.
pub trait EventLog: Send + Sync {
    fn write(&self, event: Event);

    /// # Errors
    /// When what was written cannot be put where the agent reads it.
    fn flush(&self) -> Result<()>;
}

/// What the agent or the person wrote back (`inbox.txt`).
pub trait Notes {
    /// The lines added since the last read.
    ///
    /// # Errors
    /// When they cannot be read.
    fn read(&mut self) -> Result<Vec<Event>>;
}
