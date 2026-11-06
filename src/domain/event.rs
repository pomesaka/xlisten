//! The lines of `events/`, as `AGENT.md` tells them to the agent, after `XmoQ`'s `xmoq-agent`
//! (`fixtures/agent-lines/README.md` there). `at`, the writer's clock, is added where a line is
//! written.
//!
//! WHY `Serialize` here, in the domain: the form of a line is what the agent reads, part of the
//! model and not a detail of the folder. WHY NOT a type of the same fields for the wire: the two
//! would be kept in step by hand.

use serde::Serialize;

use crate::domain::input::Input;
use crate::domain::speaker::SpeakerLabel;
use crate::domain::time::Millis;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Reader(Reader),
    Speech(Speech),
    /// A line added to `inbox.txt`.
    Note {
        text: String,
    },
    /// Something went wrong; the reader goes on.
    Error {
        message: String,
    },
}

impl Event {
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Reader {
    Began {
        /// whisper's model.
        model: String,
        inputs: Vec<Input>,
        /// Whether RE-USE goes over the microphone.
        enhance: bool,
        /// Whether the others are told apart by voice.
        diarize: bool,
    },
    Ended,
}

/// Something said aloud, as whisper read it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Speech {
    /// In the order the utterances were read, of those with words.
    pub number: u64,
    pub from: Input,
    pub text: String,
    pub start_ms: Millis,
    pub end_ms: Millis,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<SpeakerLabel>,
}
