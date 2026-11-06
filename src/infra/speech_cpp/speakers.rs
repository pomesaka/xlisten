//! Nemotron 3 Diarization, hearing the sound of the apps as it arrives, for as long as xlisten
//! runs.

use std::sync::{Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::app::ports::SpeakerFinder;
use crate::domain::speaker::{SpeakerLabel, SpeakerTurn, most_heard};
use crate::domain::time::{Span, samples_ms};
use crate::domain::utterance::AudioBlock;
use crate::infra::speech_cpp::{Diarizer, SpeechModel};

/// How long a line waits for the diarization to settle its end. In `low_latency` it trails the
/// audio by about a second (speech.cpp's docs/models/sortformer.md).
const WAIT: Duration = Duration::from_secs(3);

struct State {
    // WHY the diarizer before the model: fields are dropped in the order they are declared, and
    // speech.cpp asks for a diarization to be freed before its model.
    diarizer: Diarizer,
    _model: SpeechModel,
    turns: Vec<SpeakerTurn>,
    /// The clock of the first sample heard: where the diarization's seconds begin.
    origin: Option<f64>,
    rate: u32,
}

pub struct Speakers {
    state: Mutex<State>,
    /// Told each time a block was heard, for `speaker` to look again at what is settled.
    heard: Condvar,
}

impl Speakers {
    /// # Errors
    /// When the diarization cannot be started.
    pub fn new(model: SpeechModel, rate: u32) -> Result<Self> {
        Ok(Self {
            state: Mutex::new(State {
                diarizer: model.diarization(rate)?,
                _model: model,
                turns: Vec::new(),
                origin: None,
                rate,
            }),
            heard: Condvar::new(),
        })
    }
}

impl SpeakerFinder for Speakers {
    fn hear(&self, block: &AudioBlock) -> Result<()> {
        let mut s = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if s.origin.is_none() {
            s.origin = Some(block.end.as_f64() - samples_ms(block.samples.len(), s.rate));
        }
        let turns = s.diarizer.push(&block.samples)?;
        s.turns.extend(turns);
        drop(s);
        self.heard.notify_all();
        Ok(())
    }

    /// Waits until the diarization has settled the end of `span`, or `WAIT`; none where it has
    /// heard nothing yet, or nobody speaks within it.
    fn speaker(&self, span: Span) -> Option<SpeakerLabel> {
        let deadline = Instant::now() + WAIT;
        let mut s = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            let origin = s.origin?;
            let start_s = (span.start.as_f64() - origin) / 1000.0;
            let end_s = (span.end.as_f64() - origin) / 1000.0;
            let left = deadline.saturating_duration_since(Instant::now());
            if s.diarizer.settled() >= end_s || left.is_zero() {
                return most_heard(&s.turns, start_s, end_s);
            }
            s = self
                .heard
                .wait_timeout(s, left)
                .map_or_else(|e| e.into_inner().0, |(s, _)| s);
        }
    }
}
