use crate::domain::event::{Event, Speech};
use crate::domain::speaker::SpeakerLabel;
use crate::domain::utterance::Utterance;

/// The lines of one run, numbered in the order the utterances were read.
#[derive(Default)]
pub struct Transcript {
    next: u64,
}

impl Transcript {
    /// The line of `utterance`, read as `text`; none where no words were read in it (Silero heard
    /// no voice), and such an utterance takes no number.
    pub fn line(
        &mut self,
        utterance: &Utterance,
        text: String,
        speaker: Option<SpeakerLabel>,
    ) -> Option<Event> {
        if text.is_empty() {
            return None;
        }
        let number = self.next;
        self.next += 1;
        Some(Event::Speech(Speech {
            number,
            from: utterance.input,
            text,
            start_ms: utterance.span.start,
            end_ms: utterance.span.end,
            speaker,
        }))
    }
}
