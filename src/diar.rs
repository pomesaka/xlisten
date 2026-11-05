//! Who among `others` speaks: Nemotron 3 Diarization of speech.cpp, hearing the sound of the apps
//! as it arrives, for as long as xlisten runs. Speakers keep their label for the run, and a run
//! labels its own: an `A` of one meeting is not the `A` of another.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::speech::{Diarizer, Segment, SpeechModel};

struct State {
    // Declared before the model: dropped first, as speech.cpp asks.
    diarizer: Diarizer,
    segments: Vec<Segment>,
    /// The wall clock, ms, of the first sample pushed.
    origin_ms: Option<f64>,
    rate: u32,
}

#[derive(Clone)]
pub struct Speakers {
    state: Arc<Mutex<State>>,
    _model: Arc<Mutex<SpeechModel>>,
}

/// How long a line waits for the diarization to settle its end. In `low_latency` it trails the
/// audio by about a second (speech.cpp's docs/models/sortformer.md).
const WAIT: Duration = Duration::from_secs(3);

impl Speakers {
    /// # Errors
    /// When the model cannot be loaded.
    pub fn new(model: SpeechModel, rate: u32) -> Result<Self> {
        let diarizer = model.diarization(rate)?;
        Ok(Self {
            state: Arc::new(Mutex::new(State {
                diarizer,
                segments: Vec::new(),
                origin_ms: None,
                rate,
            })),
            _model: Arc::new(Mutex::new(model)),
        })
    }

    /// Pushes a block of the apps' sound, whose last sample is at `end_ms`.
    ///
    /// # Errors
    /// When speech.cpp fails.
    pub fn push(&self, samples: &[f32], end_ms: f64) -> Result<()> {
        let Ok(mut s) = self.state.lock() else {
            return Ok(());
        };
        if s.origin_ms.is_none() {
            #[expect(clippy::cast_precision_loss, reason = "a block is short")]
            let length_ms = samples.len() as f64 * 1000.0 / f64::from(s.rate);
            s.origin_ms = Some(end_ms - length_ms);
        }
        let new = s.diarizer.push(samples)?;
        s.segments.extend(new);
        Ok(())
    }

    /// The label (`A`, `B`, ...) of whoever speaks the most between `start_ms` and `end_ms` (wall
    /// clock), once the diarization has settled it; `None` where it finds nobody, or does not
    /// settle within `WAIT`.
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "ms since the epoch fit in f64")]
    pub fn who(&self, start_ms: i64, end_ms: i64) -> Option<String> {
        let began = Instant::now();
        loop {
            {
                let s = self.state.lock().ok()?;
                let origin = s.origin_ms?;
                let (start, end) = (
                    (start_ms as f64 - origin) / 1000.0,
                    (end_ms as f64 - origin) / 1000.0,
                );
                if s.diarizer.settled() >= end || began.elapsed() >= WAIT {
                    return label(&s.segments, start, end);
                }
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

fn label(segments: &[Segment], start: f64, end: f64) -> Option<String> {
    let mut by: Vec<(i32, f64)> = Vec::new();
    for s in segments {
        let o = s.end.min(end) - s.start.max(start);
        if o > 0.0 {
            match by.iter_mut().find(|(k, _)| *k == s.speaker) {
                Some((_, total)) => *total += o,
                None => by.push((s.speaker, o)),
            }
        }
    }
    let (speaker, _) = by.into_iter().max_by(|a, b| a.1.total_cmp(&b.1))?;
    let n = u8::try_from(speaker).ok()?;
    Some(char::from(b'A' + n).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_is_whoever_speaks_the_most_within() {
        let segments = [
            Segment {
                start: 0.0,
                end: 2.0,
                speaker: 0,
            },
            Segment {
                start: 1.5,
                end: 5.0,
                speaker: 1,
            },
        ];
        let rows: &[(&str, f64, f64, Option<&str>)] = &[
            ("one speaker alone", 0.0, 1.0, Some("A")),
            ("the one of the longer part", 1.0, 4.0, Some("B")),
            ("nobody", 6.0, 7.0, None),
        ];
        for (name, start, end, expected) in rows {
            assert_eq!(
                label(&segments, *start, *end).as_deref(),
                *expected,
                "{name}"
            );
        }
    }
}
