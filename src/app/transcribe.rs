//! A recording through the same `Pipeline` as `listen`, as fast as it reads: to compare settings
//! on the same sound.

use std::fmt;

use anyhow::{Result, bail};

use crate::app::pipeline::Pipeline;
use crate::app::ports::EventLog;
use crate::domain::event::Event;
use crate::domain::input::Input;
use crate::domain::time::{Millis, samples_ms};
use crate::domain::utterance::{AudioBlock, Segmenter};

pub struct Summary {
    seconds: f64,
    utterances: usize,
    speech_seconds: f64,
    enhanced_in: Option<f64>,
    read_in: f64,
}

/// Writes the lines of `samples`, their `start_ms` and `end_ms` from the start of the recording.
///
/// # Errors
/// At the first utterance that fails.
pub fn transcribe(
    samples: &[f32],
    rate: u32,
    input: Input,
    pipeline: &mut Pipeline,
    events: &dyn EventLog,
) -> Result<Summary> {
    let mut segmenter = Segmenter::new(input, rate);
    let block = usize::try_from(rate / 10).unwrap_or(1600);
    let mut utterances = Vec::new();
    for (i, part) in samples.chunks(block).enumerate() {
        utterances.extend(segmenter.push(&AudioBlock {
            samples: part.to_vec(),
            end: Millis::round(samples_ms(i * block + part.len(), rate)),
        }));
    }
    utterances.extend(segmenter.finish());
    let mut summary = Summary {
        seconds: samples_ms(samples.len(), rate) / 1000.0,
        utterances: utterances.len(),
        speech_seconds: 0.0,
        enhanced_in: pipeline.enhances().then_some(0.0),
        read_in: 0.0,
    };
    for u in &utterances {
        summary.speech_seconds += u.seconds();
        let out = pipeline.read(u);
        if let (Some(total), Some(t)) = (&mut summary.enhanced_in, out.enhanced_in) {
            *total += t.as_secs_f64();
        }
        summary.read_in += out.read_in.map_or(0.0, |t| t.as_secs_f64());
        for e in out.events {
            if let Event::Error { message } = e {
                bail!("{message}");
            }
            events.write(e);
        }
    }
    events.flush()?;
    Ok(summary)
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.1}s of audio, {} utterances of {:.1}s",
            self.seconds, self.utterances, self.speech_seconds
        )?;
        if let Some(t) = self.enhanced_in {
            write!(
                f,
                ", RE-USE {t:.2}s ({:.2} of their length)",
                t / self.speech_seconds
            )?;
        }
        write!(f, ", whisper {:.2}s", self.read_in)
    }
}
