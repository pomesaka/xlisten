//! One utterance to its lines: resampled for whisper, enhanced where its input is, read, and given
//! a speaker where its input has them.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::app::ports::{AudioKeeper, Enhancer, SPEECH_RATE, SpeakerFinder, SpeechToText, Stage};
use crate::domain::event::Event;
use crate::domain::transcript::Transcript;
use crate::domain::utterance::Utterance;
use crate::dsp::resample;

pub struct Pipeline {
    reader: Box<dyn SpeechToText>,
    enhancer: Option<Box<dyn Enhancer>>,
    speakers: Option<Arc<dyn SpeakerFinder>>,
    keeper: Option<Box<dyn AudioKeeper>>,
    transcript: Transcript,
}

/// What came of one utterance.
pub struct Outcome {
    pub events: Vec<Event>,
    /// How long RE-USE and whisper took, where they ran.
    pub enhanced_in: Option<Duration>,
    pub read_in: Option<Duration>,
}

impl Pipeline {
    pub fn new(
        reader: Box<dyn SpeechToText>,
        enhancer: Option<Box<dyn Enhancer>>,
        speakers: Option<Arc<dyn SpeakerFinder>>,
        keeper: Option<Box<dyn AudioKeeper>>,
    ) -> Self {
        Self {
            reader,
            enhancer,
            speakers,
            keeper,
            transcript: Transcript::default(),
        }
    }

    #[must_use]
    pub fn enhances(&self) -> bool {
        self.enhancer.is_some()
    }

    pub fn read(&mut self, utterance: &Utterance) -> Outcome {
        let mut out = Outcome {
            events: Vec::new(),
            enhanced_in: None,
            read_in: None,
        };
        let input = utterance.input;
        let mut samples = resample(&utterance.samples, utterance.rate, SPEECH_RATE);
        self.keep(utterance, Stage::Heard, &samples);
        if let Some(enhancer) = self.enhancer.as_mut().filter(|_| input.is_enhanced()) {
            let began = Instant::now();
            match enhancer.enhance(&samples, SPEECH_RATE) {
                Ok(enhanced) => {
                    out.enhanced_in = Some(began.elapsed());
                    samples = enhanced;
                    self.keep(utterance, Stage::Enhanced, &samples);
                }
                // The sound as it was heard is read instead.
                Err(e) => out.events.push(Event::error(format!("{input}: {e:#}"))),
            }
        }
        let began = Instant::now();
        let text = match self.reader.read(&samples) {
            Ok(text) => text,
            Err(e) => {
                out.events.push(Event::error(format!("{input}: {e:#}")));
                return out;
            }
        };
        out.read_in = Some(began.elapsed());
        let speaker = self
            .speakers
            .as_ref()
            .filter(|_| input.has_speakers() && !text.is_empty())
            .and_then(|s| s.speaker(utterance.span));
        out.events
            .extend(self.transcript.line(utterance, text, speaker));
        out
    }

    fn keep(&self, utterance: &Utterance, stage: Stage, samples: &[f32]) {
        if let Some(k) = &self.keeper {
            k.keep(utterance, stage, samples);
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::type_complexity, reason = "tests")]
mod tests {
    use anyhow::{Result, bail};

    use super::*;
    use crate::domain::event::Speech;
    use crate::domain::input::Input;
    use crate::domain::speaker::SpeakerLabel;
    use crate::domain::time::{Millis, Span};
    use crate::domain::utterance::AudioBlock;

    /// Reads `plain` or `enhanced`, by the first sample (`FakeEnhancer` makes it 1), or what it is
    /// given to say.
    struct FakeReader(Option<Result<String, String>>);

    impl SpeechToText for FakeReader {
        fn read(&mut self, samples: &[f32]) -> Result<String> {
            match &self.0 {
                Some(Ok(text)) => Ok(text.clone()),
                Some(Err(e)) => bail!("{e}"),
                None if samples.first() == Some(&1.0) => Ok("enhanced".into()),
                None => Ok("plain".into()),
            }
        }
    }

    struct FakeEnhancer {
        fails: bool,
    }

    impl Enhancer for FakeEnhancer {
        fn enhance(&mut self, samples: &[f32], _: u32) -> Result<Vec<f32>> {
            if self.fails {
                bail!("RE-USE failed");
            }
            Ok(vec![1.0; samples.len()])
        }
    }

    struct FakeSpeakers;

    impl SpeakerFinder for FakeSpeakers {
        fn hear(&self, _: &AudioBlock) -> Result<()> {
            Ok(())
        }

        fn speaker(&self, _: Span) -> Option<SpeakerLabel> {
            SpeakerLabel::from_index(0)
        }
    }

    fn utterance(input: Input) -> Utterance {
        Utterance {
            input,
            samples: vec![0.0; 1600],
            rate: SPEECH_RATE,
            span: Span {
                start: Millis(1000),
                end: Millis(1100),
            },
        }
    }

    fn pipeline(reads: Option<Result<&str, &str>>, enhancer_fails: bool) -> Pipeline {
        Pipeline::new(
            Box::new(FakeReader(
                reads.map(|r| r.map(str::to_owned).map_err(str::to_owned)),
            )),
            Some(Box::new(FakeEnhancer {
                fails: enhancer_fails,
            })),
            Some(Arc::new(FakeSpeakers)),
            None,
        )
    }

    fn speech(from: Input, text: &str, speaker: bool) -> Event {
        Event::Speech(Speech {
            number: 0,
            from,
            text: text.into(),
            start_ms: Millis(1000),
            end_ms: Millis(1100),
            speaker: speaker.then(|| SpeakerLabel::from_index(0).unwrap()),
        })
    }

    #[test]
    fn an_utterance_becomes_its_lines() {
        let rows: Vec<(&str, Input, Option<Result<&str, &str>>, bool, Vec<Event>)> = vec![
            (
                "me is enhanced, and has no speaker",
                Input::Me,
                None,
                false,
                vec![speech(Input::Me, "enhanced", false)],
            ),
            (
                "others is read as heard, with its speaker",
                Input::Others,
                None,
                false,
                vec![speech(Input::Others, "plain", true)],
            ),
            (
                "no words, no line",
                Input::Others,
                Some(Ok("")),
                false,
                vec![],
            ),
            (
                "a failed enhancement is told, and the sound as heard is read",
                Input::Me,
                None,
                true,
                vec![
                    Event::error("me: RE-USE failed"),
                    speech(Input::Me, "plain", false),
                ],
            ),
            (
                "a failed reading is told, and gives no line",
                Input::Others,
                Some(Err("whisper failed")),
                false,
                vec![Event::error("others: whisper failed")],
            ),
        ];
        for (name, input, reads, enhancer_fails, expected) in rows {
            let got = pipeline(reads, enhancer_fails)
                .read(&utterance(input))
                .events;
            assert_eq!(got, expected, "{name}");
        }
    }

    #[test]
    fn only_an_utterance_with_words_takes_a_number() {
        struct Words(Vec<&'static str>);
        impl SpeechToText for Words {
            fn read(&mut self, _: &[f32]) -> Result<String> {
                Ok(self.0.remove(0).into())
            }
        }
        let mut p = Pipeline::new(Box::new(Words(vec!["a", "", "b"])), None, None, None);
        let numbers: Vec<u64> = (0..3)
            .flat_map(|_| p.read(&utterance(Input::Me)).events)
            .filter_map(|e| match e {
                Event::Speech(s) => Some(s.number),
                _ => None,
            })
            .collect();
        assert_eq!(numbers, [0, 1]);
    }
}
