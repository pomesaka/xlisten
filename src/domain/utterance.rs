//! Cuts the sound of an input into utterances by its loudness.
//!
//! WHY loudness and not a model of voice activity: it is enough to find where somebody stops, which
//! is all whisper needs to be handed; Silero, before whisper, tells whether there is a voice in it
//! at all. Music or typing loud enough is taken for speech here.

use std::collections::VecDeque;

use crate::domain::input::Input;
use crate::domain::time::{Millis, Span, samples_ms};

const FRAME_MS: f64 = 30.0;
/// A frame is speech where it is this much louder than the floor of the noise, and above `ABSOLUTE_DB`.
const OVER_FLOOR_DB: f64 = 10.0;
const ABSOLUTE_DB: f64 = -55.0;
/// Frames of speech in a row that open an utterance, and the frames before them it begins with
/// (300 ms).
const OPEN_FRAMES: usize = 3;
const PRE_ROLL_FRAMES: usize = 10;
/// Quiet that ends an utterance; the latency of a line is at least this.
const CLOSE_MS: f64 = 600.0;
/// Where the sound never gets quiet (a stream with music under the voices), an utterance is cut at
/// this length, at its quietest frame of the last `CUT_WITHIN_FRAMES` (3 s), and the rest opens
/// the next one.
/// WHY 10 s: at 20 s, a line of `others` came only every 20 s through a stream with music
/// (2026-11-05); and RE-USE was trained on less than 10 s (its model card).
const LONGEST_MS: f64 = 10_000.0;
const CUT_WITHIN_FRAMES: usize = 100;
/// Speech shorter than this is a click, not an utterance.
const SHORTEST_MS: f64 = 400.0;

/// Mono samples as an input hands them, and when the last of them was heard.
pub struct AudioBlock {
    pub samples: Vec<f32>,
    pub end: Millis,
}

#[derive(Debug)]
pub struct Utterance {
    pub input: Input,
    pub samples: Vec<f32>,
    pub rate: u32,
    /// From the first frame kept before the speech, to the end of its last frame of speech.
    pub span: Span,
}

impl Utterance {
    #[must_use]
    pub fn seconds(&self) -> f64 {
        samples_ms(self.samples.len(), self.rate) / 1000.0
    }
}

struct Frame {
    samples: Vec<f32>,
    db: f64,
}

impl Frame {
    fn new(samples: &[f32]) -> Self {
        #[expect(clippy::cast_precision_loss, reason = "a frame is short")]
        let mean = samples
            .iter()
            .map(|s| f64::from(*s) * f64::from(*s))
            .sum::<f64>()
            / samples.len().max(1) as f64;
        Self {
            samples: samples.to_vec(),
            db: 20.0 * mean.sqrt().max(1e-10).log10(),
        }
    }
}

/// The loudness of the room with nobody speaking.
struct NoiseFloor(f64);

impl NoiseFloor {
    /// Whether a frame this loud is speech; a frame that is not moves the floor towards it, quickly
    /// downwards and slowly upwards.
    fn hears_speech(&mut self, db: f64) -> bool {
        let speech = db > (self.0 + OVER_FLOOR_DB).max(ABSOLUTE_DB);
        if !speech {
            let rate = if db < self.0 { 0.2 } else { 0.01 };
            self.0 += rate * (db - self.0);
        }
        speech
    }
}

/// `FRAME_MS * n`.
#[expect(clippy::cast_precision_loss, reason = "a count of frames is small")]
fn frames_ms(n: usize) -> f64 {
    n as f64 * FRAME_MS
}

enum State {
    /// Nobody speaks: the last frames, kept to begin an utterance with, and how many of them in a
    /// row were speech.
    Quiet {
        pre: VecDeque<Frame>,
        run: usize,
    },
    Speaking(Open),
}

struct Open {
    frames: Vec<Frame>,
    start_ms: f64,
    first_speech_ms: f64,
    last_speech_end_ms: f64,
    quiet_ms: f64,
}

impl Open {
    /// The utterance up to the quietest frame of the last `CUT_WITHIN_FRAMES`; the rest stays open.
    fn cut(&mut self, input: Input, rate: u32) -> Utterance {
        let from = self.frames.len().saturating_sub(CUT_WITHIN_FRAMES);
        let quietest = self
            .frames
            .iter()
            .enumerate()
            .skip(from)
            .min_by(|a, b| a.1.db.total_cmp(&b.1.db))
            .map_or(from, |(i, _)| i);
        let cut_ms = self.start_ms + frames_ms(quietest);
        let rest = self.frames.split_off(quietest);
        let head = std::mem::replace(&mut self.frames, rest);
        let utterance = Utterance {
            input,
            samples: head.into_iter().flat_map(|f| f.samples).collect(),
            rate,
            span: Span {
                start: Millis::round(self.start_ms),
                end: Millis::round(cut_ms),
            },
        };
        self.start_ms = cut_ms;
        self.first_speech_ms = cut_ms;
        utterance
    }
}

pub struct Segmenter {
    input: Input,
    rate: u32,
    frame_len: usize,
    /// The samples short of a frame, and when the first of them was heard.
    rest: Vec<f32>,
    rest_start_ms: f64,
    floor: NoiseFloor,
    state: State,
}

impl Segmenter {
    #[must_use]
    pub fn new(input: Input, rate: u32) -> Self {
        Self {
            input,
            rate,
            // FRAME_MS of samples.
            frame_len: usize::try_from(u64::from(rate) * 30 / 1000).unwrap_or(1),
            rest: Vec::new(),
            rest_start_ms: 0.0,
            floor: NoiseFloor(-60.0),
            state: State::Quiet {
                pre: VecDeque::new(),
                run: 0,
            },
        }
    }

    /// The utterances that ended within `block`.
    pub fn push(&mut self, block: &AudioBlock) -> Vec<Utterance> {
        if self.rest.is_empty() {
            self.rest_start_ms = block.end.as_f64() - samples_ms(block.samples.len(), self.rate);
        }
        let mut samples = std::mem::take(&mut self.rest);
        samples.extend_from_slice(&block.samples);
        let mut t = self.rest_start_ms;
        let mut ended = Vec::new();
        let mut frames = samples.chunks_exact(self.frame_len);
        for frame in frames.by_ref() {
            ended.extend(self.frame(Frame::new(frame), t));
            t += FRAME_MS;
        }
        self.rest = frames.remainder().to_vec();
        self.rest_start_ms = t;
        ended
    }

    /// The utterance still open when the input ends, if it is long enough.
    #[must_use]
    pub fn finish(mut self) -> Option<Utterance> {
        self.close()
    }

    fn frame(&mut self, frame: Frame, t: f64) -> Option<Utterance> {
        let speech = self.floor.hears_speech(frame.db);
        let frame_end = t + FRAME_MS;
        match &mut self.state {
            State::Quiet { pre, run } => {
                pre.push_back(frame);
                if pre.len() > PRE_ROLL_FRAMES + OPEN_FRAMES {
                    pre.pop_front();
                }
                *run = if speech { *run + 1 } else { 0 };
                if *run >= OPEN_FRAMES {
                    let frames = Vec::from(std::mem::take(pre));
                    self.state = State::Speaking(Open {
                        start_ms: frame_end - frames_ms(frames.len()),
                        first_speech_ms: frame_end - frames_ms(OPEN_FRAMES),
                        last_speech_end_ms: frame_end,
                        quiet_ms: 0.0,
                        frames,
                    });
                }
                None
            }
            State::Speaking(open) => {
                open.frames.push(frame);
                if speech {
                    open.last_speech_end_ms = frame_end;
                    open.quiet_ms = 0.0;
                } else {
                    open.quiet_ms += FRAME_MS;
                }
                if open.quiet_ms >= CLOSE_MS {
                    return self.close();
                }
                if frames_ms(open.frames.len()) >= LONGEST_MS {
                    return Some(open.cut(self.input, self.rate));
                }
                None
            }
        }
    }

    fn close(&mut self) -> Option<Utterance> {
        let quiet = State::Quiet {
            pre: VecDeque::new(),
            run: 0,
        };
        let State::Speaking(open) = std::mem::replace(&mut self.state, quiet) else {
            return None;
        };
        (open.last_speech_end_ms - open.first_speech_ms >= SHORTEST_MS).then(|| Utterance {
            input: self.input,
            samples: open.frames.into_iter().flat_map(|f| f.samples).collect(),
            rate: self.rate,
            span: Span {
                start: Millis::round(open.start_ms),
                end: Millis::round(open.last_speech_end_ms),
            },
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::type_complexity,
    reason = "tests"
)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    fn tone(seconds: f64) -> Vec<f32> {
        (0..(seconds * f64::from(RATE)) as usize)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / RATE as f32).sin())
            .collect()
    }

    /// Quiet with a little noise, as a room has.
    fn quiet(seconds: f64) -> Vec<f32> {
        let mut x: u32 = 1;
        (0..(seconds * f64::from(RATE)) as usize)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (f64::from(x) / f64::from(u32::MAX) - 0.5) as f32 * 2e-4
            })
            .collect()
    }

    fn run(parts: &[Vec<f32>]) -> Vec<(i64, i64)> {
        let samples = parts.concat();
        let mut seg = Segmenter::new(Input::Me, RATE);
        let mut out = Vec::new();
        for (i, block) in samples.chunks(1024).enumerate() {
            out.extend(seg.push(&AudioBlock {
                samples: block.to_vec(),
                end: Millis::round(samples_ms(i * 1024 + block.len(), RATE)),
            }));
        }
        out.extend(seg.finish());
        out.into_iter()
            .map(|u| (u.span.start.0, u.span.end.0))
            .collect()
    }

    #[test]
    fn segmenter() {
        // The start may be up to the pre-roll (300 ms) and a frame early. The end is within a frame,
        // or, where it is cut, anywhere in the dip of 200 ms where it is quietest.
        let rows: Vec<(&str, Vec<Vec<f32>>, Vec<(i64, i64)>, i64)> = vec![
            (
                "one utterance, from its first frame to its last",
                vec![quiet(2.0), tone(1.5), quiet(2.0)],
                vec![(2000, 3500)],
                30,
            ),
            (
                "a pause shorter than the close is inside",
                vec![quiet(2.0), tone(1.0), quiet(0.3), tone(1.0), quiet(2.0)],
                vec![(2000, 4300)],
                30,
            ),
            (
                "a pause of the close or more ends it",
                vec![quiet(2.0), tone(1.0), quiet(1.0), tone(1.0), quiet(2.0)],
                vec![(2000, 3000), (4000, 5000)],
                30,
            ),
            (
                "a click is no utterance",
                vec![quiet(2.0), tone(0.2), quiet(2.0)],
                vec![],
                30,
            ),
            (
                "the end of the stream ends it",
                vec![quiet(2.0), tone(1.5)],
                vec![(2000, 3500)],
                30,
            ),
            (
                "one that never gets quiet is cut at its quietest, and the rest goes on",
                vec![quiet(2.0), tone(8.5), quiet(0.2), tone(5.5), quiet(2.0)],
                vec![(2000, 10500), (10500, 16200)],
                200,
            ),
        ];
        for (name, parts, expected, within) in rows {
            let got = run(&parts);
            assert_eq!(got.len(), expected.len(), "{name}: {got:?}");
            for ((gs, ge), (es, ee)) in got.iter().zip(&expected) {
                assert!(
                    es - 330 <= *gs && *gs <= es + within - 30,
                    "{name}: start {gs}"
                );
                assert!((ge - ee).abs() <= within, "{name}: end {ge}");
            }
        }
    }
}
