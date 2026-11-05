//! Cuts a stream of samples into utterances by their loudness.
//!
//! WHY loudness and not a model of voice activity: it is enough to find where somebody stops, which
//! is all whisper needs to be handed, and Silero then tells whisper whether there is a voice in it
//! at all (`crate::whisper`). Music or typing loud enough is taken for speech here.

const FRAME_MS: i64 = 30;
/// A frame is speech where it is this much louder than the floor of the noise, and above `ABSOLUTE_DB`.
const OVER_FLOOR_DB: f64 = 10.0;
const ABSOLUTE_DB: f64 = -55.0;
/// Frames of speech in a row that open an utterance, and the time before them it begins with.
const OPEN_FRAMES: usize = 3;
const PRE_ROLL_MS: i64 = 300;
/// Quiet that ends an utterance; the latency of a line is at least this.
pub const CLOSE_MS: i64 = 600;
/// Where the sound never gets quiet (a stream with music under the voices), an utterance is cut at
/// this length, at its quietest frame of the last `CUT_WITHIN_MS`, and the rest opens the next one.
/// WHY 10 s: at 20 s, a line of `others` came only every 20 s through a stream with music
/// (2026-11-05); and RE-USE was trained on less than 10 s (its model card).
pub const LONGEST_MS: i64 = 10_000;
const CUT_WITHIN_MS: i64 = 3_000;
const SHORTEST_MS: f64 = 400.0;

#[derive(Debug)]
pub struct Utterance {
    pub samples: Vec<f32>,
    pub rate: u32,
    pub start_ms: i64,
    /// The end of the last frame of speech, before the quiet that closed it.
    pub end_ms: i64,
}

fn db(frame: &[f32]) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "a frame is short")]
    let mean = frame
        .iter()
        .map(|s| f64::from(*s) * f64::from(*s))
        .sum::<f64>()
        / frame.len().max(1) as f64;
    20.0 * mean.sqrt().max(1e-10).log10()
}

struct Open {
    frames: Vec<Vec<f32>>,
    dbs: Vec<f64>,
    start_ms: f64,
    first_speech_ms: f64,
    last_speech_end_ms: f64,
    quiet_ms: i64,
}

pub struct Segmenter {
    pub rate: u32,
    frame: usize,
    rest: Vec<f32>,
    rest_start_ms: f64,
    floor_db: f64,
    pre: Vec<(Vec<f32>, f64)>,
    run: usize,
    open: Option<Open>,
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "times in ms fit in f64 and back"
)]
impl Segmenter {
    #[must_use]
    pub fn new(rate: u32) -> Self {
        let frame = usize::try_from(i64::from(rate) * FRAME_MS / 1000).unwrap_or(1);
        Self {
            rate,
            frame,
            rest: Vec::new(),
            rest_start_ms: 0.0,
            floor_db: -60.0,
            pre: Vec::new(),
            run: 0,
            open: None,
        }
    }

    /// `end_ms`: the wall clock of the last sample of `samples`.
    pub fn push(&mut self, samples: &[f32], end_ms: f64) -> Vec<Utterance> {
        let start_ms = end_ms - samples.len() as f64 * 1000.0 / f64::from(self.rate);
        if self.rest.is_empty() {
            self.rest_start_ms = start_ms;
        }
        let mut buf = std::mem::take(&mut self.rest);
        buf.extend_from_slice(samples);
        let mut t = self.rest_start_ms;
        let mut out = Vec::new();
        let mut chunks = buf.chunks_exact(self.frame);
        for frame in chunks.by_ref() {
            if let Some(u) = self.frame_in(frame, t) {
                out.push(u);
            }
            t += FRAME_MS as f64;
        }
        self.rest = chunks.remainder().to_vec();
        self.rest_start_ms = t;
        out
    }

    fn frame_in(&mut self, frame: &[f32], t: f64) -> Option<Utterance> {
        let level = db(frame);
        let speech = level > (self.floor_db + OVER_FLOOR_DB).max(ABSOLUTE_DB);
        if !speech {
            // The floor follows quiet quickly downwards and slowly upwards.
            let rate = if level < self.floor_db { 0.2 } else { 0.01 };
            self.floor_db += rate * (level - self.floor_db);
        }
        let frame_end = t + FRAME_MS as f64;
        let Some(open) = self.open.as_mut() else {
            self.pre.push((frame.to_vec(), level));
            let keep = usize::try_from(PRE_ROLL_MS / FRAME_MS).unwrap_or(0) + OPEN_FRAMES;
            if self.pre.len() > keep {
                self.pre.drain(..self.pre.len() - keep);
            }
            self.run = if speech { self.run + 1 } else { 0 };
            if self.run >= OPEN_FRAMES {
                let pre = std::mem::take(&mut self.pre);
                self.open = Some(Open {
                    start_ms: frame_end - pre.len() as f64 * FRAME_MS as f64,
                    first_speech_ms: frame_end - OPEN_FRAMES as f64 * FRAME_MS as f64,
                    last_speech_end_ms: frame_end,
                    quiet_ms: 0,
                    dbs: pre.iter().map(|(_, d)| *d).collect(),
                    frames: pre.into_iter().map(|(f, _)| f).collect(),
                });
            }
            return None;
        };
        open.frames.push(frame.to_vec());
        open.dbs.push(level);
        if speech {
            open.last_speech_end_ms = frame_end;
            open.quiet_ms = 0;
        } else {
            open.quiet_ms += FRAME_MS;
        }
        let length_ms = open.frames.len() as f64 * FRAME_MS as f64;
        if open.quiet_ms >= CLOSE_MS {
            return self.close();
        }
        if length_ms >= LONGEST_MS as f64 {
            return Some(self.cut());
        }
        None
    }

    fn cut(&mut self) -> Utterance {
        let open = self
            .open
            .as_mut()
            .unwrap_or_else(|| unreachable!("cut only an open utterance"));
        let within = usize::try_from(CUT_WITHIN_MS / FRAME_MS)
            .unwrap_or(1)
            .min(open.dbs.len());
        let from = open.dbs.len() - within;
        let k = open
            .dbs
            .get(from..)
            .unwrap_or_default()
            .iter()
            .enumerate()
            .fold(
                (0, f64::INFINITY),
                |best, (i, d)| if *d < best.1 { (i, *d) } else { best },
            )
            .0
            + from;
        let cut_ms = open.start_ms + k as f64 * FRAME_MS as f64;
        let rest_frames = open.frames.split_off(k);
        let rest_dbs = open.dbs.split_off(k);
        let samples = std::mem::replace(&mut open.frames, rest_frames).concat();
        open.dbs = rest_dbs;
        let u = Utterance {
            samples,
            rate: self.rate,
            start_ms: open.start_ms.round() as i64,
            end_ms: cut_ms.round() as i64,
        };
        open.start_ms = cut_ms;
        open.first_speech_ms = cut_ms;
        u
    }

    fn close(&mut self) -> Option<Utterance> {
        let open = self.open.take()?;
        self.run = 0;
        if open.last_speech_end_ms - open.first_speech_ms < SHORTEST_MS {
            return None;
        }
        Some(Utterance {
            samples: open.frames.concat(),
            rate: self.rate,
            start_ms: open.start_ms.round() as i64,
            end_ms: open.last_speech_end_ms.round() as i64,
        })
    }

    pub fn finish(&mut self) -> Option<Utterance> {
        self.close()
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
        let mut seg = Segmenter::new(RATE);
        let mut out = Vec::new();
        for (i, block) in samples.chunks(1024).enumerate() {
            let end = (i * 1024 + block.len()) as f64 * 1000.0 / f64::from(RATE);
            out.extend(seg.push(block, end));
        }
        out.extend(seg.finish());
        out.into_iter().map(|u| (u.start_ms, u.end_ms)).collect()
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
