//! WAV files: a recording read, and the utterances kept with `--keep-audio`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};

use crate::app::log;
use crate::app::ports::{AudioKeeper, SPEECH_RATE, Stage};
use crate::domain::utterance::Utterance;
use crate::dsp::to_mono;

/// # Errors
/// When the file cannot be written.
pub fn write_wav(path: &Path, samples: &[f32], rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .with_context(|| format!("writing {}", path.display()))?;
    for s in samples {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "clamped to the range of i16 first"
        )]
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
}

/// 16-bit integer or 32-bit float PCM, any number of channels, as mono.
///
/// # Errors
/// When the file cannot be read, or is of another format.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut r =
        hound::WavReader::open(path).with_context(|| format!("reading {}", path.display()))?;
    let spec = r.spec();
    let interleaved: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => r
            .samples::<i16>()
            .map(|s| s.map(|s| f32::from(s) / 32768.0))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Float, 32) => r.samples::<f32>().collect::<Result<_, _>>()?,
        (format, bits) => bail!(
            "{}: {format:?} with {bits} bits is not read",
            path.display()
        ),
    };
    Ok((
        to_mono(&interleaved, usize::from(spec.channels)),
        spec.sample_rate,
    ))
}

/// A folder of the utterances, `<start_ms>-<input>.wav`, and `-enhanced.wav` after RE-USE beside
/// it. Nothing removes them.
pub struct WavFolder(PathBuf);

impl WavFolder {
    #[must_use]
    pub fn new(dir: PathBuf) -> Self {
        Self(dir)
    }
}

impl AudioKeeper for WavFolder {
    fn keep(&self, utterance: &Utterance, stage: Stage, samples: &[f32]) {
        let suffix = match stage {
            Stage::Heard => "",
            Stage::Enhanced => "-enhanced",
        };
        let path = self.0.join(format!(
            "{}-{}{suffix}.wav",
            utterance.span.start.0, utterance.input
        ));
        let written = fs::create_dir_all(&self.0)
            .map_err(anyhow::Error::from)
            .and_then(|()| write_wav(&path, samples, SPEECH_RATE));
        if let Err(e) = written {
            log(&format!("{}: {e:#}", path.display()));
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::cast_precision_loss, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn a_wav_reads_back() {
        let path = std::env::temp_dir().join(format!("xlisten-test-{}.wav", std::process::id()));
        let tone: Vec<f32> = (0..4800).map(|i| 0.3 * (i as f32 / 10.0).sin()).collect();
        write_wav(&path, &tone, 48_000).unwrap();
        let (back, rate) = read_wav(&path).unwrap();
        assert_eq!(rate, 48_000);
        assert!(back.iter().zip(&tone).all(|(a, b)| (a - b).abs() < 1e-4));
    }
}
