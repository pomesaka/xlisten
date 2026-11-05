//! Samples as mono `f32` in [-1, 1].

use std::path::Path;

use anyhow::{Context as _, Result, bail};

pub const WHISPER_RATE: u32 = 16_000;

/// The mean of the channels of interleaved samples.
#[must_use]
pub fn to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    #[expect(clippy::cast_precision_loss, reason = "a count of channels is small")]
    let n = channels as f32;
    interleaved
        .chunks_exact(channels)
        .map(|f| f.iter().sum::<f32>() / n)
        .collect()
}

/// Resamples a whole utterance: a windowed-sinc low-pass below the new Nyquist, then linear
/// interpolation. WHY not a polyphase crate: an utterance is resampled once, whole, so no state is
/// carried between calls, and it adds no dependency.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "lengths of audio fit in f64 and back"
)]
pub fn resample(samples: &[f32], rate: u32, to: u32) -> Vec<f32> {
    if rate == to || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = f64::from(rate) / f64::from(to);
    let filtered: Vec<f32> = if to < rate {
        let cutoff = 0.45 / ratio;
        let taps: Vec<f64> = (-32_i32..=32)
            .map(|k| {
                let k = f64::from(k);
                let x = 2.0 * cutoff * k;
                let sinc = if x == 0.0 {
                    1.0
                } else {
                    (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
                };
                let hamming = 0.54 + 0.46 * (std::f64::consts::PI * k / 32.0).cos();
                2.0 * cutoff * sinc * hamming
            })
            .collect();
        let sum: f64 = taps.iter().sum();
        let half = taps.len() / 2;
        (0..samples.len())
            .map(|i| {
                let mut acc = 0.0;
                for (j, t) in taps.iter().enumerate() {
                    if let Some(s) = (i + j).checked_sub(half).and_then(|k| samples.get(k)) {
                        acc += f64::from(*s) * t;
                    }
                }
                (acc / sum) as f32
            })
            .collect()
    } else {
        samples.to_vec()
    };
    let n = (samples.len() as f64 / ratio).round() as usize;
    let last = filtered.len() - 1;
    (0..n)
        .map(|i| {
            let pos = i as f64 * ratio;
            let k = (pos.floor() as usize).min(last);
            let frac = (pos - k as f64) as f32;
            let a = filtered.get(k).copied().unwrap_or(0.0);
            let b = filtered.get((k + 1).min(last)).copied().unwrap_or(a);
            a + (b - a) * frac
        })
        .collect()
}

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

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    reason = "tests"
)]
mod tests {
    use super::*;

    #[test]
    fn resample_keeps_the_length_and_a_low_tone() {
        let tone: Vec<f32> = (0..48_000)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 48_000.0).sin())
            .collect();
        let out = resample(&tone, 48_000, 16_000);
        assert_eq!(out.len(), 16_000);
        let peak = out[100..out.len() - 100]
            .iter()
            .fold(0.0_f32, |m, s| m.max(s.abs()));
        assert!((peak - 0.3).abs() < 0.01, "{peak}");
    }

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
