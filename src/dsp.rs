//! Samples as mono `f32` in [-1, 1]: of no layer, used by `app` and `infra` alike.

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
        low_pass(samples, 0.45 / ratio)
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

/// A windowed-sinc (Hamming, 65 taps) low-pass at `cutoff` of the rate.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a filtered sample is back in f32"
)]
fn low_pass(samples: &[f32], cutoff: f64) -> Vec<f32> {
    use std::f64::consts::PI;
    let taps: Vec<f64> = (-32_i32..=32)
        .map(|k| {
            let k = f64::from(k);
            let x = 2.0 * cutoff * k;
            let sinc = if x == 0.0 {
                1.0
            } else {
                (PI * x).sin() / (PI * x)
            };
            let hamming = 0.54 + 0.46 * (PI * k / 32.0).cos();
            2.0 * cutoff * sinc * hamming
        })
        .collect();
    let sum: f64 = taps.iter().sum();
    let half = taps.len() / 2;
    (0..samples.len())
        .map(|i| {
            let acc: f64 = taps
                .iter()
                .enumerate()
                .filter_map(|(j, t)| {
                    let s = samples.get((i + j).checked_sub(half)?)?;
                    Some(f64::from(*s) * t)
                })
                .sum();
            (acc / sum) as f32
        })
        .collect()
}

#[cfg(test)]
#[expect(
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
}
