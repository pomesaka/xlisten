//! Times in ms: on the wall clock while listening, from the start of the file for a recording.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Millis(pub i64);

impl Millis {
    /// The wall clock, since the Unix epoch.
    #[must_use]
    pub fn now() -> Self {
        Self(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX)),
        )
    }

    /// The nearest whole ms to `ms`.
    #[must_use]
    #[expect(clippy::cast_possible_truncation, reason = "a time in ms fits in i64")]
    pub fn round(ms: f64) -> Self {
        Self(ms.round() as i64)
    }

    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "ms since the epoch fit in f64")]
    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }

    /// Seconds from `earlier` to `self`.
    #[must_use]
    pub fn seconds_since(self, earlier: Self) -> f64 {
        (self.as_f64() - earlier.as_f64()) / 1000.0
    }
}

/// Where an utterance starts, and where its last speech ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: Millis,
    pub end: Millis,
}

/// How long `samples` samples last at `rate`, in ms.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "a count of samples of audio fits"
)]
pub fn samples_ms(samples: usize, rate: u32) -> f64 {
    samples as f64 * 1000.0 / f64::from(rate)
}
