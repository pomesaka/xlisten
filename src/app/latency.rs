//! How long after the end of each utterance its line was written (`at - end_ms`).

use std::collections::BTreeMap;
use std::fmt;

use crate::domain::input::Input;

pub struct Latency {
    input: Input,
    lines: usize,
    median_ms: i64,
    max_ms: i64,
}

/// One for each input, from the latencies of its lines in ms.
#[must_use]
pub fn latencies(lines: impl IntoIterator<Item = (Input, i64)>) -> Vec<Latency> {
    let mut by: BTreeMap<Input, Vec<i64>> = BTreeMap::new();
    for (input, ms) in lines {
        by.entry(input).or_default().push(ms);
    }
    by.into_iter()
        .filter_map(|(input, mut ms)| {
            ms.sort_unstable();
            Some(Latency {
                input,
                lines: ms.len(),
                median_ms: *ms.get(ms.len() / 2)?,
                max_ms: *ms.last()?,
            })
        })
        .collect()
}

impl fmt::Display for Latency {
    #[expect(clippy::cast_precision_loss, reason = "ms")]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} lines, median {:.2}s, max {:.2}s",
            self.input,
            self.lines,
            self.median_ms as f64 / 1000.0,
            self.max_ms as f64 / 1000.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_median_and_a_max_for_each_input() {
        let got: Vec<String> = latencies([
            (Input::Others, 1600),
            (Input::Me, 2150),
            (Input::Others, 900),
            (Input::Others, 2420),
        ])
        .iter()
        .map(ToString::to_string)
        .collect();
        assert_eq!(
            got,
            [
                "me: 1 lines, median 2.15s, max 2.15s",
                "others: 3 lines, median 1.60s, max 2.42s"
            ]
        );
    }
}
