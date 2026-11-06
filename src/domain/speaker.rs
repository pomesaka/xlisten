//! Who among `others` speaks. A label holds for one run: an `A` of one meeting is not the `A` of
//! another.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Serialize, Serializer};

/// `A`, `B`, ... in the order the voices were first heard; a label, not a name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SpeakerLabel(u8);

impl SpeakerLabel {
    /// The label of the diarization's speaker `index` (0 is `A`); none past `Z`.
    #[must_use]
    pub fn from_index(index: i32) -> Option<Self> {
        u8::try_from(index).ok().filter(|n| *n < 26).map(Self)
    }
}

impl fmt::Display for SpeakerLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", char::from(b'A' + self.0))
    }
}

impl Serialize for SpeakerLabel {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

/// A stretch of one voice, in seconds from the start of the sound the diarization heard.
#[derive(Clone, Copy, Debug)]
pub struct SpeakerTurn {
    pub start_s: f64,
    pub end_s: f64,
    pub speaker: SpeakerLabel,
}

/// Whoever speaks the longest between `start_s` and `end_s`; none where nobody does.
#[must_use]
pub fn most_heard(turns: &[SpeakerTurn], start_s: f64, end_s: f64) -> Option<SpeakerLabel> {
    let mut heard: BTreeMap<SpeakerLabel, f64> = BTreeMap::new();
    for t in turns {
        let overlap = t.end_s.min(end_s) - t.start_s.max(start_s);
        if overlap > 0.0 {
            *heard.entry(t.speaker).or_default() += overlap;
        }
    }
    heard
        .into_iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(speaker, _)| speaker)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_label_is_a_letter_from_a_to_z() {
        let rows: &[(i32, Option<&str>)] = &[
            (0, Some("A")),
            (1, Some("B")),
            (25, Some("Z")),
            (26, None),
            (-1, None),
        ];
        for (index, expected) in rows {
            assert_eq!(
                SpeakerLabel::from_index(*index)
                    .map(|l| l.to_string())
                    .as_deref(),
                *expected,
                "{index}"
            );
        }
    }

    #[test]
    fn most_heard_is_whoever_speaks_the_longest_within() {
        let a = SpeakerLabel::from_index(0);
        let b = SpeakerLabel::from_index(1);
        let turns: Vec<SpeakerTurn> = [(0.0, 2.0, a), (1.5, 5.0, b)]
            .into_iter()
            .filter_map(|(start_s, end_s, speaker)| {
                Some(SpeakerTurn {
                    start_s,
                    end_s,
                    speaker: speaker?,
                })
            })
            .collect();
        let rows: &[(&str, f64, f64, Option<&str>)] = &[
            ("one speaker alone", 0.0, 1.0, Some("A")),
            ("the one of the longer part", 1.0, 4.0, Some("B")),
            ("nobody", 6.0, 7.0, None),
        ];
        for (name, start, end, expected) in rows {
            assert_eq!(
                most_heard(&turns, *start, *end)
                    .map(|l| l.to_string())
                    .as_deref(),
                *expected,
                "{name}"
            );
        }
    }
}
