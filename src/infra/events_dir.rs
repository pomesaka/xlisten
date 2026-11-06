//! The files of `events/`, in the form of `xmoq-agent` of `XmoQ`.
//!
//! The form is `XmoQ`'s `fixtures/agent-lines/README.md` (read on 2026-11-05): one JSON object a line,
//! keys in the order of their names, `at` the writer's clock in ms; a file of `events/` is named
//! after the clock in 13 digits at the least, or one more than the newest there, and is moved to
//! its name only when it is whole.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use anyhow::{Context as _, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::app::ports::EventLog;
use crate::domain::event::Event;
use crate::domain::input::Input;
use crate::domain::time::Millis;

/// One line of JSON, its keys in the order of their names, with `at` where it is given.
///
/// WHY the order: `serde_json`'s `Map` is a `BTreeMap` without its feature `preserve_order`, which
/// nothing here turns on. WHY DEL and the C1 controls on top of what `serde_json` escapes
/// (U+0000..U+001F): some terminals act on them as well, and an agent may print the lines in one.
#[must_use]
pub fn encode(event: &Event, at: Option<Millis>) -> String {
    let mut value = serde_json::to_value(event)
        .unwrap_or_else(|e| serde_json::json!({ "type": "error", "message": e.to_string() }));
    if let (Some(at), Value::Object(fields)) = (at, &mut value) {
        fields.insert("at".into(), at.0.into());
    }
    value.to_string().chars().fold(String::new(), |mut out, c| {
        if ('\u{7f}'..='\u{9f}').contains(&c) {
            let _ = write!(out, "\\u{:04x}", u32::from(c));
        } else {
            out.push(c);
        }
        out
    })
}

/// The number of a file of `events/` (`<13 digits or more>.jsonl`); none for any other name, which
/// is none of the reader's.
fn event_number(name: &str) -> Option<i64> {
    let digits = name.strip_suffix(".jsonl")?;
    (digits.len() >= 13 && digits.bytes().all(|b| b.is_ascii_digit()))
        .then(|| digits.parse().ok())
        .flatten()
}

/// The name of the next file of `events/`, after those in `names`.
#[must_use]
pub fn next_event_file<S: AsRef<str>>(names: &[S], clock: Millis) -> String {
    let newest = names
        .iter()
        .filter_map(|n| event_number(n.as_ref()))
        .max()
        .unwrap_or(-1);
    format!("{:013}.jsonl", clock.0.max(newest + 1))
}

/// Gathers lines and writes them as one file of `events/` a second ([`EventLog::flush`]).
pub struct EventsDir {
    events: PathBuf,
    pending: Mutex<Vec<String>>,
    /// Whether each line is printed on stdout too.
    echo: bool,
}

impl EventsDir {
    /// # Errors
    /// When `out/events` cannot be made.
    pub fn new(out: &Path, echo: bool) -> Result<Self> {
        let events = out.join("events");
        fs::create_dir_all(&events).with_context(|| format!("making {}", events.display()))?;
        Ok(Self {
            events,
            pending: Mutex::new(Vec::new()),
            echo,
        })
    }
}

/// The files of `out/events`, in the order they were written.
fn event_files(out: &Path) -> Result<Vec<PathBuf>> {
    let events = out.join("events");
    let mut files: Vec<PathBuf> = fs::read_dir(&events)
        .with_context(|| format!("reading {}", events.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .and_then(event_number)
                .is_some()
        })
        .collect();
    files.sort();
    Ok(files)
}

/// For each `speech` line of `out/events`, its input and how long after the end of its utterance
/// it was written, in ms.
///
/// # Errors
/// When a file cannot be read, or a `speech` line lacks a field.
pub fn speech_latencies(out: &Path) -> Result<Vec<(Input, i64)>> {
    #[derive(Deserialize)]
    struct Written {
        #[serde(rename = "type")]
        kind: String,
    }
    #[derive(Deserialize)]
    struct SpeechWritten {
        from: Input,
        end_ms: i64,
        at: i64,
    }
    let mut latencies = Vec::new();
    for f in event_files(out)? {
        for raw in fs::read_to_string(&f)?.lines() {
            let context = || format!("{}: {raw}", f.display());
            if serde_json::from_str::<Written>(raw)
                .with_context(context)?
                .kind
                != "speech"
            {
                continue;
            }
            let s: SpeechWritten = serde_json::from_str(raw).with_context(context)?;
            latencies.push((s.from, s.at - s.end_ms));
        }
    }
    Ok(latencies)
}

impl EventLog for EventsDir {
    fn write(&self, event: Event) {
        let line = encode(&event, Some(Millis::now()));
        if self.echo {
            println!("{line}");
        }
        self.pending
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(line);
    }

    fn flush(&self) -> Result<()> {
        let lines =
            std::mem::take(&mut *self.pending.lock().unwrap_or_else(PoisonError::into_inner));
        if lines.is_empty() {
            return Ok(());
        }
        let names: Vec<String> = fs::read_dir(&self.events)?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .collect();
        let name = next_event_file(&names, Millis::now());
        let tmp = self.events.join(format!(".{name}.tmp"));
        let mut body = lines.join("\n");
        body.push('\n');
        fs::write(&tmp, body)?;
        fs::rename(&tmp, self.events.join(name))?;
        Ok(())
    }
}

/// The lines on stdout, without `at`: `xlisten transcribe`.
pub struct Stdout;

impl EventLog for Stdout {
    fn write(&self, event: Event) {
        println!("{}", encode(&event, None));
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;
    use crate::domain::event::{Reader, Speech};
    use crate::domain::speaker::SpeakerLabel;

    #[test]
    fn next_event_file_follows_xmoqs_rows() {
        // The rows of XmoQ's fixtures/agent-lines/event-files.tsv (read on 2026-11-05).
        let rows: &[(&str, &[&str], i64, &str)] = &[
            ("none before", &[], 1_700_000_000_000, "1700000000000.jsonl"),
            (
                "a clock of fewer digits is padded to 13",
                &[],
                5,
                "0000000000005.jsonl",
            ),
            (
                "a clock after the newest",
                &["1700000000000.jsonl"],
                1_700_000_000_500,
                "1700000000500.jsonl",
            ),
            (
                "the same millisecond as the newest",
                &["1700000000000.jsonl"],
                1_700_000_000_000,
                "1700000000001.jsonl",
            ),
            (
                "a clock that went back",
                &["1700000000500.jsonl"],
                1_700_000_000_000,
                "1700000000501.jsonl",
            ),
            (
                "the newest by its number, not by where it is listed",
                &[
                    "1700000000000.jsonl",
                    "1700000000900.jsonl",
                    "1700000000100.jsonl",
                ],
                1_700_000_000_200,
                "1700000000901.jsonl",
            ),
            (
                "other names are not counted",
                &[
                    ".1700000009999.jsonl.tmp",
                    "events.jsonl",
                    "170000000099.jsonl",
                    "1700000099999.txt",
                    "17000000x0000.jsonl",
                ],
                1_700_000_000_000,
                "1700000000000.jsonl",
            ),
            (
                "a padded name is counted by its number",
                &["0000000000005.jsonl"],
                3,
                "0000000000006.jsonl",
            ),
        ];
        for (name, names, clock, expected) in rows {
            assert_eq!(next_event_file(names, Millis(*clock)), *expected, "{name}");
        }
    }

    #[test]
    fn each_event_is_one_line_as_agent_md_tells_it() {
        let rows: Vec<(&str, Event, Option<i64>, &str)> = vec![
            (
                "reader began",
                Event::Reader(Reader::Began {
                    model: "large-v3-turbo-q5_0".into(),
                    inputs: vec![Input::Me, Input::Others],
                    enhance: true,
                    diarize: false,
                }),
                Some(1_792_883_451_092),
                r#"{"at":1792883451092,"diarize":false,"enhance":true,"inputs":["me","others"],"model":"large-v3-turbo-q5_0","state":"began","type":"reader"}"#,
            ),
            (
                "reader ended",
                Event::Reader(Reader::Ended),
                Some(5),
                r#"{"at":5,"state":"ended","type":"reader"}"#,
            ),
            (
                "speech with a speaker, without at",
                Event::Speech(Speech {
                    number: 3,
                    from: Input::Others,
                    text: "こんにちは".into(),
                    start_ms: Millis(1_792_883_451_092),
                    end_ms: Millis(1_792_883_452_000),
                    speaker: SpeakerLabel::from_index(1),
                }),
                None,
                r#"{"end_ms":1792883452000,"from":"others","number":3,"speaker":"B","start_ms":1792883451092,"text":"こんにちは","type":"speech"}"#,
            ),
            (
                "speech without a speaker has no key of it",
                Event::Speech(Speech {
                    number: 0,
                    from: Input::Me,
                    text: "はい".into(),
                    start_ms: Millis(0),
                    end_ms: Millis(400),
                    speaker: None,
                }),
                None,
                r#"{"end_ms":400,"from":"me","number":0,"start_ms":0,"text":"はい","type":"speech"}"#,
            ),
            (
                "an escape and a line break are escaped",
                Event::Note {
                    text: "a\u{1b}[2Jb\nc".into(),
                },
                None,
                r#"{"text":"a\u001b[2Jb\nc","type":"note"}"#,
            ),
            (
                "DEL and the C1 controls are escaped",
                Event::error("a\u{7f}b\u{9b}c"),
                None,
                r#"{"message":"a\u007fb\u009bc","type":"error"}"#,
            ),
        ];
        for (name, event, at, expected) in rows {
            assert_eq!(encode(&event, at.map(Millis)), expected, "{name}");
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xlisten-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn texts(dir: &Path) -> Vec<String> {
        event_files(dir)
            .unwrap()
            .iter()
            .flat_map(|f| {
                fs::read_to_string(f)
                    .unwrap()
                    .lines()
                    .map(|l| {
                        serde_json::from_str::<Value>(l).unwrap()["text"]
                            .as_str()
                            .unwrap()
                            .to_owned()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn a_flush_is_one_file_and_nothing_else() {
        let dir = scratch("flush");
        let w = EventsDir::new(&dir, false).unwrap();
        for t in ["a", "b"] {
            w.write(Event::Note { text: t.into() });
        }
        w.flush().unwrap();
        w.flush().unwrap();
        assert_eq!(fs::read_dir(dir.join("events")).unwrap().count(), 1);
        assert_eq!(texts(&dir), ["a", "b"]);
    }
}
