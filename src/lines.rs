//! The lines of `events/` and the files that hold them, in the form of `xmoq-agent` of `XmoQ`.
//!
//! The form is `XmoQ`'s `fixtures/agent-lines/README.md` (read on 2026-11-05): one JSON object a line,
//! keys in the order of their names, `at` the writer's clock in ms; a file of `events/` is named
//! after the clock in 13 digits at the least, or one more than the newest there, and is moved to
//! its name only when it is whole.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use serde_json::Value;

/// A line before it is written: its fields, `at` aside.
pub type Line = BTreeMap<&'static str, Value>;

const INBOX_LINE_BYTES: usize = 4096;

/// The wall clock in ms since the Unix epoch.
#[must_use]
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

/// One line of JSON, its keys sorted (a `BTreeMap`).
///
/// WHY DEL and the C1 controls on top of what `serde_json` escapes (U+0000..U+001F): some terminals
/// act on them as well, and an agent may print the lines in one.
#[must_use]
pub fn encode(line: &BTreeMap<&str, Value>) -> String {
    let text = serde_json::to_string(line).unwrap_or_default();
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if ('\u{7f}'..='\u{9f}').contains(&c) {
            let _ = write!(out, "\\u{:04x}", u32::from(c));
        } else {
            out.push(c);
        }
    }
    out
}

/// The name of the next file of `events/`, after those in `names`.
#[must_use]
pub fn next_event_file<S: AsRef<str>>(names: &[S], clock_ms: i64) -> String {
    let newest = names
        .iter()
        .filter_map(|n| {
            let digits = n.as_ref().strip_suffix(".jsonl")?;
            (digits.len() >= 13 && digits.bytes().all(|b| b.is_ascii_digit()))
                .then(|| digits.parse::<i64>().ok())
                .flatten()
        })
        .max()
        .unwrap_or(-1);
    format!("{:013}.jsonl", clock_ms.max(newest + 1))
}

/// Gathers lines and writes them as one file of `events/` a second ([`Self::flush`]).
pub struct EventWriter {
    events: PathBuf,
    pending: Mutex<Vec<String>>,
    echo: bool,
}

impl EventWriter {
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

    pub fn write(&self, mut line: Line) {
        line.insert("at", now_ms().into());
        let text = encode(&line.into_iter().collect());
        if self.echo {
            println!("{text}");
        }
        if let Ok(mut pending) = self.pending.lock() {
            pending.push(text);
        }
    }

    /// # Errors
    /// When the file cannot be written.
    pub fn flush(&self) -> Result<()> {
        let lines = match self.pending.lock() {
            Ok(mut pending) => std::mem::take(&mut *pending),
            Err(_) => return Ok(()),
        };
        if lines.is_empty() {
            return Ok(());
        }
        let names: Vec<String> = fs::read_dir(&self.events)?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .collect();
        let name = next_event_file(&names, now_ms());
        let tmp = self.events.join(format!(".{name}.tmp"));
        let body: String = lines.iter().fold(String::new(), |mut b, l| {
            b.push_str(l);
            b.push('\n');
            b
        });
        fs::write(&tmp, body)?;
        fs::rename(&tmp, self.events.join(name))?;
        Ok(())
    }
}

/// Lines added to `inbox.txt` become `note` lines. What is in it at the start is passed over.
pub struct Inbox {
    path: PathBuf,
    pos: u64,
    partial: Vec<u8>,
}

impl Inbox {
    /// # Errors
    /// When `inbox.txt` cannot be made.
    pub fn new(path: PathBuf) -> Result<Self> {
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        let pos = file.metadata()?.len();
        Ok(Self {
            path,
            pos,
            partial: Vec::new(),
        })
    }

    /// # Errors
    /// When `inbox.txt` cannot be read.
    pub fn poll(&mut self, writer: &EventWriter) -> Result<()> {
        let size = fs::metadata(&self.path)?.len();
        if size < self.pos {
            // Written anew: read it from the start.
            self.pos = 0;
            self.partial.clear();
        }
        if size == self.pos {
            return Ok(());
        }
        let mut file = fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.pos))?;
        let mut data = std::mem::take(&mut self.partial);
        file.take(size - self.pos).read_to_end(&mut data)?;
        self.pos = size;
        let mut parts: Vec<&[u8]> = data.split(|&b| b == b'\n').collect();
        self.partial = parts.pop().unwrap_or_default().to_vec();
        for raw in parts {
            let mut line = Line::new();
            if raw.len() > INBOX_LINE_BYTES {
                line.insert("type", "error".into());
                line.insert(
                    "message",
                    format!("a line of inbox.txt over {INBOX_LINE_BYTES} bytes is not read").into(),
                );
                writer.write(line);
                continue;
            }
            let text = String::from_utf8_lossy(raw);
            let text = text.trim_end_matches('\r');
            if !text.trim().is_empty() {
                line.insert("type", "note".into());
                line.insert("text", text.into());
                writer.write(line);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::type_complexity,
    reason = "tests"
)]
mod tests {
    use super::*;

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
            assert_eq!(next_event_file(names, *clock), *expected, "{name}");
        }
    }

    #[test]
    fn encode_sorts_keys_and_escapes_what_a_terminal_acts_on() {
        let rows: &[(&str, &[(&str, Value)], &str)] = &[
            (
                "keys in the order of their names, other characters as they are",
                &[
                    ("type", Value::from("speech")),
                    ("from", Value::from("me")),
                    ("text", Value::from("こんにちは")),
                ],
                r#"{"from":"me","text":"こんにちは","type":"speech"}"#,
            ),
            (
                "an escape and a line break are escaped",
                &[("text", Value::from("a\u{1b}[2Jb\nc"))],
                r#"{"text":"a\u001b[2Jb\nc"}"#,
            ),
            (
                "DEL and the C1 controls are escaped",
                &[("text", Value::from("a\u{7f}b\u{9b}c"))],
                r#"{"text":"a\u007fb\u009bc"}"#,
            ),
            (
                "a number without a fraction",
                &[("start_ms", Value::from(1_792_883_451_092_i64))],
                r#"{"start_ms":1792883451092}"#,
            ),
        ];
        for (name, fields, expected) in rows {
            let line: BTreeMap<&str, Value> = fields.iter().cloned().collect();
            assert_eq!(encode(&line), *expected, "{name}");
        }
    }

    fn texts(dir: &Path) -> Vec<String> {
        let mut files: Vec<_> = fs::read_dir(dir.join("events"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        files.sort();
        files
            .iter()
            .flat_map(|f| {
                fs::read_to_string(f)
                    .unwrap()
                    .lines()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .map(|l| {
                serde_json::from_str::<Value>(&l).unwrap()["text"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect()
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xlisten-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_flush_is_one_file_and_nothing_else() {
        let dir = scratch("flush");
        let w = EventWriter::new(&dir, false).unwrap();
        for t in ["a", "b"] {
            w.write(Line::from([("type", "note".into()), ("text", t.into())]));
        }
        w.flush().unwrap();
        w.flush().unwrap();
        assert_eq!(fs::read_dir(dir.join("events")).unwrap().count(), 1);
        assert_eq!(texts(&dir), ["a", "b"]);
    }

    #[test]
    fn inbox_passes_over_what_was_there_and_waits_for_a_whole_line() {
        use std::io::Write as _;
        let dir = scratch("inbox");
        let path = dir.join("inbox.txt");
        fs::write(&path, "before\n").unwrap();
        let w = EventWriter::new(&dir, false).unwrap();
        let mut inbox = Inbox::new(path.clone()).unwrap();
        let append = |s: &str| {
            fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap()
                .write_all(s.as_bytes())
                .unwrap();
        };
        append("one\n\ntw");
        inbox.poll(&w).unwrap();
        append("o\n");
        inbox.poll(&w).unwrap();
        w.flush().unwrap();
        assert_eq!(texts(&dir), ["one", "two"]);
    }
}
