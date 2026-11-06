//! `inbox.txt`: each whole line added to it is a `note`. What is in it at the start is passed over.

use std::fs;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::PathBuf;

use anyhow::Result;

use crate::app::ports::Notes;
use crate::domain::event::Event;

const LINE_BYTES: usize = 4096;

pub struct Inbox {
    path: PathBuf,
    /// Where the next read begins.
    pos: u64,
    /// The start of a line whose line feed has not come yet.
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
}

impl Notes for Inbox {
    fn read(&mut self) -> Result<Vec<Event>> {
        let size = fs::metadata(&self.path)?.len();
        if size < self.pos {
            // Written anew: read it from the start.
            self.pos = 0;
            self.partial.clear();
        }
        if size == self.pos {
            return Ok(Vec::new());
        }
        let mut file = fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.pos))?;
        let mut data = std::mem::take(&mut self.partial);
        file.take(size - self.pos).read_to_end(&mut data)?;
        self.pos = size;
        let mut lines: Vec<&[u8]> = data.split(|&b| b == b'\n').collect();
        self.partial = lines.pop().unwrap_or_default().to_vec();
        Ok(lines.into_iter().filter_map(note).collect())
    }
}

fn note(raw: &[u8]) -> Option<Event> {
    if raw.len() > LINE_BYTES {
        return Some(Event::error(format!(
            "a line of inbox.txt over {LINE_BYTES} bytes is not read"
        )));
    }
    let text = String::from_utf8_lossy(raw);
    let text = text.trim_end_matches('\r');
    (!text.trim().is_empty()).then(|| Event::Note { text: text.into() })
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use std::io::Write as _;

    use super::*;

    #[test]
    fn inbox_passes_over_what_was_there_and_waits_for_a_whole_line() {
        let dir = std::env::temp_dir().join(format!("xlisten-test-inbox-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("inbox.txt");
        fs::write(&path, "before\n").unwrap();
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
        let mut notes = inbox.read().unwrap();
        append("o\n");
        notes.extend(inbox.read().unwrap());
        assert_eq!(
            notes,
            [
                Event::Note { text: "one".into() },
                Event::Note { text: "two".into() }
            ]
        );
    }
}
