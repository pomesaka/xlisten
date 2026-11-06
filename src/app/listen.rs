//! Listening: each input cut into utterances on a thread of its own, and one thread that reads
//! them in the order they ended, until it is told to stop.

use std::any::Any;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::Result;

use crate::app::log;
use crate::app::pipeline::Pipeline;
use crate::app::ports::{EventLog, Notes, SpeakerFinder};
use crate::domain::event::{Event, Reader};
use crate::domain::input::Input;
use crate::domain::time::Millis;
use crate::domain::utterance::{AudioBlock, Segmenter, Utterance};

/// An input that has begun to hand its sound over.
///
/// WHY NOT a port for the inputs, as for the rest of the world outside: each hands blocks into a
/// channel already, and a channel is all `listen` needs of it.
pub struct Feed {
    pub input: Input,
    pub rate: u32,
    pub blocks: Receiver<AudioBlock>,
    /// What hands the blocks; dropping it stops it, and closes `blocks`.
    pub device: Box<dyn Any>,
}

pub struct Listening {
    /// In the order they were started; stopped in the same order.
    pub feeds: Vec<Feed>,
    pub pipeline: Pipeline,
    pub speakers: Option<Arc<dyn SpeakerFinder>>,
    pub log: Arc<dyn EventLog>,
    pub notes: Box<dyn Notes>,
    /// whisper's model, as the `reader` line names it.
    pub model: String,
}

/// Listens until `stop` says to, then writes what is left and the `ended` line.
///
/// # Errors
/// When the lines cannot be written.
pub fn listen(l: Listening, stop: impl Fn() -> bool) -> Result<()> {
    let Listening {
        feeds,
        pipeline,
        speakers,
        log: events,
        mut notes,
        model,
    } = l;
    events.write(Event::Reader(Reader::Began {
        model,
        inputs: feeds.iter().map(|f| f.input).collect(),
        enhance: pipeline.enhances(),
        diarize: speakers.is_some(),
    }));
    let (to_read, utterances) = mpsc::sync_channel::<Utterance>(64);
    let mut devices = Vec::new();
    let mut cutting = Vec::new();
    for feed in feeds {
        let speakers = speakers.clone().filter(|_| feed.input.has_speakers());
        devices.push(feed.device);
        cutting.push(cut(
            feed.input,
            feed.rate,
            feed.blocks,
            to_read.clone(),
            speakers,
            Arc::clone(&events),
        ));
    }
    drop(to_read);
    let reading = read(utterances, pipeline, Arc::clone(&events));

    while !stop() {
        thread::sleep(Duration::from_secs(1));
        match notes.read() {
            Ok(lines) => lines.into_iter().for_each(|e| events.write(e)),
            Err(e) => log(&format!("inbox.txt: {e:#}")),
        }
        events.flush()?;
    }

    // Stopping the devices closes their channels, which ends the threads that cut, then the one
    // that reads.
    drop(devices);
    for t in cutting {
        let _ = t.join();
    }
    let _ = reading.join();
    events.write(Event::Reader(Reader::Ended));
    events.flush()
}

fn cut(
    input: Input,
    rate: u32,
    blocks: Receiver<AudioBlock>,
    to_read: SyncSender<Utterance>,
    speakers: Option<Arc<dyn SpeakerFinder>>,
    events: Arc<dyn EventLog>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut segmenter = Segmenter::new(input, rate);
        for block in blocks {
            if let Some(s) = &speakers
                && let Err(e) = s.hear(&block)
            {
                events.write(Event::error(format!("diarization: {e:#}")));
            }
            for u in segmenter.push(&block) {
                if to_read.send(u).is_err() {
                    return;
                }
            }
        }
        if let Some(u) = segmenter.finish() {
            let _ = to_read.send(u);
        }
    })
}

/// WHY one thread for both inputs: whisper.cpp reads one at a time on the GPU anyway, and one
/// model is loaded once.
fn read(
    utterances: Receiver<Utterance>,
    mut pipeline: Pipeline,
    events: Arc<dyn EventLog>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        for u in utterances {
            let out = pipeline.read(&u);
            if let Some(read_in) = out.read_in {
                let enhanced = out.enhanced_in.map_or(String::new(), |t| {
                    let t = t.as_secs_f64();
                    format!("RE-USE {t:.2}s ({:.2} of its length), ", t / u.seconds())
                });
                log(&format!(
                    "{}: {:.1}s of speech, {enhanced}whisper {:.2}s, line {:.2}s after it ended",
                    u.input,
                    u.seconds(),
                    read_in.as_secs_f64(),
                    Millis::now().seconds_since(u.span.end),
                ));
            }
            out.events.into_iter().for_each(|e| events.write(e));
        }
    })
}
