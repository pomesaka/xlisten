//! xlisten: transcribes the microphone and the sound of meeting apps into a folder an agent reads.
//!
//! This file is the composition root: it parses the command line, makes the adapters of `infra`,
//! and hands them to a use case of `app` (`docs/adr/0001-domain-model-and-layers.md`).

mod app;
mod cli;
mod domain;
mod dsp;
mod infra;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::Parser as _;

use crate::app::listen::{Feed, Listening};
use crate::app::log;
use crate::app::pipeline::Pipeline;
use crate::app::ports::{Enhancer, SpeakerFinder};
use crate::cli::{Cli, Command};
use crate::domain::input::Input;
use crate::infra::events_dir::{EventsDir, Stdout};
use crate::infra::inbox::Inbox;
use crate::infra::microphone::Microphone;
use crate::infra::speech_cpp::speakers::Speakers;
use crate::infra::speech_cpp::{DIARIZATION, RE_USE, SpeechModel};
use crate::infra::tap::SystemAudio;
use crate::infra::wav::{WavFolder, read_wav, write_wav};
use crate::infra::whisper::{Whisper, model_path};

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Listen(a) => listen(a),
        Command::Transcribe {
            file,
            model,
            from,
            enhance,
        } => transcribe(&file, &model.name, from, enhance),
        Command::RecordMic {
            file,
            mic_device,
            seconds,
        } => record_mic(&file, mic_device.as_deref(), seconds),
        Command::Latency { dir } => latency(&dir),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            log(&format!("{e:#}"));
            ExitCode::FAILURE
        }
    }
}

/// Whether to stop: Ctrl-C, SIGTERM, or `seconds` gone by.
fn stopper(seconds: Option<f64>) -> Result<impl Fn() -> bool> {
    let signalled = Arc::new(AtomicBool::new(false));
    let s = Arc::clone(&signalled);
    ctrlc::set_handler(move || s.store(true, Ordering::SeqCst))?;
    let began = Instant::now();
    Ok(move || {
        signalled.load(Ordering::SeqCst)
            || seconds.is_some_and(|s| began.elapsed().as_secs_f64() >= s)
    })
}

/// `xlisten.pid`, there while listening: for whoever started xlisten in the background to stop it
/// by its PID (SIGTERM ends it as Ctrl-C does, with the `ended` line).
struct PidFile(PathBuf);

impl PidFile {
    fn write(path: PathBuf) -> Result<Self> {
        std::fs::write(&path, format!("{}\n", std::process::id()))?;
        Ok(Self(path))
    }
}

impl Drop for PidFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn listen(a: cli::Listen) -> Result<()> {
    std::fs::create_dir_all(&a.out)?;
    let out = a.out.canonicalize()?;
    std::fs::write(out.join("AGENT.md"), include_str!("AGENT.md"))?;
    let _pid = PidFile::write(out.join("xlisten.pid"))?;
    let stop = stopper(a.seconds)?;
    let (mic, system) = (!a.no_mic, !a.no_system);

    let reader = Whisper::open(&model_path(&a.model.name))?;
    let enhancer: Option<Box<dyn Enhancer>> = if mic && !a.no_enhance {
        Some(Box::new(SpeechModel::fetched(RE_USE)?))
    } else {
        None
    };
    let events = Arc::new(EventsDir::new(&out, true)?);
    if let Some(k) = &a.keep_audio {
        log(&format!("each utterance is kept in {}", k.display()));
    }

    let mut feeds = Vec::new();
    let mut speakers: Option<Arc<dyn SpeakerFinder>> = None;
    // WHY the microphone before the tap: with the tap's aggregate device started first,
    // AudioOutputUnitStart of the microphone waited in Core Audio and never returned (macOS 14.4,
    // 2026-11-05); in this order both start.
    if mic {
        let (tx, blocks) = mpsc::sync_channel(256);
        let m = Microphone::start(a.mic_device.as_deref(), tx)?;
        log(&format!("microphone: {}, {} Hz", m.name, m.rate));
        feeds.push(Feed {
            input: Input::Me,
            rate: m.rate,
            blocks,
            device: Box::new(m),
        });
    }
    if system {
        let (tx, blocks) = mpsc::sync_channel(256);
        let tap = SystemAudio::start(tx)?;
        if !a.no_diarize {
            speakers = Some(Arc::new(Speakers::new(
                SpeechModel::fetched(DIARIZATION)?,
                tap.rate,
            )?));
        }
        feeds.push(Feed {
            input: Input::Others,
            rate: tap.rate,
            blocks,
            device: Box::new(tap),
        });
    }

    let pipeline = Pipeline::new(
        Box::new(reader),
        enhancer,
        speakers.clone(),
        a.keep_audio
            .map(|dir| Box::new(WavFolder::new(dir)) as Box<dyn app::ports::AudioKeeper>),
    );
    let notes = Box::new(Inbox::new(out.join("inbox.txt"))?);
    log(&format!(
        "Read {}/AGENT.md and take part in the meeting as it says.",
        out.display()
    ));
    app::listen::listen(
        Listening {
            feeds,
            pipeline,
            speakers,
            log: events,
            notes,
            model: a.model.name,
        },
        stop,
    )
}

fn transcribe(file: &Path, model: &str, from: Input, enhance: bool) -> Result<()> {
    if enhance && !from.is_enhanced() {
        bail!("--enhance is for --from me: RE-USE goes over the microphone only");
    }
    let (samples, rate) = read_wav(file)?;
    let enhancer: Option<Box<dyn Enhancer>> = if enhance {
        Some(Box::new(SpeechModel::fetched(RE_USE)?))
    } else {
        None
    };
    let mut pipeline = Pipeline::new(
        Box::new(Whisper::open(&model_path(model))?),
        enhancer,
        None,
        None,
    );
    let summary = app::transcribe::transcribe(&samples, rate, from, &mut pipeline, &Stdout)?;
    log(&summary.to_string());
    Ok(())
}

fn record_mic(file: &Path, device: Option<&str>, seconds: Option<f64>) -> Result<()> {
    let stop = stopper(seconds)?;
    let (tx, blocks) = mpsc::sync_channel(1024);
    let m = Microphone::start(device, tx)?;
    log(&format!(
        "recording {} at {} Hz into {}; Ctrl-C to stop",
        m.name,
        m.rate,
        file.display()
    ));
    let rate = m.rate;
    let mut samples = Vec::new();
    while !stop() {
        samples.extend(blocks.try_iter().flat_map(|b| b.samples));
        std::thread::sleep(Duration::from_millis(100));
    }
    drop(m);
    samples.extend(blocks.try_iter().flat_map(|b| b.samples));
    write_wav(file, &samples, rate)?;
    log(&format!(
        "{:.1}s written",
        domain::time::samples_ms(samples.len(), rate) / 1000.0
    ));
    Ok(())
}

fn latency(dir: &Path) -> Result<()> {
    for l in app::latency::latencies(infra::events_dir::speech_latencies(dir)?) {
        println!("{l}");
    }
    Ok(())
}
