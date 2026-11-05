//! xlisten: transcribes the microphone and the sound of meeting apps into a folder an agent reads.

mod audio;
mod diar;
mod lines;
mod mic;
mod segmenter;
mod speech;
mod tap;
mod whisper;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail};
use serde_json::Value;

use crate::audio::{WHISPER_RATE, read_wav, resample, write_wav};
use crate::diar::Speakers;
use crate::lines::{EventWriter, Inbox, Line, encode, now_ms};
use crate::segmenter::{Segmenter, Utterance};
use crate::speech::SpeechModel;
use crate::tap::Block;
use crate::whisper::{Whisper, model_path};

const USAGE: &str = "\
usage:
  xlisten listen --out DIR [--model NAME] [--no-mic] [--no-system] [--mic-device NAME]
                 [--no-enhance] [--no-diarize] [--keep-audio DIR] [--seconds N]
      the microphone (me) and the sound of the apps (others) into DIR/events/
  xlisten transcribe FILE.wav [--model NAME] [--from me|others] [--enhance]
      a recording, as the lines listen would write, on stdout
  xlisten record-mic FILE.wav [--mic-device NAME] [--seconds N]
      the microphone into a WAV file, until Ctrl-C
  xlisten latency DIR
      how long after the end of speech each line came

--model is a name of XmoQ's cache of whisper.cpp (tiny, small-q5_1, large-v3-turbo-q5_0) or a .bin.
--keep-audio keeps each utterance there as WAV, with what RE-USE made of it; none is kept without it.";

// FIXME: measure small-q5_1 (XmoQ's default) against this on Japanese meetings, and keep the one
// that is good enough and keeps up. Both are in XmoQ's cache.
const DEFAULT_MODEL: &str = "large-v3-turbo-q5_0";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn speech_model(file: &str) -> Result<SpeechModel> {
    let path = root().join(".cache/models").join(file);
    if !path.exists() {
        bail!(
            "{} is not there: run `mise run speech` first",
            path.display()
        );
    }
    SpeechModel::load(&path)
}

const RE_USE: &str = "NVIDIA-NemotronLabs-RE-USE-10M-F32.gguf";
const DIARIZATION: &str = "Nemotron-3-Diarization-99M-Q8_0.gguf";

fn log(message: &str) {
    eprintln!("xlisten: {message}");
}

fn line(kind: &str) -> Line {
    Line::from([("type", Value::from(kind))])
}

#[derive(Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one bool a flag of the command line"
)]
struct Args {
    command: String,
    file: Option<String>,
    out: Option<String>,
    model: Option<String>,
    mic: bool,
    system: bool,
    mic_device: Option<String>,
    enhance: bool,
    diarize: bool,
    keep_audio: Option<String>,
    seconds: Option<f64>,
    from: Option<String>,
}

fn parse() -> Result<Args> {
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        mic: true,
        system: true,
        enhance: true,
        diarize: true,
        ..Args::default()
    };
    a.command = it.next().context(USAGE)?;
    if a.command == "transcribe" {
        a.enhance = false;
    }
    while let Some(arg) = it.next() {
        let mut value = || {
            it.next()
                .with_context(|| format!("{arg} needs a value\n{USAGE}"))
        };
        match arg.as_str() {
            "--out" => a.out = Some(value()?),
            "--model" => a.model = Some(value()?),
            "--mic-device" => a.mic_device = Some(value()?),
            "--keep-audio" => a.keep_audio = Some(value()?),
            "--seconds" => a.seconds = Some(value()?.parse()?),
            "--from" => a.from = Some(value()?),
            "--no-mic" => a.mic = false,
            "--no-system" => a.system = false,
            "--no-enhance" => a.enhance = false,
            "--enhance" => a.enhance = true,
            "--no-diarize" => a.diarize = false,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            s if !s.starts_with("--") && a.file.is_none() => a.file = Some(s.to_owned()),
            s => bail!("unknown {s}\n{USAGE}"),
        }
    }
    Ok(a)
}

fn main() {
    if let Err(e) = run() {
        log(&format!("{e:#}"));
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let a = parse()?;
    match a.command.as_str() {
        "listen" => listen(&a),
        "transcribe" => transcribe(&a),
        "record-mic" => record_mic(&a),
        "latency" => latency(a.file.as_deref().context(USAGE)?),
        _ => bail!("{USAGE}"),
    }
}

/// Stopped by Ctrl-C, or by SIGTERM to its PID (`xlisten.pid`).
fn stop_flag() -> Result<Arc<AtomicBool>> {
    let stop = Arc::new(AtomicBool::new(false));
    let s = Arc::clone(&stop);
    ctrlc::set_handler(move || s.store(true, Ordering::SeqCst))?;
    Ok(stop)
}

struct Work {
    from: &'static str,
    utterance: Utterance,
}

/// One thread that reads the utterances in the order they ended, both inputs together.
/// WHY one: whisper.cpp reads one at a time on the GPU anyway, and one model is loaded once.
fn transcriber(
    rx: Receiver<Work>,
    mut whisper: Whisper,
    enhancer: Option<SpeechModel>,
    speakers: Option<Speakers>,
    writer: Arc<EventWriter>,
    keep: Option<PathBuf>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut number = 0_u64;
        for Work { from, utterance: u } in rx {
            let mut samples = resample(&u.samples, u.rate, WHISPER_RATE);
            let keep_wav = |name: String, samples: &[f32]| {
                if let Some(dir) = &keep {
                    let _ = std::fs::create_dir_all(dir);
                    if let Err(e) = write_wav(&dir.join(name), samples, WHISPER_RATE) {
                        log(&format!("{e:#}"));
                    }
                }
            };
            keep_wav(format!("{}-{from}.wav", u.start_ms), &samples);
            #[expect(clippy::cast_precision_loss, reason = "an utterance is short")]
            let seconds = samples.len() as f64 / f64::from(WHISPER_RATE);
            let mut took = Vec::new();
            // WHY the microphone only: RE-USE takes noise and the echo of a room out of a recording;
            // the sound of the apps comes as the meeting app sent it.
            if let (Some(model), "me") = (&enhancer, from) {
                let began = Instant::now();
                match model.enhance(&samples, WHISPER_RATE) {
                    Ok(out) => {
                        let t = began.elapsed().as_secs_f64();
                        took.push(format!("RE-USE {t:.2}s ({:.2} of its length)", t / seconds));
                        samples = out;
                        keep_wav(format!("{}-{from}-enhanced.wav", u.start_ms), &samples);
                    }
                    Err(e) => writer.write(Line::from([
                        ("type", "error".into()),
                        ("message", format!("{from}: {e:#}").into()),
                    ])),
                }
            }
            let began = Instant::now();
            let text = match whisper.text(&samples) {
                Ok(t) => t,
                Err(e) => {
                    writer.write(Line::from([
                        ("type", "error".into()),
                        ("message", format!("{from}: {e:#}").into()),
                    ]));
                    continue;
                }
            };
            took.push(format!("whisper {:.2}s", began.elapsed().as_secs_f64()));
            let speaker = match (&speakers, from) {
                (Some(s), "others") if !text.is_empty() => s.who(u.start_ms, u.end_ms),
                _ => None,
            };
            #[expect(clippy::cast_precision_loss, reason = "ms")]
            let after = (now_ms() - u.end_ms) as f64 / 1000.0;
            log(&format!(
                "{from}: {seconds:.1}s of speech, {}, line {after:.2}s after it ended",
                took.join(", ")
            ));
            if text.is_empty() {
                continue;
            }
            let mut l = line("speech");
            l.insert("number", number.into());
            l.insert("from", from.into());
            l.insert("text", text.into());
            l.insert("start_ms", u.start_ms.into());
            l.insert("end_ms", u.end_ms.into());
            if let Some(s) = speaker {
                l.insert("speaker", s.into());
            }
            writer.write(l);
            number += 1;
        }
    })
}

/// Takes blocks off the thread of the audio and cuts them into utterances.
fn input(
    from: &'static str,
    rate: u32,
    rx: Receiver<Block>,
    to: SyncSender<Work>,
    speakers: Option<Speakers>,
    writer: Arc<EventWriter>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut seg = Segmenter::new(rate);
        for (samples, end_ms) in rx {
            if let Some(s) = &speakers
                && let Err(e) = s.push(&samples, end_ms)
            {
                writer.write(Line::from([
                    ("type", "error".into()),
                    ("message", format!("diarization: {e:#}").into()),
                ]));
            }
            for utterance in seg.push(&samples, end_ms) {
                let _ = to.send(Work { from, utterance });
            }
        }
        if let Some(utterance) = seg.finish() {
            let _ = to.send(Work { from, utterance });
        }
    })
}

fn listen(a: &Args) -> Result<()> {
    let out = PathBuf::from(a.out.as_deref().context(USAGE)?);
    std::fs::create_dir_all(&out)?;
    let out = out.canonicalize()?;
    std::fs::write(out.join("AGENT.md"), include_str!("AGENT.md"))?;
    // For whoever started xlisten in the background to stop it by its PID (SIGTERM ends it as
    // Ctrl-C does, with the `ended` line).
    let pid_file = out.join("xlisten.pid");
    std::fs::write(&pid_file, format!("{}\n", std::process::id()))?;
    let result = listen_in(a, &out);
    let _ = std::fs::remove_file(&pid_file);
    result
}

fn listen_in(a: &Args, out: &Path) -> Result<()> {
    let stop = stop_flag()?;
    let model = a.model.as_deref().unwrap_or(DEFAULT_MODEL);
    let whisper = Whisper::open(&model_path(model))?;
    let enhancer = if a.enhance && a.mic {
        Some(speech_model(RE_USE)?)
    } else {
        None
    };
    let writer = Arc::new(EventWriter::new(out, true)?);
    let keep = a.keep_audio.as_ref().map(PathBuf::from);
    if let Some(k) = &keep {
        log(&format!("each utterance is kept in {}", k.display()));
    }

    let (work_tx, work_rx) = mpsc::sync_channel::<Work>(64);
    let mut inputs = Vec::new();
    let mut mic = None;
    let mut system = None;
    let mut speakers = None;
    let mut l = line("reader");
    l.insert("state", "began".into());
    l.insert("model", model.into());
    l.insert("enhance", (a.enhance && a.mic).into());
    let names: Vec<&str> = [("me", a.mic), ("others", a.system)]
        .iter()
        .filter(|(_, on)| *on)
        .map(|(n, _)| *n)
        .collect();
    l.insert("inputs", names.into());
    // WHY the microphone before the tap: with the tap's aggregate device started first,
    // AudioOutputUnitStart of the microphone waited in Core Audio and never returned (macOS 14.4,
    // 2026-11-05); in this order both start.
    if a.mic {
        let (tx, rx) = mpsc::sync_channel::<Block>(256);
        let m = mic::Microphone::start(a.mic_device.as_deref(), tx)?;
        log(&format!("microphone: {}, {} Hz", m.name, m.rate));
        inputs.push(input(
            "me",
            m.rate,
            rx,
            work_tx.clone(),
            None,
            Arc::clone(&writer),
        ));
        mic = Some(m);
    }
    if a.system {
        let (tx, rx) = mpsc::sync_channel::<Block>(256);
        let tap = tap::SystemAudio::start(tx)?;
        if a.diarize {
            speakers = Some(Speakers::new(speech_model(DIARIZATION)?, tap.rate)?);
        }
        inputs.push(input(
            "others",
            tap.rate,
            rx,
            work_tx.clone(),
            speakers.clone(),
            Arc::clone(&writer),
        ));
        system = Some(tap);
    }
    l.insert("diarize", speakers.is_some().into());
    writer.write(l);
    drop(work_tx);
    let transcribing = transcriber(
        work_rx,
        whisper,
        enhancer,
        speakers,
        Arc::clone(&writer),
        keep,
    );
    let mut inbox = Inbox::new(out.join("inbox.txt"))?;
    log(&format!(
        "Read {}/AGENT.md and take part in the meeting as it says.",
        out.display()
    ));

    let began = Instant::now();
    while !stop.load(Ordering::SeqCst)
        && a.seconds.is_none_or(|s| began.elapsed().as_secs_f64() < s)
    {
        std::thread::sleep(Duration::from_secs(1));
        if let Err(e) = inbox.poll(&writer) {
            log(&format!("inbox.txt: {e:#}"));
        }
        writer.flush()?;
    }

    // Stopping the inputs closes their channels, which ends their threads, then the transcriber's.
    if let Some(m) = mic {
        m.stop();
    }
    drop(system);
    for i in inputs {
        let _ = i.join();
    }
    let _ = transcribing.join();
    let mut l = line("reader");
    l.insert("state", "ended".into());
    writer.write(l);
    writer.flush()
}

/// Plays a recording through the segmenter as fast as it reads, and prints `speech` lines whose
/// `start_ms` and `end_ms` are from the start of the file.
fn transcribe(a: &Args) -> Result<()> {
    let (samples, rate) = read_wav(Path::new(a.file.as_deref().context(USAGE)?))?;
    let mut whisper = Whisper::open(&model_path(a.model.as_deref().unwrap_or(DEFAULT_MODEL)))?;
    let enhancer = if a.enhance {
        Some(speech_model(RE_USE)?)
    } else {
        None
    };
    let from = a.from.as_deref().unwrap_or("me");
    let mut seg = Segmenter::new(rate);
    let block = usize::try_from(rate / 10).unwrap_or(1600);
    let mut utterances = Vec::new();
    for (i, part) in samples.chunks(block).enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "a recording fits")]
        let end = (i * block + part.len()) as f64 * 1000.0 / f64::from(rate);
        utterances.extend(seg.push(part, end));
    }
    utterances.extend(seg.finish());
    let (mut took_enhance, mut took_whisper, mut speech_s) = (0.0, 0.0, 0.0);
    for (n, u) in utterances.iter().enumerate() {
        let mut part = resample(&u.samples, u.rate, WHISPER_RATE);
        #[expect(clippy::cast_precision_loss, reason = "an utterance is short")]
        let length = part.len() as f64 / f64::from(WHISPER_RATE);
        speech_s += length;
        if let Some(m) = &enhancer {
            let began = Instant::now();
            part = m.enhance(&part, WHISPER_RATE)?;
            took_enhance += began.elapsed().as_secs_f64();
        }
        let began = Instant::now();
        let text = whisper.text(&part)?;
        took_whisper += began.elapsed().as_secs_f64();
        let mut l = line("speech");
        l.insert("number", n.into());
        l.insert("from", from.into());
        l.insert("text", text.into());
        l.insert("start_ms", u.start_ms.into());
        l.insert("end_ms", u.end_ms.into());
        println!("{}", encode(&l.into_iter().collect()));
    }
    let enhance_note = if enhancer.is_some() {
        format!(
            ", RE-USE {took_enhance:.2}s ({:.2} of their length)",
            took_enhance / speech_s
        )
    } else {
        String::new()
    };
    #[expect(clippy::cast_precision_loss, reason = "a recording fits")]
    let total = samples.len() as f64 / f64::from(rate);
    log(&format!(
        "{total:.1}s of audio, {} utterances of {speech_s:.1}s{enhance_note}, whisper {took_whisper:.2}s",
        utterances.len()
    ));
    Ok(())
}

fn record_mic(a: &Args) -> Result<()> {
    let file = PathBuf::from(a.file.as_deref().context(USAGE)?);
    let stop = stop_flag()?;
    let (tx, rx) = mpsc::sync_channel::<Block>(1024);
    let m = mic::Microphone::start(a.mic_device.as_deref(), tx)?;
    log(&format!(
        "recording {} at {} Hz into {}; Ctrl-C to stop",
        m.name,
        m.rate,
        file.display()
    ));
    let began = Instant::now();
    let mut samples = Vec::new();
    while !stop.load(Ordering::SeqCst)
        && a.seconds.is_none_or(|s| began.elapsed().as_secs_f64() < s)
    {
        while let Ok((block, _)) = rx.try_recv() {
            samples.extend(block);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let rate = m.rate;
    m.stop();
    samples.extend(rx.try_iter().flat_map(|(b, _)| b));
    write_wav(&file, &samples, rate)?;
    #[expect(clippy::cast_precision_loss, reason = "a recording fits")]
    let seconds = samples.len() as f64 / f64::from(rate);
    log(&format!("{seconds:.1}s written"));
    Ok(())
}

/// How long after the end of each utterance its `speech` line was written (`at - end_ms`).
fn latency(out: &str) -> Result<()> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(Path::new(out).join("events"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                Path::new(n).extension().is_some_and(|e| e == "jsonl")
                    && n.starts_with(|c: char| c.is_ascii_digit())
            })
        })
        .collect();
    files.sort();
    let mut by: std::collections::BTreeMap<String, Vec<i64>> = std::collections::BTreeMap::new();
    for f in files {
        for raw in std::fs::read_to_string(&f)?.lines() {
            let v: Value = serde_json::from_str(raw)?;
            let field = |k: &str| v.get(k).and_then(Value::as_i64).unwrap_or(0);
            if v.get("type").and_then(Value::as_str) == Some("speech") {
                let from = v
                    .get("from")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                by.entry(from)
                    .or_default()
                    .push(field("at") - field("end_ms"));
            }
        }
    }
    for (from, mut ms) in by {
        ms.sort_unstable();
        let median = ms.get(ms.len() / 2).copied().unwrap_or(0);
        let max = ms.last().copied().unwrap_or(0);
        #[expect(clippy::cast_precision_loss, reason = "ms")]
        let (median, max) = (median as f64 / 1000.0, max as f64 / 1000.0);
        println!(
            "{from}: {} lines, median {median:.2}s, max {max:.2}s",
            ms.len()
        );
    }
    Ok(())
}
