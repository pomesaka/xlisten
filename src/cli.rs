use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use crate::domain::input::Input;

// FIXME: measure small-q5_1 (XmoQ's default) against this on Japanese meetings, and keep the one
// that is good enough and keeps up. Both are in XmoQ's cache.
const DEFAULT_MODEL: &str = "large-v3-turbo-q5_0";

/// Transcribes the microphone and the sound of meeting apps on this Mac into files an agent reads.
#[derive(Parser)]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// The microphone (me) and the sound of the apps (others) into DIR/events/, until Ctrl-C or
    /// SIGTERM to the PID in DIR/xlisten.pid.
    Listen(Listen),
    /// A recording, as the lines listen would write, on stdout.
    Transcribe {
        file: PathBuf,
        #[command(flatten)]
        model: Model,
        /// Whose sound it is.
        #[arg(long, default_value = "me")]
        from: Input,
        /// RE-USE first (with `--from me`).
        #[arg(long)]
        enhance: bool,
    },
    /// The microphone into a WAV file, until Ctrl-C.
    RecordMic {
        file: PathBuf,
        #[arg(long)]
        mic_device: Option<String>,
        #[arg(long)]
        seconds: Option<f64>,
    },
    /// How long after the end of speech each line of DIR/events/ came.
    Latency { dir: PathBuf },
}

#[derive(Args)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one bool a flag of the command line"
)]
pub struct Listen {
    #[arg(long)]
    pub out: PathBuf,
    #[command(flatten)]
    pub model: Model,
    /// Leave the microphone out.
    #[arg(long)]
    pub no_mic: bool,
    /// Leave the sound of the apps out.
    #[arg(long)]
    pub no_system: bool,
    /// The microphone by its name, rather than the default one.
    #[arg(long)]
    pub mic_device: Option<String>,
    /// Leave RE-USE out.
    #[arg(long)]
    pub no_enhance: bool,
    /// Leave the speakers of the others untold.
    #[arg(long)]
    pub no_diarize: bool,
    /// Keep each utterance there as WAV, with what RE-USE made of it; none is kept without it.
    #[arg(long)]
    pub keep_audio: Option<PathBuf>,
    /// Stop after this many seconds.
    #[arg(long)]
    pub seconds: Option<f64>,
}

#[derive(Args)]
pub struct Model {
    /// A name of `XmoQ`'s cache of whisper.cpp (tiny, small-q5_1, large-v3-turbo-q5_0), or a .bin.
    #[arg(long = "model", default_value = DEFAULT_MODEL)]
    pub name: String,
}
