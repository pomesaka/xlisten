# ADR-0001: The domain model of xlisten, and its three layers

## Status

Accepted, 2026-11-06.

## Context

The first version in Rust (2026-11-05) put everything a module of `src/` at one level, and held the rules of a meeting's transcript in strings and maps:

- which input an utterance came from was `&'static str` (`"me"`, `"others"`), matched as strings;
- a line of `events/` was a `BTreeMap<&str, Value>` filled by hand, the `speech` line in two places and the `error` line in four;
- the rules of an utterance (RE-USE for `me` only, a speaker for `others` only, a number for a line with words only) lived inside the 90 lines of one thread's closure, untested;
- the segmenter's state was an `Option` beside fields that meant something only while it was `None`;
- times were `f64` in one place and `i64` in another.

## Decision

Three layers, depending inwards only: `infra` on `app` and `domain`, `app` on `domain`, `domain` on nothing of the other two.

```
src/
  main.rs      the composition root: parses the command line, makes the adapters, runs a use case
  cli.rs       the commands and their flags (clap)
  domain/      what a transcript of a meeting is; pure, no I/O, tested by tables
  app/         the use cases: in which order, on which thread, what a failure does; ports
  infra/       whisper.cpp, speech.cpp, Core Audio, cpal, the folder of events, WAV
  dsp.rs       mono and resampling, of no layer (a generic subdomain)
```

### Domain

| Type | Kind | What it is |
|---|---|---|
| `Input` (`Me`, `Others`) | Value | Which input an utterance came through: the microphone, or the sound the apps play. Says which inputs are enhanced (`Me`) and which are told apart by speaker (`Others`) |
| `Millis`, `Span` | Value | A time on a clock, in ms; the start and end of an utterance |
| `AudioBlock` | Value | Mono samples as an input hands them, and when the last of them was heard |
| `Utterance` | Value | A stretch of one input's sound from where somebody starts to where they stop |
| `Segmenter` | Domain service | Cuts an input's blocks into utterances: 600 ms of quiet ends one, one never quiet is cut at 10 s at its quietest |
| `SpeakerLabel`, `SpeakerTurn` | Value | `A`, `B`, ... in the order voices were first heard in a run; a stretch of one voice |
| `Transcript` | Entity (one a run) | Numbers the lines with words, in the order they are read |
| `Event` | Value | A line of `events/`: `reader`, `speech`, `note`, `error` |

`Event` derives `Serialize`: its form is this tool's published language (`AGENT.md`, after XmoQ's `xmoq-agent`), and is part of the model, not a detail of the folder. `at`, the clock of the writer, is added by the folder (`infra`).

### App

Ports, the traits the use cases call and `infra` implements: `SpeechToText`, `Enhancer`, `SpeakerFinder`, `AudioKeeper`, `EventLog`, `Notes`. The use cases: `listen` (threads, the order of utterances, stopping), `transcribe` (a recording through the same `Pipeline`), `latency`.

`Pipeline` takes one `Utterance` to its `Event`s. It asks `Input` whether to enhance and whether to find a speaker, and `Transcript` for the number; it decides nothing itself.

### Infra

One file a thing outside: `whisper.rs`, `speech_cpp/` (the C API, RE-USE, the diarization), `tap.rs` (Core Audio), `microphone.rs` (cpal), `events_dir.rs` (the files of `events/`), `inbox.rs`, `wav.rs`. Each object of Core Audio is held by a guard that destroys it, so a failure halfway through needs no hand-written undoing.

## Ubiquitous language

| Adopted | Dropped | Why |
|---|---|---|
| `Input` (`Me`, `Others`) | `from: &str`, `Work.from` | The two words the person and `AGENT.md` use. `from` stays as the key of the line (XmoQ's) |
| `Utterance` | `Work` | `Work` was a name of the queue; the thing queued is an utterance, which now carries its input |
| `Event` | `Line` (a map) | The lines are the events of `events/` |
| `SpeakerTurn` | `Segment` | A segment of what? A turn of one speaker |
| `AudioBlock` | `Block` (a tuple) | |

## Consequences

- The rules of an utterance are tested with fakes of the ports, without a model.
- `xlisten transcribe` goes through the same `Pipeline` as `listen`, and no longer prints a `speech` line without words.
- Six small traits, each with one implementation in `infra` and one fake in the tests.

## Rejected

- **Hexagonal ports for the inputs too** (a trait for the microphone and the tap). Each hands blocks into a channel already; a trait would add a name and nothing a test needs.
- **A `dto` for the lines, apart from `Event`.** The form of the line is what the agent reads; a second type of the same fields would be kept in step by hand.
- **A crate a layer (a workspace).** One binary of some 2,000 lines; modules keep the direction of the dependencies at a glance, and `rg 'crate::infra' src/domain src/app` finds a wrong one.
- **No layers, the files only tidied.** The rules of an utterance would stay where the threads are, and untested.

## When to look again

When a second way out comes (another sink than a folder, a server), when a second kind of input with rules of its own comes (a meeting app's own captions), or when `domain` needs I/O.
