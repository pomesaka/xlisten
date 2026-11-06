# xlisten

Transcribes the meetings of this Mac into a folder that an agent (Claude Code) reads: the microphone, and the sound the apps play (Google Meet in a browser, Zoom, a Slack huddle). Each input is transcribed on its own by whisper.cpp, so a line says whether it was `me` (the microphone) or `others` (the apps), and Nemotron 3 Diarization tells the others apart as `A`, `B`, ... The microphone goes through NVIDIA's RE-USE (speech enhancement) first; `--no-enhance` leaves it out. **Whether RE-USE makes the transcript better was not compared.** Nothing leaves the Mac.

One binary in Rust: whisper.cpp through `whisper-rs` (as XmoQ's `xmoq-speech`), RE-USE and the diarization through the C API of speech.cpp (`src/infra/speech_cpp/`), the process tap of Core Audio through `objc2-core-audio` (`src/infra/tap.rs`), the microphone through `cpal`.

## Running it

```sh
mise install
mise run speech                       # builds speech.cpp and fetches its two models into .cache, once
mise run listen -- --out ~/xlisten/standup
```

The models of whisper are those XmoQ fetches into `~/.cache/xmoq/whisper` (in XmoQ, `mise run speech-model -- large-v3-turbo-q5_0` and `-- silero-v5.1.2`); `--model` names another, or a `.bin`. xlisten prints the line to give the agent on stderr:

```
Read <out>/AGENT.md and take part in the meeting as it says.
```

It runs until Ctrl-C, or SIGTERM to the PID in `<out>/xlisten.pid`, and then writes the `ended` line. In Claude Code, `/listen web` or `/listen mic` starts it and reads along (the skill `listen` of dotclaude).

macOS asks, the first time, for the microphone and for recording the system's sound, for the terminal app xlisten runs in. Without the second, the tap hands only zeros and no `others` line comes.

No sound is kept. With `--keep-audio <dir>`, each utterance is kept there as WAV, with what RE-USE made of it beside it; nothing removes them. `recordings/` is out of git for such a folder.

`xlisten transcribe [--enhance] FILE.wav` writes the lines of a recording on stdout, and `xlisten record-mic FILE.wav` records the microphone, to compare settings on the same sound. `xlisten --help` has the flags. `mise run check` runs fmt, clippy and the tests.

## The folder

The same form as XmoQ's `xmoq-agent` (XmoQ's `fixtures/agent-lines/README.md`): `events/<13 digits, ms>.jsonl`, a file a second, never changed after it is moved to its name; `inbox.txt`; `AGENT.md`, the guide for the agent (`src/AGENT.md`). The lines are told in `AGENT.md`.

What was measured, and how fast it is: [docs/experiments/](docs/experiments/).

## Known gaps

- The sound of every app is tapped, not only the meeting's: music or a video playing is transcribed as `others`.
- With speakers rather than headphones, the microphone hears the others too, and their words come again as `me`.
- An utterance is cut where the sound gets quiet (`src/domain/utterance.rs`); a line comes at least `CLOSE_MS` after the speaker stops, and where it never gets quiet, every `LONGEST_MS`.
- Whisper writes words of its own (`ご視聴ありがとうございました`) where Silero hears a voice it cannot read, English among them: whisper is told the language is Japanese.
- The labels of the speakers start again with each run, and two voices may get one label.
- Meet and Zoom were not run yet.
