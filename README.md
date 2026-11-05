# xlisten

Transcribes the meetings of this Mac into a folder that an agent (Claude Code) reads: the microphone, and the sound the apps play (Google Meet in a browser, Zoom, a Slack huddle). Each input is transcribed on its own by whisper.cpp, so a line says whether it was `me` (the microphone) or `others` (the apps). Nothing leaves the Mac.

## Running it

```sh
mise install
mise run whisper                      # builds whisper-server into .cache/whisper.cpp, once
uv sync
mise run listen -- --out ~/xlisten/standup
```

The models are those XmoQ fetches into `~/.cache/xmoq/whisper` (`mise run speech-model -- large-v3-turbo-q5_0` in XmoQ); `--model` names one, or a `.bin`. xlisten prints the line to give the agent on stderr:

```
Read <out>/AGENT.md and take part in the meeting as it says.
```

macOS asks, the first time, for the microphone and for recording the system's sound, for the terminal app xlisten runs in. Without the second, the tap hands only zeros and no `others` line comes.

`uv run xlisten transcribe FILE.wav` writes the lines of a recording on stdout, and `uv run xlisten record-mic FILE.wav` records the microphone, to compare settings on the same sound.

## The folder

The same form as XmoQ's `xmoq-agent` (XmoQ's `fixtures/agent-lines/README.md`): `events/<13 digits, ms>.jsonl`, a file a second, never changed after it is moved to its name; `inbox.txt`; `AGENT.md`, the guide for the agent (`src/xlisten/AGENT.md`). The lines are told in `AGENT.md`. `logs/` holds what whisper-server writes.

## Known gaps

- The sound of every app is tapped, not only the meeting's: music or a video playing is transcribed as `others`.
- With speakers rather than headphones, the microphone hears the others too, and their words come again as `me`.
- An utterance is cut where the sound gets quiet (`src/xlisten/segmenter.py`); a line comes at least `CLOSE_MS` after the speaker stops.
