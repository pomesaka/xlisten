# The first runs of xlisten (2026-11-05)

On an Apple M3 Pro with 18 GB, macOS 14.4: whisper.cpp v1.9.5 (Metal), speech.cpp v0.10.1 (Metal), the models of XmoQ's cache. Each was run once; the seconds are those xlisten printed.

## Lines from both inputs

A stream played by Chrome through the speakers of the monitor (LG ULTRAWIDE) for 60 s, `listen` with both inputs, before RE-USE and Silero VAD were put in: 12 `speech` lines, 9 `me` (the microphone hearing the speakers) and 3 `others` (the tap). The tap handed only zeros until the person allowed the recording of the system's sound; no session had to be started again after it.

## Latency

The seconds from the end of the speech of an utterance to its line (`xlisten latency`): the quiet that closes it (`CLOSE_MS`, 0.6 s), RE-USE on the microphone, whisper, and the wait behind the other input.

| Run | `me` | `others` |
|---|---|---|
| Without RE-USE, cut at 20 s | median 1.46 s, max 1.58 s (9 lines) | median 0.94 s (3 lines) |
| With RE-USE and VAD, cut at 10 s, 40 s | 1.87 to 2.54 s (8 utterances) | 0.85 to 2.13 s (5 utterances) |

Whisper (large-v3-turbo-q5_0) took 0.78 to 0.89 s for an utterance, whatever its length up to 20 s; small-q5_1 took about half of it for 3 sentences of `say` (1.30 s against 2.89 s for the three), and wrote them without some of the commas.

Through a stream with music under the voices, the sound never got quiet, and an utterance of `others` was always cut at the longest, then 20 s: a line came only every 20 s. Hence 10 s, at the quietest frame of the last 3 s.

## RE-USE on the M3 Pro

RE-USE took 0.31 to 0.37 of the length of each utterance (14 utterances of 1.4 to 4.0 s, live), and 0.37 for 3 sentences of `say` read from a file (5.49 s for 15.0 s): it keeps up, with time to spare. It is not compared with no RE-USE: whether it makes the transcript better is not known. The person decided to keep it on the microphone without comparing it.

## Words whisper writes of its own

Over sound with no voice, whisper writes a line of its own, `ご視聴ありがとうございました` above all. Without Silero VAD, one came from 1.4 s of the microphone. With Silero VAD in whisper-server and RE-USE before it, 3 of 9 lines of the microphone were that line, where the microphone heard `say` faintly from the monitor's speakers; the tap's lines of the same `say` were right. Not run: which of RE-USE or the faint sound brings them.

The person then spoke into the microphone for about 12 s of a run of 30 s (the stream stopped): 8 utterances, none of them that line. The same utterances were read again by whisper without RE-USE and with it (`--keep-audio`); the 4 with no voice were empty both ways, and the 4 with one differed little: `もう始まってるのか?` without against `もう始まっているのか` with, `テストですね` against `テストですね。`, `うん` against `ん`, `はい` against `え`. So few words say nothing about which is better.
