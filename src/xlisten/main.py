"""xlisten: transcribes the microphone and the sound of meeting apps into a folder an agent reads.

    xlisten listen --out DIR          the two inputs into DIR/events/
    xlisten transcribe FILE.wav       a recording, as the lines `listen` would write, on stdout
    xlisten record-mic FILE.wav       the microphone into a WAV file, until Ctrl-C
    xlisten latency DIR               how long after the end of speech its line came
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
import time
from importlib import resources
from pathlib import Path

import numpy as np

from xlisten.audio import read_wav, resample, wav_bytes
from xlisten.inputs import Microphone, SystemAudio
from xlisten.lines import EventWriter, encode, now_ms
from xlisten.segmenter import Segmenter, Utterance
from xlisten.servers import Enhancer, Whisper, whisper_model

# FIXME: measure small-q5_1 (XmoQ's default) against this on Japanese meetings, and keep the one
# that is good enough and keeps up. Both are in XmoQ's cache.
DEFAULT_MODEL = "large-v3-turbo-q5_0"
INBOX_LINE_BYTES = 4096


def log(message: str) -> None:
    print(f"xlisten: {message}", file=sys.stderr, flush=True)


class Transcriber:
    """One thread that hands utterances to whisper in the order they ended, both inputs together.
    WHY one: whisper-server reads one request at a time anyway."""

    def __init__(self, whisper: Whisper, enhancer: Enhancer | None, writer: EventWriter,
                 keep_audio: Path | None) -> None:
        self._whisper = whisper
        self._enhancer = enhancer
        self._writer = writer
        self._keep = keep_audio
        self._queue: queue.Queue[tuple[str, Utterance] | None] = queue.Queue()
        self._number = 0
        self._thread = threading.Thread(target=self._run, name="transcriber", daemon=True)
        self._thread.start()

    def put(self, source: str, u: Utterance) -> None:
        self._queue.put((source, u))

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()

    def _keep_wav(self, name: str, samples: np.ndarray) -> None:
        if self._keep is not None:
            self._keep.mkdir(parents=True, exist_ok=True)
            (self._keep / name).write_bytes(wav_bytes(samples))

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            source, u = item
            samples = resample(u.samples, u.rate)
            self._keep_wav(f"{u.start_ms}-{source}.wav", samples)
            seconds = len(samples) / 16000
            took: list[str] = []
            try:
                # WHY the microphone only: RE-USE takes noise and the echo of a room out of a
                # recording; the sound of the apps comes as the meeting app sent it, with none.
                if self._enhancer is not None and source == "me":
                    began = time.monotonic()
                    samples = self._enhancer.enhance(samples)
                    t = time.monotonic() - began
                    took.append(f"RE-USE {t:.2f}s ({t / seconds:.2f} of its length)")
                    self._keep_wav(f"{u.start_ms}-{source}-enhanced.wav", samples)
                began = time.monotonic()
                text = self._whisper.transcribe(samples)
                took.append(f"whisper {time.monotonic() - began:.2f}s")
            except (OSError, ValueError) as e:
                self._writer.write({"type": "error", "message": f"{source}: {e}"})
                continue
            log(f"{source}: {seconds:.1f}s of speech, {', '.join(took)}, "
                f"line {(now_ms() - u.end_ms) / 1000:.2f}s after it ended")
            if not text:
                continue
            self._writer.write({
                "type": "speech", "number": self._number, "from": source, "text": text,
                "start_ms": u.start_ms, "end_ms": u.end_ms,
            })
            self._number += 1


class Input:
    """Takes blocks off the thread of the audio library and cuts them into utterances."""

    def __init__(self, source: str, transcriber: Transcriber) -> None:
        self.source = source
        self._transcriber = transcriber
        self._blocks: queue.SimpleQueue[tuple[np.ndarray, float, int] | None] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name=source, daemon=True)
        self._thread.start()

    def block(self, samples: np.ndarray, end_ms: float, rate: int) -> None:
        self._blocks.put((samples, end_ms, rate))

    def close(self) -> None:
        self._blocks.put(None)
        self._thread.join()

    def _run(self) -> None:
        seg: Segmenter | None = None
        while (item := self._blocks.get()) is not None:
            samples, end_ms, rate = item
            if seg is None or seg.rate != rate:
                seg = Segmenter(rate)
            for u in seg.push(samples, end_ms):
                self._transcriber.put(self.source, u)
        if seg is not None and (u := seg.finish()) is not None:
            self._transcriber.put(self.source, u)


class Inbox:
    """Lines added to inbox.txt become `note` lines. What is in it at the start is passed over."""

    def __init__(self, path: Path, writer: EventWriter) -> None:
        self._path = path
        self._writer = writer
        path.touch()
        self._pos = path.stat().st_size
        self._partial = b""

    def poll(self) -> None:
        size = self._path.stat().st_size
        if size < self._pos:  # written anew: read it from the start
            self._pos, self._partial = 0, b""
        if size == self._pos:
            return
        with self._path.open("rb") as f:
            f.seek(self._pos)
            data = self._partial + f.read(size - self._pos)
        self._pos = size
        *whole, self._partial = data.split(b"\n")
        for raw in whole:
            if len(raw) > INBOX_LINE_BYTES:
                self._writer.write({"type": "error",
                                    "message": f"a line of inbox.txt over {INBOX_LINE_BYTES} bytes is not read"})
                continue
            text = raw.decode("utf-8", errors="replace").rstrip("\r")
            if text.strip():
                self._writer.write({"type": "note", "text": text})


def write_agent_md(out: Path) -> None:
    text = resources.files("xlisten").joinpath("AGENT.md").read_text(encoding="utf-8")
    (out / "AGENT.md").write_text(text, encoding="utf-8")


def listen(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_agent_md(out)
    writer = EventWriter(out)
    whisper = Whisper(whisper_model(args.model), out / "logs" / "whisper-server.log")
    enhancer = Enhancer(out / "logs" / "speech-serve.log") if args.enhance and args.mic else None
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    keep = Path(args.keep_audio).expanduser() if args.keep_audio else None
    transcriber = Transcriber(whisper, enhancer, writer, keep)
    if keep is not None:
        log(f"each utterance is kept in {keep}")
    writer.write({"type": "reader", "state": "began", "model": args.model,
                  "enhance": enhancer is not None,
                  "inputs": [s for s, on in (("me", args.mic), ("others", args.system)) if on]})
    inputs: list[Input] = []
    devices: list[Microphone | SystemAudio] = []
    if args.mic:
        mine = Input("me", transcriber)
        mic = Microphone(lambda s, t: mine.block(s, t, mic.rate), args.mic_device)
        log(f"microphone: {mic.name}, {mic.rate} Hz")
        inputs.append(mine)
        devices.append(mic)
    if args.system:
        theirs = Input("others", transcriber)
        system = SystemAudio(lambda s, t: theirs.block(s, t, system.rate))
        inputs.append(theirs)
        devices.append(system)
    for d in devices:
        d.start()
    inbox = Inbox(out / "inbox.txt", writer)
    log(f"Read {out}/AGENT.md and take part in the meeting as it says.")

    began = time.monotonic()
    while not stop.is_set():
        stop.wait(1.0)
        inbox.poll()
        writer.flush()
        if args.seconds and time.monotonic() - began >= args.seconds:
            break

    for d in devices:
        d.stop()
    for i in inputs:
        i.close()
    transcriber.close()
    whisper.stop()
    if enhancer is not None:
        enhancer.stop()
    writer.write({"type": "reader", "state": "ended"})
    writer.flush()
    return 0


def transcribe(args: argparse.Namespace) -> int:
    """Plays a recording through the segmenter as fast as it reads, and prints `speech` lines whose
    `start_ms` and `end_ms` are from the start of the file."""
    samples, rate = read_wav(Path(args.file))
    whisper = Whisper(whisper_model(args.model), Path(args.log))
    enhancer = Enhancer(Path(args.log)) if args.enhance else None
    seg = Segmenter(rate)
    block = rate // 10
    utterances: list[Utterance] = []
    for i in range(0, len(samples), block):
        part = samples[i : i + block]
        utterances += seg.push(part, (i + len(part)) * 1000 / rate)
    if (u := seg.finish()) is not None:
        utterances.append(u)
    took_whisper = took_enhance = speech = 0.0
    for n, u in enumerate(utterances):
        part = resample(u.samples, u.rate)
        speech += len(part) / 16000
        if enhancer is not None:
            began = time.monotonic()
            part = enhancer.enhance(part)
            took_enhance += time.monotonic() - began
        began = time.monotonic()
        text = whisper.transcribe(part)
        took_whisper += time.monotonic() - began
        print(encode({"type": "speech", "number": n, "from": args.source, "text": text,
                      "start_ms": u.start_ms, "end_ms": u.end_ms}), flush=True)
    whisper.stop()
    if enhancer is not None:
        enhancer.stop()
    enhanced = (f", RE-USE {took_enhance:.2f}s ({took_enhance / speech:.2f} of their length)"
                if enhancer is not None else "")
    log(f"{len(samples) / rate:.1f}s of audio, {len(utterances)} utterances of {speech:.1f}s"
        f"{enhanced}, whisper {took_whisper:.2f}s")
    return 0


def record_mic(args: argparse.Namespace) -> int:
    blocks: list[np.ndarray] = []
    mic = Microphone(lambda s, _t: blocks.append(s), args.mic_device)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    mic.start()
    log(f"recording {mic.name} at {mic.rate} Hz into {args.file}; Ctrl-C to stop")
    stop.wait(args.seconds or None)
    mic.stop()
    samples = np.concatenate(blocks) if blocks else np.zeros(0, np.float32)
    Path(args.file).write_bytes(wav_bytes(samples, mic.rate))
    log(f"{len(samples) / mic.rate:.1f}s written")
    return 0


def latency(args: argparse.Namespace) -> int:
    """How long after the end of each utterance its `speech` line was written (`at - end_ms`)."""
    import json
    import statistics

    by: dict[str, list[int]] = {}
    for f in sorted((Path(args.out) / "events").glob("[0-9]*.jsonl")):
        for raw in f.read_text(encoding="utf-8").splitlines():
            line = json.loads(raw)
            if line["type"] == "speech":
                by.setdefault(line["from"], []).append(line["at"] - line["end_ms"])
    for source, ms in sorted(by.items()):
        print(f"{source}: {len(ms)} lines, median {statistics.median(ms) / 1000:.2f}s, "
              f"max {max(ms) / 1000:.2f}s")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="xlisten")
    sub = p.add_subparsers(dest="command", required=True)

    lp = sub.add_parser("listen", help="the microphone and the apps' sound into --out")
    lp.add_argument("--out", required=True)
    lp.add_argument("--model", default=DEFAULT_MODEL,
                    help="a name of XmoQ's cache (tiny, small-q5_1, large-v3-turbo-q5_0) or a .bin")
    lp.add_argument("--no-mic", dest="mic", action="store_false")
    lp.add_argument("--no-system", dest="system", action="store_false")
    lp.add_argument("--mic-device", default=None, help="a name or number of sounddevice")
    lp.add_argument("--no-enhance", dest="enhance", action="store_false",
                    help="hand the microphone to whisper as it is, without RE-USE")
    lp.add_argument("--keep-audio", default=None,
                    help="a folder to keep each utterance in as WAV, with what RE-USE made of it; "
                         "none is kept without it")
    lp.add_argument("--seconds", type=float, default=0, help="stop after this many seconds")
    lp.set_defaults(run=listen)

    tp = sub.add_parser("transcribe", help="a WAV file, as the lines listen would write")
    tp.add_argument("file")
    tp.add_argument("--model", default=DEFAULT_MODEL)
    tp.add_argument("--source", default="me")
    tp.add_argument("--enhance", action="store_true", help="through RE-USE first")
    tp.add_argument("--log", default="/dev/null", help="where whisper-server writes")
    tp.set_defaults(run=transcribe)

    rp = sub.add_parser("record-mic", help="the microphone into a WAV file")
    rp.add_argument("file")
    rp.add_argument("--mic-device", default=None)
    rp.add_argument("--seconds", type=float, default=0)
    rp.set_defaults(run=record_mic)

    ap = sub.add_parser("latency", help="seconds from the end of speech to its line, of --out")
    ap.add_argument("out")
    ap.set_defaults(run=latency)

    args = p.parse_args()
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
