"""The servers xlisten runs as its children, each with a model loaded once: whisper-server of
whisper.cpp (transcription) and `speech serve` of speech.cpp (RE-USE, speech enhancement).

WHY servers and not a command for each utterance: a command would load its model again for every
line.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np

from xlisten.audio import WHISPER_RATE, parse_wav, wav_bytes

ROOT = Path(__file__).resolve().parents[2]
WHISPER_SERVER = ROOT / ".cache" / "whisper.cpp" / "bin" / "whisper-server"
SPEECH = ROOT / ".cache" / "speech.cpp" / "build" / "speech"
# Fetched by scripts/build-speech.sh.
RE_USE_MODEL = ROOT / ".cache" / "models" / "NVIDIA-NemotronLabs-RE-USE-10M-F32.gguf"
# The models XmoQ fetches with `mise run speech-model` (crates/speech/scripts/fetch-model.sh).
WHISPER_MODELS = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "xmoq" / "whisper"
VAD_MODEL = WHISPER_MODELS / "ggml-silero-v5.1.2.bin"


def whisper_model(name: str) -> Path:
    p = Path(name)
    if p.suffix == ".bin":
        return p
    return WHISPER_MODELS / f"ggml-{name}.bin"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _need(path: Path, how: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} is not there: {how}")


def _multipart(fields: dict[str, str], wav: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = b""
    for name, value in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                 f"{value}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"u.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
    body += wav + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


class _Child:
    def __init__(self, argv: list[str], health: str, log: Path) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        self._log = log.open("ab")
        self.process = subprocess.Popen(
            argv, stdout=self._log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"{argv[0]} ended with {self.process.returncode}; see {log}")
            try:
                with urllib.request.urlopen(health, timeout=1):
                    return
            except OSError:
                time.sleep(0.2)
        self.stop()
        raise TimeoutError(f"{argv[0]} did not come up; see {log}")

    def post(self, url: str, fields: dict[str, str], wav: bytes) -> bytes:
        body, content_type = _multipart(fields, wav)
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": content_type})
        with urllib.request.urlopen(req, timeout=120) as res:
            return bytes(res.read())

    def stop(self) -> None:
        # Stopped by its own PID: the child this process started.
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self._log.close()


class Whisper(_Child):
    def __init__(self, model: Path, log: Path) -> None:
        _need(WHISPER_SERVER, "run `mise run whisper` first")
        _need(model, "in XmoQ, `mise run speech-model -- <name>`")
        _need(VAD_MODEL, "in XmoQ, `mise run speech-model -- silero-v5.1.2`")
        self.port = _free_port()
        # WHY --vad: whisper writes words of its own ("ご視聴ありがとうございました") over music
        # and noise; Silero VAD hands it only the parts with a voice. Without it, a 1.4 s
        # utterance of the end of a stream came back as that line (2026-11-05).
        super().__init__(
            [str(WHISPER_SERVER), "-m", str(model), "-l", "ja", "--host", "127.0.0.1",
             "--port", str(self.port), "--no-timestamps", "--vad", "-vm", str(VAD_MODEL)],
            f"http://127.0.0.1:{self.port}/health", log,
        )

    def transcribe(self, samples: np.ndarray) -> str:
        raw = self.post(f"http://127.0.0.1:{self.port}/inference",
                        {"response_format": "json", "language": "ja", "temperature": "0.0"},
                        wav_bytes(samples, WHISPER_RATE))
        return str(json.loads(raw)["text"]).strip()


class Enhancer(_Child):
    """NVIDIA's RE-USE, through `speech serve` of speech.cpp."""

    def __init__(self, log: Path) -> None:
        _need(SPEECH, "run `mise run speech` first")
        _need(RE_USE_MODEL, "run `mise run speech` first")
        self.port = _free_port()
        super().__init__(
            [str(SPEECH), "serve", str(RE_USE_MODEL), "--host", "127.0.0.1",
             "--port", str(self.port)],
            f"http://127.0.0.1:{self.port}/health", log,
        )

    def enhance(self, samples: np.ndarray) -> np.ndarray:
        raw = self.post(f"http://127.0.0.1:{self.port}/v1/audio/enhancements",
                        {"response_format": "wav"}, wav_bytes(samples, WHISPER_RATE))
        out, rate = parse_wav(raw)
        if rate != WHISPER_RATE:
            raise ValueError(f"RE-USE answered at {rate} Hz, not {WHISPER_RATE}")
        return out
