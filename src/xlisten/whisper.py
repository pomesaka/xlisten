"""whisper-server of whisper.cpp, run as a child and asked over HTTP.

WHY a server and not whisper-cli for each utterance: the server loads the model once; whisper-cli
would load it again for every line.
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

from xlisten.audio import WHISPER_RATE, wav_bytes

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / ".cache" / "whisper.cpp" / "bin" / "whisper-server"
# The models XmoQ fetches with `mise run speech-model` (crates/speech/scripts/fetch-model.sh).
MODEL_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "xmoq" / "whisper"


def model_path(name: str) -> Path:
    p = Path(name)
    if p.suffix == ".bin":
        return p
    return MODEL_DIR / f"ggml-{name}.bin"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class WhisperServer:
    def __init__(self, model: Path, log: Path) -> None:
        if not SERVER.exists():
            raise FileNotFoundError(f"{SERVER} is not there: run `mise run whisper` first")
        if not model.exists():
            raise FileNotFoundError(f"{model} is not there: in XmoQ, `mise run speech-model`")
        self.port = _free_port()
        log.parent.mkdir(parents=True, exist_ok=True)
        self._log = log.open("ab")
        self.process = subprocess.Popen(
            [str(SERVER), "-m", str(model), "-l", "ja", "--host", "127.0.0.1",
             "--port", str(self.port), "--no-timestamps"],
            stdout=self._log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )
        self._wait_ready()

    def _wait_ready(self, seconds: float = 60.0) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"whisper-server ended with {self.process.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1):
                    return
            except OSError:
                time.sleep(0.2)
        raise TimeoutError("whisper-server did not come up")

    def transcribe(self, samples: np.ndarray) -> str:
        boundary = uuid.uuid4().hex
        parts = [
            ("response_format", b"json"),
            ("language", b"ja"),
            ("temperature", b"0.0"),
        ]
        body = b""
        for name, value in parts:
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                     .encode() + value + b"\r\n")
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                 f"filename=\"u.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
        body += wav_bytes(samples, WHISPER_RATE) + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/inference", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=120) as res:
            return str(json.loads(res.read())["text"]).strip()

    def stop(self) -> None:
        # Stopped by its own PID, the child this process started.
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self._log.close()
