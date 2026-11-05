"""The two inputs: the microphone (`me`) and the sound the apps of this Mac play (`others`).

Each hands blocks of mono float32 samples, with the wall clock of the last sample, to `on_block`
on a thread of its own library.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from xlisten.audio import to_mono
from xlisten.lines import now_ms

OnBlock = Callable[[np.ndarray, float], None]


class Microphone:
    def __init__(self, on_block: OnBlock, device: str | None = None) -> None:
        import sounddevice as sd

        info = sd.query_devices(device, kind="input")
        self.name = str(info["name"])
        self.rate = int(info["default_samplerate"])

        def callback(indata: np.ndarray, frames: int, _time: object, _status: object) -> None:
            on_block(indata[:, 0].copy(), float(now_ms()))

        self._stream = sd.InputStream(
            device=device, channels=1, samplerate=self.rate, dtype="float32", callback=callback,
        )

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()


class SystemAudio:
    """A process tap of every process's output (catap's global tap).

    WHY every process and not the meeting app's: which process of Slack (Electron) plays the
    sound of a huddle is not known yet; the global tap catches it whichever it is. Whatever else
    plays (music, a video) is transcribed as `others` too.
    """

    def __init__(self, on_block: OnBlock) -> None:
        from catap import AudioBuffer, RecordingSession

        def on_buffer(buffer: AudioBuffer) -> None:
            fmt = buffer.format
            if not fmt.is_float or fmt.bits_per_sample != 32:
                raise ValueError(f"the tap gives {fmt}, not float32")
            samples = to_mono(np.frombuffer(buffer.data, "<f4"), fmt.num_channels)
            self.rate = int(fmt.sample_rate)
            on_block(samples, float(now_ms()))

        self.rate = 0
        self._session = RecordingSession.from_system_audio(mono=True, on_buffer=on_buffer)

    def start(self) -> None:
        self._session.start()

    def stop(self) -> None:
        self._session.close()
