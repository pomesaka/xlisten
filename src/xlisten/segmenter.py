"""Cuts a stream of samples into utterances by their loudness.

WHY loudness and not a model of voice activity: it is enough to find where somebody stops, which
is all whisper needs to be handed, and it adds no model or binary. Music or typing loud enough is
taken for speech; whisper then writes what it hears, often nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_MS = 30
# A frame is speech where it is this much louder than the floor of the noise, and above ABSOLUTE.
OVER_FLOOR_DB = 10.0
ABSOLUTE_DB = -55.0
# Frames of speech in a row that open an utterance, and the time before them it begins with.
OPEN_FRAMES = 3
PRE_ROLL_MS = 300
# Quiet that ends an utterance; the latency of a line is at least this.
CLOSE_MS = 600
# WHY 20 s: whisper reads 30 s at a time; a longer utterance is cut and read in pieces.
LONGEST_MS = 20_000
SHORTEST_MS = 400


@dataclass(frozen=True)
class Utterance:
    samples: np.ndarray
    rate: int
    start_ms: int
    # The end of the last frame of speech, before the quiet that closed it.
    end_ms: int


def _db(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
    return 20.0 * np.log10(max(rms, 1e-10))


class Segmenter:
    def __init__(self, rate: int) -> None:
        self.rate = rate
        self._frame = rate * FRAME_MS // 1000
        self._rest = np.zeros(0, np.float32)
        self._rest_start_ms = 0.0
        self._floor_db = -60.0
        self._pre: list[np.ndarray] = []
        self._run = 0
        self._open: list[np.ndarray] | None = None
        self._open_start_ms = 0.0
        self._first_speech_ms = 0.0
        self._last_speech_end_ms = 0.0
        self._quiet_ms = 0

    def push(self, samples: np.ndarray, end_ms: float) -> list[Utterance]:
        """`end_ms`: the wall clock of the last sample of `samples`."""
        start_ms = end_ms - len(samples) * 1000.0 / self.rate
        if len(self._rest) == 0:
            self._rest_start_ms = start_ms
        buf = np.concatenate([self._rest, samples])
        t = self._rest_start_ms
        out: list[Utterance] = []
        whole = len(buf) // self._frame * self._frame
        for i in range(0, whole, self._frame):
            frame = buf[i : i + self._frame]
            u = self._frame_in(frame, t)
            if u is not None:
                out.append(u)
            t += FRAME_MS
        self._rest = buf[whole:]
        self._rest_start_ms = t
        return out

    def _frame_in(self, frame: np.ndarray, t: float) -> Utterance | None:
        db = _db(frame)
        speech = db > max(self._floor_db + OVER_FLOOR_DB, ABSOLUTE_DB)
        if not speech:
            # The floor follows quiet quickly downwards and slowly upwards.
            rate = 0.2 if db < self._floor_db else 0.01
            self._floor_db += rate * (db - self._floor_db)

        if self._open is None:
            self._pre.append(frame)
            del self._pre[: -(PRE_ROLL_MS // FRAME_MS + OPEN_FRAMES)]
            self._run = self._run + 1 if speech else 0
            if self._run >= OPEN_FRAMES:
                self._open = list(self._pre)
                self._open_start_ms = t + FRAME_MS - len(self._pre) * FRAME_MS
                self._pre = []
                self._first_speech_ms = t + FRAME_MS - OPEN_FRAMES * FRAME_MS
                self._last_speech_end_ms = t + FRAME_MS
                self._quiet_ms = 0
            return None

        self._open.append(frame)
        if speech:
            self._last_speech_end_ms = t + FRAME_MS
            self._quiet_ms = 0
        else:
            self._quiet_ms += FRAME_MS
        length_ms = len(self._open) * FRAME_MS
        if self._quiet_ms >= CLOSE_MS or length_ms >= LONGEST_MS:
            return self._close()
        return None

    def _close(self) -> Utterance | None:
        frames, self._open = self._open, None
        self._run = 0
        assert frames is not None
        speech_ms = self._last_speech_end_ms - self._first_speech_ms
        if speech_ms < SHORTEST_MS:
            return None
        return Utterance(
            samples=np.concatenate(frames),
            rate=self.rate,
            start_ms=round(self._open_start_ms),
            end_ms=round(self._last_speech_end_ms),
        )

    def finish(self) -> Utterance | None:
        if self._open is None:
            return None
        return self._close()
