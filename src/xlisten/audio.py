"""Samples as float32 numpy arrays in [-1, 1], mono."""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path

import numpy as np

WHISPER_RATE = 16_000


def to_mono(interleaved: np.ndarray, channels: int) -> np.ndarray:
    if channels == 1:
        return interleaved
    return interleaved.reshape(-1, channels).mean(axis=1, dtype=np.float32)


def resample(samples: np.ndarray, rate: int, to: int = WHISPER_RATE) -> np.ndarray:
    """Resamples a whole utterance: a windowed-sinc low-pass below the new Nyquist, then linear
    interpolation. WHY not a polyphase library: numpy is the only dependency for audio, and an
    utterance is resampled once, whole, so no state is carried between calls."""
    if rate == to:
        return samples.astype(np.float32, copy=False)
    if to < rate:
        cutoff = 0.45 * to / rate
        taps = np.arange(-32, 33)
        kernel = 2 * cutoff * np.sinc(2 * cutoff * taps) * np.hamming(len(taps))
        samples = np.convolve(samples, kernel / kernel.sum(), mode="same")
    n = int(round(len(samples) * to / rate))
    positions = np.arange(n) * (rate / to)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)


def wav_bytes(samples: np.ndarray, rate: int = WHISPER_RATE) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    return parse_wav(path.read_bytes())


def parse_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Reads 16-bit integer or 32-bit float PCM, any number of channels, as mono.
    WHY by hand: the wave module of Python 3.12 refuses float PCM (format 3), which is what
    catap and many recorders write."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    pos = 12
    fmt: tuple[int, int, int, int] | None = None
    while pos + 8 <= len(data):
        cid, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            tag, channels, rate = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
            if tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: the real tag opens the sub-format GUID
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, channels, rate, bits)
        elif cid == b"data":
            if fmt is None:
                raise ValueError("data before fmt")
            tag, channels, rate, bits = fmt
            if tag == 1 and bits == 16:
                samples = np.frombuffer(body, "<i2").astype(np.float32) / 32768.0
            elif tag == 3 and bits == 32:
                samples = np.frombuffer(body, "<f4").astype(np.float32)
            else:
                raise ValueError(f"format {tag} with {bits} bits is not read")
            return to_mono(samples, channels), rate
        pos += 8 + size + (size & 1)
    raise ValueError("no data")
