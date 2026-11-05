from pathlib import Path

import numpy as np
import pytest

from xlisten.audio import read_wav, resample, wav_bytes
from xlisten.segmenter import Segmenter

RATE = 48_000


def tone(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def quiet(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(seconds * RATE)) * 1e-4).astype(np.float32)


def run(samples: np.ndarray, block: int = 1024) -> list[tuple[int, int]]:
    seg = Segmenter(RATE)
    out = []
    for i in range(0, len(samples), block):
        part = samples[i : i + block]
        out += seg.push(part, (i + len(part)) * 1000 / RATE)
    if (u := seg.finish()) is not None:
        out.append(u)
    return [(u.start_ms, u.end_ms) for u in out]


@pytest.mark.parametrize(
    ("samples", "expected", "end_within_ms"),
    [
        pytest.param(np.concatenate([quiet(2), tone(1.5), quiet(2)]), [(2000, 3500)], 30,
                     id="one utterance, from its first frame to its last"),
        pytest.param(np.concatenate([quiet(2), tone(1), quiet(0.3), tone(1), quiet(2)]),
                     [(2000, 4300)], 30, id="a pause shorter than the close is inside"),
        pytest.param(np.concatenate([quiet(2), tone(1), quiet(1), tone(1), quiet(2)]),
                     [(2000, 3000), (4000, 5000)], 30, id="a pause of the close or more ends it"),
        pytest.param(np.concatenate([quiet(2), tone(0.2), quiet(2)]), [], 30,
                     id="a click is no utterance"),
        pytest.param(np.concatenate([quiet(2), tone(1.5)]), [(2000, 3500)], 30,
                     id="the end of the stream ends it"),
        pytest.param(np.concatenate([quiet(2), tone(8.5), quiet(0.2), tone(5.5), quiet(2)]),
                     [(2000, 10500), (10500, 16200)], 200,
                     id="one that never gets quiet is cut at its quietest, and the rest goes on"),
    ],
)
def test_segmenter(samples: np.ndarray, expected: list[tuple[int, int]], end_within_ms: int) -> None:
    # The start may be up to the pre-roll (300 ms) and a frame early. The end is within a frame,
    # or, where it is cut, anywhere in the dip of 200 ms where it is quietest.
    got = run(samples)
    assert len(got) == len(expected)
    for (gs, ge), (es, ee) in zip(got, expected, strict=True):
        assert es - 330 <= gs <= es + end_within_ms - 30
        assert abs(ge - ee) <= end_within_ms


def test_resample_keeps_the_length_and_a_low_tone() -> None:
    out = resample(tone(1.0), RATE, 16_000)
    assert len(out) == 16_000
    assert np.max(np.abs(out[100:-100])) == pytest.approx(0.3, abs=0.01)


def test_a_wav_reads_back(tmp_path: Path) -> None:
    samples = tone(0.1)
    (tmp_path / "a.wav").write_bytes(wav_bytes(samples, RATE))
    back, rate = read_wav(tmp_path / "a.wav")
    assert rate == RATE
    assert np.max(np.abs(back - samples)) < 1e-4
