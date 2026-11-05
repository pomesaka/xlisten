import json
from pathlib import Path

import pytest

from xlisten.lines import EventWriter, encode, next_event_file
from xlisten.main import Inbox


# The rows of XmoQ's fixtures/agent-lines/event-files.tsv (read on 2026-11-05).
@pytest.mark.parametrize(
    ("names", "clock", "expected"),
    [
        pytest.param([], 1700000000000, "1700000000000.jsonl", id="none before"),
        pytest.param([], 5, "0000000000005.jsonl", id="a clock of fewer digits is padded to 13"),
        pytest.param(["1700000000000.jsonl"], 1700000000500, "1700000000500.jsonl",
                     id="a clock after the newest"),
        pytest.param(["1700000000000.jsonl"], 1700000000000, "1700000000001.jsonl",
                     id="the same millisecond as the newest"),
        pytest.param(["1700000000500.jsonl"], 1700000000000, "1700000000501.jsonl",
                     id="a clock that went back"),
        pytest.param(["1700000000000.jsonl", "1700000000900.jsonl", "1700000000100.jsonl"],
                     1700000000200, "1700000000901.jsonl",
                     id="the newest by its number, not by where it is listed"),
        pytest.param([".1700000009999.jsonl.tmp", "events.jsonl", "170000000099.jsonl",
                      "1700000099999.txt", "17000000x0000.jsonl"], 1700000000000,
                     "1700000000000.jsonl", id="other names are not counted"),
        pytest.param(["0000000000005.jsonl"], 3, "0000000000006.jsonl",
                     id="a padded name is counted by its number"),
    ],
)
def test_next_event_file(names: list[str], clock: int, expected: str) -> None:
    assert next_event_file(names, clock) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param({"type": "speech", "from": "me", "text": "こんにちは"},
                     '{"from":"me","text":"こんにちは","type":"speech"}',
                     id="keys in the order of their names, other characters as they are"),
        pytest.param({"text": "a\u001b[2Jb\nc"}, '{"text":"a\\u001b[2Jb\\nc"}',
                     id="an escape and a line break are escaped"),
        pytest.param({"text": "a\u007fb\u009bc"}, '{"text":"a\\u007fb\\u009bc"}',
                     id="DEL and the C1 controls are escaped"),
        pytest.param({"start_ms": 1792883451092}, '{"start_ms":1792883451092}',
                     id="a number without a fraction"),
    ],
)
def test_encode(line: dict[str, object], expected: str) -> None:
    assert encode(line) == expected


def test_a_flush_is_one_file_and_nothing_else(tmp_path: Path) -> None:
    w = EventWriter(tmp_path, echo=False)
    w.write({"type": "note", "text": "a"})
    w.write({"type": "note", "text": "b"})
    w.flush()
    w.flush()
    files = sorted((tmp_path / "events").iterdir())
    assert len(files) == 1
    texts = [json.loads(line)["text"] for line in files[0].read_text().splitlines()]
    assert texts == ["a", "b"]


def test_inbox_passes_over_what_was_there_and_waits_for_a_whole_line(tmp_path: Path) -> None:
    (tmp_path / "inbox.txt").write_text("before\n")
    w = EventWriter(tmp_path, echo=False)
    inbox = Inbox(tmp_path / "inbox.txt", w)
    with (tmp_path / "inbox.txt").open("a") as f:
        f.write("one\n\ntw")
    inbox.poll()
    with (tmp_path / "inbox.txt").open("a") as f:
        f.write("o\n")
    inbox.poll()
    w.flush()
    (file,) = (tmp_path / "events").iterdir()
    assert [json.loads(line)["text"] for line in file.read_text().splitlines()] == ["one", "two"]
