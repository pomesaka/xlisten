"""The lines of `events/` and the files that hold them, in the form of XmoQ's xmoq-agent.

The form is XmoQ's fixtures/agent-lines/README.md (read on 2026-11-05): one JSON object a line,
keys in the order of their names, `at` the writer's clock in ms; a file of `events/` is named
after the clock in 13 digits at the least, or one more than the newest there, and is moved to
its name only when it is whole.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

_EVENT_FILE = re.compile(r"^([0-9]+)\.jsonl$")

# WHY these on top of what json escapes (U+0000..U+001F): DEL and the C1 controls are acted on by
# some terminals as well, and an agent may print the lines in one.
_TERMINAL_CONTROLS = re.compile("[\u007f-\u009f]")


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def encode(line: dict[str, object]) -> str:
    text = json.dumps(line, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _TERMINAL_CONTROLS.sub(lambda m: f"\\u{ord(m.group()):04x}", text)


def next_event_file(names: list[str], clock_ms: int) -> str:
    newest = -1
    for name in names:
        m = _EVENT_FILE.match(name)
        if m is not None and len(m.group(1)) >= 13:
            newest = max(newest, int(m.group(1)))
    return f"{max(clock_ms, newest + 1):013d}.jsonl"


class EventWriter:
    """Gathers lines and writes them as one file of `events/` a second (`flush`)."""

    def __init__(self, out: Path, *, echo: bool = True) -> None:
        self._events = out / "events"
        self._events.mkdir(parents=True, exist_ok=True)
        self._pending: list[str] = []
        self._lock = threading.Lock()
        self._echo = echo

    def write(self, line: dict[str, object]) -> None:
        text = encode({**line, "at": now_ms()})
        with self._lock:
            self._pending.append(text)
        if self._echo:
            print(text, flush=True)

    def flush(self) -> None:
        with self._lock:
            lines, self._pending = self._pending, []
        if not lines:
            return
        name = next_event_file(os.listdir(self._events), now_ms())
        tmp = self._events / f".{name}.tmp"
        tmp.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        tmp.rename(self._events / name)
