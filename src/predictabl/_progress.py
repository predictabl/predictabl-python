"""Step-by-step progress for long operations (reading data, planning, training).

In a terminal the current step updates in place with a spinner; elsewhere
(notebooks, logs, CI) each change is printed once on its own line, so output
stays readable when it is not a TTY. `quiet=True` prints nothing; a callback
receives every update either way.
"""

from __future__ import annotations

import itertools
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Update:
    step: str  # e.g. "Training"
    status: str  # "running" | "done" | "failed"
    detail: str = ""


Callback = Callable[[Update], None]

_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Progress:
    def __init__(
        self, *, quiet: bool = False, on_update: Callback | None = None, stream=None
    ):
        self.quiet = quiet
        self.on_update = on_update
        self.stream = stream or sys.stderr
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._spin = itertools.cycle(_SPIN)
        self._current: str | None = None
        self._last_line = ""
        self._last_printed: tuple[str, str] | None = None
        self._started = time.monotonic()

    # ------------------------------------------------------------- reporting

    def running(self, step: str, detail: str = "") -> None:
        if self._current and self._current != step:
            self.done(self._current)
        self._current = step
        self._emit(Update(step, "running", detail))

    def done(self, step: str | None = None, detail: str = "") -> None:
        step = step or self._current
        if step is None:
            return
        self._emit(Update(step, "done", detail))
        if step == self._current:
            self._current = None

    def failed(self, step: str | None = None, detail: str = "") -> None:
        step = step or self._current or "Error"
        self._emit(Update(step, "failed", detail))
        self._current = None

    def tick(self) -> None:
        """Advance the spinner while waiting (terminal only)."""
        if self.tty and not self.quiet and self._last_line:
            self._write(self._last_line, final=False)

    # ---------------------------------------------------------------- output

    def _emit(self, u: Update) -> None:
        if self.on_update is not None:
            try:
                self.on_update(u)
            except Exception:  # noqa: BLE001 - a user callback must not break a build
                pass
        if self.quiet:
            return
        mark = {"done": "✓", "failed": "✗"}.get(u.status)
        line = f"{u.step:<14} {u.detail}".rstrip()
        if self.tty:
            if mark:
                self._write(f"{mark} {line}", final=True)
                self._last_line = ""
            else:
                self._last_line = line
                self._write(line, final=False)
            return
        # Not a terminal: one line per distinct change, never per poll.
        key = (u.status, line)
        if key == self._last_printed:
            return
        self._last_printed = key
        print(f"{mark or '…'} {line}", file=self.stream, flush=True)

    def _write(self, line: str, *, final: bool) -> None:
        if final:
            text = f"\r\033[2K{line}\n"
        else:
            text = f"\r\033[2K{next(self._spin)} {line}"
        self.stream.write(text)
        self.stream.flush()
