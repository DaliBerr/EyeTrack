from __future__ import annotations

import os
import sys
from typing import Optional


class NonBlockingTerminalReader:
    def __init__(self) -> None:
        self._is_windows = os.name == "nt"
        self._fd: Optional[int] = None
        self._old_term_settings = None

    def __enter__(self) -> "NonBlockingTerminalReader":
        if not self._is_windows and sys.stdin.isatty():
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._old_term_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_type, exc_val, exc_tb
        if not self._is_windows and self._fd is not None and self._old_term_settings is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term_settings)
        self._fd = None
        self._old_term_settings = None

    def poll_key(self) -> Optional[str]:
        if self._is_windows:
            import msvcrt

            if not msvcrt.kbhit():
                return None
            key = msvcrt.getwch()
            return key.lower() if key else None

        if not sys.stdin.isatty():
            return None

        import select

        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None

        key = sys.stdin.read(1)
        return key.lower() if key else None
