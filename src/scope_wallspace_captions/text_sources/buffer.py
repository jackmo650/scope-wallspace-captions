"""Thread-safe text ring buffer with rate-limited prompt output."""

from __future__ import annotations

import threading
import time
from collections import deque


class TextBuffer:
    """Thread-safe ring buffer for incoming transcription text.

    Separates two concerns:
    - Display: ``get_display_lines()`` always returns the freshest text (captions
      must be responsive).
    - Prompts: ``get_current_prompt()`` rate-limits output so the downstream
      diffusion model isn't whiplashed by every word.

    Tracks write/read sequence numbers so callers can detect unprocessed entries.
    """

    def __init__(self, capacity: int = 100) -> None:
        self._lock = threading.Lock()
        self._entries: deque[tuple[float, str]] = deque(maxlen=capacity)
        self._last_prompt_text: str = ""
        self._last_prompt_time: float = 0.0
        self._write_seq: int = 0
        self._read_seq: int = 0

    # ── Write ──

    def push(self, text: str) -> None:
        """Append a transcription segment (thread-safe)."""
        if not text or not text.strip():
            return
        with self._lock:
            self._entries.append((time.monotonic(), text.strip()))
            self._write_seq += 1

    def clear(self) -> None:
        """Remove all entries and reset prompt state."""
        with self._lock:
            self._entries.clear()
            self._last_prompt_text = ""
            self._last_prompt_time = 0.0
            self._read_seq = self._write_seq

    # ── Read ──

    @property
    def pending_count(self) -> int:
        """Number of entries pushed since last read."""
        with self._lock:
            return self._write_seq - self._read_seq

    def get_display_lines(self, max_lines: int = 3) -> list[str]:
        """Return the *max_lines* most recent text entries (always fresh)."""
        with self._lock:
            if not self._entries:
                return []
            texts = [t for _, t in self._entries]
            self._read_seq = self._write_seq
            return texts[-max_lines:]

    def get_latest_text(self) -> str:
        """Return the single most recent text entry, or empty string."""
        with self._lock:
            if not self._entries:
                return ""
            return self._entries[-1][1]

    def get_current_prompt(self, interval_sec: float = 2.0) -> str | None:
        """Return the latest text if *interval_sec* has elapsed since the last
        prompt change.  Returns ``None`` when throttled so the caller can skip
        the prompt update and let the model keep working on its current gen.
        """
        with self._lock:
            if not self._entries:
                return None
            latest_text = self._entries[-1][1]
            # Same text as last prompt — no update needed
            if latest_text == self._last_prompt_text:
                return None
            now = time.monotonic()
            if now - self._last_prompt_time < interval_sec:
                return None  # Throttled
            self._last_prompt_text = latest_text
            self._last_prompt_time = now
            self._read_seq = self._write_seq
            return latest_text

    def get_last_update_time(self) -> float:
        """Monotonic timestamp of the last push, or 0.0 if empty."""
        with self._lock:
            if not self._entries:
                return 0.0
            return self._entries[-1][0]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
