"""Caption Event System — convert raw text into structured visual triggers.

Instead of treating transcription as a flat string, this module parses it
into semantic events that downstream visual effects can react to:

    "Hello world!"  →  SENTENCE_START, WORD("Hello"), WORD("world"),
                        EXCLAMATION, SENTENCE_END

Event types drive different visual behaviours — flashes on words, colour
shifts on questions, intensity spikes on exclamations, fades on pauses.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CaptionEventType(str, Enum):
    """Structured event types derived from transcription text."""

    SENTENCE_START = "sentence_start"
    SENTENCE_END = "sentence_end"
    WORD = "word"
    QUESTION = "question"
    EXCLAMATION = "exclamation"
    PAUSE = "pause"
    EMPHASIS = "emphasis"
    SPEAKER_CHANGE = "speaker_change"


@dataclass(frozen=True)
class CaptionEvent:
    """A single parsed caption event."""

    type: CaptionEventType
    text: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for downstream consumption / output dict."""
        return {
            "type": self.type.value,
            "text": self.text,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ── Parser ──────────────────────────────────────────────────────────────────


class CaptionEventParser:
    """Stateful parser that converts transcription text into CaptionEvents.

    Compares each incoming text against previous state to detect deltas,
    pauses, speaker changes, and punctuation events.
    """

    def __init__(self, pause_threshold_sec: float = 2.0) -> None:
        self.pause_threshold_sec = pause_threshold_sec
        self._prev_text: str = ""
        self._prev_time: float = 0.0
        self._prev_speaker: str | None = None
        self._prev_words: list[str] = []

    def parse(self, text: str, speaker: str | None = None) -> list[CaptionEvent]:
        """Parse *text* into a list of events relative to previous state.

        Returns an empty list when the text hasn't changed.
        """
        now = time.monotonic()
        text = text.strip()
        if not text:
            return []

        # No change — nothing to emit
        if text == self._prev_text and speaker == self._prev_speaker:
            return []

        events: list[CaptionEvent] = []

        # ── Pause detection ──
        if self._prev_time > 0 and (now - self._prev_time) >= self.pause_threshold_sec:
            events.append(CaptionEvent(
                type=CaptionEventType.PAUSE,
                text="",
                timestamp=now,
                metadata={"gap_sec": round(now - self._prev_time, 2)},
            ))

        # ── Speaker change ──
        if speaker and speaker != self._prev_speaker and self._prev_speaker is not None:
            events.append(CaptionEvent(
                type=CaptionEventType.SPEAKER_CHANGE,
                text=speaker,
                timestamp=now,
                metadata={"previous_speaker": self._prev_speaker},
            ))

        # ── Word-level events ──
        words = text.split()
        new_words = self._get_new_words(words)

        if new_words:
            # Sentence start — first word or follows a sentence-ending punctuation
            if not self._prev_words or self._is_sentence_boundary():
                events.append(CaptionEvent(
                    type=CaptionEventType.SENTENCE_START,
                    text=new_words[0],
                    timestamp=now,
                ))

            for i, word in enumerate(new_words):
                clean = word.strip(".,!?;:\"'()[]{}—–-")

                # EMPHASIS — ALL CAPS words (3+ chars to avoid acronyms like "I")
                if clean.isupper() and len(clean) >= 3:
                    events.append(CaptionEvent(
                        type=CaptionEventType.EMPHASIS,
                        text=clean,
                        timestamp=now,
                        metadata={"word_index": len(self._prev_words) + i},
                    ))

                # WORD event
                events.append(CaptionEvent(
                    type=CaptionEventType.WORD,
                    text=clean,
                    timestamp=now,
                    metadata={
                        "word_index": len(self._prev_words) + i,
                        "word_count": len(words),
                    },
                ))

        # ── Punctuation events (from the full text tail) ──
        if text.endswith("?"):
            events.append(CaptionEvent(
                type=CaptionEventType.QUESTION,
                text=text,
                timestamp=now,
                metadata={"intensity": self._punctuation_intensity(text)},
            ))
        elif text.endswith("!"):
            events.append(CaptionEvent(
                type=CaptionEventType.EXCLAMATION,
                text=text,
                timestamp=now,
                metadata={"intensity": self._punctuation_intensity(text)},
            ))

        # Sentence end — period, question mark, or exclamation at end
        if text and text[-1] in ".?!":
            events.append(CaptionEvent(
                type=CaptionEventType.SENTENCE_END,
                text=text,
                timestamp=now,
            ))

        # ── Update state ──
        self._prev_text = text
        self._prev_time = now
        self._prev_speaker = speaker
        self._prev_words = words

        return events

    def reset(self) -> None:
        """Clear parser state."""
        self._prev_text = ""
        self._prev_time = 0.0
        self._prev_speaker = None
        self._prev_words = []

    # ── Helpers ──

    def _get_new_words(self, current_words: list[str]) -> list[str]:
        """Return words that are new relative to the previous word list."""
        prev_len = len(self._prev_words)
        if not self._prev_words:
            return current_words
        # If the current text starts with the previous words, the delta is the tail
        if current_words[:prev_len] == self._prev_words:
            return current_words[prev_len:]
        # Text changed entirely — treat all words as new
        return current_words

    def _is_sentence_boundary(self) -> bool:
        """Check if previous text ended with sentence-ending punctuation."""
        return bool(self._prev_text and self._prev_text[-1] in ".?!")

    @staticmethod
    def _punctuation_intensity(text: str) -> float:
        """Derive an intensity value (0–1) from punctuation density.

        Multiple exclamation marks ("WOW!!!") = higher intensity.
        """
        if not text:
            return 0.0
        count = 0
        for ch in reversed(text):
            if ch in "!?":
                count += 1
            else:
                break
        return min(count / 3.0, 1.0)


# ── Event Buffer ────────────────────────────────────────────────────────────


class CaptionEventBuffer:
    """Thread-safe ring buffer of CaptionEvents for visual reactivity."""

    def __init__(self, capacity: int = 200) -> None:
        self._lock = threading.Lock()
        self._events: deque[CaptionEvent] = deque(maxlen=capacity)

    def push(self, event: CaptionEvent) -> None:
        with self._lock:
            self._events.append(event)

    def push_many(self, events: list[CaptionEvent]) -> None:
        with self._lock:
            self._events.extend(events)

    def get_active_events(self, window_sec: float = 1.0) -> list[CaptionEvent]:
        """Return events within the last *window_sec* seconds."""
        cutoff = time.monotonic() - window_sec
        with self._lock:
            return [e for e in self._events if e.timestamp >= cutoff]

    def get_current_event_type(self) -> CaptionEventType | None:
        """Return the type of the most recent event, or None."""
        with self._lock:
            if not self._events:
                return None
            return self._events[-1].type

    def get_latest_events(self, count: int = 5) -> list[CaptionEvent]:
        """Return the *count* most recent events."""
        with self._lock:
            return list(self._events)[-count:]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
