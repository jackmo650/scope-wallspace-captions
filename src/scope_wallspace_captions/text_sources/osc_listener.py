"""OSC UDP listener for live caption text.

Listens for incoming OSC messages matching the WallSpace/A.EYE.ECHO format:
  /caption/text   [string]  — transcription segment
  /caption/clear  []        — clear caption buffer
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

if TYPE_CHECKING:
    from .buffer import TextBuffer

logger = logging.getLogger(__name__)


class OscListener:
    """Receives caption text via OSC (UDP) and pushes it to a TextBuffer."""

    def __init__(
        self,
        port: int,
        buffer: TextBuffer,
        address: str = "/caption/text",
        clear_address: str = "/caption/clear",
    ) -> None:
        self._port = port
        self._buffer = buffer
        self._running = False

        dispatcher = Dispatcher()
        dispatcher.map(address, self._on_text)
        dispatcher.map(clear_address, self._on_clear)

        try:
            self._server = ThreadingOSCUDPServer(("0.0.0.0", port), dispatcher)
        except OSError as e:
            logger.error(
                f"[WS Captions] Cannot bind OSC port {port}: {e}. "
                f"Is another instance already using this port? "
                f"Try a different osc_port value."
            )
            self._server = None
            self._thread = None
            return

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name=f"osc-caption-{port}",
        )

    # ── Lifecycle ──

    def start(self) -> None:
        if self._running or self._server is None:
            return
        self._running = True
        self._thread.start()
        logger.info(f"[WS Captions] OSC listener started on 0.0.0.0:{self._port} (address: /caption/text)")

    def stop(self) -> None:
        if not self._running or self._server is None:
            return
        self._running = False
        self._server.shutdown()
        logger.info(f"[WS Captions] OSC listener stopped (port {self._port})")

    # ── Handlers ──

    def _on_text(self, address: str, *args: object) -> None:
        """Handle /caption/text [string]."""
        if args:
            text = str(args[0])
            self._buffer.push(text)

    def _on_clear(self, address: str, *args: object) -> None:
        """Handle /caption/clear []."""
        self._buffer.clear()
        logger.debug("[WS Captions] Caption buffer cleared via OSC")
