"""OSC UDP listener for live caption text.

Listens for incoming OSC messages matching the WallSpace/A.EYE.ECHO format:
  /caption/text   [string]  — transcription segment
  /caption/clear  []        — clear caption buffer

Uses an internal queue to decouple the UDP server thread from the buffer lock,
preventing message drops during burst traffic (>10 msg/s).
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from typing import TYPE_CHECKING

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

if TYPE_CHECKING:
    from .buffer import TextBuffer

logger = logging.getLogger(__name__)

# Internal ingestion queue size — handles burst of up to 500 messages
_QUEUE_MAXSIZE = 500
# Socket receive buffer size (256 KB) for UDP burst absorption
_SOCKET_RCVBUF = 262144
# Throttle drop warnings to once per this interval
_DROP_WARN_INTERVAL = 5.0


class _ReusableOSCUDPServer(ThreadingOSCUDPServer):
    """ThreadingOSCUDPServer with SO_REUSEADDR + SO_REUSEPORT.

    Prevents [Errno 48] Address already in use when a pipeline is
    reloaded and the previous socket hasn't been fully released yet.
    Also increases the socket receive buffer for burst absorption.
    """

    allow_reuse_address = True

    def server_bind(self) -> None:
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass  # SO_REUSEPORT not available on all platforms
        # Increase receive buffer to absorb UDP bursts
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _SOCKET_RCVBUF)
        except OSError:
            pass
        super().server_bind()


class OscListener:
    """Receives caption text via OSC (UDP) and pushes it to a TextBuffer.

    Uses an internal queue to decouple the UDP handler from buffer writes,
    preventing drops under burst traffic.
    """

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
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._dropped_count: int = 0
        self._last_drop_warn: float = 0.0

        dispatcher = Dispatcher()
        dispatcher.map(address, self._on_text)
        dispatcher.map(clear_address, self._on_clear)

        try:
            self._server = _ReusableOSCUDPServer(("0.0.0.0", port), dispatcher)
        except OSError as e:
            logger.error(
                f"[WS Captions] Cannot bind OSC port {port}: {e}. "
                f"Is another instance already using this port? "
                f"Try a different osc_port value."
            )
            self._server = None
            self._thread = None
            self._drain_thread = None
            return

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name=f"osc-caption-{port}",
        )
        self._drain_thread = threading.Thread(
            target=self._drain_loop,
            daemon=True,
            name=f"osc-drain-{port}",
        )

    # ── Lifecycle ──

    def start(self) -> None:
        if self._running or self._server is None:
            return
        self._running = True
        self._thread.start()
        self._drain_thread.start()
        logger.info(f"[WS Captions] OSC listener started on 0.0.0.0:{self._port} (address: /caption/text)")

    def stop(self) -> None:
        if not self._running or self._server is None:
            return
        self._running = False
        self._server.shutdown()
        # Signal drain thread to exit
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._dropped_count > 0:
            logger.warning(
                f"[WS Captions] OSC listener stopped (port {self._port}). "
                f"Total dropped messages: {self._dropped_count}"
            )
        else:
            logger.info(f"[WS Captions] OSC listener stopped (port {self._port})")

    @property
    def dropped_count(self) -> int:
        """Total number of messages dropped due to queue overflow."""
        return self._dropped_count

    # ── Handlers (called from UDP server thread — must be fast) ──

    def _on_text(self, address: str, *args: object) -> None:
        """Handle /caption/text [string]. Enqueue without blocking."""
        if args:
            text = str(args[0])
            try:
                self._queue.put_nowait(text)
            except queue.Full:
                self._dropped_count += 1
                now = time.monotonic()
                if now - self._last_drop_warn >= _DROP_WARN_INTERVAL:
                    self._last_drop_warn = now
                    logger.warning(
                        f"[WS Captions] OSC queue full — dropped {self._dropped_count} message(s) total. "
                        f"Caption sources may be sending faster than {_QUEUE_MAXSIZE} msg/s."
                    )

    def _on_clear(self, address: str, *args: object) -> None:
        """Handle /caption/clear []."""
        self._buffer.clear()
        logger.debug("[WS Captions] Caption buffer cleared via OSC")

    # ── Drain loop (separate thread writes to buffer) ──

    def _drain_loop(self) -> None:
        """Pull messages from the queue and write to the buffer."""
        while self._running:
            try:
                text = self._queue.get(timeout=0.1)
                if text is None:
                    break  # Shutdown signal
                self._buffer.push(text)
            except queue.Empty:
                continue
