"""WebSocket listener for live caption text.

Accepts plain-text messages or JSON ``{"text": "...", "speaker": "..."}``.
Runs an asyncio event loop in a daemon thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import TYPE_CHECKING

import websockets

if TYPE_CHECKING:
    from .buffer import TextBuffer

logger = logging.getLogger(__name__)


class WebSocketListener:
    """Receives caption text via WebSocket and pushes it to a TextBuffer."""

    def __init__(self, port: int, buffer: TextBuffer) -> None:
        self._port = port
        self._buffer = buffer
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    # ── Lifecycle ──

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=f"ws-caption-{self._port}",
        )
        self._thread.start()
        logger.info(f"[WS Captions] WebSocket listener started on port {self._port}")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        logger.info(f"[WS Captions] WebSocket listener stopped (port {self._port})")

    # ── Internal ──

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception:
            logger.exception("[WS Captions] WebSocket listener error")

    async def _serve(self) -> None:
        try:
            async with websockets.serve(self._handler, "0.0.0.0", self._port):
                logger.info(f"[WS Captions] WebSocket server listening on ws://0.0.0.0:{self._port}")
                while self._running:
                    await asyncio.sleep(0.5)
        except OSError as exc:
            logger.error(f"[WS Captions] WebSocket port {self._port} unavailable: {exc}")

    async def _handler(self, websocket: websockets.WebSocketServerProtocol) -> None:
        peer = websocket.remote_address
        logger.info(f"[WS Captions] WebSocket client connected: {peer}")
        try:
            async for message in websocket:
                text = self._parse_message(message)
                if text:
                    self._buffer.push(text)
        except websockets.ConnectionClosed:
            pass
        logger.info(f"[WS Captions] WebSocket client disconnected: {peer}")

    @staticmethod
    def _parse_message(message: str | bytes) -> str:
        """Parse a WebSocket message as plain text or JSON.

        Accepts:
        - Plain text string: used directly
        - JSON: ``{"text": "...", "speaker": "..."}`` — ``text`` field extracted
        """
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        message = message.strip()
        if not message:
            return ""
        # Try JSON first
        if message.startswith("{"):
            try:
                data = json.loads(message)
                return str(data.get("text", "")).strip()
            except (json.JSONDecodeError, AttributeError):
                pass
        # Fall back to plain text
        return message
