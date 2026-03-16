"""WallSpace Captions pipeline — live transcription overlay + prompt forwarding.

Two variants:
- **Pre** (preprocessor): Text baked into frames *before* AI generation —
  the diffusion model sees and stylises the text.
- **Post** (postprocessor): Clean text overlay *after* AI generation —
  readable captions on top of AI visuals.

Both receive text from configurable sources (Scope prompt, manual, OSC,
WebSocket), parse it into caption events, render an overlay, and optionally
forward the text as prompts for downstream generation.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .events import CaptionEventBuffer, CaptionEventParser
from .overlay import OverlayConfig, load_font, render_caption_overlay
from .schema import WallspaceCaptionsPostConfig, WallspaceCaptionsPreConfig
from .text_sources import OscListener, TextBuffer, WebSocketListener

logger = logging.getLogger(__name__)

# ── Position Presets ────────────────────────────────────────────────────────

POSITION_PRESETS: dict[str, tuple[float, float]] = {
    "bottom": (50.0, 90.0),
    "top": (50.0, 10.0),
    "center": (50.0, 50.0),
}


# ── Base Pipeline ───────────────────────────────────────────────────────────


class _WallspaceCaptionsBase(Pipeline):
    """Shared implementation for Pre and Post caption pipelines."""

    def __init__(self, **kwargs: Any) -> None:
        # ── Device ──
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info(f"[WS Captions] Using device: {self.device}")

        # ── Load-time params ──
        self._text_source: str = kwargs.get("text_source", "prompt")
        self._font_path: str = kwargs.get("font_path", "")
        self._osc_address: str = kwargs.get("osc_address", "/caption/text")

        # ── Shared state ──
        self._buffer = TextBuffer(capacity=100)
        self._event_parser = CaptionEventParser(
            pause_threshold_sec=kwargs.get("pause_threshold", 2.0),
        )
        self._event_buffer = CaptionEventBuffer(capacity=200)

        # ── Font (loaded once, reloaded if size changes) ──
        self._current_font_size: int = 48
        self._font = load_font(self._font_path, self._current_font_size)

        # ── Listeners ──
        self._osc_listener: OscListener | None = None
        self._ws_listener: WebSocketListener | None = None

        if self._text_source == "osc":
            osc_port = int(kwargs.get("osc_port", 9001))
            self._osc_listener = OscListener(
                port=osc_port,
                buffer=self._buffer,
                address=self._osc_address,
            )
            self._osc_listener.start()
        elif self._text_source == "websocket":
            ws_port = int(kwargs.get("ws_port", 9100))
            self._ws_listener = WebSocketListener(port=ws_port, buffer=self._buffer)
            self._ws_listener.start()

        logger.info(f"[WS Captions] Initialised (source={self._text_source})")

    def prepare(self, **kwargs: Any) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs: Any) -> dict:
        """Process a video frame: overlay captions + forward prompts.

        Args:
            **kwargs: Runtime parameters including video, prompts, and all
                schema-defined fields.

        Returns:
            Dict with ``video`` (tensor), optional ``prompts``, and optional
            ``events`` for downstream consumption.
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None")

        # ── 1. Stack input frames ──
        if isinstance(video, list):
            frames = torch.cat(video, dim=0)  # (T, H, W, C) in [0, 255]
        else:
            frames = video
        frames = frames.float() / 255.0  # → [0, 1]
        frames = frames.to(self.device)

        # ── 2. Resolve text from selected source ──
        text = ""
        if self._text_source == "prompt":
            prompts = kwargs.get("prompts", [])
            if prompts:
                first = prompts[0]
                if isinstance(first, dict):
                    text = first.get("text", "")
                else:
                    text = str(first)
            if text:
                self._buffer.push(text)
        elif self._text_source == "manual":
            text = kwargs.get("transcription_text", "")
            if text:
                self._buffer.push(text)
        else:
            # osc / websocket — buffer already populated by background listener
            text = self._buffer.get_latest_text()

        # ── 3. Parse events ──
        events_enabled = kwargs.get("events_enabled", True)
        active_events = []
        if events_enabled and text:
            self._event_parser.pause_threshold_sec = kwargs.get("pause_threshold", 2.0)
            parsed = self._event_parser.parse(text)
            if parsed:
                self._event_buffer.push_many(parsed)
            active_events = self._event_buffer.get_active_events(window_sec=1.0)

        # ── 4. Get display lines ──
        max_lines = kwargs.get("max_lines", 3)
        display_lines = self._buffer.get_display_lines(max_lines)

        # ── 5. Get rate-limited prompt text ──
        update_interval = kwargs.get("update_interval", 2.0)
        prompt_text = self._buffer.get_current_prompt(update_interval)

        # ── 6. Resolve position ──
        preset = kwargs.get("position_preset", "bottom")
        if preset in POSITION_PRESETS:
            pos_x, pos_y = POSITION_PRESETS[preset]
        else:
            pos_x = kwargs.get("pos_x", 50.0)
            pos_y = kwargs.get("pos_y", 90.0)

        # ── 7. Render overlay ──
        overlay_enabled = kwargs.get("overlay_enabled", True)
        if overlay_enabled and display_lines:
            font_size = kwargs.get("font_size", 48)
            if font_size != self._current_font_size:
                self._font = load_font(self._font_path, font_size)
                self._current_font_size = font_size

            config = OverlayConfig(
                pos_x=pos_x,
                pos_y=pos_y,
                text_align=kwargs.get("text_align", "center"),
                max_width_pct=kwargs.get("max_width", 90.0),
                max_lines=max_lines,
                line_spacing=kwargs.get("line_spacing", 1.3),
                font_size=font_size,
                font=self._font,
                text_color=(
                    kwargs.get("text_color_r", 255),
                    kwargs.get("text_color_g", 255),
                    kwargs.get("text_color_b", 255),
                ),
                text_opacity=kwargs.get("text_opacity", 100.0),
                outline_enabled=kwargs.get("outline_enabled", True),
                outline_width=kwargs.get("outline_width", 2),
                outline_color=(
                    kwargs.get("outline_color_r", 0),
                    kwargs.get("outline_color_g", 0),
                    kwargs.get("outline_color_b", 0),
                ),
                bg_enabled=kwargs.get("bg_enabled", True),
                bg_color=(
                    kwargs.get("bg_color_r", 0),
                    kwargs.get("bg_color_g", 0),
                    kwargs.get("bg_color_b", 0),
                ),
                bg_opacity=kwargs.get("bg_opacity", 50.0),
                bg_padding=kwargs.get("bg_padding", 12),
                bg_corner_radius=kwargs.get("bg_corner_radius", 8),
                events=active_events,
                event_intensity=kwargs.get("event_intensity", 0.5),
                word_flash_enabled=kwargs.get("word_flash_enabled", False),
                punctuation_react=kwargs.get("punctuation_react", True),
                event_color_shift=kwargs.get("event_color_shift", False),
            )

            for i in range(frames.shape[0]):
                frames[i] = render_caption_overlay(frames[i], display_lines, config)

        # ── 8. Build output ──
        result = frames.clamp(0, 1)
        if result.device.type != "cpu":
            result = result.cpu()

        output: dict[str, Any] = {"video": result}

        # Forward prompts
        if kwargs.get("prompt_enabled", True) and prompt_text:
            style = kwargs.get("style_prefix", "")
            template = kwargs.get("prompt_template", "{style} {text}")
            try:
                formatted = template.format(style=style, text=prompt_text).strip()
            except (KeyError, IndexError):
                formatted = f"{style} {prompt_text}".strip()
            weight = kwargs.get("prompt_weight", 1.0)
            output["prompts"] = [{"text": formatted, "weight": weight}]

        # Forward events
        if events_enabled and active_events:
            output["events"] = [e.to_dict() for e in active_events]

        return output

    def __del__(self) -> None:
        if self._osc_listener:
            self._osc_listener.stop()
        if self._ws_listener:
            self._ws_listener.stop()


# ── Concrete Pipelines ──────────────────────────────────────────────────────


class WallspaceCaptionsPrePipeline(_WallspaceCaptionsBase):
    """Preprocessor — text baked into frames before AI generation.

    The AI model sees and stylises/integrates the caption text into its
    generation. Prompt forwarding works here: the ``prompts`` key in our
    output dict is forwarded as parameters to the main pipeline via
    Scope's ``extra_params`` mechanism.
    """

    @classmethod
    def get_config_class(cls) -> type:
        return WallspaceCaptionsPreConfig


class WallspaceCaptionsPostPipeline(_WallspaceCaptionsBase):
    """Postprocessor — clean text overlay after AI generation.

    Readable captions rendered on top of AI output. Note: prompt
    forwarding from a postprocessor does NOT update the main pipeline's
    prompt (Scope only forwards extra params downstream, not upstream).
    Use the Pre variant or send prompts via OSC / WebRTC data channel.
    """

    @classmethod
    def get_config_class(cls) -> type:
        return WallspaceCaptionsPostConfig
