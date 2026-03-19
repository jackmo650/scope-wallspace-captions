"""Advanced caption overlay rendering using PIL.

Renders styled text onto torch tensors with:
- XY coordinate placement (percentage-based, resolution-independent)
- Font control (family, size, weight)
- Foreground colour with opacity
- Text outline/stroke
- Background box with colour, opacity, padding, corner radius
- Text alignment (left, center, right)
- Word wrap + multi-line scrolling
- Event-reactive effects (word flash, punctuation reactions, emphasis)
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from .events import CaptionEvent, CaptionEventType

logger = logging.getLogger(__name__)

# ── Font Loading ────────────────────────────────────────────────────────────

_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def load_font(path: str = "", size: int = 48) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font, falling back to PIL default."""
    key = (path, size)
    if key in _font_cache:
        return _font_cache[key]

    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    if path and Path(path).is_file():
        try:
            font = ImageFont.truetype(path, size)
            _font_cache[key] = font
            return font
        except Exception:
            logger.warning(f"[WS Captions] Could not load font '{path}', using default")

    # Try common system fonts
    for fallback in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf", "FreeSans.ttf"):
        try:
            font = ImageFont.truetype(fallback, size)
            _font_cache[key] = font
            return font
        except OSError:
            continue

    # Absolute fallback — PIL's built-in bitmap font (not scalable)
    font = ImageFont.load_default()
    _font_cache[key] = font
    return font


# ── Configuration ───────────────────────────────────────────────────────────


@dataclass
class OverlayConfig:
    """All parameters for rendering a caption overlay."""

    # Position
    pos_x: float = 50.0          # % of frame width (anchor = text center)
    pos_y: float = 90.0          # % of frame height (anchor = text top)
    text_align: str = "center"   # left | center | right
    max_width_pct: float = 90.0  # % of frame width for word wrap
    max_lines: int = 3
    line_spacing: float = 1.3

    # Font
    font_size: int = 48
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None

    # Text colour
    text_color: tuple[int, int, int] = (255, 255, 255)
    text_opacity: float = 100.0  # 0-100

    # Outline
    outline_enabled: bool = True
    outline_width: int = 2
    outline_color: tuple[int, int, int] = (0, 0, 0)

    # Background box
    bg_enabled: bool = True
    bg_color: tuple[int, int, int] = (0, 0, 0)
    bg_opacity: float = 50.0     # 0-100
    bg_padding: int = 12
    bg_corner_radius: int = 8

    # Events
    events: list[CaptionEvent] = field(default_factory=list)
    event_intensity: float = 0.5
    word_flash_enabled: bool = False
    punctuation_react: bool = True
    event_color_shift: bool = False

    # Responsive layout
    responsive_layout: bool = True
    reference_height: int = 1080
    safe_zone_pct: float = 5.0

    # Token-level rendering
    token_rendering: bool = False
    token_fade_in: float = 0.3
    karaoke_enabled: bool = False
    karaoke_color: tuple[int, int, int] = (0, 200, 255)


# ── Event Colour Map ────────────────────────────────────────────────────────

EVENT_COLORS: dict[str, tuple[int, int, int]] = {
    "question": (100, 180, 255),     # Blue tint
    "exclamation": (255, 100, 100),  # Red/hot
    "pause": (150, 150, 150),        # Faded grey
    "emphasis": (255, 255, 100),     # Bright yellow
}


# ── Responsive Scaling ─────────────────────────────────────────────────────


def _scale_value(value: int | float, frame_height: int, ref_height: int) -> int:
    """Scale a pixel value proportionally to frame resolution."""
    return max(1, int(value * frame_height / ref_height))


def _apply_responsive_scaling(config: OverlayConfig, frame_height: int) -> OverlayConfig:
    """Return a copy of config with font size, outline, padding scaled to frame resolution."""
    if not config.responsive_layout or frame_height == config.reference_height:
        return config
    ref = config.reference_height
    # Shallow copy via dataclass replace
    import copy
    scaled = copy.copy(config)
    scaled.font_size = _scale_value(config.font_size, frame_height, ref)
    scaled.outline_width = _scale_value(config.outline_width, frame_height, ref)
    scaled.bg_padding = _scale_value(config.bg_padding, frame_height, ref)
    scaled.bg_corner_radius = _scale_value(config.bg_corner_radius, frame_height, ref)
    # Reload font at new size
    scaled.font = None
    return scaled


# ── Main Render Function ───────────────────────────────────────────────────


def render_caption_overlay(
    frame: torch.Tensor,
    lines: list[str],
    config: OverlayConfig,
) -> torch.Tensor:
    """Render caption text onto a video frame tensor.

    Args:
        frame: ``(H, W, C)`` float32 tensor in [0, 1].
        lines: Text lines to display (most recent last).
        config: Full styling + event configuration.

    Returns:
        ``(H, W, C)`` float32 tensor in [0, 1] with overlay composited.
    """
    if not lines:
        return frame

    h, w, c = frame.shape

    # Apply responsive scaling based on frame resolution
    config = _apply_responsive_scaling(config, h)

    font = config.font or load_font(size=config.font_size)

    # ── Word-wrap lines to fit max_width ──
    max_px = int(w * config.max_width_pct / 100)
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(_word_wrap(line, font, max_px))

    # Trim to max_lines (keep newest)
    if len(wrapped) > config.max_lines:
        wrapped = wrapped[-config.max_lines:]

    if not wrapped:
        return frame

    # ── Measure text block ──
    line_heights: list[int] = []
    line_widths: list[int] = []
    for line in wrapped:
        bbox = font.getbbox(line)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        line_widths.append(lw)
        line_heights.append(lh)

    line_h = int(max(line_heights) * config.line_spacing) if line_heights else config.font_size
    block_h = line_h * len(wrapped)
    block_w = max(line_widths) if line_widths else 0

    # ── Compute position ──
    anchor_x = int(w * config.pos_x / 100)
    anchor_y = int(h * config.pos_y / 100)

    # ── Determine event-reactive colour ──
    text_rgb = config.text_color
    if config.event_color_shift and config.events:
        text_rgb = _event_tinted_color(text_rgb, config.events, config.event_intensity)

    text_alpha = int(config.text_opacity * 2.55)

    # Apply pause fade
    if config.punctuation_react and config.events:
        from .events import CaptionEventType
        for ev in reversed(config.events):
            if ev.type == CaptionEventType.PAUSE:
                fade = max(0.3, 1.0 - config.event_intensity * 0.7)
                text_alpha = int(text_alpha * fade)
                break

    # ── Convert frame to PIL ──
    np_frame = (frame.cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(np_frame, mode="RGB").convert("RGBA")

    # ── Create overlay layer (text region only for performance) ──
    pad = config.bg_padding if config.bg_enabled else 0
    overlay_w = block_w + pad * 2
    overlay_h = block_h + pad * 2
    overlay = Image.new("RGBA", (overlay_w, overlay_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # ── Background box ──
    if config.bg_enabled:
        bg_alpha = int(config.bg_opacity * 2.55)
        bg_rgba = (*config.bg_color, bg_alpha)
        if config.bg_corner_radius > 0:
            draw.rounded_rectangle(
                [(0, 0), (overlay_w - 1, overlay_h - 1)],
                radius=config.bg_corner_radius,
                fill=bg_rgba,
            )
        else:
            draw.rectangle([(0, 0), (overlay_w - 1, overlay_h - 1)], fill=bg_rgba)

    # ── Draw text lines ──
    text_rgba = (*text_rgb, text_alpha)
    outline_rgba = (*config.outline_color, text_alpha) if config.outline_enabled else None

    # Track newest word for flash effect
    flash_word = _get_flash_word(config.events) if config.word_flash_enabled else None

    for i, line in enumerate(wrapped):
        # Horizontal alignment within the overlay
        lw = line_widths[min(i, len(line_widths) - 1)] if i < len(line_widths) else 0
        if config.text_align == "left":
            tx = pad
        elif config.text_align == "right":
            tx = overlay_w - pad - lw
        else:  # center
            tx = (overlay_w - lw) // 2

        ty = pad + i * line_h

        # Outline (stroke)
        if config.outline_enabled and config.outline_width > 0:
            draw.text(
                (tx, ty), line, font=font, fill=outline_rgba,
                stroke_width=config.outline_width,
                stroke_fill=outline_rgba,
            )

        # ── Per-word rendering ──
        if config.token_rendering and config.events:
            # Token-level: per-word fade-in, karaoke, independent effects
            _draw_tokens_with_effects(draw, line, tx, ty, font, text_rgba, config)
        elif flash_word or (config.punctuation_react and config.events):
            _draw_words_with_effects(
                draw, line, tx, ty, font, text_rgba, config, flash_word,
            )
        else:
            draw.text((tx, ty), line, font=font, fill=text_rgba)

    # ── Composite onto frame ──
    # Position the overlay on the full-size image
    if config.text_align == "center":
        paste_x = anchor_x - overlay_w // 2
    elif config.text_align == "right":
        paste_x = anchor_x - overlay_w
    else:
        paste_x = anchor_x

    paste_y = anchor_y - overlay_h // 2

    # Clamp to frame bounds with safe zone
    safe_margin_x = int(w * config.safe_zone_pct / 100) if config.responsive_layout else 0
    safe_margin_y = int(h * config.safe_zone_pct / 100) if config.responsive_layout else 0
    paste_x = max(safe_margin_x, min(paste_x, w - overlay_w - safe_margin_x))
    paste_y = max(safe_margin_y, min(paste_y, h - overlay_h - safe_margin_y))

    img.paste(overlay, (paste_x, paste_y), overlay)

    # ── Convert back to tensor ──
    result = img.convert("RGB")
    np_result = np.array(result, dtype=np.float32) / 255.0
    return torch.from_numpy(np_result).to(frame.device)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _word_wrap(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_px: int) -> list[str]:
    """Wrap text to fit within *max_px* pixels using the given font."""
    if not text:
        return []

    # Estimate chars per line from average char width
    avg_char_w = max(1, _text_width(font, "M"))
    chars_per_line = max(10, max_px // avg_char_w)

    lines = textwrap.wrap(text, width=chars_per_line)
    # Refine: if any line is still too wide, re-wrap tighter
    result: list[str] = []
    for line in lines:
        if _text_width(font, line) <= max_px:
            result.append(line)
        else:
            # Binary search for the right wrap width
            result.extend(textwrap.wrap(line, width=max(5, chars_per_line - 5)))
    return result if result else [text]


def _text_width(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> int:
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]


def _event_tinted_color(
    base: tuple[int, int, int],
    events: list[CaptionEvent],
    intensity: float,
) -> tuple[int, int, int]:
    """Blend base colour toward an event-specific tint."""
    from .events import CaptionEventType

    # Find the most recent high-priority event
    priority = {
        CaptionEventType.EXCLAMATION: 4,
        CaptionEventType.QUESTION: 3,
        CaptionEventType.EMPHASIS: 2,
        CaptionEventType.PAUSE: 1,
    }
    best_event = None
    best_pri = 0
    for ev in reversed(events):
        pri = priority.get(ev.type, 0)
        if pri > best_pri:
            best_pri = pri
            best_event = ev
    if best_event is None:
        return base

    tint = EVENT_COLORS.get(best_event.type.value, base)
    t = intensity * 0.6  # Don't fully replace the base colour
    return (
        int(base[0] * (1 - t) + tint[0] * t),
        int(base[1] * (1 - t) + tint[1] * t),
        int(base[2] * (1 - t) + tint[2] * t),
    )


def _get_flash_word(events: list[CaptionEvent]) -> str | None:
    """Return the most recent WORD event's text for highlight flash."""
    from .events import CaptionEventType

    for ev in reversed(events):
        if ev.type == CaptionEventType.WORD:
            return ev.text
    return None


def _draw_words_with_effects(
    draw: ImageDraw.ImageDraw,
    line: str,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    base_rgba: tuple[int, int, int, int],
    config: OverlayConfig,
    flash_word: str | None,
) -> None:
    """Draw a line word-by-word, applying per-word effects."""
    from .events import CaptionEventType

    words = line.split()
    cursor_x = x

    # Check for exclamation scale effect
    exclamation_active = False
    if config.punctuation_react and config.events:
        for ev in config.events:
            if ev.type == CaptionEventType.EXCLAMATION:
                exclamation_active = True
                break

    for i, word in enumerate(words):
        display = word + (" " if i < len(words) - 1 else "")
        rgba = base_rgba

        # Flash the newest word
        clean = word.strip(".,!?;:\"'()[]{}—–-")
        if flash_word and clean.lower() == flash_word.lower():
            # Bright white flash
            rgba = (255, 255, 255, 255)

        # Emphasis — ALL CAPS
        if clean.isupper() and len(clean) >= 3:
            rgba = (*EVENT_COLORS.get("emphasis", base_rgba[:3]), base_rgba[3])

        draw.text((cursor_x, y), display, font=font, fill=rgba)
        cursor_x += _text_width(font, display)


def _draw_tokens_with_effects(
    draw: ImageDraw.ImageDraw,
    line: str,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    base_rgba: tuple[int, int, int, int],
    config: OverlayConfig,
) -> None:
    """Draw a line token-by-token with fade-in, karaoke highlight, and per-word effects."""
    import time as _time
    from .events import CaptionEventType

    words = line.split()
    cursor_x = x
    now = _time.monotonic()

    # Build a map of word → event for token effects
    word_events: dict[str, CaptionEvent] = {}
    for ev in config.events:
        if ev.type == CaptionEventType.WORD:
            word_events[ev.text.lower()] = ev

    for i, word in enumerate(words):
        display = word + (" " if i < len(words) - 1 else "")
        clean = word.strip(".,!?;:\"'()[]{}—–-")
        rgba = base_rgba

        # Token fade-in: compute opacity based on age
        ev = word_events.get(clean.lower())
        if ev and config.token_fade_in > 0:
            age = now - ev.timestamp
            if age < config.token_fade_in:
                fade_alpha = int(rgba[3] * (age / config.token_fade_in))
                rgba = (rgba[0], rgba[1], rgba[2], max(0, fade_alpha))

        # Karaoke highlight: most recent word gets highlight colour
        if config.karaoke_enabled and ev:
            # The most recent WORD event is the "active" karaoke word
            latest_word_ev = None
            for rev_ev in reversed(config.events):
                if rev_ev.type == CaptionEventType.WORD:
                    latest_word_ev = rev_ev
                    break
            if latest_word_ev and clean.lower() == latest_word_ev.text.lower():
                rgba = (*config.karaoke_color, rgba[3])

        # Flash the newest word
        if config.word_flash_enabled:
            flash = _get_flash_word(config.events)
            if flash and clean.lower() == flash.lower():
                rgba = (255, 255, 255, 255)

        # Emphasis — ALL CAPS
        if clean.isupper() and len(clean) >= 3:
            rgba = (*EVENT_COLORS.get("emphasis", base_rgba[:3]), rgba[3])

        draw.text((cursor_x, y), display, font=font, fill=rgba)
        cursor_x += _text_width(font, display)
