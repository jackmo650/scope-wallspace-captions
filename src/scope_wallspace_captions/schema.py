"""Pydantic configuration schema for WallSpace Captions pipelines."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)

# ── Base Config ─────────────────────────────────────────────────────────────


class _WallspaceCaptionsBaseConfig(BasePipelineConfig):
    """Shared config for both Pre and Post caption pipelines."""

    pipeline_description: ClassVar[str] = (
        "Live transcription overlay + prompt forwarding. Receives caption text "
        "via Scope prompts, manual input, OSC, or WebSocket. Renders styled "
        "text overlay with event-reactive effects and forwards text as prompts "
        "for AI visual generation."
    )
    pipeline_version: ClassVar[str] = "0.1.0"
    docs_url: ClassVar[str] = "https://github.com/jackmo650/scope-wallspace-captions"
    estimated_vram_gb: ClassVar[float] = 0.1
    supports_prompts: ClassVar[bool] = True
    supports_lora: ClassVar[bool] = False
    supports_vace: ClassVar[bool] = False
    supports_cache_management: ClassVar[bool] = False
    supports_quantization: ClassVar[bool] = False
    modes: ClassVar[dict[str, ModeDefaults]] = {
        "video": ModeDefaults(default=True),
    }

    # ── Load-time parameters ────────────────────────────────────────────

    text_source: Literal["prompt", "manual", "osc", "websocket"] = Field(
        default="prompt",
        description=(
            "Text input method. 'prompt' = Scope prompt field, 'manual' = typed text, "
            "'osc' = UDP OSC listener, 'websocket' = WebSocket server"
        ),
        json_schema_extra=ui_field_config(order=1, label="Text Source", is_load_param=True),
    )

    osc_port: int = Field(
        default=9001,
        ge=1024,
        le=65535,
        description="UDP port for OSC listener — matches WallSpace broadcast port (default 9001)",
        json_schema_extra=ui_field_config(order=2, label="OSC Port", is_load_param=True),
    )

    osc_address: str = Field(
        default="/caption/text",
        description="OSC address to listen on",
        json_schema_extra=ui_field_config(order=3, label="OSC Address", is_load_param=True),
    )

    ws_port: int = Field(
        default=9100,
        ge=1024,
        le=65535,
        description="TCP port for WebSocket listener (text_source='websocket')",
        json_schema_extra=ui_field_config(order=4, label="WebSocket Port", is_load_param=True),
    )

    font_path: str = Field(
        default="",
        description="Path to .ttf/.otf font file (leave empty for system default)",
        json_schema_extra=ui_field_config(order=5, label="Font Path", is_load_param=True),
    )

    # ── Text Input (Input & Controls panel) ─────────────────────────────

    transcription_text: str = Field(
        default="",
        description="Manual text input (when Text Source = 'manual')",
        json_schema_extra=ui_field_config(
            order=1, label="Transcription Text", category="input",
        ),
    )

    # ── Caption Placement ───────────────────────────────────────────────

    overlay_enabled: bool = Field(
        default=True,
        description="Enable caption text overlay on video",
        json_schema_extra=ui_field_config(order=10, label="Overlay Enabled"),
    )

    position_preset: Literal["bottom", "top", "center", "custom"] = Field(
        default="bottom",
        description="Quick position preset ('custom' uses X/Y coordinates below)",
        json_schema_extra=ui_field_config(order=11, label="Position Preset"),
    )

    pos_x: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Horizontal position (% of frame width, text anchor = center)",
        json_schema_extra=ui_field_config(order=12, label="Position X %"),
    )

    pos_y: float = Field(
        default=90.0,
        ge=0.0,
        le=100.0,
        description="Vertical position (% of frame height, text anchor = middle)",
        json_schema_extra=ui_field_config(order=13, label="Position Y %"),
    )

    text_align: Literal["left", "center", "right"] = Field(
        default="center",
        description="Text alignment within the overlay",
        json_schema_extra=ui_field_config(order=14, label="Text Align"),
    )

    max_width: float = Field(
        default=90.0,
        ge=10.0,
        le=100.0,
        description="Maximum text width (% of frame) for word wrap",
        json_schema_extra=ui_field_config(order=15, label="Max Width %"),
    )

    max_lines: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum visible caption lines",
        json_schema_extra=ui_field_config(order=16, label="Max Lines"),
    )

    line_spacing: float = Field(
        default=1.3,
        ge=0.8,
        le=3.0,
        description="Line height multiplier",
        json_schema_extra=ui_field_config(order=17, label="Line Spacing"),
    )

    # ── Font & Text Style ───────────────────────────────────────────────

    font_size: int = Field(
        default=48,
        ge=8,
        le=200,
        description="Font size in pixels",
        json_schema_extra=ui_field_config(order=20, label="Font Size"),
    )

    font_weight: Literal["normal", "bold"] = Field(
        default="normal",
        description="Font weight (uses bold variant if available)",
        json_schema_extra=ui_field_config(order=21, label="Font Weight"),
    )

    text_color_r: int = Field(
        default=255, ge=0, le=255,
        description="Text colour — red channel",
        json_schema_extra=ui_field_config(order=22, label="Text Red"),
    )

    text_color_g: int = Field(
        default=255, ge=0, le=255,
        description="Text colour — green channel",
        json_schema_extra=ui_field_config(order=23, label="Text Green"),
    )

    text_color_b: int = Field(
        default=255, ge=0, le=255,
        description="Text colour — blue channel",
        json_schema_extra=ui_field_config(order=24, label="Text Blue"),
    )

    text_opacity: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Text opacity (0 = invisible, 100 = fully opaque)",
        json_schema_extra=ui_field_config(order=25, label="Text Opacity %"),
    )

    # ── Text Outline ────────────────────────────────────────────────────

    outline_enabled: bool = Field(
        default=True,
        description="Enable text outline/stroke for readability",
        json_schema_extra=ui_field_config(order=30, label="Outline Enabled"),
    )

    outline_width: int = Field(
        default=2, ge=0, le=10,
        description="Outline thickness in pixels",
        json_schema_extra=ui_field_config(order=31, label="Outline Width"),
    )

    outline_color_r: int = Field(
        default=0, ge=0, le=255,
        description="Outline colour — red channel",
        json_schema_extra=ui_field_config(order=32, label="Outline Red"),
    )

    outline_color_g: int = Field(
        default=0, ge=0, le=255,
        description="Outline colour — green channel",
        json_schema_extra=ui_field_config(order=33, label="Outline Green"),
    )

    outline_color_b: int = Field(
        default=0, ge=0, le=255,
        description="Outline colour — blue channel",
        json_schema_extra=ui_field_config(order=34, label="Outline Blue"),
    )

    # ── Background Box ──────────────────────────────────────────────────

    bg_enabled: bool = Field(
        default=True,
        description="Enable background box behind text",
        json_schema_extra=ui_field_config(order=40, label="Background Enabled"),
    )

    bg_color_r: int = Field(
        default=0, ge=0, le=255,
        description="Background colour — red channel",
        json_schema_extra=ui_field_config(order=41, label="BG Red"),
    )

    bg_color_g: int = Field(
        default=0, ge=0, le=255,
        description="Background colour — green channel",
        json_schema_extra=ui_field_config(order=42, label="BG Green"),
    )

    bg_color_b: int = Field(
        default=0, ge=0, le=255,
        description="Background colour — blue channel",
        json_schema_extra=ui_field_config(order=43, label="BG Blue"),
    )

    bg_opacity: float = Field(
        default=50.0, ge=0.0, le=100.0,
        description="Background opacity %",
        json_schema_extra=ui_field_config(order=44, label="BG Opacity %"),
    )

    bg_padding: int = Field(
        default=12, ge=0, le=50,
        description="Padding inside background box (pixels)",
        json_schema_extra=ui_field_config(order=45, label="BG Padding"),
    )

    bg_corner_radius: int = Field(
        default=8, ge=0, le=30,
        description="Rounded corner radius (pixels)",
        json_schema_extra=ui_field_config(order=46, label="BG Corner Radius"),
    )

    # ── Prompt Forwarding ───────────────────────────────────────────────

    prompt_enabled: bool = Field(
        default=True,
        description="Forward transcription text as prompt to downstream pipeline",
        json_schema_extra=ui_field_config(order=50, label="Prompt Forwarding"),
    )

    style_prefix: str = Field(
        default="",
        description="Prepend to transcription text for prompt styling (e.g. 'cinematic neon')",
        json_schema_extra=ui_field_config(order=51, label="Style Prefix"),
    )

    prompt_template: str = Field(
        default="{style} {text}",
        description="Template for formatting the final prompt ({style} and {text} placeholders)",
        json_schema_extra=ui_field_config(order=52, label="Prompt Template"),
    )

    prompt_weight: float = Field(
        default=1.0, ge=0.0, le=2.0,
        description="Weight of the forwarded prompt",
        json_schema_extra=ui_field_config(order=53, label="Prompt Weight"),
    )

    update_interval: float = Field(
        default=2.0, ge=0.5, le=10.0,
        description="Minimum seconds between prompt updates (rate limiting)",
        json_schema_extra=ui_field_config(order=54, label="Update Interval (sec)"),
    )

    # ── Caption Event System ────────────────────────────────────────────

    events_enabled: bool = Field(
        default=True,
        description="Parse text into structured events (WORD, QUESTION, EXCLAMATION, PAUSE, etc.)",
        json_schema_extra=ui_field_config(order=60, label="Events Enabled"),
    )

    pause_threshold: float = Field(
        default=2.0, ge=0.5, le=5.0,
        description="Seconds of silence before a PAUSE event is triggered",
        json_schema_extra=ui_field_config(order=61, label="Pause Threshold (sec)"),
    )

    event_intensity: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Global intensity multiplier for event-driven visual effects",
        json_schema_extra=ui_field_config(order=62, label="Event Intensity"),
    )

    word_flash_enabled: bool = Field(
        default=False,
        description="Flash/highlight the newest word on each WORD event",
        json_schema_extra=ui_field_config(order=63, label="Word Flash"),
    )

    punctuation_react: bool = Field(
        default=True,
        description="Visual reaction to punctuation events (? ! . and pauses)",
        json_schema_extra=ui_field_config(order=64, label="Punctuation React"),
    )

    event_color_shift: bool = Field(
        default=False,
        description="Shift overlay colour based on event type (blue=question, red=exclamation)",
        json_schema_extra=ui_field_config(order=65, label="Event Colour Shift"),
    )

    timing_smoothing: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Word timing smoothing level (0=off, 1=max). Reduces jitter from speech engines",
        json_schema_extra=ui_field_config(order=66, label="Timing Smoothing"),
    )

    timing_outlier_threshold: float = Field(
        default=3.0, ge=1.0, le=10.0,
        description="Standard deviations before a word timing is treated as an outlier",
        json_schema_extra=ui_field_config(order=67, label="Timing Outlier Threshold"),
    )

    # ── Token-Level Rendering ────────────────────────────────────────────

    token_rendering: bool = Field(
        default=False,
        description="Enable per-word token rendering with independent timing, opacity, and colour",
        json_schema_extra=ui_field_config(order=70, label="Token Rendering"),
    )

    token_fade_in: float = Field(
        default=0.3, ge=0.0, le=2.0,
        description="Seconds to fade in new tokens (0 = instant appear)",
        json_schema_extra=ui_field_config(order=71, label="Token Fade-In (sec)"),
    )

    karaoke_enabled: bool = Field(
        default=False,
        description="Karaoke-style: highlight words sequentially based on timestamps",
        json_schema_extra=ui_field_config(order=72, label="Karaoke Mode"),
    )

    karaoke_color_r: int = Field(
        default=0, ge=0, le=255,
        description="Karaoke highlight colour — red channel",
        json_schema_extra=ui_field_config(order=73, label="Karaoke Red"),
    )

    karaoke_color_g: int = Field(
        default=200, ge=0, le=255,
        description="Karaoke highlight colour — green channel",
        json_schema_extra=ui_field_config(order=74, label="Karaoke Green"),
    )

    karaoke_color_b: int = Field(
        default=255, ge=0, le=255,
        description="Karaoke highlight colour — blue channel",
        json_schema_extra=ui_field_config(order=75, label="Karaoke Blue"),
    )

    # ── Responsive Layout ────────────────────────────────────────────────

    responsive_layout: bool = Field(
        default=True,
        description="Scale fonts, outlines, and padding proportionally to frame resolution",
        json_schema_extra=ui_field_config(order=18, label="Responsive Layout"),
    )

    reference_height: int = Field(
        default=1080, ge=360, le=4320,
        description="Reference resolution for font sizes (fonts are authored for this height)",
        json_schema_extra=ui_field_config(order=19, label="Reference Height"),
    )

    safe_zone_pct: float = Field(
        default=5.0, ge=0.0, le=20.0,
        description="Safe zone margin (% of frame) — captions won't render in the outer margin",
        json_schema_extra=ui_field_config(order=20, label="Safe Zone %"),
    )


# ── Pre/Post Variants ──────────────────────────────────────────────────────


class WallspaceCaptionsPreConfig(_WallspaceCaptionsBaseConfig):
    """Preprocessor config — text baked into frames before AI generation."""

    pipeline_id: ClassVar[str] = "wallspace-captions-pre"
    pipeline_name: ClassVar[str] = "WS Captions (Pre)"
    usage: ClassVar[list[UsageType]] = [UsageType.PREPROCESSOR]


class WallspaceCaptionsPostConfig(_WallspaceCaptionsBaseConfig):
    """Postprocessor config — clean text overlay after AI generation."""

    pipeline_id: ClassVar[str] = "wallspace-captions-post"
    pipeline_name: ClassVar[str] = "WS Captions (Post)"
    usage: ClassVar[list[UsageType]] = [UsageType.POSTPROCESSOR]
