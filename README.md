# scope-wallspace-captions

Live transcription-to-visuals node for [Daydream Scope](https://daydream.live) — receives caption text from WallSpace, A.EYE.ECHO, OSC, or WebSocket sources, overlays styled text on video, and forwards transcription as prompts to drive real-time AI visual generation.

**Co-authored by [Jack Morgan](https://github.com/jackmo650) and [Ciborg / Matt](https://github.com/dmtelf)**

## Features

- **4 text input methods**: Scope prompt field, manual text, OSC (UDP), WebSocket
- **Pre + Post pipelines**: Pre bakes text into frames so AI stylises it; Post overlays clean captions after AI generation
- **Advanced caption placement**: XY coordinate positioning (percentage-based), preset positions (top/center/bottom), text alignment
- **Full styling control**: Font size, colour (RGB), opacity, text outline with colour/width, background box with colour/opacity/padding/corner radius
- **Caption Event System**: Parses text into structured events (WORD, SENTENCE_START/END, QUESTION, EXCLAMATION, PAUSE, EMPHASIS, SPEAKER_CHANGE) that drive visual behaviours
- **Event-reactive effects**: Per-word flash, punctuation colour reactions, pause fade, emphasis highlighting
- **Prompt forwarding**: Transcription text forwarded as prompts with style prefix, template formatting, and rate limiting

## Use Cases

- **Accessibility**: Live captions for deaf/hard-of-hearing audiences at live events (A.EYE.ECHO integration)
- **VJ performance**: Spoken word → AI-generated reactive visuals in real-time
- **Live events**: Audience speech drives projected visuals
- **Art installations**: Text-reactive generative art

## Installation

### From GitHub

```bash
uv pip install "scope-wallspace-captions @ git+https://github.com/jackmo650/scope-wallspace-captions"
```

### Local development

```bash
git clone https://github.com/jackmo650/scope-wallspace-captions.git
cd scope-wallspace-captions
pip install -e .
```

### Via Scope

Settings → Nodes → Browse → select the `scope-wallspace-captions` directory.

## Pipelines

| Pipeline | ID | Usage | Description |
|---|---|---|---|
| **WS Captions (Pre)** | `wallspace-captions-pre` | Preprocessor | Text baked into frames before AI model — AI sees and stylises the text |
| **WS Captions (Post)** | `wallspace-captions-post` | Postprocessor | Clean text overlay after AI generation — readable captions on top |

## Text Input Methods

| Source | Setting | How it works |
|---|---|---|
| **Scope Prompt** | `text_source=prompt` | Text from Scope's built-in prompt field (default) |
| **Manual** | `text_source=manual` | Type into the `transcription_text` field |
| **OSC** | `text_source=osc` | UDP listener on configurable port (default 9000), address `/caption/text` |
| **WebSocket** | `text_source=websocket` | WebSocket server on configurable port (default 9100), accepts plain text or JSON |

### OSC Format

Matches the WallSpace/A.EYE.ECHO OSC bridge:
- `/caption/text [string]` — caption text
- `/caption/clear []` — clear buffer

### WebSocket Format

Accepts plain text strings or JSON:
```json
{"text": "Hello world", "speaker": "Jack"}
```

## Caption Event System

Raw transcription text is parsed into structured events:

| Event | Trigger | Visual Effect |
|---|---|---|
| `SENTENCE_START` | New sentence begins | Flash/pulse |
| `SENTENCE_END` | Period/question/exclamation at end | Fade transition |
| `WORD` | Each word extracted | Per-word animation (flash) |
| `QUESTION` | `?` detected | Blue colour shift |
| `EXCLAMATION` | `!` detected | Red intensity spike |
| `PAUSE` | Gap > threshold | Opacity fade |
| `EMPHASIS` | ALL CAPS word (3+ chars) | Yellow highlight |
| `SPEAKER_CHANGE` | Different speaker tag | Style switch |

Events are also forwarded in the output dict as `{"events": [...]}` for downstream nodes.

## Parameters

### Load-time (require pipeline reload)

| Parameter | Default | Description |
|---|---|---|
| `text_source` | `prompt` | Input method: prompt, manual, osc, websocket |
| `osc_port` | `9000` | OSC UDP port |
| `osc_address` | `/caption/text` | OSC address |
| `ws_port` | `9100` | WebSocket TCP port |
| `font_path` | (empty) | Path to .ttf/.otf font |

### Runtime — Caption Placement

| Parameter | Default | Description |
|---|---|---|
| `overlay_enabled` | `true` | Toggle overlay |
| `position_preset` | `bottom` | bottom, top, center, custom |
| `pos_x` | `50` | X position (% of width) |
| `pos_y` | `90` | Y position (% of height) |
| `text_align` | `center` | left, center, right |
| `max_width` | `90` | Max text width (%) |
| `max_lines` | `3` | Visible lines |

### Runtime — Text Style

| Parameter | Default | Description |
|---|---|---|
| `font_size` | `48` | Font size (px) |
| `text_color_r/g/b` | `255/255/255` | Text RGB |
| `text_opacity` | `100` | Text opacity (%) |
| `outline_enabled` | `true` | Text outline |
| `outline_width` | `2` | Outline thickness (px) |
| `outline_color_r/g/b` | `0/0/0` | Outline RGB |
| `bg_enabled` | `true` | Background box |
| `bg_color_r/g/b` | `0/0/0` | Background RGB |
| `bg_opacity` | `50` | Background opacity (%) |
| `bg_padding` | `12` | Box padding (px) |
| `bg_corner_radius` | `8` | Corner radius (px) |

### Runtime — Prompt Forwarding

| Parameter | Default | Description |
|---|---|---|
| `prompt_enabled` | `true` | Forward text as prompt |
| `style_prefix` | (empty) | Style prefix (e.g. "cinematic neon") |
| `prompt_template` | `{style} {text}` | Prompt format template |
| `prompt_weight` | `1.0` | Prompt weight |
| `update_interval` | `2.0` | Rate limit (seconds) |

### Runtime — Events

| Parameter | Default | Description |
|---|---|---|
| `events_enabled` | `true` | Enable event parsing |
| `pause_threshold` | `2.0` | Silence duration for PAUSE (sec) |
| `event_intensity` | `0.5` | Effect intensity multiplier |
| `word_flash_enabled` | `false` | Flash newest word |
| `punctuation_react` | `true` | React to ? ! . |
| `event_color_shift` | `false` | Colour shift by event type |

## Requirements

- Python 3.12+
- PyTorch (ships with Scope)
- python-osc
- websockets
- Pillow

## License

MIT

## Development Workflow

This project follows a **human-in-the-loop** development process:

1. **All requests start as GitHub Issues** — bugs, features, tasks, and experiments are logged using the provided issue templates.
2. **Issues are reviewed and triaged** — the maintainer reviews each issue, adjusts scope, and assigns priority.
3. **Only `approved` issues move forward** — no implementation begins until an issue is explicitly labeled `approved`.
4. **Implementation happens on explicit instruction** — coding agents and contributors only work on approved, assigned work.
5. **Pull requests reference an approved issue** — every PR must link back to the issue it addresses.

