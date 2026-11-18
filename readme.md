# YouTube Shorts Agent

YouTube Shorts Agent is a Python CLI that converts a YouTube video into
9:16, captioned short-form clips. It downloads the source, transcribes speech,
uses an LLM to find standalone highlights, tracks people for portrait
reframing, creates animated subtitles, and renders MP4 files.

## Pipeline

```text
YouTube URL
  -> Download video and metadata
  -> Extract 16 kHz mono WAV audio
  -> Transcribe with word timestamps
  -> Mine 25-60 second highlight candidates
  -> Detect scene cuts and track the largest person
  -> Generate karaoke-style ASS subtitles
  -> Crop to 9:16 and render MP4 Shorts
```

The pipeline can be run as one command or as separate stages.

## Requirements

- Python 3.10 or newer
- FFmpeg on `PATH`
- Internet access for YouTube downloads and Gemini
- A CUDA-capable PyTorch installation is recommended
- Gemini API access, or a local GGUF model supported by `llama-cpp-python`
- H.264/NVENC support for GPU rendering; CPU `libx264` rendering is available

The project lists its Python dependencies in
[`requirements.txt`](requirements.txt). FFmpeg is an external dependency.
`llama-cpp-python`, CUDA builds of PyTorch, and some model assets may require
platform-specific installation.

## Installation

Create and activate a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install FFmpeg with your operating system's package manager and verify that it
is available:

```bash
ffmpeg -version
```

For Gemini, create a `.env` file in the repository root:

```dotenv
GEMINI_API_KEY=your_api_key
```

Keep API keys out of version control. `.env` is loaded by
[`src/config.py`](src/config.py).

Verify the setup:

```bash
python src/cli.py test-env
python src/cli.py --help
```

Run commands from the repository root because paths in the configuration are
resolved relative to the current working directory.

## Quick start

Generate up to two Shorts from a YouTube video:

```bash
python src/cli.py run "https://www.youtube.com/watch?v=VIDEO_ID" \
  --max-shorts 2 \
  --provider gemini \
  --whisper-size small
```

Each candidate is processed sequentially. The final paths are printed in a
summary table.

CLI options for `run`:

| Option | Description |
|---|---|
| `--max-shorts` | Maximum number of mined candidates to render. Default: `2`. |
| `--provider` | Override mining provider: `gemini` or `local_gguf`. |
| `--whisper-size` | Override the configured Whisper model size. |

If Gemini has no API key or the API request fails, mining attempts to fall back
to the configured local GGUF model. That fallback requires a valid
`llama-cpp-python` installation and model file.

## CLI commands

Use `python src/cli.py COMMAND --help` for command-specific help.

### Environment check

```bash
python src/cli.py test-env
```

Reports the resolved workspace directory, configured LLM provider, CUDA
availability, GPU name, and Gemini API-key status.

### Stage 1: ingest

```bash
python src/cli.py ingest "https://www.youtube.com/watch?v=VIDEO_ID"
```

Downloads the video and metadata with `yt-dlp`, then uses FFmpeg to create
`audio.wav` as 16 kHz, mono, 16-bit PCM.

Browser cookies can be enabled in [`configs/config.yaml`](configs/config.yaml)
by setting `ingest.cookies_from_browser` to `chrome`, `firefox`, `edge`, or
another browser supported by `yt-dlp`. Keep this configuration local if it
contains authentication preferences.

Generated files:

```text
workspace/<video-id>/raw.mp4
workspace/<video-id>/audio.wav
workspace/<video-id>/metadata.json
```

### Stage 2: transcribe

```bash
python src/cli.py transcribe VIDEO_ID --model-size small
```

The target may be a video ID or YouTube URL. The default model is set in
`configs/config.yaml`; valid CLI overrides include `tiny`, `base`, `small`,
`medium`, and `large-v3`.

Generated files:

```text
workspace/<video-id>/transcript.json
workspace/<video-id>/aligned_words.json
```

### Stage 3: mine candidates

```bash
python src/cli.py mine VIDEO_ID \
  --max-clips 3 \
  --provider gemini
```

The miner asks Gemini or a local GGUF model for structured clip candidates,
snaps start and end times to aligned spoken words, sorts the candidates by
score, and writes `clips.json`.

Candidate mining requires prior `transcribe` output. The CLI prints a preview
table with each score, timestamp range, duration, title, and hook.

Generated file:

```text
workspace/<video-id>/clips.json
```

### Stage 4: reframe

```bash
python src/cli.py reframe VIDEO_ID --clip-index 0
```

Extracts the selected candidate, detects scene boundaries, and uses YOLOv8n
person detection to build a smooth 9:16 crop trajectory. The selected target
is the largest detected person; when no person is detected, the crop retains
the previous/center position.

Generated files:

```text
workspace/<video-id>/clips/clip_00_raw.mp4
workspace/<video-id>/clips/clip_00_trajectory.json
```

YOLOv8n weights may be downloaded automatically by Ultralytics on the first run.

### Stage 5: subtitles

```bash
python src/cli.py subtitles VIDEO_ID \
  --clip-index 0 \
  --words-per-group 3
```

Creates a vertical-video ASS file from the aligned transcript words. Words are
grouped into readability windows and highlighted in sequence.

Generated file:

```text
workspace/<video-id>/clips/clip_00_subtitles.ass
```

### Stage 6: render

```bash
# Prefer NVENC
python src/cli.py render VIDEO_ID --clip-index 0

# Force CPU encoding
python src/cli.py render VIDEO_ID --clip-index 0 --cpu
```

FFmpeg applies the crop trajectory, burns in the ASS subtitles, scales the
video to 1080x1920, and normalizes audio to approximately -14 LUFS. NVENC is
used by default for the standalone render command and in the end-to-end
pipeline. If NVENC fails, the renderer retries with `libx264`.

Generated file:

```text
workspace/<video-id>/out/short_00.mp4
```

## Configuration

The runtime loader reads [`configs/config.yaml`](configs/config.yaml) into
Pydantic models. The following shape is currently recognized:

```yaml
project:
  workspace_dir: "./workspace"
  models_dir: "./models"

llm:
  provider: "gemini" # gemini | local_gguf
  gemini:
    model: "gemini-4-hydrogen"
    temperature: 0.2
  local_gguf:
    model_path: "./models/Qwen3.5-4B-Q4_K_M.gguf"
    n_ctx: 8192
    n_gpu_layers: -1

whisper:
  model_size: "medium"
  device: "cuda"
  compute_type: "float16"

ingest:
  max_height: 1080
  cookies_from_browser: "chrome"
  player_clients: ["tv", "web_safari", "web"]

reframe:
  target_aspect_ratio: [9, 16]
  scene_threshold: 27.0
  deadband_pixels: 25
  ema_alpha: 0.15

subtitles:
  font_name: "Montserrat Black"
  font_size: 24
  words_per_group: 3
  bottom_margin_percent: 0.25
```

`GEMINI_API_KEY` is read from the environment or `.env`; it is not a YAML key.

### Important configuration status

The YAML keys above are recognized by [`src/config.py`](src/config.py).
However, `reframe` and `subtitles` values are not yet passed from
`PipelineConfig` into every stage: reframing currently uses module defaults,
and subtitle generation reads CLI/module defaults rather than the
`subtitles` configuration section. The other main settings, including workspace
paths, ingest options, Whisper settings, and LLM providers, are consumed by
the CLI pipeline.

## Generated workspace layout

```text
workspace/<video-id>/
  raw.mp4
  audio.wav
  metadata.json
  transcript.json
  aligned_words.json
  clips.json
  clips/
    clip_00_raw.mp4
    clip_00_trajectory.json
    clip_00_subtitles.ass
  out/
    short_00.mp4
```

Video, audio, workspace artifacts, Whisper models, and GGUF files can consume
significant disk space. The `workspace/` and `models/` directories should not
be committed unless they are intentionally managed.

## Project structure

| Path | Responsibility |
|---|---|
| [`src/cli.py`](src/cli.py) | Typer CLI and end-to-end pipeline orchestration |
| [`src/config.py`](src/config.py) | Environment loading and Pydantic configuration |
| [`src/modules/ingest.py`](src/modules/ingest.py) | YouTube download, metadata, and audio extraction |
| [`src/modules/transcribe.py`](src/modules/transcribe.py) | Faster-Whisper transcription |
| [`src/modules/miner.py`](src/modules/miner.py) | Gemini/local highlight mining and clip schema |
| [`src/modules/detector.py`](src/modules/detector.py) | Scene-cut detection |
| [`src/modules/reframer.py`](src/modules/reframer.py) | Subclip extraction and YOLO crop trajectories |
| [`src/modules/subtitler.py`](src/modules/subtitler.py) | ASS subtitle generation |
| [`src/modules/render.py`](src/modules/render.py) | FFmpeg portrait rendering and audio normalization |
| [`src/utils/shell.py`](src/utils/shell.py) | External command execution |
| [`src/utils/logger.py`](src/utils/logger.py) | Shared Rich console theme |
| [`configs/config.yaml`](configs/config.yaml) | Runtime configuration |
| [`requirements.txt`](requirements.txt) | Python dependencies |

## Current limitations

- Reframe settings such as aspect ratio, scene threshold, deadband, and EMA
  smoothing are currently passed as hard-coded/module defaults.
- Subtitle font, font size, word grouping, and safe margin are partially
  hard-coded; CLI currently exposes only `--words-per-group`.
- The `run` command does not expose a cookies-from-browser argument.
- Browser cookies are configured in YAML for the ingestion module, not through
  a CLI flag.
- The local GGUF model must be downloaded or placed in `models/` separately.
- Gemini failures fall back to a local model, so local inference must be usable
  when fallback is required.
- Person tracking chooses the largest detected person and does not determine
  which person is speaking.
- Output quality depends on source resolution, video bitrate, track availability,
  Whisper model, and GPU/CPU performance.
