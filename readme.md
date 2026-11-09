<div align="center">

# YouTube Shorts Agent

An in-progress Python CLI that turns a YouTube video into one or more vertical,
captioned Shorts. It downloads the source, transcribes speech, asks an LLM to
select candidate moments, tracks people to create portrait crops, generates
animated subtitles, and renders MP4 files.

</div>

## What is implemented

The end-to-end command is implemented as six stages:

```text
YouTube URL
    │
    ├─ 1. Ingest video, metadata, and normalized audio
    ├─ 2. Transcribe audio with word timestamps
    ├─ 3. Ask Gemini or a local GGUF model for clip candidates
    ├─ 4. Extract each candidate and calculate a person-aware 9:16 crop
    ├─ 5. Generate word-highlighted ASS subtitles
    └─ 6. Crop, subtitle, normalize audio, and encode each Short
         └─ workspace/<video-id>/out/short_XX.mp4
```

The individual stages can also be run separately from the CLI. This is useful
for rerunning a stage or inspecting its intermediate files. This project is
still under development: see [Current limitations](#current-limitations).

## Requirements

- Python with the packages listed in [`requirements.txt`](requirements.txt).
- FFmpeg available on `PATH`. It is used to demux audio, extract subclips, and
  render the final video.
- A CUDA-capable PyTorch setup is recommended for Whisper and YOLO inference.
  The configured defaults request CUDA.
- For Gemini mining, a Gemini API key in `GEMINI_API_KEY`.
- Or, for local mining, `llama-cpp-python` installed and a compatible GGUF model
  available at the configured path.
- NVIDIA NVENC support is used by default for final encoding. The standalone
  `render` command can be told to use CPU encoding instead.

The Python requirements file does not currently list `llama-cpp-python`,
PyTorch, or FFmpeg itself. Install/configure those separately as appropriate
for your system and GPU.

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install FFmpeg using your operating system's package manager and ensure the
`ffmpeg` executable is on `PATH`. Configure CUDA/PyTorch and `llama-cpp-python`
if you plan to use GPU inference or the local GGUF provider.

For Gemini, create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your_api_key
```

Do not commit API keys. The application loads `.env` in
[`src/config.py`](src/config.py).

Check the environment and show CLI help:

```bash
python src/cli.py test-env
python src/cli.py --help
```

The CLI can also be invoked as a module:

```bash
python -m src.cli --help
```

Run commands from the repository root because configuration and model paths
are relative to the current working directory.

## End-to-end use

```bash
python src/cli.py run "https://www.youtube.com/watch?v=VIDEO_ID" \
  --max-shorts 2 \
  --provider gemini \
  --whisper-size small
```

Options:

- `--max-shorts`: maximum number of candidates to render. Defaults to `2`.
- `--provider`: `gemini` or `local_gguf`; if omitted, the configured provider
  is used. Gemini failures or a missing key trigger a local-model fallback.
- `--whisper-size`: override the configured Whisper model size for this run.

The selected candidates are processed one at a time through reframing,
subtitling, and rendering. At the end, the CLI prints a table with each output's
title, duration, and path.

## Pipeline stages

### 1. Ingest: download video and prepare audio

Implemented by [`src/modules/ingest.py`](src/modules/ingest.py), called by the
`ingest` command and by `run`.

The stage:

1. Extracts the 11-character YouTube video ID from the URL.
2. Creates `workspace/<video-id>/`.
3. Uses `yt-dlp` to download a video up to 1080p, preferring H.264 video and
   M4A audio where available.
4. Saves selected video metadata as JSON.
5. Invokes FFmpeg to produce 16 kHz, mono, 16-bit PCM WAV audio.

Files created:

```text
workspace/<video-id>/
├── raw.mp4
├── audio.wav
└── metadata.json
```

The downloader has retry, timeout, buffering, chunking, and fragment
concurrency options configured. Download speed still depends on the selected
stream, YouTube/CDN delivery, and the network connection.

### 2. Transcribe: speech recognition and word timing

Implemented by [`src/modules/transcribe.py`](src/modules/transcribe.py), using
`faster-whisper`.

The stage loads the selected Whisper model and transcribes `audio.wav` with
word timestamps and voice-activity filtering. It records the detected language,
transcript segments, and individual word times/confidence values.

Files created:

```text
workspace/<video-id>/
├── transcript.json
└── aligned_words.json
```

`transcript.json` contains language information and timestamped transcript
segments. `aligned_words.json` is a flat list of words with their start, end,
and probability; later stages use it to refine clip boundaries and create
subtitles.

### 3. Mine: select highlight candidates

Implemented by [`src/modules/miner.py`](src/modules/miner.py).

The transcript is formatted with timestamps and sent to one of two providers:

- **Gemini**: uses the configured Gemini model and requests structured JSON.
- **Local GGUF**: loads the configured GGUF model with `llama-cpp-python` and
  requests JSON output according to the clip schema.

The prompt asks for self-contained, engaging clips with a hook and a
narrative arc. Candidate fields include start/end time, duration, hook, title,
topic, score, and reasoning. Candidate times are snapped to nearby word
boundaries when aligned word data is present, sorted by score, and saved in
`clips.json`.

The CLI prints a preview table with each candidate's score, timestamps,
duration, title, and hook.

File created:

```text
workspace/<video-id>/clips.json
```

### 4. Reframe: extract and follow the subject

Orchestrated by [`src/cli.py`](src/cli.py), with clip extraction and crop
tracking in [`src/modules/reframer.py`](src/modules/reframer.py). Scene
boundaries are detected by [`src/modules/detector.py`](src/modules/detector.py).

For each selected candidate:

1. FFmpeg extracts and re-encodes the source interval as a normalized H.264
   subclip.
2. PySceneDetect detects shot boundaries in that subclip.
3. YOLOv8n detects people on sampled frames. The largest detected person is
   used as the crop target; if no person is detected, the previous/center
   position is retained.
4. A deadband and exponential smoothing reduce small crop movements. The crop
   resets at detected scene cuts.
5. The module saves per-frame crop keyframes and scene information as JSON.

The crop is portrait 9:16, based on the source frame height. YOLO weights may
be downloaded automatically by Ultralytics the first time the model is used.

Files created for candidate index `00`:

```text
workspace/<video-id>/clips/
├── clip_00_raw.mp4
└── clip_00_trajectory.json
```

### 5. Subtitles: generate animated captions

Implemented by [`src/modules/subtitler.py`](src/modules/subtitler.py).

The module selects aligned words that overlap the chosen clip, converts their
timestamps to clip-relative times, and writes an ASS subtitle file. Words are
shown in groups (three by default); each word is highlighted in sequence while
the other words in that group remain visible. The generated ASS file uses a
vertical-video script resolution and a bottom safe-zone margin.

File created for candidate index `00`:

```text
workspace/<video-id>/clips/clip_00_subtitles.ass
```

### 6. Render: produce the final Short

Implemented by [`src/modules/render.py`](src/modules/render.py).

FFmpeg combines the extracted subclip, crop trajectory, and ASS subtitles. It
applies a 9:16 portrait crop, scales the result to 1080 × 1920, burns in the
subtitles, normalizes audio to approximately -14 LUFS, and encodes the output.
The default encoder is NVIDIA NVENC; the standalone `render --cpu` option uses
CPU `libx264`. If NVENC fails inside the render helper, it retries with CPU
encoding.

Output:

```text
workspace/<video-id>/out/short_00.mp4
```

## CLI commands

Run `python src/cli.py COMMAND --help` for the latest options.

### Check the environment

```bash
python src/cli.py test-env
```

Displays the resolved workspace path, configured LLM provider, PyTorch CUDA
availability/device name, and whether a Gemini key is present when Gemini is
selected. It does not download a video or run the full pipeline.

### Run individual stages

```bash
# Download source, metadata, and normalized audio
python src/cli.py ingest "https://www.youtube.com/watch?v=VIDEO_ID"

# Transcribe an already-ingested video by ID or URL
python src/cli.py transcribe VIDEO_ID --model-size small

# Mine an existing transcript
python src/cli.py mine VIDEO_ID --max-clips 3 --provider local_gguf

# Reframe one mined candidate (index 0 is the first candidate)
python src/cli.py reframe VIDEO_ID --clip-index 0

# Generate that candidate's subtitle file
python src/cli.py subtitles VIDEO_ID --clip-index 0 --words-per-group 3

# Render it; --cpu forces libx264 instead of NVENC
python src/cli.py render VIDEO_ID --clip-index 0 --cpu
```

The `transcribe`, `mine`, `reframe`, `subtitles`, and `render` commands expect
their preceding stage's files to exist. The `run` command performs all stages
automatically.

## Configuration

Configuration is declared in [`configs/config.yaml`](configs/config.yaml) and
loaded into Pydantic models by [`src/config.py`](src/config.py). The model
currently recognizes these top-level fields:

```yaml
workspace_dir: "./workspace"
models_dir: "./models"

llm:
  provider: "gemini" # gemini | local_gguf
  gemini:
    model: "gemini-3.5-flash-lite"
    temperature: 0.2
  local_gguf:
    model_path: "./models/Qwen3.5-4B-Q4_K_M.gguf"
    n_ctx: 8192
    n_gpu_layers: -1

whisper:
  model_size: "medium"
  device: "cuda"
  compute_type: "float16"
```

`GEMINI_API_KEY` is read from the environment or `.env`, not from YAML.

**Important: the current YAML and loader do not fully match.** The checked-in
YAML nests `workspace_dir` and `models_dir` under `project:`. The loader
expects them at the top level, so those nested values are ignored and the
defaults are used. The YAML also contains `reframe:` and `subtitles:` sections,
but the current `PipelineConfig` does not define those settings. The pipeline
uses values hard-coded in its modules/CLI for those behaviors; changing those
YAML entries currently does not change reframing or subtitle output.

## Generated workspace layout

After running the pipeline, a video workspace typically looks like:

```text
workspace/<video-id>/
├── raw.mp4
├── audio.wav
├── metadata.json
├── transcript.json
├── aligned_words.json
├── clips.json
├── clips/
│   ├── clip_00_raw.mp4
│   ├── clip_00_trajectory.json
│   ├── clip_00_subtitles.ass
│   └── ...
└── out/
    ├── short_00.mp4
    └── ...
```

The workspace is an intermediate/output area. Large video, audio, and model
files can require substantial disk space.

## Project files

| File | Responsibility |
|---|---|
| [`src/cli.py`](src/cli.py) | Typer CLI, stage commands, and end-to-end orchestration. |
| [`src/config.py`](src/config.py) | Loads `.env` and YAML configuration and validates supported settings. |
| [`src/modules/ingest.py`](src/modules/ingest.py) | YouTube ID parsing, yt-dlp download, metadata, and FFmpeg audio extraction. |
| [`src/modules/transcribe.py`](src/modules/transcribe.py) | Faster-Whisper transcription and word-level timestamp files. |
| [`src/modules/miner.py`](src/modules/miner.py) | Gemini/local GGUF highlight selection, schema validation, and word-boundary snapping. |
| [`src/modules/detector.py`](src/modules/detector.py) | Detects shot boundaries using PySceneDetect. |
| [`src/modules/reframer.py`](src/modules/reframer.py) | Extracts subclips and generates person-aware portrait crop trajectories. |
| [`src/modules/subtitler.py`](src/modules/subtitler.py) | Generates karaoke-style ASS subtitle files. |
| [`src/modules/render.py`](src/modules/render.py) | Builds FFmpeg crop expressions and renders the final video. |
| [`src/utils/shell.py`](src/utils/shell.py) | Runs external commands without a shell and raises on failure. |
| [`src/utils/logger.py`](src/utils/logger.py) | Shared Rich console and output theme. |
| [`configs/config.yaml`](configs/config.yaml) | User-editable configuration (subject to the loader mismatch above). |
| [`requirements.txt`](requirements.txt) | Python package requirements. |

The `__init__.py` files mark `src`, `src.modules`, and `src.utils` as Python
packages; they are currently empty.

## Current limitations

- The configuration model does not yet include all keys present in
  `configs/config.yaml`; in particular, `project`, `reframe`, and `subtitles`
  are not wired into the runtime settings.
- `llama-cpp-python`, PyTorch, FFmpeg, and model assets are not all installed or
  provisioned by `requirements.txt`.
- The local provider requires a compatible `llama-cpp-python` install and an
  accessible GGUF file. If Gemini fails, the code attempts local fallback, so
  that local setup must also be available for fallback to succeed.
- YOLO person tracking can only center on detected people; it does not identify
  which person is speaking.
- The standalone `detector` module is used by reframing to find scene cuts; it
  is not a separate CLI stage.
- The `run` command processes up to the number of clips returned by mining and
  renders them sequentially.