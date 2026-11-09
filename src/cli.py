from pathlib import Path
import sys
from typing import Optional
import typer
import json
from rich.table import Table

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.modules.miner import mine_highlights
from src.modules.ingest import extract_video_id, ingest_video
from src.modules.transcribe import transcribe_audio
from src.modules.reframer import extract_raw_clip, generate_crop_trajectory
from src.modules.subtitler import generate_clip_subtitles
from src.modules.render import render_short
from src.utils.logger import console

app = typer.Typer(
    help="AI YouTube Shorts Agent CLI",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def test_env():
    """Verify system configuration and GPU initialization."""
    config = load_config()
    console.print("[stage]=== Environment Check ===[/stage]")
    console.print(f"[info]Workspace Dir:[/info] {config.workspace_dir.resolve()}")
    console.print(f"[info]Configured LLM Provider:[/info] {config.llm.provider}")

    import torch

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "None"
    console.print(
        f"[info]CUDA Available:[/info] {cuda_available} ({device_name})"
    )

    if config.llm.provider == "gemini":
        has_key = bool(config.llm.gemini.api_key)
        status = (
            "[success]Found[/success]"
            if has_key
            else "[error]Missing GEMINI_API_KEY in .env[/error]"
        )
        console.print(f"[info]Gemini API Key:[/info] {status}")


@app.command()
def ingest(
    url: str = typer.Argument(..., help="YouTube video URL to download and process")
):
    """Stage 1: Download YouTube video, fetch metadata, and demux 16kHz WAV."""
    config = load_config()
    result = ingest_video(url, config.workspace_dir)
    console.print(f"[stage]=== Ingestion Complete ===[/stage]")
    console.print(f"Directory: {result.workspace_dir}")
    console.print(f"Video: {result.raw_video_path.name} | Audio: {result.audio_wav_path.name}")

@app.command()
def transcribe(
    target: str = typer.Argument(
        ..., help="Video ID or YouTube URL that was already ingested"
    ),
    model_size: Optional[str] = typer.Option(
        None, help="Whisper model size: tiny, base, small, medium, large-v3"
    ),
):
    """Stage 2: Transcribe audio and generate word-level timestamps."""
    config = load_config()

    # Determine video_id from URL or raw ID
    if "youtube.com" in target or "youtu.be" in target:
        video_id = extract_video_id(target)
    else:
        video_id = target

    workspace_dir = config.workspace_dir / video_id
    audio_path = workspace_dir / "audio.wav"

    if not audio_path.exists():
        console.print(
            f"[error]Audio file does not exist at {audio_path}. Run 'ingest' first.[/error]"
        )
        raise typer.Exit(code=1)

    chosen_model = model_size or config.whisper.model_size

    res = transcribe_audio(
        audio_path=audio_path,
        workspace_dir=workspace_dir,
        model_size=chosen_model,
        device=config.whisper.device,
        compute_type=config.whisper.compute_type,
    )

    console.print(f"[stage]=== Transcription Complete ===[/stage]")
    console.print(f"Transcript JSON: {res.transcript_path}")
    console.print(f"Aligned Words JSON: {res.aligned_words_path}")

@app.command()
def mine(
    target: str = typer.Argument(..., help="Video ID or URL already transcribed"),
    max_clips: int = typer.Option(3, help="Number of clips to extract"),
    provider: Optional[str] = typer.Option(
        None, help="Override provider: 'gemini' or 'local_gguf'"
    ),
):
    """Stage 3: Mine transcript for viral short-form candidates."""
    config = load_config()
    video_id = (
        extract_video_id(target)
        if ("youtube.com" in target or "youtu.be" in target)
        else target
    )
    workspace_dir = config.workspace_dir / video_id

    clips_path = mine_highlights(
        workspace_dir=workspace_dir,
        config=config,
        max_clips=max_clips,
        override_provider=provider,
    )

    console.print(f"[stage]=== Highlight Mining Complete ===[/stage]")
    console.print(f"Clips File: {clips_path}")

    # Display preview table in console
    with open(clips_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    table = Table(title=f"Mined Clips for: {data.get('video_title', video_id)}")
    table.add_column("Score", style="bold yellow", justify="center")
    table.add_column("Timestamps", style="cyan")
    table.add_column("Duration", style="green")
    table.add_column("Title / Hook", style="white")

    for c in data.get("clips", []):
        table.add_row(
            f"{c['score']:.1f}",
            f"{c['start_time']}s - {c['end_time']}s",
            f"{c['duration']}s",
            f"[bold]{c['title']}[/bold]\n[dim]Hook: \"{c['hook']}\"[/dim]",
        )

    console.print(table)


@app.command()
def reframe(
    target: str = typer.Argument(..., help="Video ID or URL with mined clips"),
    clip_index: int = typer.Option(0, help="Index of clip candidate in clips.json to process (0 = top clip)"),
):
    """Stage 4: Extract subclip, detect shot cuts, track subject, and build 9:16 crop path."""
    config = load_config()
    video_id = extract_video_id(target) if ("youtube.com" in target or "youtu.be" in target) else target
    workspace_dir = config.workspace_dir / video_id

    raw_video = workspace_dir / "raw.mp4"
    clips_path = workspace_dir / "clips.json"

    if not raw_video.exists():
        console.print(f"[error]Source video raw.mp4 not found in {workspace_dir}[/error]")
        raise typer.Exit(code=1)
    if not clips_path.exists():
        console.print(f"[error]clips.json not found. Run 'mine' first.[/error]")
        raise typer.Exit(code=1)

    with open(clips_path, "r", encoding="utf-8") as f:
        clips_data = json.load(f)

    available_clips = clips_data.get("clips", [])
    if not available_clips or clip_index >= len(available_clips):
        console.print(f"[error]Clip index {clip_index} not found in clips.json[/error]")
        raise typer.Exit(code=1)
    
    selected_clip = available_clips[clip_index]
    start_t = float(selected_clip["start_time"])
    dur = float(selected_clip["duration"])

    clips_output_dir = workspace_dir / "clips"
    clips_output_dir.mkdir(parents=True, exist_ok=True)

    subclip_path = clips_output_dir / f"clip_{clip_index:02d}_raw.mp4"
    trajectory_path = clips_output_dir / f"clip_{clip_index:02d}_trajectory.json"

    # Step 1: Extract trimmed subclip
    extract_raw_clip(
        source_video=raw_video,
        output_clip_path=subclip_path,
        start_time=start_t,
        duration=dur,
    )

    # Step 2: Track subject and build smooth crop path
    generate_crop_trajectory(
        clip_video_path=subclip_path,
        output_trajectory_path=trajectory_path,
        device="cuda",
    )

    console.print(f"[stage]=== Reframing Complete ===[/stage]")
    console.print(f"Subclip: {subclip_path}")
    console.print(f"Crop Path: {trajectory_path}")

@app.command()
def subtitles(
    target: str = typer.Argument(..., help="Video ID or URL with transcribed words"),
    clip_index: int = typer.Option(0, help="Index of clip candidate to subtitle"),
    words_per_group: int = typer.Option(3, help="Words shown simultaneously per line"),
):
    """Stage 5: Generate dynamic karaoke-style ASS subtitles for a clip."""
    config = load_config()
    video_id = extract_video_id(target) if ("youtube.com" in target or "youtu.be" in target) else target
    workspace_dir = config.workspace_dir / video_id

    words_path = workspace_dir / "aligned_words.json"
    clips_path = workspace_dir / "clips.json"
    output_ass = workspace_dir / "clips" / f"clip_{clip_index:02d}_subtitles.ass"

    generate_clip_subtitles(
        aligned_words_path=words_path,
        clips_json_path=clips_path,
        output_ass_path=output_ass,
        clip_index=clip_index,
        words_per_group=words_per_group,
    )

    console.print(f"[stage]=== Subtitles Complete ===[/stage]")
    console.print(f"Subtitle File: {output_ass}")

@app.command()
def render(
    target: str = typer.Argument(..., help="Video ID or YouTube URL"),
    clip_index: int = typer.Option(0, help="Clip index to composite"),
    cpu: bool = typer.Option(False, "--cpu", help="Force CPU libx264 encoding instead of NVENC"),
):
    """Stage 6: Render final vertical Short with dynamic crop, subtitles, and audio normalization."""
    config = load_config()
    video_id = extract_video_id(target) if ("youtube.com" in target or "youtu.be" in target) else target
    workspace_dir = config.workspace_dir / video_id

    clip_dir = workspace_dir / "clips"
    out_dir = workspace_dir / "out"

    subclip_path = clip_dir / f"clip_{clip_index:02d}_raw.mp4"
    trajectory_path = clip_dir / f"clip_{clip_index:02d}_trajectory.json"
    ass_path = clip_dir / f"clip_{clip_index:02d}_subtitles.ass"
    out_path = out_dir / f"short_{clip_index:02d}.mp4"

    render_short(
        clip_video_path=subclip_path,
        trajectory_path=trajectory_path,
        ass_subtitles_path=ass_path,
        output_video_path=out_path,
        use_nvenc=not cpu,
    )
    console.print(f"[stage]=== Render Complete ===[/stage] -> {out_path}")

@app.command()
def run(
    url: str = typer.Argument(..., help="YouTube video URL"),
    max_shorts: int = typer.Option(2, help="Number of viral Shorts to generate"),
    provider: Optional[str] = typer.Option(None, help="LLM provider: 'gemini' or 'local_gguf'"),
    whisper_size: Optional[str] = typer.Option(None, help="Whisper model size"),
):
    """Full End-to-End Pipeline: Ingest -> Transcribe -> Mine -> Reframe -> Subtitle -> Render."""
    config = load_config()
    console.print(f"[stage]Starting YouTube Shorts Agent DAG for:[/stage] {url}")

    # Stage 1: Ingest
    ingest_res = ingest_video(url, config.workspace_dir)
    workspace_dir = ingest_res.workspace_dir
    video_id = ingest_res.video_id

    # Stage 2: Transcribe
    chosen_whisper = whisper_size or config.whisper.model_size
    transcribe_audio(
        audio_path=ingest_res.audio_wav_path,
        workspace_dir=workspace_dir,
        model_size=chosen_whisper,
        device=config.whisper.device,
        compute_type=config.whisper.compute_type,
    )

    # Stage 3: Highlight Mining
    clips_path = mine_highlights(
        workspace_dir=workspace_dir,
        config=config,
        max_clips=max_shorts,
        override_provider=provider,
    )

    with open(clips_path, "r", encoding="utf-8") as f:
        clips_data = json.load(f)

    clips_list = clips_data.get("clips", [])
    num_to_process = min(max_shorts, len(clips_list))
    rendered_files = []

    # Stages 4, 5, 6 per clip
    for idx in range(num_to_process):
        clip = clips_list[idx]
        console.print(f"\n[stage]=== Processing Short #{idx + 1}/{num_to_process}: {clip['title']} ===[/stage]")

        clip_dir = workspace_dir / "clips"
        subclip_path = clip_dir / f"clip_{idx:02d}_raw.mp4"
        trajectory_path = clip_dir / f"clip_{idx:02d}_trajectory.json"
        ass_path = clip_dir / f"clip_{idx:02d}_subtitles.ass"
        out_path = workspace_dir / "out" / f"short_{idx:02d}.mp4"

        # Stage 4: Reframe
        extract_raw_clip(
            source_video=ingest_res.raw_video_path,
            output_clip_path=subclip_path,
            start_time=float(clip["start_time"]),
            duration=float(clip["duration"]),
        )
        generate_crop_trajectory(clip_video_path=subclip_path, output_trajectory_path=trajectory_path)

        # Stage 5: Subtitles
        generate_clip_subtitles(
            aligned_words_path=workspace_dir / "aligned_words.json",
            clips_json_path=clips_path,
            output_ass_path=ass_path,
            clip_index=idx,
        )

        # Stage 6: Render
        rendered_path = render_short(
            clip_video_path=subclip_path,
            trajectory_path=trajectory_path,
            ass_subtitles_path=ass_path,
            output_video_path=out_path,
        )
        rendered_files.append((clip, rendered_path))

    # Summary Output
    console.print("\n[stage]================ ALL SHORTS GENERATED ================ [/stage]")
    table = Table(title=f"Completed Shorts for: {ingest_res.title}")
    table.add_column("Index", style="cyan", justify="center")
    table.add_column("Title", style="bold white")
    table.add_column("Duration", style="green")
    table.add_column("Path", style="yellow")

    for idx, (clip, path) in enumerate(rendered_files):
        table.add_row(f"{idx + 1}", clip["title"], f"{clip['duration']}s", str(path))

    console.print(table)

if __name__ == "__main__":
    app()