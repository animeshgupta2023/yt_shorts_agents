from pathlib import Path
import sys
from typing import Optional
import typer
import json

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.modules.miner import mine_highlights
from src.modules.ingest import extract_video_id, ingest_video
from src.modules.transcribe import transcribe_audio
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

    from rich.table import Table

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


if __name__ == "__main__":
    app()