from pathlib import Path
import sys
from typing import Optional
import typer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.modules.ingest import ingest_video
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

if __name__ == "__main__":
    app()