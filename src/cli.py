from pathlib import Path
from typing import Optional
import typer
from src.config import load_config
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
def run(
    url: str = typer.Argument(..., help="YouTube video URL"),
    max_shorts: int = typer.Option(3, help="Max clips to extract"),
    provider: Optional[str] = typer.Option(
        None, help="Override LLM provider (gemini | local_gguf)"
    ),
):
    """Full execution pipeline."""
    console.print(f"[stage]Starting pipeline for:[/stage] {url}")
    # Will coordinate stages sequentially


if __name__ == "__main__":
    app()