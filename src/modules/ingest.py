import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel
import yt_dlp

from src.config import IngestConfig
from src.utils.logger import console
from src.utils.shell import run_command

# Ensure Deno is discoverable for yt-dlp-ejs challenge solving
_deno_bin = str(Path.home() / ".deno" / "bin")
if _deno_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_deno_bin}:{os.environ.get('PATH', '')}"


class IngestResult(BaseModel):
    video_id: str
    title: str
    duration: float
    workspace_dir: Path
    raw_video_path: Path
    audio_wav_path: Path
    metadata_path: Path


def extract_video_id(url: str) -> str:
    """Extract standard 11-character YouTube video ID."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if not match:
        raise ValueError(f"Could not parse valid YouTube video ID from: {url}")
    return match.group(1)


def demux_audio(raw_video_path: Path, output_wav_path: Path) -> Path:
    """Demux video into 16kHz mono PCM 16-bit WAV for Whisper."""
    run_command(
        [
            "ffmpeg", "-y",
            "-i", raw_video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_wav_path,
        ],
        desc="Demuxing 16kHz mono audio",
    )
    return output_wav_path


def ingest_video(
    url: str,
    base_workspace: Path,
    ingest_config: Optional[IngestConfig] = None,
    force_download: bool = False,
) -> IngestResult:
    """Download video and metadata using yt-dlp defaults (with cookie fallback), then demux audio."""
    cfg = ingest_config or IngestConfig()
    h = cfg.max_height
    video_id = extract_video_id(url)
    target_dir = base_workspace / video_id
    target_dir.mkdir(parents=True, exist_ok=True)

    raw_video_path = target_dir / "raw.mp4"
    audio_wav_path = target_dir / "audio.wav"
    metadata_path = target_dir / "metadata.json"

    # Skip download if valid artifacts already exist
    if (
        not force_download
        and raw_video_path.exists() and raw_video_path.stat().st_size > 0
        and audio_wav_path.exists() and audio_wav_path.stat().st_size > 0
        and metadata_path.exists()
    ):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        console.print(f"[info]Using cached ingest artifacts for {video_id}[/info]")
        return IngestResult(
            video_id=video_id,
            title=meta.get("title", ""),
            duration=float(meta.get("duration", 0)),
            workspace_dir=target_dir,
            raw_video_path=raw_video_path,
            audio_wav_path=audio_wav_path,
            metadata_path=metadata_path,
        )

    # Single unified format selector: prefer H.264+AAC up to max_height, else best available
    base_opts: Dict[str, Any] = {
        "format": f"bv*[height<={h}][vcodec^=avc1]+ba[ext=m4a]/bv*[height<={h}]+ba/b[height<={h}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(raw_video_path),
        "overwrites": True,
        "quiet": True,
        "no_warnings": True,
        "remote_components": ["ejs:github"],
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 4,
    }

    console.print(f"[stage]Ingesting YouTube Video:[/stage] [info]{video_id}[/info]")

    # Pass 1: Clean yt-dlp defaults (no cookies, no manual player_client overrides)
    try:
        with yt_dlp.YoutubeDL(base_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as err:
        # Pass 2: Retry with browser cookies only if Pass 1 fails (age-gate / IP check)
        if not cfg.cookies_from_browser:
            raise RuntimeError(f"yt-dlp failed for {url}: {err}") from err
        console.print(f"[warning]Default ingest failed ({err}). Retrying with browser cookies...[/warning]")
        auth_opts = {**base_opts, "cookiesfrombrowser": (cfg.cookies_from_browser,)}
        with yt_dlp.YoutubeDL(auth_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)

    if info_dict is None:
        raise RuntimeError(f"yt-dlp failed to fetch info for {url}")

    metadata = {
        "id": video_id,
        "title": info_dict.get("title", ""),
        "description": info_dict.get("description", ""),
        "duration": float(info_dict.get("duration", 0)),
        "channel": info_dict.get("uploader", ""),
        "chapters": info_dict.get("chapters") or [],
        "heatmap": info_dict.get("heatmap") or [],
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    console.print(f"[success]Video downloaded:[/success] {metadata['title']} ({metadata['duration']}s)")

    demux_audio(raw_video_path, audio_wav_path)
    console.print(f"[success]Audio extracted:[/success] {audio_wav_path.name}")

    return IngestResult(
        video_id=video_id,
        title=metadata["title"],
        duration=metadata["duration"],
        workspace_dir=target_dir,
        raw_video_path=raw_video_path,
        audio_wav_path=audio_wav_path,
        metadata_path=metadata_path,
    )