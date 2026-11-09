import json
import re
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel
import yt_dlp

from src.utils.logger import console
from src.utils.shell import run_command


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
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if not match:
        raise ValueError(f"Could not parse valid YouTube video ID from: {url}")
    return match.group(1)


def demux_audio(raw_video_path: Path, output_wav_path: Path) -> Path:
    """Demux video into 16kHz mono PCM 16-bit WAV for Whisper acoustic models."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", raw_video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path,
    ]
    run_command(cmd, desc="Demuxing 16kHz mono audio")
    return output_wav_path


def ingest_video(url: str, base_workspace: Path) -> IngestResult:
    """Download video and metadata, then demux normalized audio."""
    video_id = extract_video_id(url)
    target_dir = base_workspace / video_id
    target_dir.mkdir(parents=True, exist_ok=True)

    raw_video_path = target_dir / "raw.mp4"
    audio_wav_path = target_dir / "audio.wav"
    metadata_path = target_dir / "metadata.json"

    # Configure yt-dlp: cap at 1080p, prefer standard H.264 / AAC MP4 container
    # Prioritize H.264 (avc1) video up to 1080p, fall back to any format if unavailable
    ydl_opts: Dict[str, Any] = {
        "format": (
            "bestvideo[height<=1080][vcodec^=avc1]+bestaudio[ext=m4a]/"
            "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
            "best[height<=1080][ext=mp4]/best"
        ),
        "outtmpl": str(raw_video_path),
        "overwrites": True,
        "quiet": True,
        "no_warnings": True,
    }

    console.print(f"[stage]Ingesting YouTube Video:[/stage] [info]{video_id}[/info]")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        if info_dict is None:
            raise RuntimeError(f"yt-dlp failed to fetch info for {url}")

    # Save relevant metadata
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

    # Extract clean audio track
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