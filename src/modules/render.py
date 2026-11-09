import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from src.utils.logger import console
from src.utils.shell import run_command


def build_crop_expression(trajectory_path: Path) -> Dict[str, Any]:
    """
    Parse trajectory keyframes and shots to construct an FFmpeg crop expression.
    Computes a stable median anchor for each detected scene cut to eliminate
    jitter while instantly adapting when the camera angle switches.
    """
    with open(trajectory_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    res = data.get("resolution", {})
    crop_w = res.get("crop_w", 608)
    crop_h = res.get("crop_h", 1080)
    scenes = data.get("scenes", [])
    keyframes = data.get("keyframes", [])

    if not keyframes:
        raise ValueError(f"No keyframes found in {trajectory_path.name}")

    # Map keyframes to their respective scenes
    scene_crops: List[Dict[str, Any]] = []
    for s_idx, (s_start, s_end) in enumerate(scenes):
        scene_kfs = [
            k for k in keyframes if k.get("scene_idx") == s_idx or (s_start <= k["timestamp"] <= s_end)
        ]
        if scene_kfs:
            x_vals = [k["crop_x"] for k in scene_kfs]
            median_x = int(np.median(x_vals))
        else:
            median_x = keyframes[0]["crop_x"]

        scene_crops.append({
            "start": s_start,
            "end": s_end,
            "x": median_x,
        })

    # Construct nested FFmpeg ternary expression: if(between(t, s0, e0), x0, if(...))
    if len(scene_crops) == 1:
        x_expr = f"{scene_crops[0]['x']}"
    else:
        # Build backward for clean nested if-expressions
        x_expr = f"{scene_crops[-1]['x']}"
        for sc in reversed(scene_crops[:-1]):
            x_expr = f"if(between(t\\,{sc['start']:.2f}\\,{sc['end']:.2f})\\,{sc['x']}\\,{x_expr})"

    return {
        "crop_w": crop_w,
        "crop_h": crop_h,
        "x_expr": x_expr,
        "scene_count": len(scene_crops),
    }


def render_short(
    clip_video_path: Path,
    trajectory_path: Path,
    ass_subtitles_path: Path,
    output_video_path: Path,
    use_nvenc: bool = True,
    video_bitrate: str = "8M",
) -> Path:
    """
    Render vertical 9:16 Short with dynamic crop, burned ASS subtitles,
    EBU R128 audio loudness normalization, and NVENC GPU encoding.
    """
    if not clip_video_path.exists():
        raise FileNotFoundError(f"Missing subclip: {clip_video_path}")
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing trajectory: {trajectory_path}")
    if not ass_subtitles_path.exists():
        raise FileNotFoundError(f"Missing subtitle file: {ass_subtitles_path}")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    crop_info = build_crop_expression(trajectory_path)

    # Escape path characters for FFmpeg's filter parser
    escaped_ass = (
        str(ass_subtitles_path.resolve())
        .replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
    )

    # Build filtergraph: dynamic crop -> scale to 1080x1920 -> burn subtitles
    filtergraph = (
        f"crop=w={crop_info['crop_w']}:h={crop_info['crop_h']}:x='{crop_info['x_expr']}':y=0,"
        f"scale=1080:1920:flags=lanczos,"
        f"ass='{escaped_ass}'"
    )

    # Target loudness: -14 LUFS (YouTube Shorts standard), -1.5 dB True Peak
    audio_filter = "loudnorm=I=-14:LRA=7:TP=-1.5"

    console.print(
        f"[stage]Compositing Vertical Short:[/stage] {output_video_path.name} "
        f"(NVENC: {use_nvenc}, Shots: {crop_info['scene_count']})"
    )

    # Base FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(clip_video_path),
        "-vf", filtergraph,
        "-af", audio_filter,
    ]

    if use_nvenc:
        cmd.extend([
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-b:v", video_bitrate,
            "-pix_fmt", "yuv420p",
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
        ])

    cmd.extend([
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_video_path),
    ])

    try:
        run_command(cmd, desc="Compositing and hardware encoding Short")
    except RuntimeError:
        if use_nvenc:
            console.print("[warning]NVENC encode failed. Retrying with CPU libx264...[/warning]")
            return render_short(
                clip_video_path=clip_video_path,
                trajectory_path=trajectory_path,
                ass_subtitles_path=ass_subtitles_path,
                output_video_path=output_video_path,
                use_nvenc=False,
                video_bitrate=video_bitrate,
            )
        raise

    console.print(f"[success]Short generated successfully:[/success] {output_video_path}")
    return output_video_path