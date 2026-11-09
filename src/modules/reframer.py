import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import torch
from pydantic import BaseModel
from ultralytics import YOLO

from src.modules.detector import detect_scene_cuts
from src.utils.logger import console
from src.utils.shell import run_command


class CropKeyframe(BaseModel):
    timestamp: float
    frame_idx: int
    scene_idx: int
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int


def extract_raw_clip(
    source_video: Path,
    output_clip_path: Path,
    start_time: float,
    duration: float,
) -> Path:
    """
    Extract candidate slice and normalize to standard H.264 (yuv420p).
    Avoids '-c copy' to prevent OpenCV decoding failures on AV1/VP9 streams.
    """
    output_clip_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_time:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(source_video),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_clip_path),
    ]
    run_command(cmd, desc=f"Extracting subclip ({duration:.1f}s)")
    return output_clip_path


def compute_target_x(
    boxes: List[Any],
    frame_width: int,
    crop_width: int,
    prev_center_x: float,
) -> float:
    """
    Select active subject and return ideal crop box center X coordinate.
    Chooses the person with the largest bounding box area.
    """
    if not boxes or len(boxes) == 0:
        return prev_center_x

    best_box = None
    max_area = 0.0

    for box in boxes:
        # box.xyxy: [x1, y1, x2, y2]
        coords = box.xyxy[0].cpu().numpy()
        w = coords[2] - coords[0]
        h = coords[3] - coords[1]
        area = w * h
        if area > max_area:
            max_area = area
            best_box = coords

    if best_box is not None:
        center_x = (best_box[0] + best_box[2]) / 2.0
        return float(center_x)

    return prev_center_x


def generate_crop_trajectory(
    clip_video_path: Path,
    output_trajectory_path: Path,
    device: str = "cuda",
    deadband_px: int = 25,
    ema_alpha: float = 0.12,
    sample_stride: int = 2,  # Process every 2nd frame for 2x faster tracking
) -> Path:
    """
    Track subjects per frame using YOLOv8n, apply scene-aware EMA smoothing,
    and save keyframe crop coordinates.
    """
    cap = cv2.VideoCapture(str(clip_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video clip: {clip_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Target 9:16 aspect ratio relative to original height (1080 -> 608w)
    crop_h = orig_h
    crop_w = int(orig_h * (9 / 16))
    if crop_w % 2 != 0:
        crop_w += 1  # Codecs require even dimensions
    max_x = orig_w - crop_w
    default_center_x = orig_w / 2.0

    # Detect shot cuts in this clip
    scene_cuts = detect_scene_cuts(clip_video_path)

    console.print(f"[stage]Tracking subjects with YOLOv8 on GPU:[/stage] {clip_video_path.name}")
    # Load YOLOv8n (auto-downloads lightweight weights on first run)
    model = YOLO("yolov8n.pt")
    if device == "cuda" and torch.cuda.is_available():
        model.to("cuda")

    keyframes: List[CropKeyframe] = []
    current_scene_idx = 0
    smooth_center_x = default_center_x
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = round(frame_idx / fps, 3)

        # Check if we transitioned to a new shot/scene
        if current_scene_idx < len(scene_cuts) - 1:
            if timestamp >= scene_cuts[current_scene_idx][1]:
                current_scene_idx += 1
                # On cut: reset EMA filter to snap immediately without panning
                smooth_center_x = default_center_x

        # Run detection on sampled frames, interpolate between steps
        if frame_idx % sample_stride == 0:
            # Class 0: person
            results = model(frame, classes=[0], conf=0.35, verbose=False)
            boxes = results[0].boxes if results else []
            target_x = compute_target_x(boxes, orig_w, crop_w, smooth_center_x)

            # Apply deadband threshold: ignore micro-movements
            if abs(target_x - smooth_center_x) > deadband_px:
                # Apply Exponential Moving Average
                smooth_center_x = (ema_alpha * target_x) + ((1.0 - ema_alpha) * smooth_center_x)

        # Convert center point to clamped top-left crop coordinate
        crop_x = int(smooth_center_x - (crop_w / 2.0))
        crop_x = max(0, min(crop_x, max_x))

        keyframes.append(
            CropKeyframe(
                timestamp=timestamp,
                frame_idx=frame_idx,
                scene_idx=current_scene_idx,
                crop_x=crop_x,
                crop_y=0,
                crop_w=crop_w,
                crop_h=crop_h,
            )
        )
        frame_idx += 1

    cap.release()

    # Save trajectory file
    output_payload = {
        "source_clip": str(clip_video_path),
        "fps": fps,
        "total_frames": len(keyframes),
        "resolution": {"orig_w": orig_w, "orig_h": orig_h, "crop_w": crop_w, "crop_h": crop_h},
        "scene_count": len(scene_cuts),
        "scenes": scene_cuts,
        "keyframes": [k.model_dump() for k in keyframes],
    }

    with open(output_trajectory_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    console.print(
        f"[success]Generated crop trajectory ({len(keyframes)} frames, {len(scene_cuts)} shots):[/success] "
        f"{output_trajectory_path.name}"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_trajectory_path