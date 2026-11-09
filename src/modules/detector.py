from pathlib import Path
from typing import List, Tuple
from scenedetect import ContentDetector, SceneManager, open_video
from src.utils.logger import console


def detect_scene_cuts(
    video_path: Path,
    threshold: float = 27.0,
    min_scene_len_sec: float = 0.8,
) -> List[Tuple[float, float]]:
    """
    Detect hard camera cuts and shot boundaries in a video clip.
    Returns a list of (start_sec, end_sec) tuples for each distinct shot.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=threshold,
            min_scene_len=int(video.frame_rate * min_scene_len_sec),
        )
    )

    console.print(f"[stage]Detecting shot boundaries:[/stage] {video_path.name}")
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()

    scenes: List[Tuple[float, float]] = []
    for scene in scene_list:
        start_sec = round(scene[0].get_seconds(), 3)
        end_sec = round(scene[1].get_seconds(), 3)
        scenes.append((start_sec, end_sec))

    if not scenes:
        # Fallback if no cuts detected: single shot encompassing the full clip
        total_duration = video.duration.get_seconds()
        scenes = [(0.0, round(total_duration, 3))]

    console.print(f"[info]Detected {len(scenes)} shots in clip.[/info]")
    return scenes