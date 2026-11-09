import json
from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel

from src.utils.logger import console


class SubtitleWord(BaseModel):
    word: str
    start: float
    end: float


def format_ass_time(seconds: float) -> str:
    """Format seconds into ASS timestamp standard: H:MM:SS.cs"""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int(round((seconds - int(seconds)) * 100))
    if centisecs >= 100:
        secs += 1
        centisecs = 0
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def build_ass_header(font_name: str = "Montserrat", font_size: int = 70) -> str:
    """
    Construct ASS script header and styles for vertical 1080x1920 video.
    Safe zone: Alignment=2 (bottom center), MarginV=480 (25% above bottom UI).
    Colors in &HAABBGGRR&: White=&H00FFFFFF&, Yellow=&H0000FFFF&, Black=&H00000000&.
    """
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,2,60,60,480,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def generate_clip_subtitles(
    aligned_words_path: Path,
    clips_json_path: Path,
    output_ass_path: Path,
    clip_index: int = 0,
    words_per_group: int = 3,
    highlight_color_bgr: str = "00FFFF",  # Bright Yellow in BGR hex
    normal_color_bgr: str = "FFFFFF",     # White in BGR hex
) -> Path:
    """
    Filter words for target clip, calculate clip-relative timestamps,
    and render kinetic word-highlighting ASS subtitle lines.
    """
    if not aligned_words_path.exists():
        raise FileNotFoundError(f"Missing {aligned_words_path}")
    if not clips_json_path.exists():
        raise FileNotFoundError(f"Missing {clips_json_path}")

    with open(clips_json_path, "r", encoding="utf-8") as f:
        clips_data = json.load(f)

    available_clips = clips_data.get("clips", [])
    if clip_index >= len(available_clips):
        raise IndexError(f"Clip index {clip_index} out of range in {clips_json_path.name}")

    selected_clip = available_clips[clip_index]
    clip_start = float(selected_clip["start_time"])
    clip_end = float(selected_clip["end_time"])

    with open(aligned_words_path, "r", encoding="utf-8") as f:
        raw_words = json.load(f)

    # Filter words belonging to this clip
    clip_words: List[SubtitleWord] = []
    for w in raw_words:
        w_start = float(w["start"])
        w_end = float(w["end"])
        if w_end >= clip_start and w_start <= clip_end:
            rel_start = max(0.0, round(w_start - clip_start, 2))
            rel_end = max(rel_start + 0.05, round(w_end - clip_start, 2))
            clip_words.append(
                SubtitleWord(
                    word=w["word"].strip().upper(),
                    start=rel_start,
                    end=rel_end,
                )
            )

    console.print(f"[stage]Generating ASS subtitles:[/stage] {len(clip_words)} spoken words")

    # Group words into small readability windows (default 3 words)
    grouped_windows: List[List[SubtitleWord]] = []
    for i in range(0, len(clip_words), words_per_group):
        grouped_windows.append(clip_words[i : i + words_per_group])

    dialogue_lines: List[str] = []

    for group in grouped_windows:
        # Create an event for each word within the group as the highlighted target
        for idx, active_word in enumerate(group):
            line_start = format_ass_time(active_word.start)
            # Active duration stays until the next word starts or this group window ends
            line_end = format_ass_time(
                group[idx + 1].start if idx + 1 < len(group) else active_word.end
            )

            # Build line text with color overrides
            line_parts = []
            for w_idx, w in enumerate(group):
                if w_idx == idx:
                    # Highlight color
                    line_parts.append(r"{\c&H" + highlight_color_bgr + r"&}" + w.word + r"{\c&H" + normal_color_bgr + r"&}")
                else:
                    line_parts.append(w.word)

            styled_text = " ".join(line_parts)
            dialogue_lines.append(
                f"Dialogue: 0,{line_start},{line_end},Default,,0,0,0,,{styled_text}"
            )

    output_ass_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(build_ass_header())
        f.write("\n".join(dialogue_lines))
        f.write("\n")

    console.print(f"[success]Generated dynamic ASS subtitles:[/success] {output_ass_path.name}")
    return output_ass_path