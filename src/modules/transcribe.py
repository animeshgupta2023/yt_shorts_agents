import gc
import json
from pathlib import Path
from typing import Any, Dict, List
import torch
from faster_whisper import WhisperModel
from pydantic import BaseModel

from src.utils.logger import console


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    probability: float


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: List[WordTimestamp]


class TranscriptionResult(BaseModel):
    language: str
    language_probability: float
    duration: float
    segments: List[Segment]
    words: List[WordTimestamp]
    transcript_path: Path
    aligned_words_path: Path


def transcribe_audio(
    audio_path: Path,
    workspace_dir: Path,
    model_size: str = "medium",
    device: str = "cuda",
    compute_type: str = "float16",
) -> TranscriptionResult:
    """
    Transcribe audio with faster-whisper using GPU acceleration,
    Silero VAD filtering, and sub-second word-level alignment.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    transcript_path = workspace_dir / "transcript.json"
    aligned_words_path = workspace_dir / "aligned_words.json"

    console.print(
        f"[stage]Loading faster-whisper model:[/stage] [info]{model_size}[/info] on [info]{device}[/info] ({compute_type})"
    )

    # Initialize model on RTX 4050
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=4,
    )

    console.print(f"[stage]Transcribing & Aligning:[/stage] {audio_path.name}")

    # Run transcription with word timestamps and Silero VAD to strip silent gaps
    segments_raw, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    console.print(
        f"[info]Detected language:[/info] [success]{info.language}[/success] "
        f"(probability: {info.language_probability:.2f})"
    )

    parsed_segments: List[Segment] = []
    all_words: List[WordTimestamp] = []

    for seg in segments_raw:
        seg_words: List[WordTimestamp] = []
        if seg.words:
            for w in seg.words:
                word_obj = WordTimestamp(
                    word=w.word.strip(),
                    start=round(w.start, 3),
                    end=round(w.end, 3),
                    probability=round(w.probability, 3),
                )
                seg_words.append(word_obj)
                all_words.append(word_obj)

        parsed_segments.append(
            Segment(
                id=seg.id,
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=seg.text.strip(),
                words=seg_words,
            )
        )

    # Save transcript.json
    transcript_payload: Dict[str, Any] = {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": round(info.duration, 3),
        "segments": [s.model_dump() for s in parsed_segments],
    }

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript_payload, f, indent=2, ensure_ascii=False)

    # Save aligned_words.json
    with open(aligned_words_path, "w", encoding="utf-8") as f:
        json.dump([w.model_dump() for w in all_words], f, indent=2, ensure_ascii=False)

    console.print(
        f"[success]Extracted {len(parsed_segments)} segments & {len(all_words)} aligned words.[/success]"
    )

    # Free GPU VRAM for downstream modules
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return TranscriptionResult(
        language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
        segments=parsed_segments,
        words=all_words,
        transcript_path=transcript_path,
        aligned_words_path=aligned_words_path,
    )