import gc
import json
import os
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from faster_whisper import WhisperModel, decode_audio
from huggingface_hub import snapshot_download
from pydantic import BaseModel

from src.utils.logger import console
from src.utils.shell import run_command

CT2_REPO_MAP = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}


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


def get_local_model_path(models_dir: Path, model_size: str) -> Path:
    """Return local path inside models/, downloading CTranslate2 weights directly there if missing."""
    target_dir = models_dir / f"whisper-{model_size}"
    if (target_dir / "model.bin").exists():
        return target_dir

    repo_id = CT2_REPO_MAP.get(model_size)
    if not repo_id:
        raise FileNotFoundError(
            f"Model '{target_dir}' not found and no default Hugging Face repo mapped for '{model_size}'."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[info]Downloading {repo_id} directly into {target_dir}...[/info]")
    snapshot_download(repo_id=repo_id, local_dir=str(target_dir))
    return target_dir


def extract_audio_probe(audio_path: Path, workspace_dir: Path, probe_seconds: int = 30) -> np.ndarray:
    """
    Read a 30s 16kHz mono float32 probe without decoding the entire file.
    Uses zero-overhead wave byte seeking for standard 16kHz mono WAVs, with FFmpeg fallback.
    """
    target_sr = 16000
    target_frames = target_sr * probe_seconds

    if audio_path.suffix.lower() == ".wav":
        with wave.open(str(audio_path), "rb") as wf:
            if wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == target_sr:
                total_frames = wf.getnframes()
                # Skip the first 45s on longer videos to avoid intro music or sponsor jingles
                start_frame = target_sr * 45 if total_frames > target_sr * 90 else 0
                wf.setpos(start_frame)
                raw_pcm = wf.readframes(target_frames)
                return np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0

    # Fallback for non-WAV inputs or non-standard sample rates
    probe_path = workspace_dir / "audio_probe.wav"
    try:
        run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-t",
                str(probe_seconds),
                "-ar",
                str(target_sr),
                "-ac",
                "1",
                str(probe_path),
            ],
            desc="Extracting 30s language probe",
        )
        return decode_audio(str(probe_path), sampling_rate=target_sr)
    finally:
        if probe_path.exists():
            os.remove(probe_path)


def _free_vram(model: Optional[WhisperModel]) -> None:
    """Delete CTranslate2 model reference and release CUDA VRAM."""
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe_audio(
    audio_path: Path,
    workspace_dir: Path,
    model_size: str = "medium",
    device: str = "cuda",
    compute_type: str = "float16",
    models_dir: Path = Path("models"),
) -> TranscriptionResult:
    """
    Transcribe audio with faster-whisper using local models/ storage,
    automatic Hindi model routing, Silero VAD filtering, and word-level alignment.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    transcript_path = workspace_dir / "transcript.json"
    aligned_words_path = workspace_dir / "aligned_words.json"

    base_model_dir = get_local_model_path(models_dir, model_size)
    hindi_model_dir = models_dir / "whisper-hindi-small"
    has_hindi_model = (hindi_model_dir / "model.bin").exists()

    console.print(
        f"[stage]Loading faster-whisper model:[/stage] [info]{base_model_dir}[/info] on [info]{device}[/info] ({compute_type})"
    )

    model: Optional[WhisperModel] = WhisperModel(
        str(base_model_dir),
        device=device,
        compute_type=compute_type,
        cpu_threads=4,
        local_files_only=True,
    )

    try:
        forced_language: Optional[str] = None

        # Probe 30s of speech to check if audio is Hindi and swap to fine-tuned Hindi model
        if has_hindi_model:
            sample_30s = extract_audio_probe(audio_path, workspace_dir, probe_seconds=30)
            detected_lang, lang_prob, _ = model.detect_language(sample_30s)
            console.print(f"[info]Language probe:[/info] {detected_lang} (probability: {lang_prob:.2f})")

            if detected_lang == "hi" and lang_prob >= 0.65:
                _free_vram(model)
                model = None
                console.print(
                    f"[stage]Switching to fine-tuned Hindi model:[/stage] [info]{hindi_model_dir}[/info]"
                )
                model = WhisperModel(
                    str(hindi_model_dir),
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=4,
                    local_files_only=True,
                )
                forced_language = "hi"

        console.print(f"[stage]Transcribing & Aligning:[/stage] {audio_path.name}")

        segments_raw, info = model.transcribe(
            str(audio_path),
            language=forced_language,
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

        transcript_payload: Dict[str, Any] = {
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 3),
            "segments": [s.model_dump() for s in parsed_segments],
        }

        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript_payload, f, indent=2, ensure_ascii=False)

        with open(aligned_words_path, "w", encoding="utf-8") as f:
            json.dump([w.model_dump() for w in all_words], f, indent=2, ensure_ascii=False)

        console.print(
            f"[success]Extracted {len(parsed_segments)} segments & {len(all_words)} aligned words.[/success]"
        )

        return TranscriptionResult(
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            segments=parsed_segments,
            words=all_words,
            transcript_path=transcript_path,
            aligned_words_path=aligned_words_path,
        )
    finally:
        _free_vram(model)