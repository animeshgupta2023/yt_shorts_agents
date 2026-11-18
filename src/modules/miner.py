import gc
import json
import re
from pathlib import Path
from typing import List, Optional
import torch
from pydantic import BaseModel, Field

from src.config import PipelineConfig
from src.utils.logger import console


class ClipCandidate(BaseModel):
    topic: str = Field(..., description="Short 1-3 word subject tag")
    value_proposition: str = Field(
        ..., description="What specific insight, skill, or mental model the viewer gains from this clip"
    )
    reasoning: str = Field(
        ..., description="How the clip progresses from premise to mechanism to takeaway"
    )
    hook: str = Field(..., description="Exact verbatim opening sentence in the first 3-5 seconds")
    banner_text: str = Field(
        ..., description="Punchy 4-7 word top-screen overlay framing the takeaway (no clickbait)"
    )
    title: str = Field(..., description="Clear, high-signal YouTube Shorts title under 60 characters")
    visual_cues: List[str] = Field(
        default_factory=list,
        description="2-3 concrete concepts or objects mentioned in the clip suitable for B-roll/chart overlays",
    )
    start_time: float = Field(..., description="Exact start timestamp in seconds matching the opening hook")
    end_time: float = Field(..., description="Exact end timestamp in seconds right after the takeaway")
    duration: float = Field(..., description="Duration in seconds (end_time - start_time)")
    score: float = Field(..., description="Value density + retention score from 1.0 to 10.0")


class ClipExtraction(BaseModel):
    clips: List[ClipCandidate]


def format_transcript_with_timestamps(segments: List[dict]) -> str: 
    """Format segments into compact timestamped lines: [MM:SS] text."""
    lines = []
    for s in segments:
        start_min = int(s["start"] // 60)
        start_sec = int(s["start"] % 60)
        lines.append(f"[{start_min:02d}:{start_sec:02d}] (sec: {s['start']:.1f}-{s['end']:.1f}) {s['text']}")
    return "\n".join(lines)


def snap_to_word_boundaries(clips: List[ClipCandidate], words_path: Path) -> List[ClipCandidate]:
    """Snap start and end times to the closest spoken word boundaries."""
    if not words_path.exists():
        return clips

    with open(words_path, "r", encoding="utf-8") as f:
        words = json.load(f)

    if not words:
        return clips

    snapped_clips = []
    for clip in clips:
        # Snap start_time to closest word start
        best_start = min(words, key=lambda w: abs(w["start"] - clip.start_time))["start"]
        # Snap end_time to closest word end
        best_end = min(words, key=lambda w: abs(w["end"] - clip.end_time))["end"]

        # Ensure minimum clip duration
        if best_end - best_start < 15.0:
            best_end = best_start + 15.0

        snapped_clips.append(
            ClipCandidate(
                start_time=round(best_start, 2),
                end_time=round(best_end, 2),
                duration=round(best_end - best_start, 2),
                hook=clip.hook,
                value_proposition=clip.value_proposition,
                banner_text=clip.banner_text,
                visual_cues=clip.visual_cues,
                title=clip.title,
                topic=clip.topic,
                score=clip.score,
                reasoning=clip.reasoning,
            )
        )
    return snapped_clips


def mine_with_gemini(
    transcript_text: str,
    video_title: str,
    api_key: str,
    model_name: str = "gemini-4-hydrogen",
    max_clips: int = 3,
    temperature: float = 0.2
) -> ClipExtraction:
    """Extract high-value, standalone Shorts using structured Pydantic output."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    system_instruction = (
        "You are an expert educational and editorial producer for short-form video (YouTube Shorts, Reels). "
        "Your job is to extract self-contained segments with high signal-to-noise ratio—clips that teach a "
        "clear concept, break down a framework, or reveal a non-obvious insight without fluff."
    )

    context_blocks = [f"Video Title: {video_title}"]
    metadata_context = "\n\n".join(context_blocks)

    prompt = f"""Analyze the timestamped transcript below and extract the top {max_clips} distinct, high-value standalone Shorts.

{metadata_context}

### Content Quality & Selection Criteria:
1. **High Signal, Zero Fluff:** Prioritize segments containing concrete frameworks, step-by-step explanations, root-cause reasoning, or memorable analogies. Reject empty banter, vague platitudes, and meandering stories without a clear point.
2. **Value-Driven Hook (0-5s):** `start_time` MUST begin on a sentence that immediately frames a specific problem, counterintuitive truth, or clear promise—never mid-thought or after a slow throat-clearing intro.
3. **Complete Micro-Arc:** Every clip must stand alone for a viewer who has not seen the full video:
   - *Setup:* What question or problem is being addressed?
   - *Core Insight:* Why does it work or happen?
   - *Resolution:* A crisp concluding takeaway or punchline (`end_time` stops cleanly right after it).
4. **No Unresolved References:** Do not start clips with dangling pronouns ("That's why he did it...") unless the context is immediately clear within 2 seconds.
5. **Strict Duration & Non-Overlap:** Every clip MUST be between **25.0 and 58.0 seconds** long (`end_time - start_time`), and all {max_clips} clips must cover completely separate, non-overlapping sections of the transcript.
6. **Exclude Non-Content:** Skip sponsor reads, channel intros, merch plugs, and "like & subscribe" outros.
7. **Exact Timestamps:** Base `start_time` and `end_time` strictly on the `(sec: X.X-Y.Y)` markers in the transcript.

Transcript:
{transcript_text}
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=ClipExtraction,
            temperature=temperature
        ),
    )

    if hasattr(response, "parsed") and response.parsed:
        return response.parsed

    data = json.loads(response.text)
    return ClipExtraction.model_validate(data)


def mine_with_local_gguf(
    transcript_text: str,
    video_title: str,
    config: PipelineConfig,
    max_clips: int = 3,
    temperature: float = 0.2
) -> ClipExtraction:
    """Extract viral clips using local llama-cpp-python with CUDA offload."""
    from llama_cpp import Llama

    model_path = Path(config.llm.local_gguf.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Local GGUF model not found at: {model_path}")

    console.print(f"[stage]Loading local GGUF model onto GPU:[/stage] {model_path.name}")

    llm = Llama(
        model_path=str(model_path),
        n_ctx=config.llm.local_gguf.n_ctx,
        n_gpu_layers=config.llm.local_gguf.n_gpu_layers,
        verbose=False,
    )

    system_instruction = (
        "You are a viral short-form video editor for YouTube Shorts and TikTok. "
        "Analyze the transcript and identify the top engaging clips. "
        "Each clip must be between 25 and 60 seconds, have a compelling hook, and a complete standalone story. "
        "Respond strictly with valid JSON matching the requested schema."
    )

    user_prompt = f"""Video Title: "{video_title}"
Find the top {max_clips} viral shorts (strictly 25-60 seconds long).
Ensure exact numeric start_time and end_time timestamps based on the transcript.

Transcript:
{transcript_text}
"""

    console.print("[info]Inferencing with local model...[/info]")
    
    # Use create_chat_completion with structured schema enforcement
    schema = ClipExtraction.model_json_schema()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_object",
            "schema": schema,
        },
        temperature=temperature,
        max_tokens=1500,
    )

    raw_text = response["choices"][0]["message"]["content"].strip()

    # Strip markdown if present
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\n", "", raw_text)
        raw_text = re.sub(r"\n```$", "", raw_text)

    # Parse and validate against Pydantic schema
    data = json.loads(raw_text)

    # Free GPU VRAM immediately
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ClipExtraction.model_validate(data)


def mine_highlights(
    workspace_dir: Path,
    config: PipelineConfig,
    max_clips: int = 3,
    override_provider: Optional[str] = None,
) -> Path:
    """Orchestrates candidate mining, falls back if needed, snaps timestamps, and saves clips.json."""
    transcript_path = workspace_dir / "transcript.json"
    metadata_path = workspace_dir / "metadata.json"
    words_path = workspace_dir / "aligned_words.json"
    clips_path = workspace_dir / "clips.json"

    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found at {transcript_path}. Run transcribe first.")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    video_title = "YouTube Video"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            video_title = meta.get("title", video_title)

    # start min, start sec, end min, end sec, text -> formatted_transcript
    formatted_transcript = format_transcript_with_timestamps(transcript_data.get("segments", []))
    provider = override_provider or config.llm.provider

    result: Optional[ClipExtraction] = None

    if provider == "gemini":
        api_key = config.llm.gemini.api_key
        if not api_key:
            console.print("[warning]No GEMINI_API_KEY found. Falling back to local GGUF...[/warning]")
            provider = "local_gguf"
        else:
            try:
                console.print(f"[stage]Mining highlights with Gemini API:[/stage] {config.llm.gemini.model}")
                result = mine_with_gemini(
                    transcript_text=formatted_transcript,
                    video_title=video_title,
                    api_key=api_key,
                    model_name=config.llm.gemini.model,
                    max_clips=max_clips,
                    temperature=config.llm.gemini.temperature

                )
            except Exception as e:
                console.print(f"[warning]Gemini API failed: {e}. Falling back to local GGUF...[/warning]")
                provider = "local_gguf"

    if provider == "local_gguf":
        result = mine_with_local_gguf(
            transcript_text=formatted_transcript,
            video_title=video_title,
            config=config,
            max_clips=max_clips,
            temperature=config.llm.local_gguf.temperature
        )

    if not result or not result.clips:
        raise RuntimeError("No highlight clips could be mined from transcript.")

    # Snap timestamps to actual spoken words
    snapped_clips = snap_to_word_boundaries(result.clips, words_path)

    # Sort clips by score descending
    snapped_clips.sort(key=lambda c: c.score, reverse=True)

    # Save to clips.json
    output_payload = {
        "video_title": video_title,
        "clip_count": len(snapped_clips),
        "clips": [c.model_dump() for c in snapped_clips[:max_clips]],
    }

    with open(clips_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    console.print(f"[success]Mined {len(output_payload['clips'])} viral clips successfully![/success]")
    return clips_path