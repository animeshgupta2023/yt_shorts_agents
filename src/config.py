import os
from pathlib import Path
from typing import Literal
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class GeminiConfig(BaseModel):
    model: str = "gemini-4-hydrogen"
    temperature: float = 0.2 
    api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

class LocalGGUFConfig(BaseModel):
    model_path: Path = Path("./models/Qwen3.5-4B-Q4_K_M.gguf")
    n_ctx: int = 4096
    n_gpu_layers: int = -1


class LLMConfig(BaseModel):
    provider: Literal["gemini", "local_gguf"] = "gemini"
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    local_gguf: LocalGGUFConfig = Field(default_factory=LocalGGUFConfig)


class WhisperConfig(BaseModel):
    model_size: str = "medium"
    device: str = "cuda"
    compute_type: str = "float16"

class ProjectConfig(BaseModel):
    workspace_dir: Path = Path("./workspace")
    models_dir: Path = Path("./models")

class ReframeConfig(BaseModel):
    target_aspect_ratio: tuple[int, int] = (9, 16)
    scene_threshold: float = 27.0
    deadband_pixels: int = 25
    ema_alpha: float = 0.15

class SubtitleConfig(BaseModel):
    font_name: str = "Montserrat Black"
    font_size: int = 24
    words_per_group: int = 3
    bottom_margin_percent: float = 0.25

class IngestConfig(BaseModel):
    max_height: int = 1080
    cookies_from_browser: str | None = "chrome"
    player_clients: list[str] = Field(default_factory=lambda: ["tv", "web_safari", "web"])

class PipelineConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    reframe: ReframeConfig = Field(default_factory=ReframeConfig)
    subtitles: SubtitleConfig = Field(default_factory=SubtitleConfig)

def load_config(config_path: str = "configs/config.yaml") -> PipelineConfig:
    p = Path(config_path)
    if not p.exists():
        return PipelineConfig()
    with open(p, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}
    return PipelineConfig(**raw_data)