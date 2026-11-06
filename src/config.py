import os
from pathlib import Path
from typing import Literal
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class GeminiConfig(BaseModel):
    model: str = "gemini-3.5-flash-lite"
    temperature: float = 0.2
    api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))


class LocalGGUFConfig(BaseModel):
    model_path: str = "./models/qwen.gguf"
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


class PipelineConfig(BaseModel):
    workspace_dir: Path = Path("./workspace")
    models_dir: Path = Path("./models")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)


def load_config(config_path: str = "configs/config.yaml") -> PipelineConfig:
    p = Path(config_path)
    if not p.exists():
        return PipelineConfig()
    with open(p, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}
    return PipelineConfig(**raw_data)