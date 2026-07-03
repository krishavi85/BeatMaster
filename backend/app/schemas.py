from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

class AudioFileOut(BaseModel):
    id: str
    project_id: str
    kind: str
    label: str
    original_name: str | None
    mime_type: str | None
    size_bytes: int
    duration_seconds: float | None
    sample_rate: int | None
    channels: int | None
    codec: str | None
    created_at: datetime
    metadata_json: dict[str, Any] | None
    stream_url: str
    download_url: str

class JobOut(BaseModel):
    id: str
    project_id: str | None
    type: str
    status: str
    progress: float
    stage: str
    request_json: dict[str, Any]
    result_json: dict[str, Any] | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    model_config = {"from_attributes": True}

class ProjectOut(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    analysis: dict[str, Any] | None
    files: list[AudioFileOut]
    jobs: list[JobOut]

class SeparationRequest(BaseModel):
    source_file_id: str
    model: Literal["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra_q"] = "htdemucs"
    two_stems: Literal["vocals", "drums", "bass", "other"] | None = None
    output_format: Literal["wav", "flac", "mp3"] = "wav"

class MixTrack(BaseModel):
    file_id: str
    gain_db: float = Field(0.0, ge=-60.0, le=18.0)
    pan: float = Field(0.0, ge=-1.0, le=1.0)
    mute: bool = False

class MixRequest(BaseModel):
    tracks: list[MixTrack] = Field(min_length=1)
    name: str = "mix"
    output_format: Literal["wav", "flac", "mp3"] = "wav"

class MasterRequest(BaseModel):
    source_file_id: str
    name: str = "master"
    target_lufs: float = Field(-14.0, ge=-24.0, le=-5.0)
    true_peak_db: float = Field(-1.0, ge=-3.0, le=-0.1)
    loudness_range: float = Field(11.0, ge=1.0, le=20.0)
    style: Literal["transparent", "warm", "bright", "punchy", "wide"] = "transparent"
    output_format: Literal["wav", "flac", "mp3"] = "wav"

class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=8, max_length=1000)
    duration_seconds: int = Field(12, ge=4, le=30)
    name: str = "generated"
    seed: int | None = None

    @field_validator("prompt")
    @classmethod
    def strip_prompt(cls, value: str) -> str:
        return value.strip()

class CapabilityOut(BaseModel):
    ffmpeg: bool
    ffprobe: bool
    demucs: bool
    musicgen_enabled: bool
    musicgen_runtime: bool
    cuda_available: bool
    separation_models: list[str]
    generation_model: str
