from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .culture_profiles import get_profile


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


class CultureAwareRequest(BaseModel):
    language: str | None = None
    culture_profile_id: str | None = None

    @field_validator("culture_profile_id")
    @classmethod
    def validate_culture_profile(cls, value: str | None) -> str | None:
        if value and get_profile(value) is None:
            raise ValueError("Unknown culture profile")
        return value


class GenerationRequest(CultureAwareRequest):
    prompt: str = Field(min_length=8, max_length=2000)
    duration_seconds: int = Field(30, ge=4, le=300)
    name: str = "generated"
    seed: int | None = None
    guidance_scale: float = Field(3.0, ge=1.0, le=10.0)
    sections: str | None = Field(None, max_length=1000)

    @field_validator("prompt")
    @classmethod
    def strip_prompt(cls, value: str) -> str:
        return value.strip()


class LyricsRequest(CultureAwareRequest):
    prompt: str = Field(min_length=8, max_length=3000)
    title: str = Field("Untitled", min_length=1, max_length=200)
    mood: str | None = Field(None, max_length=200)
    structure: str | None = Field(None, max_length=500)


class ChordRequest(BaseModel):
    source_file_id: str
    name: str = Field("Chord Map", min_length=1, max_length=200)


class MidiRequest(BaseModel):
    source_file_id: str
    name: str = Field("MIDI Transcription", min_length=1, max_length=200)
    tempo_bpm: float | None = Field(None, ge=30.0, le=300.0)


class SingingRequest(BaseModel):
    lyrics_file_id: str
    midi_file_id: str | None = None
    name: str = Field("Lead Vocals", min_length=1, max_length=200)
    title: str = Field("BeatMaster Song", min_length=1, max_length=200)
    language: str = Field("English", min_length=1, max_length=100)
    voice_id: str | None = Field(None, max_length=200)


class DawExportRequest(BaseModel):
    file_ids: list[str] | None = None
    name: str = Field("DAW Package", min_length=1, max_length=200)
    tempo_bpm: float | None = Field(None, ge=30.0, le=300.0)
    time_signature: str = Field("4/4", pattern=r"^[1-9][0-9]?/[1-9][0-9]?$", max_length=8)


class CompleteSongRequest(CultureAwareRequest):
    prompt: str = Field(min_length=8, max_length=3000)
    title: str = Field("BeatMaster Song", min_length=1, max_length=200)
    mood: str | None = Field(None, max_length=200)
    structure: str | None = Field(None, max_length=1000)
    duration_seconds: int = Field(180, ge=4, le=300)
    seed: int | None = None
    guidance_scale: float = Field(3.0, ge=1.0, le=10.0)
    include_lyrics: bool = True
    render_vocals: bool = False
    voice_id: str | None = Field(None, max_length=200)
    vocal_gain_db: float = Field(-3.0, ge=-24.0, le=12.0)
    separate_stems: bool = True
    extract_chords: bool = True
    extract_midi: bool = True
    export_daw: bool = True
    separation_model: Literal["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra_q"] = "htdemucs"
    output_format: Literal["wav", "flac", "mp3"] = "wav"
    time_signature: str = Field("4/4", pattern=r"^[1-9][0-9]?/[1-9][0-9]?$", max_length=8)


class CapabilityOut(BaseModel):
    ffmpeg: bool
    ffprobe: bool
    demucs: bool
    chord_detection: bool
    midi_transcription: bool
    daw_export: bool
    lyrics_provider_configured: bool
    lyrics_provider: str | None
    singing_provider_configured: bool
    singing_provider: str | None
    musicgen_enabled: bool
    musicgen_runtime: bool
    complete_song_pipeline: bool
    cuda_available: bool
    separation_models: list[str]
    generation_model: str
    culture_profile_count: int
    culture_model_count: int
