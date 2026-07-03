import importlib.util
import os
import shutil

from .config import settings
from .culture_profiles import list_profiles
from .lyrics_engine import provider_status as lyrics_provider_status
from .model_registry import registry_status
from .singing_engine import provider_status as singing_provider_status


def inspect_capabilities() -> dict:
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    lyrics = lyrics_provider_status()
    singing = singing_provider_status()
    registry = registry_status()
    transformer_runtime = importlib.util.find_spec("transformers") is not None
    torch_runtime = importlib.util.find_spec("torch") is not None
    musicgen_runtime = transformer_runtime and torch_runtime
    ffmpeg_ready = shutil.which(settings.ffmpeg_bin) is not None
    ffprobe_ready = shutil.which(settings.ffprobe_bin) is not None
    demucs_ready = importlib.util.find_spec("demucs") is not None
    chord_ready = importlib.util.find_spec("librosa") is not None
    midi_ready = importlib.util.find_spec("mido") is not None and chord_ready
    musicgen_ready = settings.enable_local_musicgen and musicgen_runtime and bool(os.getenv("MUSICGEN_MODEL"))
    return {
        "ffmpeg": ffmpeg_ready,
        "ffprobe": ffprobe_ready,
        "demucs": demucs_ready,
        "chord_detection": chord_ready,
        "midi_transcription": midi_ready,
        "daw_export": True,
        "lyrics_provider_configured": bool(lyrics["configured"]),
        "lyrics_provider": lyrics["provider"],
        "singing_provider_configured": bool(singing["configured"]),
        "singing_provider": singing["provider"],
        "musicgen_enabled": settings.enable_local_musicgen,
        "musicgen_runtime": musicgen_runtime,
        "complete_song_pipeline": musicgen_ready and ffmpeg_ready and ffprobe_ready and demucs_ready and chord_ready and midi_ready,
        "cuda_available": cuda_available,
        "separation_models": ["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra_q"],
        "generation_model": os.getenv("MUSICGEN_MODEL", "not configured"),
        "culture_profile_count": len(list_profiles()),
        "culture_model_count": int(registry["culture_model_count"]),
    }
