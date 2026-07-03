import importlib.util
import os
import shutil

from .config import settings
from .culture_profiles import list_profiles
from .lyrics_engine import provider_status
from .model_registry import registry_status


def inspect_capabilities() -> dict:
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    lyrics = provider_status()
    registry = registry_status()
    transformer_runtime = importlib.util.find_spec("transformers") is not None
    torch_runtime = importlib.util.find_spec("torch") is not None
    musicgen_runtime = transformer_runtime and torch_runtime
    musicgen_ready = settings.enable_local_musicgen and musicgen_runtime and bool(os.getenv("MUSICGEN_MODEL"))
    return {
        "ffmpeg": shutil.which(settings.ffmpeg_bin) is not None,
        "ffprobe": shutil.which(settings.ffprobe_bin) is not None,
        "demucs": importlib.util.find_spec("demucs") is not None,
        "chord_detection": importlib.util.find_spec("librosa") is not None,
        "midi_transcription": importlib.util.find_spec("mido") is not None and importlib.util.find_spec("librosa") is not None,
        "daw_export": True,
        "lyrics_provider_configured": bool(lyrics["configured"]),
        "lyrics_provider": lyrics["provider"],
        "musicgen_enabled": settings.enable_local_musicgen,
        "musicgen_runtime": musicgen_runtime,
        "complete_song_pipeline": musicgen_ready,
        "cuda_available": cuda_available,
        "separation_models": ["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra_q"],
        "generation_model": os.getenv("MUSICGEN_MODEL", "not configured"),
        "culture_profile_count": len(list_profiles()),
        "culture_model_count": int(registry["culture_model_count"]),
    }
