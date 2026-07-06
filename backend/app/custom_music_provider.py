import os
from pathlib import Path

from .beatmaster_model_provider import generate_with_beatmaster_model


def provider_name():
    return os.getenv("MUSIC_GENERATION_PROVIDER", "transformers").strip().lower()


def generate_if_selected(request, output: Path, progress):
    if provider_name() not in {"beatmaster", "beatmaster-model-server", "custom"}:
        return None
    progress(5, "Calling BeatMaster custom model server")
    metadata = generate_with_beatmaster_model(request, output)
    progress(86, "BeatMaster custom model audio received")
    return metadata
