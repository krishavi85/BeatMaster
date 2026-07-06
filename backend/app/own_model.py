import os
from pathlib import Path

from .audio_probe import AudioProcessingError
from .beatmaster_model_provider import generate_with_beatmaster_model
from .culture_profiles import enhance_prompt, get_profile
from .storage import project_dir, safe_filename


def selected():
    return os.getenv("MUSIC_GENERATION_PROVIDER", "transformers").strip().lower() in {"beatmaster", "custom"}


def generate(request, project, job_id, progress):
    if not selected():
        return None
    profile_id = request.get("culture_profile_id")
    prompt = enhance_prompt(request["prompt"], profile_id, request.get("language"))
    output = project_dir(project.id) / "generated" / f"{job_id}_{safe_filename(request.get('name') or request.get('title') or 'generated')}.wav"
    progress(5, "Calling BeatMaster model server")
    metadata = generate_with_beatmaster_model({**request, "prompt": prompt}, output)
    if not output.exists():
        raise AudioProcessingError("BeatMaster model server did not create audio")
    progress(86, "BeatMaster model audio received")
    return output, {
        "prompt": request["prompt"],
        "enhanced_prompt": prompt,
        "culture_profile_id": profile_id,
        "culture_profile": get_profile(profile_id),
        "language": request.get("language"),
        **metadata,
    }
