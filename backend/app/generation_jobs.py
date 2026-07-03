from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import Session

from .audio_probe import AudioProcessingError, analyze_audio
from .config import settings
from .culture_profiles import enhance_prompt, get_profile
from .file_registry import register_file
from .model_registry import select_generation_model
from .models import Job, Project
from .storage import project_dir, safe_filename

ProgressCallback = Callable[[float, str], None]


def generate_audio(request: dict[str, Any], project: Project, job_id: str, progress: ProgressCallback) -> tuple[Path, dict[str, Any]]:
    if not settings.enable_local_musicgen:
        raise AudioProcessingError("Local music generation is disabled on this worker")
    try:
        import torch
        from scipy.io.wavfile import write as write_wav
        from transformers import AutoProcessor, MusicgenForConditionalGeneration, set_seed
    except ImportError as exc:
        raise AudioProcessingError("Generation dependencies are not installed") from exc
    culture_profile_id = request.get("culture_profile_id")
    selection = select_generation_model(culture_profile_id)
    culture_profile = get_profile(culture_profile_id)
    prompt_text = enhance_prompt(request["prompt"], culture_profile_id, request.get("language"))
    device = settings.musicgen_device if settings.musicgen_device != "cuda" or torch.cuda.is_available() else "cpu"
    progress(5, f"Loading {selection.model_id} on {device}")
    processor = AutoProcessor.from_pretrained(selection.model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(selection.model_id).to(device)
    if request.get("seed") is not None:
        set_seed(int(request["seed"]))
    inputs = processor(text=[prompt_text], padding=True, return_tensors="pt").to(device)
    duration_seconds = int(request.get("duration_seconds", 12))
    progress(20, "Generating audio tokens")
    with torch.inference_mode():
        values = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=float(request.get("guidance_scale", 3.0)),
            max_new_tokens=max(128, duration_seconds * 50),
        )
    sample_rate = int(model.config.audio_encoder.sampling_rate)
    output = project_dir(project.id) / "generated" / f"{job_id}_{safe_filename(request.get('name', 'generated'))}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    waveform = values[0, 0].detach().cpu().numpy()
    peak = max(float(abs(waveform).max()), 1e-9)
    write_wav(output, sample_rate, (waveform / peak * 0.95 * 32767).astype("int16"))
    metadata = {
        "prompt": request["prompt"],
        "enhanced_prompt": prompt_text,
        "model": selection.model_id,
        "model_source": selection.source,
        "culture_profile_id": culture_profile_id,
        "culture_profile": culture_profile,
        "fine_tuned_for_profile": selection.fine_tuned_for_profile,
        "language": request.get("language"),
        "seed": request.get("seed"),
        "duration_seconds_requested": duration_seconds,
        "device": device,
    }
    return output, metadata


def process_generate(session: Session, job: Job, progress):
    request = job.request_json
    project = session.get(Project, job.project_id) if job.project_id else None
    if project is None:
        project = Project(name=request.get("name", "Generated music"))
        session.add(project)
        session.flush()
        job.project_id = project.id
    output, metadata = generate_audio(request, project, job.id, progress)
    progress(90, "Analyzing generated audio")
    analysis = analyze_audio(output)
    item = register_file(session, project.id, output, "generated", request.get("name", "Generated music"), metadata_json={**metadata, "analysis": analysis})
    project.analysis = {**(project.analysis or {}), **analysis, "culture_profile_id": request.get("culture_profile_id")}
    return {"file_id": item.id, "project_id": project.id, "analysis": analysis, **metadata}
