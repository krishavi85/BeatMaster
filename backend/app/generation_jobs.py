from __future__ import annotations

import math
import os
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


def _crossfade(segments, sample_rate: int, seconds: float = 1.0):
    import numpy as np

    if not segments:
        raise AudioProcessingError("The generation model returned no audio")
    combined = segments[0]
    for segment in segments[1:]:
        overlap = min(int(sample_rate * seconds), len(combined) // 4, len(segment) // 4)
        if overlap <= 0:
            combined = np.concatenate([combined, segment])
            continue
        fade_out = np.linspace(1.0, 0.0, overlap, endpoint=False, dtype=np.float32)
        fade_in = 1.0 - fade_out
        blended = combined[-overlap:] * fade_out + segment[:overlap] * fade_in
        combined = np.concatenate([combined[:-overlap], blended, segment[overlap:]])
    return combined


def generate_audio(request: dict[str, Any], project: Project, job_id: str, progress: ProgressCallback) -> tuple[Path, dict[str, Any]]:
    if not settings.enable_local_musicgen:
        raise AudioProcessingError("Local music generation is disabled on this worker")
    try:
        import numpy as np
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
    sample_rate = int(model.config.audio_encoder.sampling_rate)
    duration_seconds = int(request.get("duration_seconds", 12))
    segment_limit = max(8, min(30, int(os.getenv("MUSICGEN_SEGMENT_SECONDS", "30"))))
    segment_count = max(1, int(math.ceil(duration_seconds / segment_limit)))
    remaining = duration_seconds
    segments = []
    section_names = request.get("sections") or request.get("structure") or ""
    for index in range(segment_count):
        segment_seconds = min(segment_limit, remaining)
        remaining -= segment_seconds
        if request.get("seed") is not None:
            set_seed(int(request["seed"]) + index)
        section_prompt = (
            f"{prompt_text}\n\nLong-form section {index + 1} of {segment_count}. "
            f"Maintain musical continuity with the previous and next sections. "
            f"Requested structure: {section_names or 'intro, development, climax and outro'}."
        )
        inputs = processor(text=[section_prompt], padding=True, return_tensors="pt").to(device)
        progress(12 + (index / segment_count) * 72, f"Generating section {index + 1} of {segment_count}")
        with torch.inference_mode():
            values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=float(request.get("guidance_scale", 3.0)),
                max_new_tokens=max(128, segment_seconds * 50),
            )
        waveform = values[0, 0].detach().cpu().float().numpy()
        target_samples = max(1, int(segment_seconds * sample_rate))
        if len(waveform) > target_samples:
            waveform = waveform[:target_samples]
        segments.append(np.asarray(waveform, dtype=np.float32))
    progress(86, "Crossfading generated song sections")
    waveform = _crossfade(segments, sample_rate, seconds=float(os.getenv("MUSICGEN_CROSSFADE_SECONDS", "1.0")))
    peak = max(float(np.max(np.abs(waveform))), 1e-9)
    output = project_dir(project.id) / "generated" / f"{job_id}_{safe_filename(request.get('name') or request.get('title') or 'generated')}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
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
        "duration_seconds_rendered": round(len(waveform) / sample_rate, 3),
        "segment_count": segment_count,
        "segment_limit_seconds": segment_limit,
        "device": device,
    }
    return output, metadata


def process_generate(session: Session, job: Job, progress):
    request = job.request_json
    project = session.get(Project, job.project_id) if job.project_id else None
    if project is None:
        project = Project(name=request.get("name") or request.get("title") or "Generated music")
        session.add(project)
        session.flush()
        job.project_id = project.id
    output, metadata = generate_audio(request, project, job.id, progress)
    progress(90, "Analyzing generated audio")
    analysis = analyze_audio(output)
    item = register_file(session, project.id, output, "generated", request.get("name") or request.get("title") or "Generated music", metadata_json={**metadata, "analysis": analysis})
    project.analysis = {**(project.analysis or {}), **analysis, "culture_profile_id": request.get("culture_profile_id")}
    return {"file_id": item.id, "project_id": project.id, "analysis": analysis, **metadata}
