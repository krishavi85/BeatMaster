import os
from sqlalchemy.orm import Session
from .audio_probe import AudioProcessingError, analyze_audio
from .config import settings
from .file_registry import register_file
from .models import Job, Project
from .storage import project_dir, safe_filename


def process_generate(session: Session, job: Job, progress):
    if not settings.enable_local_musicgen:
        raise AudioProcessingError("Local music generation is disabled on this worker")
    model_id = os.getenv("MUSICGEN_MODEL", "").strip()
    if not model_id:
        raise AudioProcessingError("MUSICGEN_MODEL is required")
    try:
        import torch
        from scipy.io.wavfile import write as write_wav
        from transformers import AutoProcessor, MusicgenForConditionalGeneration, set_seed
    except ImportError as exc:
        raise AudioProcessingError("Generation dependencies are not installed") from exc
    request = job.request_json
    project = Project(name=request.get("name", "Generated music"))
    session.add(project)
    session.flush()
    job.project_id = project.id
    device = settings.musicgen_device if settings.musicgen_device != "cuda" or torch.cuda.is_available() else "cpu"
    progress(5, f"Loading generation model on {device}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id).to(device)
    if request.get("seed") is not None:
        set_seed(int(request["seed"]))
    inputs = processor(text=[request["prompt"]], padding=True, return_tensors="pt").to(device)
    progress(20, "Generating audio tokens")
    with torch.inference_mode():
        values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=max(128, int(request.get("duration_seconds", 12) * 50)))
    sample_rate = int(model.config.audio_encoder.sampling_rate)
    output = project_dir(project.id) / "generated" / f"{job.id}_{safe_filename(request.get('name', 'generated'))}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    waveform = values[0, 0].detach().cpu().numpy()
    peak = max(float(abs(waveform).max()), 1e-9)
    write_wav(output, sample_rate, (waveform / peak * 0.95 * 32767).astype("int16"))
    progress(92, "Analyzing generated audio")
    item = register_file(session, project.id, output, "generated", request.get("name", "Generated music"), metadata_json={"prompt": request["prompt"], "model": model_id, "seed": request.get("seed")})
    return {"file_id": item.id, "project_id": project.id, "analysis": analyze_audio(output), "model": model_id}
