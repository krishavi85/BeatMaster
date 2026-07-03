from typing import Annotated
from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from .api_helpers import create_job, get_db, load_project
from .config import settings

router = APIRouter(tags=["ui"])

def back(project_id: str):
    return RedirectResponse(f"/ui/projects/{project_id}", status_code=303)

@router.post("/ui/projects/{project_id}/separate", include_in_schema=False)
def separate_ui(
    project_id: str,
    source_file_id: Annotated[str, Form()],
    model: Annotated[str, Form()] = "htdemucs",
    two_stems: Annotated[str, Form()] = "",
    output_format: Annotated[str, Form()] = "wav",
    session: Session = Depends(get_db),
):
    load_project(session, project_id)
    if model not in {"htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra_q"}:
        raise HTTPException(status_code=400, detail="Unsupported Demucs model")
    if output_format not in {"wav", "flac", "mp3"}:
        raise HTTPException(status_code=400, detail="Unsupported output format")
    create_job(session, project_id, "separate", {"source_file_id": source_file_id, "model": model, "two_stems": two_stems or None, "output_format": output_format})
    return back(project_id)

@router.post("/ui/projects/{project_id}/mix", include_in_schema=False)
def mix_ui(
    project_id: str,
    file_ids: Annotated[list[str], Form()],
    name: Annotated[str, Form()] = "BeatMaster Mix",
    gain_db: Annotated[float, Form()] = 0.0,
    pan: Annotated[float, Form()] = 0.0,
    output_format: Annotated[str, Form()] = "wav",
    session: Session = Depends(get_db),
):
    load_project(session, project_id)
    if not file_ids:
        raise HTTPException(status_code=400, detail="Select at least one track")
    if not -60 <= gain_db <= 18 or not -1 <= pan <= 1:
        raise HTTPException(status_code=400, detail="Gain or pan is outside the supported range")
    tracks = [{"file_id": file_id, "gain_db": gain_db, "pan": pan, "mute": False} for file_id in file_ids]
    create_job(session, project_id, "mix", {"tracks": tracks, "name": name.strip() or "Mix", "output_format": output_format})
    return back(project_id)

@router.post("/ui/projects/{project_id}/master", include_in_schema=False)
def master_ui(
    project_id: str,
    source_file_id: Annotated[str, Form()],
    name: Annotated[str, Form()] = "BeatMaster Master",
    target_lufs: Annotated[float, Form()] = -14.0,
    true_peak_db: Annotated[float, Form()] = -1.0,
    loudness_range: Annotated[float, Form()] = 11.0,
    style: Annotated[str, Form()] = "transparent",
    output_format: Annotated[str, Form()] = "wav",
    session: Session = Depends(get_db),
):
    load_project(session, project_id)
    if style not in {"transparent", "warm", "bright", "punchy", "wide"}:
        raise HTTPException(status_code=400, detail="Unsupported mastering style")
    create_job(session, project_id, "master", {"source_file_id": source_file_id, "name": name.strip() or "Master", "target_lufs": target_lufs, "true_peak_db": true_peak_db, "loudness_range": loudness_range, "style": style, "output_format": output_format})
    return back(project_id)

@router.post("/ui/generate", include_in_schema=False)
def generate_ui(
    prompt: Annotated[str, Form(min_length=8, max_length=1000)],
    duration_seconds: Annotated[int, Form()] = 12,
    name: Annotated[str, Form()] = "Generated music",
    seed: Annotated[int | None, Form()] = None,
    session: Session = Depends(get_db),
):
    if not settings.enable_local_musicgen:
        raise HTTPException(status_code=503, detail="Music generation is disabled")
    job = create_job(session, None, "generate", {"prompt": prompt.strip(), "duration_seconds": duration_seconds, "name": name.strip() or "Generated music", "seed": seed})
    return RedirectResponse(f"/ui/jobs/{job.id}", status_code=303)
