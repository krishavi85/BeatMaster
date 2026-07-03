import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .api_helpers import create_job, get_db, load_project
from .config import settings
from .lyrics_engine import provider_status
from .models import Job
from .runtime_capabilities import inspect_capabilities
from .schemas import (
    ChordRequest,
    CompleteSongRequest,
    DawExportRequest,
    GenerationRequest,
    JobOut,
    LyricsRequest,
    MasterRequest,
    MidiRequest,
    MixRequest,
    SeparationRequest,
)

router = APIRouter(tags=["processing"])


@router.post("/api/projects/{project_id}/separate", response_model=JobOut, status_code=202)
def queue_separation(project_id: str, request: SeparationRequest, session: Session = Depends(get_db)):
    load_project(session, project_id)
    return create_job(session, project_id, "separate", request.model_dump())


@router.post("/api/projects/{project_id}/mix", response_model=JobOut, status_code=202)
def queue_mix(project_id: str, request: MixRequest, session: Session = Depends(get_db)):
    load_project(session, project_id)
    return create_job(session, project_id, "mix", request.model_dump())


@router.post("/api/projects/{project_id}/master", response_model=JobOut, status_code=202)
def queue_master(project_id: str, request: MasterRequest, session: Session = Depends(get_db)):
    load_project(session, project_id)
    return create_job(session, project_id, "master", request.model_dump())


@router.post("/api/projects/{project_id}/chords", response_model=JobOut, status_code=202)
def queue_chords(project_id: str, request: ChordRequest, session: Session = Depends(get_db)):
    load_project(session, project_id)
    return create_job(session, project_id, "chords", request.model_dump())


@router.post("/api/projects/{project_id}/midi", response_model=JobOut, status_code=202)
def queue_midi(project_id: str, request: MidiRequest, session: Session = Depends(get_db)):
    load_project(session, project_id)
    return create_job(session, project_id, "midi", request.model_dump())


@router.post("/api/projects/{project_id}/lyrics", response_model=JobOut, status_code=202)
def queue_lyrics(project_id: str, request: LyricsRequest, session: Session = Depends(get_db)):
    load_project(session, project_id)
    if not provider_status()["configured"]:
        raise HTTPException(status_code=503, detail="No lyrics provider is configured")
    return create_job(session, project_id, "lyrics", request.model_dump())


@router.post("/api/projects/{project_id}/daw-export", response_model=JobOut, status_code=202)
def queue_daw_export(project_id: str, request: DawExportRequest, session: Session = Depends(get_db)):
    load_project(session, project_id)
    return create_job(session, project_id, "daw_export", request.model_dump())


@router.post("/api/generate", response_model=JobOut, status_code=202)
def queue_generation(request: GenerationRequest, session: Session = Depends(get_db)):
    if not settings.enable_local_musicgen or not os.getenv("MUSICGEN_MODEL"):
        raise HTTPException(status_code=503, detail="Music generation is not configured on this deployment")
    return create_job(session, None, "generate", request.model_dump())


@router.post("/api/song-packages", response_model=JobOut, status_code=202)
def queue_song_package(request: CompleteSongRequest, session: Session = Depends(get_db)):
    capabilities = inspect_capabilities()
    if not capabilities["complete_song_pipeline"]:
        raise HTTPException(status_code=503, detail="The complete music-production runtime is not configured")
    if request.include_lyrics and not capabilities["lyrics_provider_configured"]:
        raise HTTPException(status_code=503, detail="A lyrics provider is required when include_lyrics is true")
    if request.render_vocals:
        if not request.include_lyrics:
            raise HTTPException(status_code=400, detail="render_vocals requires include_lyrics")
        if not capabilities["singing_provider_configured"]:
            raise HTTPException(status_code=503, detail="A singing provider is required when render_vocals is true")
    return create_job(session, None, "song_package", request.model_dump())


@router.get("/api/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, session: Session = Depends(get_db)):
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
