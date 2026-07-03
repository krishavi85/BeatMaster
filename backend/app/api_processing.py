import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .api_helpers import create_job, get_db, load_project
from .config import settings
from .models import Job
from .schemas import GenerationRequest, JobOut, MasterRequest, MixRequest, SeparationRequest

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

@router.post("/api/generate", response_model=JobOut, status_code=202)
def queue_generation(request: GenerationRequest, session: Session = Depends(get_db)):
    if not settings.enable_local_musicgen or not os.getenv("MUSICGEN_MODEL"):
        raise HTTPException(status_code=503, detail="Music generation is not configured on this deployment")
    return create_job(session, None, "generate", request.model_dump())

@router.get("/api/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, session: Session = Depends(get_db)):
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
