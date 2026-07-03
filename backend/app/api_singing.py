from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .api_helpers import create_job, get_db, load_project
from .schemas import JobOut, SingingRequest
from .singing_engine import provider_status

router = APIRouter(tags=["singing"])


@router.post("/api/projects/{project_id}/singing", response_model=JobOut, status_code=202)
def queue_singing(project_id: str, request: SingingRequest, session: Session = Depends(get_db)):
    load_project(session, project_id)
    if not provider_status()["configured"]:
        raise HTTPException(status_code=503, detail="No singing synthesis provider is configured")
    return create_job(session, project_id, "singing", request.model_dump())
