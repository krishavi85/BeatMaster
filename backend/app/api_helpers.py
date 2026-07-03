from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload
from .database import SessionLocal
from .models import AudioFile, Job, Project
from .schemas import JobOut, ProjectOut


def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def file_out(item: AudioFile) -> dict:
    return {"id": item.id, "project_id": item.project_id, "kind": item.kind, "label": item.label, "original_name": item.original_name, "mime_type": item.mime_type, "size_bytes": item.size_bytes, "duration_seconds": item.duration_seconds, "sample_rate": item.sample_rate, "channels": item.channels, "codec": item.codec, "created_at": item.created_at, "metadata_json": item.metadata_json, "stream_url": f"/api/files/{item.id}/stream", "download_url": f"/api/files/{item.id}/download"}


def project_out(project: Project) -> ProjectOut:
    return ProjectOut(id=project.id, name=project.name, created_at=project.created_at, updated_at=project.updated_at, analysis=project.analysis, files=[file_out(item) for item in sorted(project.files, key=lambda value: value.created_at)], jobs=[JobOut.model_validate(job) for job in sorted(project.jobs, key=lambda value: value.created_at, reverse=True)])


def load_project(session: Session, project_id: str) -> Project:
    project = session.scalar(select(Project).where(Project.id == project_id).options(selectinload(Project.files), selectinload(Project.jobs)))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def create_job(session: Session, project_id: str | None, job_type: str, request: dict) -> Job:
    job = Job(project_id=project_id, type=job_type, request_json=request, status="queued", stage="Queued")
    session.add(job)
    session.commit()
    session.refresh(job)
    return job
