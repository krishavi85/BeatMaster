import shutil
from typing import Annotated
from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload
from .api_helpers import get_db, load_project, project_out
from .audio_probe import probe_audio
from .config import settings
from .models import AudioFile, Job, Project
from .schemas import ProjectOut
from .storage import relative_to_data, save_upload

router = APIRouter(prefix="/api/projects", tags=["projects"])

@router.get("", response_model=list[ProjectOut])
def list_projects(session: Session = Depends(get_db)):
    projects = session.scalars(select(Project).options(selectinload(Project.files), selectinload(Project.jobs)).order_by(Project.updated_at.desc())).unique().all()
    return [project_out(project) for project in projects]

@router.post("", response_model=ProjectOut, status_code=201)
async def create_project(name: Annotated[str, Form(min_length=1, max_length=200)], audio: Annotated[UploadFile, File()], session: Session = Depends(get_db)):
    project = Project(name=name.strip())
    session.add(project)
    session.flush()
    path, size = await save_upload(project.id, audio)
    metadata = probe_audio(path)
    source = AudioFile(project_id=project.id, kind="source", label="Original", relative_path=relative_to_data(path), original_name=audio.filename, mime_type=audio.content_type, size_bytes=size, duration_seconds=metadata.get("duration_seconds"), sample_rate=metadata.get("sample_rate"), channels=metadata.get("channels"), codec=metadata.get("codec"), metadata_json=metadata)
    session.add(source)
    session.flush()
    session.add(Job(project_id=project.id, type="analyze", request_json={"source_file_id": source.id}, stage="Queued"))
    session.commit()
    return project_out(load_project(session, project.id))

@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: str, session: Session = Depends(get_db)):
    return project_out(load_project(session, project_id))

@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, session: Session = Depends(get_db)):
    project = load_project(session, project_id)
    folder = settings.data_dir / "projects" / project.id
    session.delete(project)
    session.commit()
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
