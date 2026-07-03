from typing import Annotated
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from .api_helpers import get_db, load_project
from .audio_probe import probe_audio
from .models import AudioFile, Job, Project
from .pages import page
from .project_view import render_project
from .storage import relative_to_data, save_upload

router = APIRouter(tags=["ui"])

@router.post("/ui/projects", include_in_schema=False)
async def create_project_ui(
    name: Annotated[str, Form(min_length=1, max_length=200)],
    audio: Annotated[UploadFile, File()],
    session: Session = Depends(get_db),
):
    original_name = audio.filename
    mime_type = audio.content_type
    project = Project(name=name.strip())
    session.add(project)
    session.flush()
    path, size = await save_upload(project.id, audio)
    metadata = probe_audio(path)
    source = AudioFile(
        project_id=project.id,
        kind="source",
        label="Original",
        relative_path=relative_to_data(path),
        original_name=original_name,
        mime_type=mime_type,
        size_bytes=size,
        duration_seconds=metadata.get("duration_seconds"),
        sample_rate=metadata.get("sample_rate"),
        channels=metadata.get("channels"),
        codec=metadata.get("codec"),
        metadata_json=metadata,
    )
    session.add(source)
    session.flush()
    session.add(Job(project_id=project.id, type="analyze", request_json={"source_file_id": source.id}, stage="Queued"))
    session.commit()
    return RedirectResponse(f"/ui/projects/{project.id}", status_code=303)

@router.get("/ui/projects/{project_id}", include_in_schema=False)
def project_workspace(project_id: str, session: Session = Depends(get_db)):
    project = load_project(session, project_id)
    body, active = render_project(project)
    return page(project.name, body, refresh=active)
