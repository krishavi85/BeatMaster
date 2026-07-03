from html import escape
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from .api_helpers import get_db, load_project
from .asset_registry import register_asset
from .audio_probe import probe_audio
from .models import AudioFile, Job, Project
from .pages import page
from .project_view import render_project
from .storage import absolute_from_relative, project_dir, relative_to_data, safe_filename, save_upload

router = APIRouter(tags=["ui"])
EDITABLE_KINDS = {"lyrics", "chord_sheet", "chord_timeline", "notes"}


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


def _editable_asset(session: Session, project_id: str, file_id: str | None) -> AudioFile | None:
    if not file_id:
        return None
    item = session.get(AudioFile, file_id)
    if item is None or item.project_id != project_id or item.kind not in EDITABLE_KINDS:
        raise HTTPException(status_code=404, detail="Editable document not found")
    return item


@router.get("/ui/projects/{project_id}/document-editor", include_in_schema=False)
def document_editor(
    project_id: str,
    file_id: str | None = Query(None),
    kind: str = Query("lyrics"),
    label: str = Query("Lyrics"),
    session: Session = Depends(get_db),
):
    project = load_project(session, project_id)
    item = _editable_asset(session, project_id, file_id)
    selected_kind = item.kind if item else kind
    if selected_kind not in EDITABLE_KINDS:
        raise HTTPException(status_code=400, detail="Unsupported document type")
    selected_label = item.label if item else label
    content = ""
    if item:
        content = absolute_from_relative(item.relative_path).read_text(encoding="utf-8")
    options = "".join(
        f'<option value="{escape(value)}" {"selected" if value == selected_kind else ""}>{escape(value.replace("_", " ").title())}</option>'
        for value in sorted(EDITABLE_KINDS)
    )
    source_id = item.id if item else ""
    body = f'''<div class="actions"><a class="button secondary" href="/ui/projects/{project.id}">← Project</a></div><section class="hero"><span class="pill">DOCUMENT EDITOR</span><h1>{escape(selected_label)}</h1><p>Saving creates a new version, so earlier lyrics and chord documents remain available.</p></section><section class="card"><form action="/ui/projects/{project.id}/documents" method="post"><input type="hidden" name="source_file_id" value="{escape(source_id)}"><label>Document type</label><select name="kind">{options}</select><label>Label</label><input name="label" value="{escape(selected_label)}" required maxlength="200"><label>Content</label><textarea name="content" required style="min-height:520px;font-family:ui-monospace,monospace">{escape(content)}</textarea><button type="submit">Save new version</button></form></section>'''
    return page(f"Edit {selected_label}", body)


@router.post("/ui/projects/{project_id}/documents", include_in_schema=False)
def save_document(
    project_id: str,
    kind: Annotated[str, Form()],
    label: Annotated[str, Form(min_length=1, max_length=200)],
    content: Annotated[str, Form(min_length=1)],
    source_file_id: Annotated[str | None, Form()] = None,
    session: Session = Depends(get_db),
):
    load_project(session, project_id)
    if kind not in EDITABLE_KINDS:
        raise HTTPException(status_code=400, detail="Unsupported document type")
    source = _editable_asset(session, project_id, source_file_id or None)
    output_dir = project_dir(project_id) / "edits"
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = ".lab" if kind == "chord_timeline" else ".txt"
    version = len([item for item in session.query(AudioFile).filter(AudioFile.project_id == project_id, AudioFile.kind == kind).all()]) + 1
    output = output_dir / f"{safe_filename(label)}_v{version}{extension}"
    output.write_text(content.rstrip() + "\n", encoding="utf-8")
    register_asset(session, project_id, output, kind, f"{label.strip()} v{version}", metadata_json={"edited_from_file_id": source.id if source else None, "manual_edit": True, "version": version})
    session.commit()
    return RedirectResponse(f"/ui/projects/{project_id}", status_code=303)
