from html import escape
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .api_helpers import get_db
from .models import AudioFile, Job
from .pages import page

router = APIRouter(tags=["ui"])

@router.get("/ui/jobs/{job_id}", include_in_schema=False)
def job_page(job_id: str, session: Session = Depends(get_db)):
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    refresh = job.status in {"queued", "running"}
    result = job.result_json or {}
    output = ""
    file_id = result.get("file_id")
    if file_id:
        item = session.get(AudioFile, file_id)
        if item:
            project_link = f'<a class="button" href="/ui/projects/{item.project_id}">Open project</a>' if item.project_id else ""
            output = f'<section class="card"><h2>Processor output</h2><audio controls src="/api/files/{item.id}/stream"></audio><div class="actions"><a class="button secondary" href="/api/files/{item.id}/download">Download audio</a>{project_link}</div></section>'
    error = f'<section class="notice"><strong>Processor error</strong><pre>{escape(job.error)}</pre></section>' if job.error else ""
    body = f'<a class="button secondary" href="/">← Dashboard</a><section class="hero"><span class="pill">{escape(job.type.upper())}</span><h1>{escape(job.stage)}</h1><p>Status: {escape(job.status)} · {round(job.progress, 1)}%</p><div class="progress"><i style="width:{max(0, min(100, job.progress))}%"></i></div></section>{output}{error}'
    return page(f"Job {job.id[:8]}", body, refresh=refresh)
