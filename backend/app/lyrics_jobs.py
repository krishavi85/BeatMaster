from __future__ import annotations

import json
from sqlalchemy.orm import Session

from .asset_registry import register_asset
from .lyrics_engine import generate_lyrics
from .models import Job, Project
from .storage import project_dir, safe_filename


def process_lyrics(session: Session, job: Job, progress):
    request = job.request_json
    project = session.get(Project, job.project_id) if job.project_id else None
    if project is None:
        project = Project(name=request.get("title") or "Songwriting Project")
        session.add(project)
        session.flush()
        job.project_id = project.id
    progress(10, "Preparing songwriting request")
    text, metadata = generate_lyrics(request)
    output_dir = project_dir(project.id) / "lyrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    base = safe_filename(request.get("title") or "lyrics")
    text_path = output_dir / f"{job.id}_{base}.txt"
    json_path = output_dir / f"{job.id}_{base}.json"
    text_path.write_text(text.strip() + "\n", encoding="utf-8")
    json_path.write_text(json.dumps({"text": text, "metadata": metadata}, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(80, "Registering songwriting assets")
    text_asset = register_asset(session, project.id, text_path, "lyrics", request.get("title") or "Lyrics", metadata_json=metadata)
    metadata_asset = register_asset(session, project.id, json_path, "lyrics_metadata", "Lyrics metadata", metadata_json=metadata)
    project.analysis = {**(project.analysis or {}), "latest_lyrics_file_id": text_asset.id, "lyrics_language": request.get("language"), "culture_profile_id": request.get("culture_profile_id")}
    progress(95, "Songwriting assets ready")
    return {"project_id": project.id, "lyrics_file_id": text_asset.id, "metadata_file_id": metadata_asset.id}
