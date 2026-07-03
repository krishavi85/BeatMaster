from __future__ import annotations

from sqlalchemy.orm import Session

from .file_registry import get_file, register_file
from .models import Job
from .singing_engine import render_singing
from .storage import absolute_from_relative, project_dir, safe_filename


def process_singing(session: Session, job: Job, progress):
    request = job.request_json
    lyrics_item = get_file(session, request["lyrics_file_id"], job.project_id)
    midi_item = get_file(session, request["midi_file_id"], job.project_id) if request.get("midi_file_id") else None
    lyrics_path = absolute_from_relative(lyrics_item.relative_path)
    midi_path = absolute_from_relative(midi_item.relative_path) if midi_item else None
    output = project_dir(job.project_id) / "vocals" / f"{job.id}_{safe_filename(request.get('name') or 'Lead Vocals')}.wav"
    progress(10, "Sending lyrics and melody to singing provider")
    metadata = render_singing(
        lyrics_path=lyrics_path,
        midi_path=midi_path,
        output_path=output,
        language=request.get("language") or "English",
        title=request.get("title") or request.get("name") or "BeatMaster Song",
        voice_id=request.get("voice_id"),
    )
    progress(90, "Registering rendered vocal track")
    item = register_file(session, job.project_id, output, "vocal", request.get("name") or "Lead Vocals", metadata_json={"lyrics_file_id": lyrics_item.id, "midi_file_id": midi_item.id if midi_item else None, **metadata})
    return {"vocal_file_id": item.id, **metadata}
