from pathlib import Path
from sqlalchemy import select
from sqlalchemy.orm import Session
from .audio_probe import AudioProcessingError, probe_audio
from .models import AudioFile
from .storage import relative_to_data


def register_file(session: Session, project_id: str, path: Path, kind: str, label: str, original_name: str | None = None, metadata_json: dict | None = None) -> AudioFile:
    metadata = probe_audio(path)
    item = AudioFile(project_id=project_id, kind=kind, label=label, relative_path=relative_to_data(path), original_name=original_name, size_bytes=path.stat().st_size, duration_seconds=metadata.get("duration_seconds"), sample_rate=metadata.get("sample_rate"), channels=metadata.get("channels"), codec=metadata.get("codec"), metadata_json=metadata_json)
    session.add(item)
    session.flush()
    return item


def get_file(session: Session, file_id: str, project_id: str | None = None) -> AudioFile:
    query = select(AudioFile).where(AudioFile.id == file_id)
    if project_id:
        query = query.where(AudioFile.project_id == project_id)
    item = session.scalar(query)
    if not item:
        raise AudioProcessingError(f"Audio file not found: {file_id}")
    return item
