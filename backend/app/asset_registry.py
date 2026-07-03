from __future__ import annotations

from pathlib import Path
from sqlalchemy.orm import Session

from .models import AudioFile
from .storage import relative_to_data

MIME_BY_SUFFIX = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".mid": "audio/midi",
    ".midi": "audio/midi",
    ".json": "application/json",
    ".txt": "text/plain; charset=utf-8",
    ".srt": "application/x-subrip",
    ".vtt": "text/vtt; charset=utf-8",
    ".lab": "text/plain; charset=utf-8",
    ".rpp": "text/plain; charset=utf-8",
    ".zip": "application/zip",
}


def register_asset(
    session: Session,
    project_id: str,
    path: Path,
    kind: str,
    label: str,
    *,
    mime_type: str | None = None,
    metadata_json: dict | None = None,
) -> AudioFile:
    item = AudioFile(
        project_id=project_id,
        kind=kind,
        label=label,
        relative_path=relative_to_data(path),
        original_name=path.name,
        mime_type=mime_type or MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream"),
        size_bytes=path.stat().st_size,
        duration_seconds=None,
        sample_rate=None,
        channels=None,
        codec=path.suffix.lower().lstrip(".") or "file",
        metadata_json=metadata_json or {},
    )
    session.add(item)
    session.flush()
    return item


def latest_text_asset(project_files: list[AudioFile], kinds: set[str]) -> AudioFile | None:
    candidates = [item for item in project_files if item.kind in kinds]
    return max(candidates, key=lambda item: item.created_at) if candidates else None
