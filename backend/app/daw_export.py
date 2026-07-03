from __future__ import annotations

import json
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from .asset_registry import register_asset
from .models import AudioFile, Job, Project
from .storage import absolute_from_relative, project_dir, safe_filename

AUDIO_KINDS = {"source", "stem", "mix", "master", "generated"}
DOCUMENT_KINDS = {"midi", "chords", "chord_sheet", "chord_timeline", "lyrics", "lyrics_metadata", "captions"}


def _safe_text(value: str) -> str:
    return value.replace("\\", "_").replace('"', "'").replace("\n", " ").strip()


def _unique_name(path: Path, used: set[str]) -> str:
    base = safe_filename(path.name)
    stem = Path(base).stem
    suffix = Path(base).suffix
    candidate = base
    index = 2
    while candidate.lower() in used:
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    used.add(candidate.lower())
    return candidate


def _reaper_project(project: Project, audio_entries: list[dict[str, Any]], tempo_bpm: float) -> str:
    lines = [
        '<REAPER_PROJECT 0.1 "7.0" 0',
        "  RIPPLE 0",
        "  GROUPOVERRIDE 0 0 0",
        f"  TEMPO {tempo_bpm:.6f} 4 4",
        f'  AUTHOR "BeatMaster"',
        f'  NOTES 0 2 "{_safe_text(project.name)} exported by BeatMaster"',
    ]
    for entry in audio_entries:
        duration = max(0.001, float(entry.get("duration_seconds") or 0.001))
        track_guid = "{" + str(uuid.uuid4()).upper() + "}"
        item_guid = "{" + str(uuid.uuid4()).upper() + "}"
        lines.extend([
            f"  <TRACK {track_guid}",
            f'    NAME "{_safe_text(entry["label"])}"',
            "    PEAKCOL 16576",
            "    VOLPAN 1 0 -1 -1 1",
            "    MUTESOLO 0 0 0",
            f"    <ITEM",
            "      POSITION 0",
            f"      LENGTH {duration:.9f}",
            f"      IGUID {item_guid}",
            f'      NAME "{_safe_text(entry["label"])}"',
            "      VOLPAN 1 0 1 -1",
            "      SOFFS 0",
            "      PLAYRATE 1 1 0 -1 0 0.0025",
            "      <SOURCE WAVE",
            f'        FILE "Media/{entry["archive_name"]}"',
            "      >",
            "    >",
            "  >",
        ])
    lines.append(">")
    return "\n".join(lines) + "\n"


def process_daw_export(session: Session, job: Job, progress):
    project = session.get(Project, job.project_id)
    if project is None:
        raise RuntimeError("Project not found")
    request = job.request_json
    requested_ids = set(request.get("file_ids") or [])
    files = [item for item in project.files if not requested_ids or item.id in requested_ids]
    files = [item for item in files if item.kind in AUDIO_KINDS | DOCUMENT_KINDS]
    if not files:
        raise RuntimeError("No project files are available for DAW export")
    output_dir = project_dir(project.id) / "daw"
    staging = output_dir / f"{job.id}_staging"
    media_dir = staging / "Media"
    documents_dir = staging / "Documents"
    shutil.rmtree(staging, ignore_errors=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)
    progress(10, "Collecting project assets")
    used_audio_names: set[str] = set()
    used_document_names: set[str] = set()
    audio_entries: list[dict[str, Any]] = []
    document_entries: list[dict[str, Any]] = []
    for index, item in enumerate(files):
        source = absolute_from_relative(item.relative_path)
        if not source.exists():
            continue
        if item.kind in AUDIO_KINDS:
            archive_name = _unique_name(source, used_audio_names)
            destination = media_dir / archive_name
            audio_entries.append({"file_id": item.id, "label": item.label, "kind": item.kind, "archive_name": archive_name, "duration_seconds": item.duration_seconds, "sample_rate": item.sample_rate, "channels": item.channels, "codec": item.codec})
        else:
            archive_name = _unique_name(source, used_document_names)
            destination = documents_dir / archive_name
            document_entries.append({"file_id": item.id, "label": item.label, "kind": item.kind, "archive_name": archive_name})
        shutil.copy2(source, destination)
        progress(10 + (index + 1) / max(len(files), 1) * 45, f"Copying {item.label}")
    if not audio_entries:
        raise RuntimeError("DAW export requires at least one audio file")
    tempo = float(request.get("tempo_bpm") or (project.analysis or {}).get("tempo_bpm") or 120.0)
    manifest = {
        "format": "BeatMaster DAW Interchange 1.0",
        "project_id": project.id,
        "project_name": project.name,
        "tempo_bpm": tempo,
        "time_signature": request.get("time_signature") or "4/4",
        "sample_start_seconds": 0,
        "audio_tracks": audio_entries,
        "documents": document_entries,
        "notes": "All audio tracks start at 00:00. Import the Media folder into any DAW or open the included Reaper project.",
    }
    (staging / "beatmaster-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (staging / "README.txt").write_text(
        "BeatMaster DAW Interchange\n\n"
        "1. Open BeatMaster_Project.rpp in REAPER, or import every file in Media into another DAW at timeline position 00:00.\n"
        "2. Import MIDI and chord/lyric documents from Documents.\n"
        "3. Confirm the project tempo shown in beatmaster-manifest.json.\n",
        encoding="utf-8",
    )
    (staging / "BeatMaster_Project.rpp").write_text(_reaper_project(project, audio_entries, tempo), encoding="utf-8")
    progress(70, "Writing DAW project and manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    package_name = safe_filename(request.get("name") or f"{project.name}_DAW_Package")
    zip_path = output_dir / f"{job.id}_{package_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in staging.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(staging))
    shutil.rmtree(staging, ignore_errors=True)
    progress(94, "Registering DAW package")
    asset = register_asset(session, project.id, zip_path, "daw_package", request.get("name") or "DAW package", metadata_json=manifest)
    return {"daw_package_file_id": asset.id, "track_count": len(audio_entries), "document_count": len(document_entries), "tempo_bpm": tempo, "formats": ["REAPER RPP", "generic aligned stems", "MIDI and documents"]}
