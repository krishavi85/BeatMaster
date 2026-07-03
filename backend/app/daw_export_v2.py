from __future__ import annotations

import json
import shutil
import uuid
import zipfile
from pathlib import Path

from sqlalchemy.orm import Session

from .asset_registry import register_asset
from .models import Job, Project
from .storage import absolute_from_relative, project_dir, safe_filename

AUDIO_KINDS = {"source", "stem", "mix", "master", "generated", "vocal"}
DOCUMENT_KINDS = {"midi", "chords", "chord_sheet", "chord_timeline", "lyrics", "lyrics_metadata", "captions"}


def unique_name(path: Path, used: set[str]) -> str:
    base = safe_filename(path.name)
    stem, suffix = Path(base).stem, Path(base).suffix
    candidate, index = base, 2
    while candidate.lower() in used:
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    used.add(candidate.lower())
    return candidate


def reaper_project(project_name: str, tracks: list[dict], tempo: float, signature: str) -> str:
    try:
        numerator, denominator = [int(value) for value in signature.split("/", 1)]
    except (ValueError, AttributeError):
        numerator, denominator = 4, 4
    lines = ['<REAPER_PROJECT 0.1 "7.0" 0', f"  TEMPO {tempo:.6f} {numerator} {denominator}", '  AUTHOR "BeatMaster"']
    for track in tracks:
        track_guid = "{" + str(uuid.uuid4()).upper() + "}"
        item_guid = "{" + str(uuid.uuid4()).upper() + "}"
        label = str(track["label"]).replace('"', "'").replace("\n", " ")
        duration = max(0.001, float(track.get("duration_seconds") or 0.001))
        lines.extend([
            f"  <TRACK {track_guid}",
            f'    NAME "{label}"',
            "    VOLPAN 1 0 -1 -1 1",
            "    <ITEM",
            "      POSITION 0",
            f"      LENGTH {duration:.9f}",
            f"      IGUID {item_guid}",
            f'      NAME "{label}"',
            "      <SOURCE WAVE",
            f'        FILE "Media/{track["archive_name"]}"',
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
    files = [item for item in project.files if (not requested_ids or item.id in requested_ids) and item.kind in AUDIO_KINDS | DOCUMENT_KINDS]
    if not files:
        raise RuntimeError("No project assets are available for DAW export")
    output_dir = project_dir(project.id) / "daw"
    staging = output_dir / f"{job.id}_staging"
    media_dir, documents_dir = staging / "Media", staging / "Documents"
    shutil.rmtree(staging, ignore_errors=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)
    audio_tracks, documents = [], []
    used_audio: set[str] = set()
    used_documents: set[str] = set()
    for index, item in enumerate(files):
        source = absolute_from_relative(item.relative_path)
        if not source.exists():
            continue
        if item.kind in AUDIO_KINDS:
            archive_name = unique_name(source, used_audio)
            destination = media_dir / archive_name
            audio_tracks.append({"file_id": item.id, "label": item.label, "kind": item.kind, "archive_name": archive_name, "duration_seconds": item.duration_seconds, "sample_rate": item.sample_rate, "channels": item.channels, "codec": item.codec})
        else:
            archive_name = unique_name(source, used_documents)
            destination = documents_dir / archive_name
            documents.append({"file_id": item.id, "label": item.label, "kind": item.kind, "archive_name": archive_name})
        shutil.copy2(source, destination)
        progress(10 + (index + 1) / max(1, len(files)) * 45, f"Copying {item.label}")
    if not audio_tracks:
        raise RuntimeError("DAW export requires at least one audio track")
    tempo = float(request.get("tempo_bpm") or (project.analysis or {}).get("tempo_bpm") or 120.0)
    signature = request.get("time_signature") or "4/4"
    manifest = {"format": "BeatMaster DAW Interchange 1.1", "project_id": project.id, "project_name": project.name, "tempo_bpm": tempo, "time_signature": signature, "culture_profile_id": (project.analysis or {}).get("culture_profile_id"), "sample_start_seconds": 0, "audio_tracks": audio_tracks, "documents": documents}
    (staging / "beatmaster-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (staging / "README.txt").write_text("Open BeatMaster_Project.rpp in REAPER or import every file in Media at 00:00. Import MIDI, chord and lyric assets from Documents.\n", encoding="utf-8")
    (staging / "BeatMaster_Project.rpp").write_text(reaper_project(project.name, audio_tracks, tempo, signature), encoding="utf-8")
    progress(75, "Writing DAW package")
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{job.id}_{safe_filename(request.get('name') or project.name + '_DAW_Package')}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in staging.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(staging))
    shutil.rmtree(staging, ignore_errors=True)
    asset = register_asset(session, project.id, zip_path, "daw_package", request.get("name") or "DAW package", metadata_json=manifest)
    return {"daw_package_file_id": asset.id, "track_count": len(audio_tracks), "document_count": len(documents), "tempo_bpm": tempo, "time_signature": signature, "formats": ["REAPER RPP", "aligned audio stems", "MIDI and documents"]}
