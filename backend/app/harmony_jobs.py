from __future__ import annotations

from sqlalchemy.orm import Session

from .asset_registry import register_asset
from .file_registry import get_file
from .harmony_engine import detect_chords, transcribe_midi, write_chord_assets
from .models import Job
from .storage import absolute_from_relative, project_dir, safe_filename


def process_chords(session: Session, job: Job, progress):
    request = job.request_json
    source_item = get_file(session, request["source_file_id"], job.project_id)
    source = absolute_from_relative(source_item.relative_path)
    progress(10, "Calculating harmonic chroma")
    result = detect_chords(source)
    output_dir = project_dir(job.project_id) / "harmony"
    base = safe_filename(request.get("name") or source.stem)
    progress(70, "Writing chord timeline")
    paths = write_chord_assets(result, output_dir, f"{job.id}_{base}")
    json_asset = register_asset(session, job.project_id, paths["json"], "chords", f"{request.get('name') or source_item.label} chords", metadata_json=result)
    text_asset = register_asset(session, job.project_id, paths["text"], "chord_sheet", f"{request.get('name') or source_item.label} chord sheet", metadata_json=result)
    lab_asset = register_asset(session, job.project_id, paths["lab"], "chord_timeline", f"{request.get('name') or source_item.label} chord timeline", metadata_json=result)
    progress(95, "Chord assets ready")
    return {"chords_file_id": json_asset.id, "chord_sheet_file_id": text_asset.id, "timeline_file_id": lab_asset.id, "analysis": result}


def process_midi(session: Session, job: Job, progress):
    request = job.request_json
    source_item = get_file(session, request["source_file_id"], job.project_id)
    source = absolute_from_relative(source_item.relative_path)
    output_dir = project_dir(job.project_id) / "midi"
    output_dir.mkdir(parents=True, exist_ok=True)
    base = safe_filename(request.get("name") or source.stem)
    output = output_dir / f"{job.id}_{base}.mid"
    progress(10, "Detecting pitched notes")
    metadata = transcribe_midi(source, output, request.get("tempo_bpm"))
    progress(90, "Registering MIDI asset")
    item = register_asset(session, job.project_id, output, "midi", request.get("name") or f"{source_item.label} MIDI", metadata_json={"source_file_id": source_item.id, **metadata})
    return {"midi_file_id": item.id, **metadata}
