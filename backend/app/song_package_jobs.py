from __future__ import annotations

from uuid import uuid4
from sqlalchemy.orm import Session

from .daw_export import process_daw_export
from .file_registry import register_file
from .generation_jobs import generate_audio
from .harmony_jobs import process_chords, process_midi
from .lyrics_jobs import process_lyrics
from .models import Job, Project
from .separation_jobs import process_separate
from .audio_probe import analyze_audio


def _child_job(project_id: str, job_type: str, request: dict) -> Job:
    return Job(id=str(uuid4()), project_id=project_id, type=job_type, request_json=request, status="running", stage="Running")


def process_song_package(session: Session, job: Job, progress):
    request = job.request_json
    project = session.get(Project, job.project_id) if job.project_id else None
    if project is None:
        project = Project(name=request.get("title") or request.get("name") or "BeatMaster Song")
        session.add(project)
        session.flush()
        job.project_id = project.id
    results: dict = {"project_id": project.id, "pipeline": []}
    if request.get("include_lyrics", True):
        progress(3, "Writing multilingual lyrics")
        lyrics_job = _child_job(project.id, "lyrics", {
            "title": request.get("title") or project.name,
            "prompt": request["prompt"],
            "language": request.get("language") or "English",
            "culture_profile_id": request.get("culture_profile_id"),
            "mood": request.get("mood"),
            "structure": request.get("structure"),
        })
        lyrics_result = process_lyrics(session, lyrics_job, lambda value, stage: progress(3 + value * 0.12, stage))
        results["lyrics"] = lyrics_result
        results["pipeline"].append("lyrics")
    progress(18, "Generating complete music bed")
    audio_path, generation_metadata = generate_audio(request, project, job.id, lambda value, stage: progress(18 + value * 0.34, stage))
    analysis = analyze_audio(audio_path)
    audio_item = register_file(session, project.id, audio_path, "generated", request.get("title") or project.name, metadata_json={**generation_metadata, "analysis": analysis})
    project.analysis = {**(project.analysis or {}), **analysis, "culture_profile_id": request.get("culture_profile_id")}
    results["audio_file_id"] = audio_item.id
    results["generation"] = generation_metadata
    results["analysis"] = analysis
    results["pipeline"].append("audio_generation")
    if request.get("separate_stems", True):
        progress(55, "Separating editable stems")
        separation_job = _child_job(project.id, "separate", {
            "source_file_id": audio_item.id,
            "model": request.get("separation_model") or "htdemucs",
            "two_stems": None,
            "output_format": request.get("output_format") or "wav",
        })
        results["separation"] = process_separate(session, separation_job, lambda value, stage: progress(55 + value * 0.16, stage))
        results["pipeline"].append("stems")
    if request.get("extract_chords", True):
        progress(72, "Extracting chord map")
        chord_job = _child_job(project.id, "chords", {"source_file_id": audio_item.id, "name": project.name})
        results["chords"] = process_chords(session, chord_job, lambda value, stage: progress(72 + value * 0.08, stage))
        results["pipeline"].append("chords")
    if request.get("extract_midi", True):
        progress(81, "Transcribing MIDI")
        midi_job = _child_job(project.id, "midi", {"source_file_id": audio_item.id, "name": f"{project.name} melody", "tempo_bpm": analysis.get("tempo_bpm")})
        try:
            results["midi"] = process_midi(session, midi_job, lambda value, stage: progress(81 + value * 0.08, stage))
            results["pipeline"].append("midi")
        except Exception as exc:
            results["midi_warning"] = str(exc)
    session.flush()
    if request.get("export_daw", True):
        progress(90, "Building DAW interchange package")
        daw_job = _child_job(project.id, "daw_export", {"name": f"{project.name} DAW Package", "tempo_bpm": analysis.get("tempo_bpm"), "time_signature": request.get("time_signature") or "4/4"})
        results["daw_export"] = process_daw_export(session, daw_job, lambda value, stage: progress(90 + value * 0.08, stage))
        results["pipeline"].append("daw_export")
    results["transparency"] = {
        "audio_type": "AI-generated music bed",
        "sung_vocals_included": False,
        "lyrics_are_separate_editable_assets": bool(request.get("include_lyrics", True)),
        "culture_conditioning": generation_metadata.get("model_source"),
        "fine_tuned_culture_model": generation_metadata.get("fine_tuned_for_profile", False),
    }
    progress(99, "Complete production package ready")
    return results
