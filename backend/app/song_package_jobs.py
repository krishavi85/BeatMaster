from __future__ import annotations

from uuid import uuid4

from sqlalchemy.orm import Session

from .audio_probe import analyze_audio
from .daw_export import process_daw_export
from .file_registry import register_file
from .generation_entry import generate_audio
from .harmony_jobs import process_chords, process_midi
from .lyrics_jobs import process_lyrics
from .models import Job, Project
from .render_jobs import process_mix
from .separation_jobs import process_separate
from .singing_jobs import process_singing


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
    lyrics_result: dict | None = None
    midi_result: dict | None = None
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
        lyrics_result = process_lyrics(session, lyrics_job, lambda value, stage: progress(3 + value * 0.1, stage))
        results["lyrics"] = lyrics_result
        results["pipeline"].append("lyrics")
    progress(15, "Generating complete music bed")
    audio_path, generation_metadata = generate_audio(request, project, job.id, lambda value, stage: progress(15 + value * 0.3, stage))
    analysis = analyze_audio(audio_path)
    audio_item = register_file(session, project.id, audio_path, "generated", request.get("title") or project.name, metadata_json={**generation_metadata, "analysis": analysis})
    project.analysis = {**(project.analysis or {}), **analysis, "culture_profile_id": request.get("culture_profile_id")}
    results["audio_file_id"] = audio_item.id
    results["generation"] = generation_metadata
    results["analysis"] = analysis
    results["pipeline"].append("audio_generation")
    if request.get("separate_stems", True):
        progress(47, "Separating editable stems")
        separation_job = _child_job(project.id, "separate", {
            "source_file_id": audio_item.id,
            "model": request.get("separation_model") or "htdemucs",
            "two_stems": None,
            "output_format": request.get("output_format") or "wav",
        })
        results["separation"] = process_separate(session, separation_job, lambda value, stage: progress(47 + value * 0.13, stage))
        results["pipeline"].append("stems")
    if request.get("extract_chords", True):
        progress(61, "Extracting chord map")
        chord_job = _child_job(project.id, "chords", {"source_file_id": audio_item.id, "name": project.name})
        results["chords"] = process_chords(session, chord_job, lambda value, stage: progress(61 + value * 0.08, stage))
        results["pipeline"].append("chords")
    if request.get("extract_midi", True):
        progress(70, "Transcribing MIDI")
        midi_job = _child_job(project.id, "midi", {"source_file_id": audio_item.id, "name": f"{project.name} melody", "tempo_bpm": analysis.get("tempo_bpm")})
        try:
            midi_result = process_midi(session, midi_job, lambda value, stage: progress(70 + value * 0.08, stage))
            results["midi"] = midi_result
            results["pipeline"].append("midi")
        except Exception as exc:
            results["midi_warning"] = str(exc)
    final_audio_file_id = audio_item.id
    if request.get("render_vocals", False):
        if not lyrics_result:
            raise RuntimeError("Singing synthesis requires generated lyrics")
        progress(79, "Rendering singing vocals")
        singing_job = _child_job(project.id, "singing", {
            "lyrics_file_id": lyrics_result["lyrics_file_id"],
            "midi_file_id": midi_result.get("midi_file_id") if midi_result else None,
            "name": "Lead Vocals",
            "title": request.get("title") or project.name,
            "language": request.get("language") or "English",
            "voice_id": request.get("voice_id"),
        })
        singing_result = process_singing(session, singing_job, lambda value, stage: progress(79 + value * 0.08, stage))
        results["singing"] = singing_result
        results["pipeline"].append("singing_vocals")
        progress(88, "Mixing music bed and vocals")
        final_mix_job = _child_job(project.id, "mix", {
            "tracks": [
                {"file_id": audio_item.id, "gain_db": -2.0, "pan": 0.0, "mute": False},
                {"file_id": singing_result["vocal_file_id"], "gain_db": float(request.get("vocal_gain_db", -3.0)), "pan": 0.0, "mute": False},
            ],
            "name": f"{project.name} Complete Mix",
            "output_format": request.get("output_format") or "wav",
        })
        mix_result = process_mix(session, final_mix_job, lambda value, stage: progress(88 + value * 0.05, stage))
        final_audio_file_id = mix_result["file_id"]
        results["complete_mix"] = mix_result
        results["pipeline"].append("complete_vocal_mix")
    session.flush()
    if request.get("export_daw", True):
        progress(94, "Building DAW interchange package")
        daw_job = _child_job(project.id, "daw_export", {"name": f"{project.name} DAW Package", "tempo_bpm": analysis.get("tempo_bpm"), "time_signature": request.get("time_signature") or "4/4"})
        results["daw_export"] = process_daw_export(session, daw_job, lambda value, stage: progress(94 + value * 0.05, stage))
        results["pipeline"].append("daw_export")
    results["final_audio_file_id"] = final_audio_file_id
    results["transparency"] = {
        "audio_type": "AI-generated music bed with optional provider-rendered singing vocals",
        "sung_vocals_included": bool(request.get("render_vocals", False)),
        "lyrics_are_separate_editable_assets": bool(request.get("include_lyrics", True)),
        "culture_conditioning": generation_metadata.get("model_source") or generation_metadata.get("provider"),
        "fine_tuned_culture_model": generation_metadata.get("fine_tuned_for_profile", False),
    }
    progress(99, "Complete production package ready")
    return results
