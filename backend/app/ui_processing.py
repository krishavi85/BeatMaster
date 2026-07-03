import json
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from .api_helpers import create_job, get_db, load_project
from .config import settings
from .culture_profiles import get_profile
from .lyrics_engine import provider_status
from .models import AudioFile
from .runtime_capabilities import inspect_capabilities

router = APIRouter(tags=["ui"])


def back(project_id: str):
    return RedirectResponse(f"/ui/projects/{project_id}", status_code=303)


def _validate_project_file(session: Session, project_id: str, file_id: str) -> AudioFile:
    item = session.get(AudioFile, file_id)
    if item is None or item.project_id != project_id:
        raise HTTPException(status_code=400, detail=f"File does not belong to this project: {file_id}")
    return item


@router.post("/ui/projects/{project_id}/separate", include_in_schema=False)
def separate_ui(
    project_id: str,
    source_file_id: Annotated[str, Form()],
    model: Annotated[str, Form()] = "htdemucs",
    two_stems: Annotated[str, Form()] = "",
    output_format: Annotated[str, Form()] = "wav",
    session: Session = Depends(get_db),
):
    load_project(session, project_id)
    _validate_project_file(session, project_id, source_file_id)
    if model not in {"htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra_q"}:
        raise HTTPException(status_code=400, detail="Unsupported Demucs model")
    if output_format not in {"wav", "flac", "mp3"}:
        raise HTTPException(status_code=400, detail="Unsupported output format")
    create_job(session, project_id, "separate", {"source_file_id": source_file_id, "model": model, "two_stems": two_stems or None, "output_format": output_format})
    return back(project_id)


@router.post("/ui/projects/{project_id}/mix", include_in_schema=False)
def mix_ui(
    project_id: str,
    tracks_json: Annotated[str | None, Form()] = None,
    file_ids: Annotated[list[str] | None, Form()] = None,
    name: Annotated[str, Form()] = "BeatMaster Mix",
    gain_db: Annotated[float, Form()] = 0.0,
    pan: Annotated[float, Form()] = 0.0,
    output_format: Annotated[str, Form()] = "wav",
    session: Session = Depends(get_db),
):
    load_project(session, project_id)
    tracks: list[dict]
    if tracks_json:
        try:
            parsed = json.loads(tracks_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid mixer data") from exc
        if not isinstance(parsed, list):
            raise HTTPException(status_code=400, detail="Mixer data must be a list")
        tracks = []
        for raw in parsed:
            if not isinstance(raw, dict) or not raw.get("file_id"):
                continue
            file_id = str(raw["file_id"])
            _validate_project_file(session, project_id, file_id)
            track_gain = float(raw.get("gain_db", 0.0))
            track_pan = float(raw.get("pan", 0.0))
            if not -60 <= track_gain <= 18 or not -1 <= track_pan <= 1:
                raise HTTPException(status_code=400, detail="A track gain or pan value is outside the supported range")
            tracks.append({"file_id": file_id, "gain_db": track_gain, "pan": track_pan, "mute": bool(raw.get("mute", False))})
    else:
        ids = file_ids or []
        if not -60 <= gain_db <= 18 or not -1 <= pan <= 1:
            raise HTTPException(status_code=400, detail="Gain or pan is outside the supported range")
        tracks = []
        for file_id in ids:
            _validate_project_file(session, project_id, file_id)
            tracks.append({"file_id": file_id, "gain_db": gain_db, "pan": pan, "mute": False})
    if not tracks or all(track["mute"] for track in tracks):
        raise HTTPException(status_code=400, detail="Select at least one unmuted track")
    if output_format not in {"wav", "flac", "mp3"}:
        raise HTTPException(status_code=400, detail="Unsupported output format")
    create_job(session, project_id, "mix", {"tracks": tracks, "name": name.strip() or "Mix", "output_format": output_format})
    return back(project_id)


@router.post("/ui/projects/{project_id}/master", include_in_schema=False)
def master_ui(
    project_id: str,
    source_file_id: Annotated[str, Form()],
    name: Annotated[str, Form()] = "BeatMaster Master",
    target_lufs: Annotated[float, Form()] = -14.0,
    true_peak_db: Annotated[float, Form()] = -1.0,
    loudness_range: Annotated[float, Form()] = 11.0,
    style: Annotated[str, Form()] = "transparent",
    output_format: Annotated[str, Form()] = "wav",
    session: Session = Depends(get_db),
):
    load_project(session, project_id)
    _validate_project_file(session, project_id, source_file_id)
    if style not in {"transparent", "warm", "bright", "punchy", "wide"}:
        raise HTTPException(status_code=400, detail="Unsupported mastering style")
    create_job(session, project_id, "master", {"source_file_id": source_file_id, "name": name.strip() or "Master", "target_lufs": target_lufs, "true_peak_db": true_peak_db, "loudness_range": loudness_range, "style": style, "output_format": output_format})
    return back(project_id)


@router.post("/ui/projects/{project_id}/chords", include_in_schema=False)
def chords_ui(project_id: str, source_file_id: Annotated[str, Form()], name: Annotated[str, Form()] = "Chord Map", session: Session = Depends(get_db)):
    load_project(session, project_id)
    _validate_project_file(session, project_id, source_file_id)
    create_job(session, project_id, "chords", {"source_file_id": source_file_id, "name": name.strip() or "Chord Map"})
    return back(project_id)


@router.post("/ui/projects/{project_id}/midi", include_in_schema=False)
def midi_ui(project_id: str, source_file_id: Annotated[str, Form()], name: Annotated[str, Form()] = "MIDI Transcription", tempo_bpm: Annotated[float | None, Form()] = None, session: Session = Depends(get_db)):
    load_project(session, project_id)
    _validate_project_file(session, project_id, source_file_id)
    create_job(session, project_id, "midi", {"source_file_id": source_file_id, "name": name.strip() or "MIDI Transcription", "tempo_bpm": tempo_bpm})
    return back(project_id)


@router.post("/ui/projects/{project_id}/lyrics", include_in_schema=False)
def lyrics_ui(
    project_id: str,
    title: Annotated[str, Form()],
    prompt: Annotated[str, Form(min_length=8, max_length=3000)],
    language: Annotated[str, Form()] = "English",
    culture_profile_id: Annotated[str, Form()] = "",
    mood: Annotated[str | None, Form()] = None,
    structure: Annotated[str | None, Form()] = None,
    session: Session = Depends(get_db),
):
    load_project(session, project_id)
    if not provider_status()["configured"]:
        raise HTTPException(status_code=503, detail="No lyrics provider is configured")
    if culture_profile_id and get_profile(culture_profile_id) is None:
        raise HTTPException(status_code=400, detail="Unknown culture profile")
    create_job(session, project_id, "lyrics", {"title": title.strip() or "Untitled", "prompt": prompt.strip(), "language": language, "culture_profile_id": culture_profile_id or None, "mood": mood, "structure": structure})
    return back(project_id)


@router.post("/ui/projects/{project_id}/daw-export", include_in_schema=False)
def daw_export_ui(
    project_id: str,
    name: Annotated[str, Form()] = "DAW Package",
    tempo_bpm: Annotated[float | None, Form()] = None,
    time_signature: Annotated[str, Form()] = "4/4",
    session: Session = Depends(get_db),
):
    load_project(session, project_id)
    create_job(session, project_id, "daw_export", {"name": name.strip() or "DAW Package", "tempo_bpm": tempo_bpm, "time_signature": time_signature, "file_ids": None})
    return back(project_id)


@router.post("/ui/generate", include_in_schema=False)
def generate_ui(
    prompt: Annotated[str, Form(min_length=8, max_length=2000)],
    duration_seconds: Annotated[int, Form()] = 12,
    name: Annotated[str, Form()] = "Generated music",
    language: Annotated[str, Form()] = "English",
    culture_profile_id: Annotated[str, Form()] = "",
    seed: Annotated[int | None, Form()] = None,
    session: Session = Depends(get_db),
):
    if not settings.enable_local_musicgen:
        raise HTTPException(status_code=503, detail="Music generation is disabled")
    if culture_profile_id and get_profile(culture_profile_id) is None:
        raise HTTPException(status_code=400, detail="Unknown culture profile")
    job = create_job(session, None, "generate", {"prompt": prompt.strip(), "duration_seconds": duration_seconds, "name": name.strip() or "Generated music", "language": language, "culture_profile_id": culture_profile_id or None, "seed": seed})
    return RedirectResponse(f"/ui/jobs/{job.id}", status_code=303)


@router.post("/ui/song-package", include_in_schema=False)
def song_package_ui(
    title: Annotated[str, Form()],
    prompt: Annotated[str, Form(min_length=8, max_length=3000)],
    language: Annotated[str, Form()] = "English",
    culture_profile_id: Annotated[str, Form()] = "",
    duration_seconds: Annotated[int, Form()] = 20,
    seed: Annotated[int | None, Form()] = None,
    include_lyrics: Annotated[bool, Form()] = False,
    session: Session = Depends(get_db),
):
    capabilities = inspect_capabilities()
    if not capabilities["complete_song_pipeline"]:
        raise HTTPException(status_code=503, detail="The music generation runtime is not configured")
    if include_lyrics and not capabilities["lyrics_provider_configured"]:
        raise HTTPException(status_code=503, detail="No lyrics provider is configured")
    if culture_profile_id and get_profile(culture_profile_id) is None:
        raise HTTPException(status_code=400, detail="Unknown culture profile")
    job = create_job(session, None, "song_package", {"title": title.strip() or "BeatMaster Song", "name": title.strip() or "BeatMaster Song", "prompt": prompt.strip(), "language": language, "culture_profile_id": culture_profile_id or None, "duration_seconds": duration_seconds, "seed": seed, "include_lyrics": include_lyrics, "separate_stems": True, "extract_chords": True, "extract_midi": True, "export_daw": True, "separation_model": "htdemucs", "output_format": "wav", "time_signature": "4/4"})
    return RedirectResponse(f"/ui/jobs/{job.id}", status_code=303)
