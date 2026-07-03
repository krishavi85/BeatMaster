from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from .api_helpers import create_job, get_db, load_project
from .culture_profiles import get_profile
from .models import AudioFile
from .runtime_capabilities import inspect_capabilities
from .singing_engine import provider_status

router = APIRouter(tags=["ui"])


def _project_file(session: Session, project_id: str, file_id: str) -> AudioFile:
    item = session.get(AudioFile, file_id)
    if item is None or item.project_id != project_id:
        raise HTTPException(status_code=400, detail="Selected file does not belong to this project")
    return item


@router.post("/ui/projects/{project_id}/singing", include_in_schema=False)
def singing_ui(
    project_id: str,
    lyrics_file_id: Annotated[str, Form()],
    midi_file_id: Annotated[str | None, Form()] = None,
    name: Annotated[str, Form()] = "Lead Vocals",
    title: Annotated[str, Form()] = "BeatMaster Song",
    language: Annotated[str, Form()] = "English",
    voice_id: Annotated[str | None, Form()] = None,
    session: Session = Depends(get_db),
):
    load_project(session, project_id)
    _project_file(session, project_id, lyrics_file_id)
    if midi_file_id:
        _project_file(session, project_id, midi_file_id)
    if not provider_status()["configured"]:
        raise HTTPException(status_code=503, detail="No singing synthesis provider is configured")
    create_job(session, project_id, "singing", {"lyrics_file_id": lyrics_file_id, "midi_file_id": midi_file_id or None, "name": name.strip() or "Lead Vocals", "title": title.strip() or "BeatMaster Song", "language": language, "voice_id": voice_id or None})
    return RedirectResponse(f"/ui/projects/{project_id}", status_code=303)


@router.post("/ui/song-package-complete", include_in_schema=False)
def complete_song_ui(
    title: Annotated[str, Form()],
    prompt: Annotated[str, Form(min_length=8, max_length=3000)],
    language: Annotated[str, Form()] = "English",
    culture_profile_id: Annotated[str, Form()] = "",
    structure: Annotated[str | None, Form()] = None,
    duration_seconds: Annotated[int, Form()] = 180,
    seed: Annotated[int | None, Form()] = None,
    include_lyrics: Annotated[bool, Form()] = False,
    render_vocals: Annotated[bool, Form()] = False,
    voice_id: Annotated[str | None, Form()] = None,
    session: Session = Depends(get_db),
):
    capabilities = inspect_capabilities()
    if not capabilities["complete_song_pipeline"]:
        raise HTTPException(status_code=503, detail="The complete production pipeline is not configured")
    if not 4 <= duration_seconds <= 300:
        raise HTTPException(status_code=400, detail="Duration must be between 4 and 300 seconds")
    if culture_profile_id and get_profile(culture_profile_id) is None:
        raise HTTPException(status_code=400, detail="Unknown culture profile")
    if include_lyrics and not capabilities["lyrics_provider_configured"]:
        raise HTTPException(status_code=503, detail="No lyrics provider is configured")
    if render_vocals and not include_lyrics:
        raise HTTPException(status_code=400, detail="Singing vocals require generated lyrics")
    if render_vocals and not capabilities["singing_provider_configured"]:
        raise HTTPException(status_code=503, detail="No singing synthesis provider is configured")
    job = create_job(session, None, "song_package", {"title": title.strip() or "BeatMaster Song", "name": title.strip() or "BeatMaster Song", "prompt": prompt.strip(), "language": language, "culture_profile_id": culture_profile_id or None, "structure": structure, "duration_seconds": duration_seconds, "seed": seed, "include_lyrics": include_lyrics, "render_vocals": render_vocals, "voice_id": voice_id or None, "vocal_gain_db": -3.0, "separate_stems": True, "extract_chords": True, "extract_midi": True, "export_daw": True, "separation_model": "htdemucs", "output_format": "wav", "time_signature": "4/4"})
    return RedirectResponse(f"/ui/jobs/{job.id}", status_code=303)
