from html import escape

from sqlalchemy import select
from sqlalchemy.orm import Session

from .dashboard_generation import generation_panel
from .models import Project
from .runtime_capabilities import inspect_capabilities


def render_dashboard(session: Session) -> str:
    projects = session.scalars(select(Project).order_by(Project.updated_at.desc())).all()
    capabilities = inspect_capabilities()
    project_cards = "".join(
        f'<a class="card project" href="/ui/projects/{item.id}"><span class="pill">PROJECT</span><h3>{escape(item.name)}</h3><p class="muted">Updated {item.updated_at.strftime("%Y-%m-%d %H:%M")}</p></a>'
        for item in projects
    )
    if not project_cards:
        project_cards = '<div class="card"><h3>No projects yet</h3><p class="muted">Upload a real audio file below.</p></div>'
    upload = '<section id="new" class="card"><h2>New project</h2><form action="/ui/projects" method="post" enctype="multipart/form-data"><label>Project name</label><input name="name" required maxlength="200" placeholder="My production"><label>Audio file</label><input name="audio" type="file" accept="audio/*" required><button type="submit">Upload and analyze</button></form></section>'
    cards = [
        ("FFMPEG", capabilities["ffmpeg"], "Mixing, mastering and analysis"),
        ("DEMUCS", capabilities["demucs"], "Real editable source separation"),
        ("HARMONY", capabilities["chord_detection"] and capabilities["midi_transcription"], "Chord maps and MIDI transcription"),
        ("LYRICS", capabilities["lyrics_provider_configured"], f"Provider: {capabilities['lyrics_provider'] or 'not configured'}"),
        ("DAW EXPORT", capabilities["daw_export"], "REAPER project and aligned interchange ZIP"),
        ("CULTURE", capabilities["culture_profile_count"] > 0, f"{capabilities['culture_profile_count']} transparent culture profiles"),
    ]
    status_cards = "".join(
        f'<div class="card"><span class="pill {"ok" if ready else "off"}">{label}</span><div class="metric">{"Ready" if ready else "Unavailable"}</div><p class="muted">{escape(detail)}</p></div>'
        for label, ready, detail in cards
    )
    generation_ready = bool(capabilities["complete_song_pipeline"])
    lyrics_ready = bool(capabilities["lyrics_provider_configured"])
    return f'''<section class="hero"><span class="pill ok">MULTILINGUAL AI MUSIC STUDIO</span><h1>Produce, edit and export without placeholders.</h1><p>Create music beds, editable stems, chord maps, MIDI, multilingual lyrics and DAW interchange packages. Caribbean and Surinamese profiles are explicit and inspectable; BeatMaster only labels a model as culturally fine-tuned when you register that model.</p></section>
<div class="grid">{status_cards}</div>
<div class="grid" style="margin-top:18px">{upload}{generation_panel(generation_ready, lyrics_ready)}</div>
<section style="margin-top:18px"><h2>Your projects</h2><div class="grid">{project_cards}</div></section>'''
