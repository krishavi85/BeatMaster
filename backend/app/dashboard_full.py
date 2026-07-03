import importlib.util
import os
import shutil
from html import escape
from sqlalchemy import select
from sqlalchemy.orm import Session
from .config import settings
from .dashboard_generation import generation_panel
from .models import Project


def render_dashboard(session: Session) -> str:
    projects = session.scalars(select(Project).order_by(Project.updated_at.desc())).all()
    demucs = importlib.util.find_spec("demucs") is not None
    ffmpeg = shutil.which(settings.ffmpeg_bin) is not None
    generation = settings.enable_local_musicgen and importlib.util.find_spec("transformers") is not None and bool(os.getenv("MUSICGEN_MODEL"))
    project_cards = "".join(f'<a class="card project" href="/ui/projects/{item.id}"><span class="pill">PROJECT</span><h3>{escape(item.name)}</h3><p class="muted">Updated {item.updated_at.strftime("%Y-%m-%d %H:%M")}</p></a>' for item in projects)
    if not project_cards:
        project_cards = '<div class="card"><h3>No projects yet</h3><p class="muted">Upload a real audio file below.</p></div>'
    upload = '<section id="new" class="card"><h2>New project</h2><form action="/ui/projects" method="post" enctype="multipart/form-data"><label>Project name</label><input name="name" required maxlength="200" placeholder="My production"><label>Audio file</label><input name="audio" type="file" accept="audio/*" required><button type="submit">Upload and analyze</button></form></section>'
    return f'<section class="hero"><span class="pill ok">AUDIO WORKSTATION</span><h1>Produce without placeholders.</h1><p>Upload audio, inspect measured values, separate stems, mix tracks, master to a loudness target, and generate audio when a model is configured.</p></section><div class="grid"><div class="card"><span class="pill {"ok" if ffmpeg else "off"}">FFMPEG</span><div class="metric">{"Ready" if ffmpeg else "Unavailable"}</div><p class="muted">Mixing, mastering and analysis</p></div><div class="card"><span class="pill {"ok" if demucs else "off"}">DEMUCS</span><div class="metric">{"Ready" if demucs else "Unavailable"}</div><p class="muted">Real source separation</p></div><div class="card"><span class="pill {"ok" if generation else "off"}">GENERATION</span><div class="metric">{"Ready" if generation else "Disabled"}</div><p class="muted">Configured local model only</p></div></div><div class="grid" style="margin-top:18px">{upload}{generation_panel(generation)}</div><section style="margin-top:18px"><h2>Your projects</h2><div class="grid">{project_cards}</div></section>'
