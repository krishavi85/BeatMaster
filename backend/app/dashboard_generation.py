from html import escape

from .culture_profiles import SUPPORTED_LANGUAGES, list_profiles


def _language_options() -> str:
    return "".join(f'<option value="{escape(item["name"])}">{escape(item["name"])}</option>' for item in SUPPORTED_LANGUAGES)


def _culture_options() -> str:
    return '<option value="">No cultural profile</option>' + "".join(
        f'<option value="{escape(item["id"])}">{escape(item["name"])} · {escape(item["region"])}</option>'
        for item in list_profiles()
    )


def generation_panel(ready: bool, lyrics_ready: bool = False, singing_ready: bool = False) -> str:
    disabled = "" if ready else "disabled"
    notice = "" if ready else '<div class="notice">Generation is disabled until ENABLE_LOCAL_MUSICGEN is true, MUSICGEN_MODEL is configured, and the AI worker dependencies are installed.</div>'
    lyric_note = "" if lyrics_ready else '<div class="notice">The package can run without lyrics. Configure a lyrics provider to include multilingual songwriting.</div>'
    singing_note = "" if singing_ready else '<div class="notice">Singing vocals remain disabled until a compatible synthesis provider is connected.</div>'
    languages = _language_options()
    profiles = _culture_options()
    music_form = f'''<section class="card"><h2>Generate music</h2><p class="muted">Runs the configured local MusicGen-compatible model. Culture profiles add transparent rhythm and instrumentation conditioning; mapped fine-tuned models are used only when registered by the deployment owner.</p><form action="/ui/generate" method="post"><label>Prompt</label><textarea name="prompt" required minlength="8" maxlength="2000" placeholder="Describe genre, instruments, mood and arrangement"></textarea><label>Name</label><input name="name" value="Generated music"><label>Language context</label><select name="language">{languages}</select><label>Culture profile</label><select name="culture_profile_id">{profiles}</select><label>Duration in seconds</label><input name="duration_seconds" type="number" value="12" min="4" max="30"><label>Seed</label><input name="seed" type="number" placeholder="Optional"><button type="submit" {disabled}>Generate real audio</button></form>{notice}</section>'''
    package_form = f'''<section class="card"><h2>Complete song production package</h2><p class="muted">Creates a music bed, editable stems, chord map, MIDI, lyrics and a DAW package. A connected singing provider can also render a vocal track and final mix.</p><form action="/ui/song-package-complete" method="post"><label>Title</label><input name="title" value="BeatMaster Song" required><label>Creative brief</label><textarea name="prompt" required minlength="8" maxlength="3000" placeholder="Describe the song, arrangement, emotion, instruments and audience"></textarea><label>Language</label><select name="language">{languages}</select><label>Culture profile</label><select name="culture_profile_id">{profiles}</select><label>Duration in seconds</label><input name="duration_seconds" type="number" value="20" min="4" max="30"><label>Seed</label><input name="seed" type="number" placeholder="Optional"><label><input type="checkbox" name="include_lyrics" value="true" {'checked' if lyrics_ready else ''}> Include AI-written lyrics</label><br><label><input type="checkbox" name="render_vocals" value="true" {'disabled' if not singing_ready else ''}> Render singing vocals</label><label>Voice ID</label><input name="voice_id" placeholder="Provider voice or singer model ID" {'disabled' if not singing_ready else ''}><br><button type="submit" {disabled}>Build production package</button></form>{notice}{lyric_note}{singing_note}</section>'''
    return music_form + package_form
