from __future__ import annotations

from html import escape

from .culture_profiles import SUPPORTED_LANGUAGES, list_profiles
from .models import AudioFile, Project
from .runtime_capabilities import inspect_capabilities
from .storage import absolute_from_relative

AUDIO_KINDS = {"source", "stem", "mix", "master", "generated", "vocal"}
EDITABLE_KINDS = {"lyrics", "chord_sheet", "chord_timeline", "notes"}


def option_files(project: Project, kinds: set[str] | None = None) -> str:
    items = [item for item in project.files if not kinds or item.kind in kinds]
    return "".join(f'<option value="{item.id}">{escape(item.label)} · {escape(item.kind)}</option>' for item in items)


def metric(label: str, value, suffix: str = "") -> str:
    shown = "—" if value is None else escape(str(value))
    return f'<div class="card"><span class="muted">{escape(label)}</span><div class="metric">{shown}{escape(suffix)}</div></div>'


def _text_preview(item: AudioFile) -> str:
    if not (item.mime_type or "").startswith("text/") and item.kind not in EDITABLE_KINDS:
        return ""
    try:
        text = absolute_from_relative(item.relative_path).read_text(encoding="utf-8")[:2500]
    except Exception:
        return ""
    return f'<details><summary>Preview</summary><pre style="white-space:pre-wrap;max-height:320px;overflow:auto">{escape(text)}</pre></details>'


def _file_card(item: AudioFile) -> str:
    details = f'{escape(item.codec or "file")} · {item.size_bytes / 1024 / 1024:.2f} MB'
    if item.duration_seconds is not None:
        details += f' · {item.sample_rate or 0} Hz · {item.channels or 0} ch · {round(item.duration_seconds, 2)} s'
    if item.kind in AUDIO_KINDS or (item.mime_type or "").startswith("audio/"):
        preview = f'<audio controls preload="metadata" src="/api/files/{item.id}/stream"></audio>'
    else:
        preview = _text_preview(item)
    edit_button = f'<a class="button" href="/ui/projects/{item.project_id}/document-editor?file_id={item.id}">Edit</a>' if item.kind in EDITABLE_KINDS else ""
    return f'''<div class="card"><span class="pill">{escape(item.kind.upper())}</span><h3>{escape(item.label)}</h3><p class="muted">{details}</p>{preview}<div class="actions"><a class="button secondary" href="/api/files/{item.id}/download">Download</a>{edit_button}</div></div>'''


def _language_options() -> str:
    return "".join(f'<option value="{escape(item["name"])}">{escape(item["name"])}</option>' for item in SUPPORTED_LANGUAGES)


def _culture_options(selected: str | None = None) -> str:
    options = ['<option value="">No cultural profile</option>']
    for item in list_profiles():
        chosen = " selected" if selected == item["id"] else ""
        options.append(f'<option value="{escape(item["id"])}"{chosen}>{escape(item["name"])} · {escape(item["region"])}</option>')
    return "".join(options)


def render_project(project: Project) -> tuple[str, bool]:
    analysis = project.analysis or {}
    capabilities = inspect_capabilities()
    metrics = "".join([
        metric("Integrated loudness", analysis.get("integrated_lufs"), " LUFS"),
        metric("True peak", analysis.get("true_peak_dbfs"), " dBFS"),
        metric("Tempo", analysis.get("tempo_bpm"), " BPM"),
        metric("Estimated key", analysis.get("estimated_key")),
        metric("Sample rate", analysis.get("sample_rate"), " Hz"),
        metric("Duration", analysis.get("duration_seconds"), " s"),
    ])
    file_cards = "".join(_file_card(item) for item in project.files) or '<div class="card"><p class="muted">No audio or production assets yet.</p></div>'
    jobs = sorted(project.jobs, key=lambda item: item.created_at, reverse=True)
    active = any(item.status in {"queued", "running"} for item in jobs)
    job_rows = "".join(
        f'''<tr><td>{escape(item.type)}</td><td><span class="pill {'ok' if item.status == 'succeeded' else 'off' if item.status == 'failed' else ''}">{escape(item.status)}</span></td><td>{round(item.progress, 1)}%</td><td>{escape(item.stage)}</td><td>{escape((item.error or '')[:220])}</td></tr>'''
        for item in jobs[:40]
    ) or '<tr><td colspan="5">No processing jobs.</td></tr>'
    audio_files = [item for item in project.files if item.kind in AUDIO_KINDS or (item.mime_type or "").startswith("audio/")]
    sources = option_files(project, {item.kind for item in audio_files})
    mixer_rows = "".join(
        f'''<tr class="mix-row" data-file-id="{item.id}"><td><strong>{escape(item.label)}</strong><br><span class="muted">{escape(item.kind)}</span></td><td><input class="mix-gain" type="number" value="0" min="-60" max="18" step="0.1"></td><td><input class="mix-pan" type="number" value="0" min="-1" max="1" step="0.05"></td><td><input class="mix-mute" type="checkbox"></td></tr>'''
        for item in audio_files if item.kind in {"source", "stem", "generated", "vocal"}
    ) or '<tr><td colspan="4">No mixable files.</td></tr>'
    culture_selected = analysis.get("culture_profile_id")
    lyrics_disabled = "" if capabilities["lyrics_provider_configured"] else "disabled"
    lyrics_notice = "" if capabilities["lyrics_provider_configured"] else '<div class="notice">Connect Ollama, an OpenAI-compatible endpoint or a local Transformers text model to enable AI songwriting.</div>'
    lyrics_options = option_files(project, {"lyrics"})
    midi_options = '<option value="">No melody MIDI</option>' + option_files(project, {"midi"})
    singing_ready = capabilities["singing_provider_configured"] and bool(lyrics_options)
    singing_disabled = "" if singing_ready else "disabled"
    if not capabilities["singing_provider_configured"]:
        singing_notice = '<div class="notice">Connect a compatible singing synthesis REST provider to render vocals.</div>'
    elif not lyrics_options:
        singing_notice = '<div class="notice">Generate or write a lyrics asset before rendering vocals.</div>'
    else:
        singing_notice = ""
    forms = f'''
<div class="grid" style="margin-top:18px">
<section class="card"><h2>Separate editable stems</h2><form action="/ui/projects/{project.id}/separate" method="post"><label>Source</label><select name="source_file_id" required>{sources}</select><label>Demucs model</label><select name="model"><option>htdemucs</option><option>htdemucs_ft</option><option>htdemucs_6s</option><option>mdx_extra_q</option></select><label>Mode</label><select name="two_stems"><option value="">4 or 6 stems</option><option value="vocals">Vocals + accompaniment</option><option value="drums">Drums + remainder</option><option value="bass">Bass + remainder</option></select><label>Format</label><select name="output_format"><option>wav</option><option>flac</option><option>mp3</option></select><button type="submit">Run real separation</button></form></section>
<section class="card"><h2>Editable stem mixer</h2><form action="/ui/projects/{project.id}/mix" method="post" onsubmit="return collectMixer(this)"><input type="hidden" name="tracks_json" value=""><table><tr><th>Track</th><th>Gain dB</th><th>Pan</th><th>Mute</th></tr>{mixer_rows}</table><label>Mix name</label><input name="name" value="BeatMaster Mix"><label>Format</label><select name="output_format"><option>wav</option><option>flac</option><option>mp3</option></select><button type="submit">Render real mix</button></form></section>
<section class="card"><h2>Master audio</h2><form action="/ui/projects/{project.id}/master" method="post"><label>Source</label><select name="source_file_id" required>{sources}</select><label>Name</label><input name="name" value="BeatMaster Master"><label>Target LUFS</label><input name="target_lufs" type="number" value="-14" min="-24" max="-5" step="0.1"><label>True peak ceiling</label><input name="true_peak_db" type="number" value="-1" min="-3" max="-0.1" step="0.1"><label>Loudness range</label><input name="loudness_range" type="number" value="11" min="1" max="20" step="0.1"><label>Style</label><select name="style"><option>transparent</option><option>warm</option><option>bright</option><option>punchy</option><option>wide</option></select><label>Format</label><select name="output_format"><option>wav</option><option>flac</option><option>mp3</option></select><button type="submit">Run two-pass master</button></form></section>
<section class="card"><h2>Chords and MIDI</h2><form action="/ui/projects/{project.id}/chords" method="post"><label>Audio source</label><select name="source_file_id" required>{sources}</select><label>Chord-map name</label><input name="name" value="Chord Map"><button type="submit">Extract chord timeline</button></form><hr style="border-color:#29243d"><form action="/ui/projects/{project.id}/midi" method="post"><label>Audio source</label><select name="source_file_id" required>{sources}</select><label>MIDI name</label><input name="name" value="MIDI Transcription"><label>Tempo override</label><input name="tempo_bpm" type="number" min="30" max="300" step="0.1" placeholder="Use detected tempo"><button type="submit">Transcribe MIDI</button></form></section>
<section class="card"><h2>Multilingual songwriting</h2><div class="actions"><a class="button secondary" href="/ui/projects/{project.id}/document-editor?kind=lyrics&label={escape(project.name)}%20Lyrics">Write lyrics manually</a></div><br><form action="/ui/projects/{project.id}/lyrics" method="post"><label>Title</label><input name="title" value="{escape(project.name)}" required><label>Theme and story</label><textarea name="prompt" minlength="8" maxlength="3000" required placeholder="Describe the story, message and emotional progression"></textarea><label>Language</label><select name="language">{_language_options()}</select><label>Culture profile</label><select name="culture_profile_id">{_culture_options(culture_selected)}</select><label>Mood</label><input name="mood" placeholder="Hopeful, intimate, celebratory..."><label>Structure</label><input name="structure" placeholder="Verse, Pre-Chorus, Chorus, Verse, Bridge, Final Chorus"><button type="submit" {lyrics_disabled}>Generate editable lyrics</button></form>{lyrics_notice}</section>
<section class="card"><h2>Singing vocals</h2><p class="muted">Sends your lyrics and optional melody MIDI to the configured synthesis provider and stores the returned vocal track.</p><form action="/ui/projects/{project.id}/singing" method="post"><label>Lyrics</label><select name="lyrics_file_id" required>{lyrics_options}</select><label>Melody MIDI</label><select name="midi_file_id">{midi_options}</select><label>Track name</label><input name="name" value="Lead Vocals"><label>Song title</label><input name="title" value="{escape(project.name)}"><label>Language</label><select name="language">{_language_options()}</select><label>Voice ID</label><input name="voice_id" placeholder="Provider voice or singer model ID"><button type="submit" {singing_disabled}>Render singing track</button></form>{singing_notice}</section>
<section class="card"><h2>DAW integration</h2><p class="muted">Exports aligned audio, MIDI, chords and lyrics with a REAPER project and a generic interchange manifest.</p><form action="/ui/projects/{project.id}/daw-export" method="post"><label>Package name</label><input name="name" value="{escape(project.name)} DAW Package"><label>Tempo</label><input name="tempo_bpm" type="number" min="30" max="300" step="0.1" value="{escape(str(analysis.get('tempo_bpm') or ''))}" placeholder="120"><label>Time signature</label><input name="time_signature" value="4/4" pattern="[1-9][0-9]?/[1-9][0-9]?"><button type="submit">Export DAW package</button></form></section>
</div>'''
    script = '''<script>function collectMixer(form){const tracks=[];form.querySelectorAll('.mix-row').forEach(row=>{tracks.push({file_id:row.dataset.fileId,gain_db:Number(row.querySelector('.mix-gain').value),pan:Number(row.querySelector('.mix-pan').value),mute:row.querySelector('.mix-mute').checked});});form.querySelector('[name=tracks_json]').value=JSON.stringify(tracks);return tracks.some(track=>!track.mute);}</script>'''
    provenance = ""
    if culture_selected:
        provenance = f'<div class="notice">Culture profile: <strong>{escape(str(culture_selected))}</strong>. Check each generated asset metadata to see whether a registered fine-tuned model or transparent prompt conditioning was used.</div>'
    body = f'''<div class="actions"><a class="button secondary" href="/">← Dashboard</a></div><section class="hero"><span class="pill">PROJECT</span><h1>{escape(project.name)}</h1><p>Every value and output below comes from the connected worker and stored project assets.</p></section>{provenance}
<h2>Measured audio</h2><div class="grid">{metrics}</div>
<h2>Files and production assets</h2><div class="grid">{file_cards}</div>
{forms}
<section class="card" style="margin-top:18px"><h2>Processing jobs</h2><table><tr><th>Processor</th><th>Status</th><th>Progress</th><th>Stage</th><th>Error</th></tr>{job_rows}</table></section>{script}'''
    return body, active
