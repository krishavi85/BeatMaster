from html import escape
from .models import Project


def option_files(project: Project, kinds: set[str] | None = None) -> str:
    items = [item for item in project.files if not kinds or item.kind in kinds]
    return "".join(f'<option value="{item.id}">{escape(item.label)} · {escape(item.kind)}</option>' for item in items)


def metric(label: str, value, suffix: str = "") -> str:
    shown = "—" if value is None else escape(str(value))
    return f'<div class="card"><span class="muted">{escape(label)}</span><div class="metric">{shown}{escape(suffix)}</div></div>'


def render_project(project: Project) -> tuple[str, bool]:
    analysis = project.analysis or {}
    metrics = "".join([
        metric("Integrated loudness", analysis.get("integrated_lufs"), " LUFS"),
        metric("True peak", analysis.get("true_peak_dbfs"), " dBFS"),
        metric("Tempo", analysis.get("tempo_bpm"), " BPM"),
        metric("Estimated key", analysis.get("estimated_key")),
        metric("Sample rate", analysis.get("sample_rate"), " Hz"),
        metric("Duration", analysis.get("duration_seconds"), " s"),
    ])
    file_cards = "".join(
        f'''<div class="card"><span class="pill">{escape(item.kind.upper())}</span><h3>{escape(item.label)}</h3><p class="muted">{escape(item.codec or "unknown")} · {item.sample_rate or 0} Hz · {item.channels or 0} ch · {round((item.duration_seconds or 0), 2)} s</p><audio controls preload="metadata" src="/api/files/{item.id}/stream"></audio><div><a class="button secondary" href="/api/files/{item.id}/download">Download</a></div></div>'''
        for item in project.files
    ) or '<div class="card"><p class="muted">No audio outputs yet.</p></div>'
    jobs = sorted(project.jobs, key=lambda item: item.created_at, reverse=True)
    active = any(item.status in {"queued", "running"} for item in jobs)
    job_rows = "".join(
        f'''<tr><td>{escape(item.type)}</td><td><span class="pill {'ok' if item.status == 'succeeded' else 'off' if item.status == 'failed' else ''}">{escape(item.status)}</span></td><td>{round(item.progress, 1)}%</td><td>{escape(item.stage)}</td><td>{escape((item.error or '')[:220])}</td></tr>'''
        for item in jobs[:25]
    ) or '<tr><td colspan="5">No processing jobs.</td></tr>'
    sources = option_files(project)
    mix_candidates = [item for item in project.files if item.kind in {"source", "stem"}]
    mixer_rows = "".join(
        f'''<tr><td><input type="checkbox" name="file_ids" value="{item.id}" {'checked' if item.kind == 'stem' else ''}> {escape(item.label)}</td><td>{escape(item.kind)}</td></tr>'''
        for item in mix_candidates
    ) or '<tr><td colspan="2">No mixable files.</td></tr>'
    body = f'''<div class="actions"><a class="button secondary" href="/">← Dashboard</a></div><section class="hero"><span class="pill">PROJECT</span><h1>{escape(project.name)}</h1><p>Every value and output below comes from the connected worker and stored audio files.</p></section>
<h2>Measured audio</h2><div class="grid">{metrics}</div>
<h2>Files and renders</h2><div class="grid">{file_cards}</div>
<div class="grid" style="margin-top:18px"><section class="card"><h2>Separate stems</h2><form action="/ui/projects/{project.id}/separate" method="post"><label>Source</label><select name="source_file_id" required>{sources}</select><label>Demucs model</label><select name="model"><option>htdemucs</option><option>htdemucs_ft</option><option>htdemucs_6s</option><option>mdx_extra_q</option></select><label>Mode</label><select name="two_stems"><option value="">4 or 6 stems</option><option value="vocals">Vocals + accompaniment</option><option value="drums">Drums + remainder</option><option value="bass">Bass + remainder</option></select><label>Format</label><select name="output_format"><option>wav</option><option>flac</option><option>mp3</option></select><button type="submit">Run real separation</button></form></section>
<section class="card"><h2>Mix tracks</h2><form action="/ui/projects/{project.id}/mix" method="post"><table><tr><th>Track</th><th>Type</th></tr>{mixer_rows}</table><label>Mix name</label><input name="name" value="BeatMaster Mix"><label>Global gain</label><input name="gain_db" type="number" value="0" min="-60" max="18" step="0.1"><label>Global pan</label><input name="pan" type="number" value="0" min="-1" max="1" step="0.05"><label>Format</label><select name="output_format"><option>wav</option><option>flac</option><option>mp3</option></select><button type="submit">Render real mix</button></form></section>
<section class="card"><h2>Master audio</h2><form action="/ui/projects/{project.id}/master" method="post"><label>Source</label><select name="source_file_id" required>{sources}</select><label>Name</label><input name="name" value="BeatMaster Master"><label>Target LUFS</label><input name="target_lufs" type="number" value="-14" min="-24" max="-5" step="0.1"><label>True peak ceiling</label><input name="true_peak_db" type="number" value="-1" min="-3" max="-0.1" step="0.1"><label>Loudness range</label><input name="loudness_range" type="number" value="11" min="1" max="20" step="0.1"><label>Style</label><select name="style"><option>transparent</option><option>warm</option><option>bright</option><option>punchy</option><option>wide</option></select><label>Format</label><select name="output_format"><option>wav</option><option>flac</option><option>mp3</option></select><button type="submit">Run two-pass master</button></form></section></div>
<section class="card" style="margin-top:18px"><h2>Processing jobs</h2><table><tr><th>Processor</th><th>Status</th><th>Progress</th><th>Stage</th><th>Error</th></tr>{job_rows}</table></section>'''
    return body, active
