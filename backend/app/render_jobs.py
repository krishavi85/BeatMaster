from sqlalchemy.orm import Session
from .audio_probe import analyze_audio
from .audio_render import master_audio, mix_audio
from .file_registry import get_file, register_file
from .models import Job
from .storage import absolute_from_relative, project_dir, safe_filename


def process_mix(session: Session, job: Job, progress):
    request = job.request_json
    selected = []
    source_ids = []
    for track in request["tracks"]:
        if track.get("mute"):
            continue
        item = get_file(session, track["file_id"], job.project_id)
        selected.append((absolute_from_relative(item.relative_path), float(track.get("gain_db", 0)), float(track.get("pan", 0))))
        source_ids.append(item.id)
    progress(20, "Building mixer graph")
    suffix = request.get("output_format", "wav")
    output = project_dir(job.project_id) / "mixes" / f"{job.id}_{safe_filename(request.get('name', 'mix'))}.{suffix}"
    mix_audio(selected, output)
    progress(90, "Measuring mixed output")
    item = register_file(session, job.project_id, output, "mix", request.get("name", "Mix"), metadata_json={"source_file_ids": source_ids, "tracks": request["tracks"]})
    return {"file_id": item.id, "analysis": analyze_audio(output)}


def process_master(session: Session, job: Job, progress):
    request = job.request_json
    source_item = get_file(session, request["source_file_id"], job.project_id)
    source = absolute_from_relative(source_item.relative_path)
    suffix = request.get("output_format", "wav")
    output = project_dir(job.project_id) / "masters" / f"{job.id}_{safe_filename(request.get('name', 'master'))}.{suffix}"
    progress(10, "First-pass loudness measurement")
    result = master_audio(source, output, float(request["target_lufs"]), float(request["true_peak_db"]), float(request["loudness_range"]), request.get("style", "transparent"))
    progress(92, "Registering mastered output")
    item = register_file(session, job.project_id, output, "master", request.get("name", "Master"), metadata_json={"source_file_id": source_item.id, **result, "settings": request})
    return {"file_id": item.id, **result}
