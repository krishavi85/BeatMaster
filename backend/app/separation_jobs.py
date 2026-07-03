from pathlib import Path
import re
import subprocess
from sqlalchemy.orm import Session
from .audio_probe import AudioProcessingError
from .config import settings
from .file_registry import get_file, register_file
from .models import Job
from .storage import absolute_from_relative, project_dir


def process_separate(session: Session, job: Job, progress):
    request = job.request_json
    file_item = get_file(session, request["source_file_id"], job.project_id)
    source = absolute_from_relative(file_item.relative_path)
    model = request.get("model", "htdemucs")
    output_format = request.get("output_format", "wav")
    root = project_dir(job.project_id) / "stems" / job.id
    root.mkdir(parents=True, exist_ok=True)
    command = ["python", "-m", "demucs.separate", "-n", model, "-d", settings.demucs_device, "--out", str(root)]
    if output_format == "mp3":
        command += ["--mp3", "--mp3-bitrate", "320"]
    elif output_format == "flac":
        command += ["--flac"]
    else:
        command += ["--int24"]
    if request.get("two_stems"):
        command += [f"--two-stems={request['two_stems']}"]
    command.append(str(source))
    progress(5, f"Loading Demucs model {model}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip())
        match = re.search(r"(\d{1,3})%", line)
        if match:
            progress(min(92, 10 + int(match.group(1)) * 0.8), "Separating sources")
    if process.wait() != 0:
        raise AudioProcessingError("\n".join(lines[-100:]))
    candidates = [path for path in root.rglob("*") if path.suffix.lower() in {".wav", ".flac", ".mp3"}]
    if not candidates:
        raise AudioProcessingError("Demucs finished without stem files")
    progress(94, "Registering separated stems")
    outputs = []
    for path in sorted(candidates):
        stem_name = path.stem.lower()
        item = register_file(session, job.project_id, path, "stem", stem_name.replace("_", " ").title(), metadata_json={"model": model, "source_file_id": file_item.id, "stem": stem_name})
        outputs.append({"file_id": item.id, "stem": stem_name})
    return {"model": model, "outputs": outputs, "log_tail": lines[-25:]}
