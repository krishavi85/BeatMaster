from pathlib import Path
import re
from uuid import uuid4
from fastapi import HTTPException, UploadFile
from .config import settings

SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
ALLOWED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif"}


def safe_filename(value: str) -> str:
    name = Path(value).name
    cleaned = SAFE_RE.sub("_", name).strip("._")
    return cleaned[:180] or "audio"


def project_dir(project_id: str) -> Path:
    path = settings.data_dir / "projects" / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_to_data(path: Path) -> str:
    return str(path.resolve().relative_to(settings.data_dir.resolve()))


def absolute_from_relative(relative_path: str) -> Path:
    root = settings.data_dir.resolve()
    candidate = (root / relative_path).resolve()
    if root not in candidate.parents and candidate != root:
        raise HTTPException(status_code=400, detail="Invalid file path")
    return candidate


async def save_upload(project_id: str, upload: UploadFile) -> tuple[Path, int]:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Unsupported audio extension: {suffix or 'none'}")
    target_dir = project_dir(project_id) / "source"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{uuid4().hex}_{safe_filename(upload.filename or 'audio' + suffix)}"
    total = 0
    try:
        with target.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Audio file exceeds configured upload limit")
                handle.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return target, total
