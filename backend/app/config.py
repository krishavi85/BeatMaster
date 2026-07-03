from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("BEATMASTER_DATA_DIR", "/data"))
    database_url: str = os.getenv("DATABASE_URL", "sqlite:////data/beatmaster.db")
    cors_origins: tuple[str, ...] = tuple(x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8080").split(",") if x.strip())
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
    worker_poll_seconds: float = float(os.getenv("WORKER_POLL_SECONDS", "1.0"))
    enable_local_musicgen: bool = os.getenv("ENABLE_LOCAL_MUSICGEN", "false").lower() == "true"
    musicgen_device: str = os.getenv("MUSICGEN_DEVICE", "cuda")
    ffmpeg_bin: str = os.getenv("FFMPEG_BIN", "ffmpeg")
    ffprobe_bin: str = os.getenv("FFPROBE_BIN", "ffprobe")
    demucs_device: str = os.getenv("DEMUCS_DEVICE", "cpu")

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "projects").mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
