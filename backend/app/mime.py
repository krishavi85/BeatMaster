from pathlib import Path

AUDIO_MIME = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
}


def audio_mime(path: Path) -> str:
    return AUDIO_MIME.get(path.suffix.lower(), "application/octet-stream")
