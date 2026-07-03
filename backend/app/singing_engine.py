from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx


class SingingProviderError(RuntimeError):
    pass


def provider_status() -> dict[str, Any]:
    provider = os.getenv("SINGING_PROVIDER", "").strip().lower()
    if provider == "rest":
        return {
            "provider": provider,
            "configured": bool(os.getenv("SINGING_API_URL")),
            "default_voice_id": os.getenv("SINGING_VOICE_ID") or None,
        }
    return {"provider": provider or None, "configured": False, "default_voice_id": None}


def render_singing(
    *,
    lyrics_path: Path,
    output_path: Path,
    language: str,
    title: str,
    voice_id: str | None = None,
    midi_path: Path | None = None,
) -> dict[str, Any]:
    status = provider_status()
    if not status["configured"] or status["provider"] != "rest":
        raise SingingProviderError("No singing synthesis provider is configured")
    url = os.getenv("SINGING_API_URL", "").strip()
    headers: dict[str, str] = {}
    token = os.getenv("SINGING_API_KEY", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    selected_voice = voice_id or os.getenv("SINGING_VOICE_ID", "").strip()
    fields = {
        "language": language,
        "title": title,
        "voice_id": selected_voice,
        "output_format": "wav",
    }
    files: dict[str, tuple[str, bytes, str]] = {
        "lyrics": (lyrics_path.name, lyrics_path.read_bytes(), "text/plain"),
    }
    if midi_path is not None:
        files["midi"] = (midi_path.name, midi_path.read_bytes(), "audio/midi")
    with httpx.Client(timeout=600.0, follow_redirects=True) as client:
        response = client.post(url, headers=headers, data=fields, files=files)
        if response.status_code >= 400:
            raise SingingProviderError(f"Singing provider returned {response.status_code}: {response.text[-1500:]}")
        content_type = response.headers.get("content-type", "").lower()
        if content_type.startswith("audio/") or content_type == "application/octet-stream":
            audio_bytes = response.content
            metadata = {"provider": "rest", "voice_id": selected_voice, "language": language, "response_type": content_type}
        else:
            try:
                data = response.json()
            except json.JSONDecodeError as exc:
                raise SingingProviderError("Singing provider returned neither audio nor JSON") from exc
            audio_url = str(data.get("audio_url") or "").strip()
            if not audio_url:
                raise SingingProviderError("Singing provider JSON response is missing audio_url")
            audio_response = client.get(audio_url)
            if audio_response.status_code >= 400:
                raise SingingProviderError(f"Could not download rendered singing audio: {audio_response.status_code}")
            audio_bytes = audio_response.content
            metadata = {"provider": "rest", "voice_id": selected_voice, "language": language, "response": data}
    if len(audio_bytes) < 1024:
        raise SingingProviderError("Rendered singing audio is unexpectedly small")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)
    return metadata
