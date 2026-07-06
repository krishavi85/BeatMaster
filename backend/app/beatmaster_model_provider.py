from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

from .audio_probe import AudioProcessingError


def provider_status() -> dict[str, Any]:
    url = os.getenv("BEATMASTER_MODEL_API_URL", "").strip()
    return {
        "provider": "beatmaster-model-server",
        "configured": bool(url),
        "url": url or None,
    }


def generate_with_beatmaster_model(request: dict[str, Any], output: Path) -> dict[str, Any]:
    url = os.getenv("BEATMASTER_MODEL_API_URL", "").strip().rstrip("/")
    if not url:
        raise AudioProcessingError("BEATMASTER_MODEL_API_URL is not configured")
    token = os.getenv("BEATMASTER_MODEL_API_KEY", "").strip()
    headers = {"Accept": "audio/wav"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {
        "prompt": request["prompt"],
        "duration_seconds": int(request.get("duration_seconds", 30)),
        "seed": request.get("seed"),
        "temperature": float(request.get("temperature", 1.0)),
        "top_k": int(request.get("top_k", 100)),
    }
    try:
        response = httpx.post(f"{url}/generate", json=payload, headers=headers, timeout=float(os.getenv("BEATMASTER_MODEL_TIMEOUT", "1800")))
    except httpx.HTTPError as exc:
        raise AudioProcessingError(f"BeatMaster model server could not be reached: {exc}") from exc
    if response.status_code >= 400:
        detail = response.text[-2000:]
        raise AudioProcessingError(f"BeatMaster model server returned {response.status_code}: {detail}")
    content_type = response.headers.get("content-type", "").lower()
    if "audio" not in content_type and content_type != "application/octet-stream":
        raise AudioProcessingError(f"BeatMaster model server returned unexpected content type: {content_type}")
    if len(response.content) < 1024:
        raise AudioProcessingError("BeatMaster model server returned an unexpectedly small audio file")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)
    return {
        "provider": "beatmaster-model-server",
        "server_url": url,
        "model": response.headers.get("x-beatmaster-model") or "BeatMaster custom model",
        "duration_seconds_requested": payload["duration_seconds"],
        "seed": payload["seed"],
        "temperature": payload["temperature"],
        "top_k": payload["top_k"],
    }
