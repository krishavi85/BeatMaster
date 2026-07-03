from __future__ import annotations

import os
from typing import Any

import httpx

from .culture_profiles import get_profile


class LyricsProviderError(RuntimeError):
    pass


def provider_status() -> dict[str, Any]:
    provider = os.getenv("LYRICS_PROVIDER", "").strip().lower()
    if provider == "ollama":
        return {"provider": provider, "configured": bool(os.getenv("OLLAMA_URL") and os.getenv("OLLAMA_LYRICS_MODEL")), "model": os.getenv("OLLAMA_LYRICS_MODEL") or None}
    if provider == "openai-compatible":
        return {"provider": provider, "configured": bool(os.getenv("LYRICS_API_URL") and os.getenv("LYRICS_MODEL")), "model": os.getenv("LYRICS_MODEL") or None}
    if provider == "local-transformers":
        return {"provider": provider, "configured": bool(os.getenv("LYRICS_LOCAL_MODEL")), "model": os.getenv("LYRICS_LOCAL_MODEL") or None}
    return {"provider": provider or None, "configured": False, "model": None}


def build_prompt(request: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You are BeatMaster's multilingual songwriter. Create original, singable lyrics. "
        "Preserve the requested language and cultural context, avoid stereotypes, and return only labeled song sections."
    )
    profile = get_profile(request.get("culture_profile_id"))
    profile_text = ""
    if profile:
        profile_text = (
            f"Cultural profile: {profile['name']} from {profile['region']}. "
            f"Rhythm: {profile['rhythm']}. Instruments: {', '.join(profile['instruments'])}. "
            f"Guidance: {profile['production_notes']}"
        )
    user = (
        f"Title: {request.get('title') or 'Untitled'}\n"
        f"Language: {request.get('language') or 'English'}\n"
        f"Theme: {request['prompt']}\n"
        f"Mood: {request.get('mood') or 'emotionally clear'}\n"
        f"Structure: {request.get('structure') or 'Verse, Pre-Chorus, Chorus, Verse, Chorus, Bridge, Final Chorus'}\n"
        f"{profile_text}\n"
        "Write a complete performance-ready lyric without commentary."
    )
    return system, user


def _ollama(system: str, user: str) -> str:
    base = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434").rstrip("/")
    model = os.getenv("OLLAMA_LYRICS_MODEL", "").strip()
    if not model:
        raise LyricsProviderError("OLLAMA_LYRICS_MODEL is required")
    response = httpx.post(f"{base}/api/generate", json={"model": model, "system": system, "prompt": user, "stream": False}, timeout=300.0)
    if response.status_code >= 400:
        raise LyricsProviderError(f"Ollama returned {response.status_code}: {response.text[-1000:]}")
    text = str(response.json().get("response") or "").strip()
    if not text:
        raise LyricsProviderError("Ollama returned an empty lyric")
    return text


def _openai_compatible(system: str, user: str) -> str:
    url = os.getenv("LYRICS_API_URL", "").strip()
    model = os.getenv("LYRICS_MODEL", "").strip()
    if not url or not model:
        raise LyricsProviderError("LYRICS_API_URL and LYRICS_MODEL are required")
    headers = {"Content-Type": "application/json"}
    token = os.getenv("LYRICS_API_KEY", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = httpx.post(url, headers=headers, json={"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": 0.85}, timeout=300.0)
    if response.status_code >= 400:
        raise LyricsProviderError(f"Lyrics API returned {response.status_code}: {response.text[-1000:]}")
    data = response.json()
    try:
        text = str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LyricsProviderError("Lyrics API returned an unexpected response") from exc
    if not text:
        raise LyricsProviderError("Lyrics API returned an empty lyric")
    return text


def _local(system: str, user: str) -> str:
    model = os.getenv("LYRICS_LOCAL_MODEL", "").strip()
    if not model:
        raise LyricsProviderError("LYRICS_LOCAL_MODEL is required")
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise LyricsProviderError("Transformers is not installed") from exc
    generator = pipeline("text-generation", model=model, device_map="auto")
    result = generator(f"{system}\n\n{user}\n\nLyrics:\n", max_new_tokens=900, do_sample=True, temperature=0.85, return_full_text=False)
    text = str(result[0].get("generated_text") or "").strip()
    if not text:
        raise LyricsProviderError("Local lyrics model returned an empty lyric")
    return text


def generate_lyrics(request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    status = provider_status()
    if not status["configured"]:
        raise LyricsProviderError("No lyrics provider is configured")
    system, user = build_prompt(request)
    provider = status["provider"]
    if provider == "ollama":
        text = _ollama(system, user)
    elif provider == "openai-compatible":
        text = _openai_compatible(system, user)
    elif provider == "local-transformers":
        text = _local(system, user)
    else:
        raise LyricsProviderError(f"Unsupported lyrics provider: {provider}")
    return text, {"provider": provider, "model": status.get("model"), "language": request.get("language"), "culture_profile_id": request.get("culture_profile_id"), "prompt": request.get("prompt"), "title": request.get("title")}
