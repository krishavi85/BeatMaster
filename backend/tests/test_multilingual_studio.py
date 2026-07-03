from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from scipy.io.wavfile import write as write_wav

from app.culture_profiles import enhance_prompt, get_profile, list_profiles
from app.daw_export_v2 import reaper_project
from app.harmony_engine import detect_chords, transcribe_midi
from app.model_registry import registry_status, select_generation_model
from app.runtime_capabilities import inspect_capabilities
from app.server import application


def test_culture_profiles_are_explicit_and_prompt_is_enhanced():
    profiles = list_profiles()
    assert len(profiles) >= 9
    assert len({item["id"] for item in profiles}) == len(profiles)
    profile = get_profile("suriname-kaseko")
    assert profile is not None
    assert profile["region"] == "Suriname"
    prompt = enhance_prompt("A joyful homecoming song", "suriname-kaseko", "Sranan Tongo")
    assert "Surinamese Kaseko" in prompt
    assert "Sranan Tongo" in prompt
    assert "avoid parody" in prompt


def test_model_registry_distinguishes_prompt_conditioning_and_fine_tuning(monkeypatch):
    monkeypatch.setenv("MUSICGEN_MODEL", "example/base-model")
    monkeypatch.delenv("CULTURE_MODEL_MAP", raising=False)
    base = select_generation_model("caribbean-soca")
    assert base.model_id == "example/base-model"
    assert base.fine_tuned_for_profile is False
    monkeypatch.setenv("CULTURE_MODEL_MAP", '{"caribbean-soca":"example/soca-model"}')
    mapped = select_generation_model("caribbean-soca")
    assert mapped.model_id == "example/soca-model"
    assert mapped.fine_tuned_for_profile is True
    assert registry_status()["culture_model_count"] == 1


def _write_test_tone(path: Path, frequencies: list[float], seconds: float = 2.0, sample_rate: int = 22050):
    time = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
    signal = sum(np.sin(2 * np.pi * frequency * time) for frequency in frequencies)
    signal = signal / max(float(np.max(np.abs(signal))), 1e-9) * 0.6
    write_wav(path, sample_rate, (signal * 32767).astype(np.int16))


def test_chord_detection_and_midi_create_real_assets(tmp_path: Path):
    chord_audio = tmp_path / "c_major.wav"
    _write_test_tone(chord_audio, [261.63, 329.63, 392.00])
    chords = detect_chords(chord_audio)
    assert chords["events"]
    assert any(event["chord"].startswith("C") for event in chords["events"])
    melody_audio = tmp_path / "melody.wav"
    _write_test_tone(melody_audio, [440.0], seconds=1.5)
    midi_path = tmp_path / "melody.mid"
    result = transcribe_midi(melody_audio, midi_path, tempo_bpm=120.0)
    assert midi_path.exists()
    assert midi_path.stat().st_size > 20
    assert result["note_count"] >= 1


def test_reaper_project_contains_aligned_tracks_and_time_signature():
    text = reaper_project(
        "Test Project",
        [{"label": "Lead Vocals", "archive_name": "lead.wav", "duration_seconds": 12.5}],
        108.0,
        "6/8",
    )
    assert "TEMPO 108.000000 6 8" in text
    assert 'FILE "Media/lead.wav"' in text
    assert 'NAME "Lead Vocals"' in text


def test_api_reports_real_capabilities_and_profiles():
    client = TestClient(application)
    health = client.get("/health")
    assert health.status_code == 200
    capabilities = client.get("/api/capabilities")
    assert capabilities.status_code == 200
    data = capabilities.json()
    for key in ("chord_detection", "midi_transcription", "daw_export", "lyrics_provider_configured", "singing_provider_configured", "culture_profile_count"):
        assert key in data
    profiles = client.get("/api/culture-profiles")
    assert profiles.status_code == 200
    assert len(profiles.json()["profiles"]) >= 9
    direct = inspect_capabilities()
    assert direct["daw_export"] is True
