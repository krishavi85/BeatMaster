from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

from .audio_probe import AudioProcessingError

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _chord_templates() -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    templates: list[np.ndarray] = []
    for root in range(12):
        for suffix, intervals in (("", (0, 4, 7)), ("m", (0, 3, 7))):
            vector = np.zeros(12, dtype=np.float32)
            for interval in intervals:
                vector[(root + interval) % 12] = 1.0
            vector /= max(float(np.linalg.norm(vector)), 1e-9)
            names.append(f"{PITCH_CLASSES[root]}{suffix}")
            templates.append(vector)
    return names, np.stack(templates)


CHORD_NAMES, CHORD_TEMPLATES = _chord_templates()


def _compress_chords(labels: list[str], starts: np.ndarray, duration: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        start = float(starts[index])
        end = float(starts[index + 1]) if index + 1 < len(starts) else float(duration)
        if end <= start:
            continue
        if events and events[-1]["chord"] == label:
            events[-1]["end_seconds"] = round(end, 3)
            continue
        events.append({"chord": label, "start_seconds": round(start, 3), "end_seconds": round(end, 3)})
    return events


def detect_chords(path: Path) -> dict[str, Any]:
    audio, sample_rate = librosa.load(path, sr=22050, mono=True)
    if audio.size < sample_rate:
        raise AudioProcessingError("Audio is too short for chord analysis")
    harmonic = librosa.effects.harmonic(audio)
    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sample_rate, hop_length=hop_length)
    tempo_value, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate, hop_length=hop_length)
    tempo = float(np.atleast_1d(tempo_value)[0])
    if beat_frames.size >= 2:
        synced = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        starts = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)
    else:
        frames_per_slice = max(1, int(round(sample_rate / hop_length * 0.5)))
        boundaries = np.arange(0, chroma.shape[1], frames_per_slice, dtype=int)
        synced = librosa.util.sync(chroma, boundaries, aggregate=np.median)
        starts = librosa.frames_to_time(boundaries, sr=sample_rate, hop_length=hop_length)
    labels: list[str] = []
    confidences: list[float] = []
    for frame in synced.T:
        norm = float(np.linalg.norm(frame))
        if norm < 0.08:
            labels.append("N")
            confidences.append(0.0)
            continue
        normalized = frame / norm
        scores = CHORD_TEMPLATES @ normalized
        best = int(np.argmax(scores))
        labels.append(CHORD_NAMES[best])
        confidences.append(round(float(scores[best]), 4))
    duration = float(librosa.get_duration(y=audio, sr=sample_rate))
    events = _compress_chords(labels, starts, duration)
    return {
        "tempo_bpm": round(tempo, 2),
        "duration_seconds": round(duration, 3),
        "events": events,
        "mean_confidence": round(float(np.mean(confidences)) if confidences else 0.0, 4),
        "method": "beat-synchronous chroma template matching",
    }


def write_chord_assets(result: dict[str, Any], output_dir: Path, base_name: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{base_name}_chords.json"
    text_path = output_dir / f"{base_name}_chords.txt"
    lab_path = output_dir / f"{base_name}_chords.lab"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"Tempo: {result['tempo_bpm']} BPM", "", "Time       Chord"]
    lab_lines: list[str] = []
    for event in result["events"]:
        start = float(event["start_seconds"])
        minutes = int(start // 60)
        seconds = start - minutes * 60
        lines.append(f"{minutes:02d}:{seconds:05.2f}   {event['chord']}")
        lab_lines.append(f"{event['start_seconds']}\t{event['end_seconds']}\t{event['chord']}")
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    lab_path.write_text("\n".join(lab_lines) + "\n", encoding="utf-8")
    return {"json": json_path, "text": text_path, "lab": lab_path}


def _midi_from_basic_pitch(source: Path, output: Path) -> dict[str, Any] | None:
    try:
        from basic_pitch.inference import predict
    except ImportError:
        return None
    try:
        _, midi_data, note_events = predict(str(source))
        midi_data.write(str(output))
        return {"method": "basic-pitch", "note_count": len(note_events)}
    except Exception:
        return None


def _segment_pitch(f0: np.ndarray, voiced: np.ndarray, times: np.ndarray) -> list[tuple[float, float, int]]:
    notes: list[tuple[float, float, int]] = []
    current_note: int | None = None
    start_time = 0.0
    last_time = 0.0
    for pitch, is_voiced, time_value in zip(f0, voiced, times):
        midi_note = int(round(float(librosa.hz_to_midi(pitch)))) if is_voiced and np.isfinite(pitch) else None
        if midi_note is not None:
            midi_note = max(0, min(127, midi_note))
        if midi_note != current_note:
            if current_note is not None and last_time - start_time >= 0.08:
                notes.append((start_time, max(last_time, start_time + 0.08), current_note))
            current_note = midi_note
            start_time = float(time_value)
        last_time = float(time_value)
    if current_note is not None and last_time - start_time >= 0.08:
        notes.append((start_time, max(last_time, start_time + 0.08), current_note))
    return notes


def _write_mido(notes: list[tuple[float, float, int]], output: Path, tempo_bpm: float) -> None:
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    midi.tracks.append(track)
    safe_tempo = tempo_bpm if math.isfinite(tempo_bpm) and tempo_bpm > 0 else 120.0
    track.append(MetaMessage("track_name", name="BeatMaster Transcription", time=0))
    track.append(MetaMessage("set_tempo", tempo=bpm2tempo(safe_tempo), time=0))
    events: list[tuple[int, Message]] = []
    seconds_per_beat = 60.0 / safe_tempo
    for start, end, note in notes:
        start_tick = int(round(start / seconds_per_beat * midi.ticks_per_beat))
        end_tick = max(start_tick + 1, int(round(end / seconds_per_beat * midi.ticks_per_beat)))
        events.append((start_tick, Message("note_on", note=note, velocity=88, time=0)))
        events.append((end_tick, Message("note_off", note=note, velocity=0, time=0)))
    events.sort(key=lambda item: (item[0], 0 if item[1].type == "note_off" else 1))
    previous_tick = 0
    for absolute_tick, message in events:
        message.time = max(0, absolute_tick - previous_tick)
        track.append(message)
        previous_tick = absolute_tick
    track.append(MetaMessage("end_of_track", time=0))
    midi.save(output)


def transcribe_midi(source: Path, output: Path, tempo_bpm: float | None = None) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    basic_pitch_result = _midi_from_basic_pitch(source, output)
    if basic_pitch_result:
        return basic_pitch_result
    audio, sample_rate = librosa.load(source, sr=22050, mono=True)
    hop_length = 256
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sample_rate, hop_length=hop_length)
    notes = _segment_pitch(f0, voiced_flag, times)
    if not notes:
        raise AudioProcessingError("No stable pitched notes were detected for MIDI transcription")
    if tempo_bpm is None:
        tempo_value, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
        tempo_bpm = float(np.atleast_1d(tempo_value)[0])
    _write_mido(notes, output, tempo_bpm)
    return {"method": "librosa-pyin-monophonic", "note_count": len(notes), "tempo_bpm": round(float(tempo_bpm), 2)}
