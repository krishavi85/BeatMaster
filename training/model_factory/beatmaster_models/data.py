from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset


class DatasetValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SegmentRecord:
    id: str
    recording_id: str
    audio_path: str
    split: str
    caption: str
    profile_id: str
    start_seconds: float
    duration_seconds: float
    lyrics_path: str | None = None
    midi_path: str | None = None
    alignment_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"Invalid JSON on line {line_number} of {path}") from exc
        if not isinstance(value, dict):
            raise DatasetValidationError(f"Line {line_number} of {path} must be a JSON object")
        records.append(value)
    return records


def write_jsonl(path: Path, records: Iterator[dict[str, Any]] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_audio(path: Path, sample_rate: int, channels: int = 1) -> np.ndarray:
    waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.T
    if waveform.shape[0] > channels:
        waveform = waveform.mean(axis=0, keepdims=True) if channels == 1 else waveform[:channels]
    elif waveform.shape[0] < channels:
        waveform = np.repeat(waveform, channels, axis=0)
    if source_rate != sample_rate:
        divisor = math.gcd(source_rate, sample_rate)
        waveform = resample_poly(waveform, sample_rate // divisor, source_rate // divisor, axis=-1).astype(np.float32)
    return np.clip(waveform, -1.0, 1.0)


def audio_fingerprint(waveform: np.ndarray) -> str:
    mono = waveform.mean(axis=0)
    if mono.size == 0:
        return hashlib.sha256(b"").hexdigest()
    target = 16000
    if mono.size > target:
        indices = np.linspace(0, mono.size - 1, target).astype(np.int64)
        mono = mono[indices]
    normalized = mono / max(float(np.max(np.abs(mono))), 1e-8)
    quantized = np.round(normalized * 32767).astype("<i2")
    return hashlib.sha256(quantized.tobytes()).hexdigest()


def build_caption(recording: dict[str, Any], profile_id: str) -> str:
    parts: list[str] = []
    title = str(recording.get("title") or "").strip()
    if title:
        parts.append(f"Title: {title}.")
    subgenre = str(recording.get("subgenre") or profile_id.replace("-", " ")).strip()
    parts.append(f"Style: {subgenre}.")
    region = str(recording.get("region") or "").strip()
    if region:
        parts.append(f"Region: {region}.")
    languages = recording.get("languages") or []
    if languages:
        parts.append(f"Languages: {', '.join(str(value) for value in languages)}.")
    instruments = recording.get("instruments") or []
    if instruments:
        parts.append(f"Instruments: {', '.join(str(value) for value in instruments)}.")
    mood = str(recording.get("mood") or "").strip()
    if mood:
        parts.append(f"Mood: {mood}.")
    tempo = recording.get("tempo_bpm")
    if tempo is not None:
        parts.append(f"Tempo: {tempo} BPM.")
    meter = str(recording.get("time_signature") or "").strip()
    if meter:
        parts.append(f"Meter: {meter}.")
    description = str(recording.get("description") or "").strip()
    if description:
        parts.append(description)
    return " ".join(parts)


def validate_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    required = ["dataset_name", "version", "profile_id", "community_reviewers", "recordings"]
    missing = [field for field in required if not manifest.get(field)]
    if missing:
        raise DatasetValidationError(f"Manifest is missing: {', '.join(missing)}")
    if not isinstance(manifest["recordings"], list) or not manifest["recordings"]:
        raise DatasetValidationError("Manifest must contain at least one recording")
    seen_ids: set[str] = set()
    for index, recording in enumerate(manifest["recordings"]):
        recording_id = str(recording.get("id") or "").strip()
        if not recording_id:
            raise DatasetValidationError(f"Recording {index} has no id")
        if recording_id in seen_ids:
            raise DatasetValidationError(f"Duplicate recording id: {recording_id}")
        seen_ids.add(recording_id)
        if recording.get("ml_training_consent") is not True:
            raise DatasetValidationError(f"Recording {recording_id} lacks explicit ML training consent")
        if recording.get("withdrawn") is not False:
            raise DatasetValidationError(f"Recording {recording_id} is withdrawn or missing withdrawal status")
        if not str(recording.get("license") or "").strip():
            raise DatasetValidationError(f"Recording {recording_id} has no license")
        relative = Path(str(recording.get("path") or ""))
        source = relative if relative.is_absolute() else manifest_path.parent / relative
        if not source.exists():
            raise DatasetValidationError(f"Recording file does not exist: {source}")
        if recording.get("split") not in {"train", "validation", "test"}:
            raise DatasetValidationError(f"Recording {recording_id} has invalid split")


def prepare_dataset(
    manifest_path: Path,
    output_dir: Path,
    *,
    sample_rate: int = 32000,
    channels: int = 1,
    segment_seconds: float = 10.0,
    overlap_seconds: float = 1.0,
    minimum_rms: float = 0.002,
    maximum_clip_fraction: float = 0.02,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    segment_samples = int(round(segment_seconds * sample_rate))
    step_samples = int(round((segment_seconds - overlap_seconds) * sample_rate))
    if segment_samples <= 0 or step_samples <= 0:
        raise DatasetValidationError("Segment and overlap settings are invalid")
    records: list[SegmentRecord] = []
    fingerprints: dict[str, str] = {}
    rejected: list[dict[str, str]] = []
    profile_id = str(manifest["profile_id"])
    for recording in manifest["recordings"]:
        recording_id = str(recording["id"])
        source_path = Path(str(recording["path"]))
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        waveform = load_audio(source_path, sample_rate, channels)
        fingerprint = audio_fingerprint(waveform)
        if fingerprint in fingerprints:
            rejected.append({"recording_id": recording_id, "reason": f"duplicate of {fingerprints[fingerprint]}"})
            continue
        fingerprints[fingerprint] = recording_id
        total_samples = waveform.shape[-1]
        starts = list(range(0, max(1, total_samples - segment_samples + 1), step_samples))
        if not starts or starts[-1] + segment_samples < total_samples:
            starts.append(max(0, total_samples - segment_samples))
        for segment_index, start in enumerate(sorted(set(starts))):
            segment = waveform[:, start : start + segment_samples]
            if segment.shape[-1] < segment_samples:
                segment = np.pad(segment, ((0, 0), (0, segment_samples - segment.shape[-1])))
            rms = float(np.sqrt(np.mean(np.square(segment))))
            clip_fraction = float(np.mean(np.abs(segment) >= 0.999))
            if rms < minimum_rms:
                rejected.append({"recording_id": recording_id, "reason": f"segment {segment_index} is near-silent"})
                continue
            if clip_fraction > maximum_clip_fraction:
                rejected.append({"recording_id": recording_id, "reason": f"segment {segment_index} is clipped"})
                continue
            segment_id = f"{recording_id}_{segment_index:05d}"
            destination = audio_dir / f"{segment_id}.wav"
            sf.write(destination, segment.T, sample_rate, subtype="PCM_24")
            def resolve_optional(field: str) -> str | None:
                value = recording.get(field)
                if not value:
                    return None
                path = Path(str(value))
                return str(path if path.is_absolute() else (manifest_path.parent / path).resolve())
            records.append(SegmentRecord(
                id=segment_id,
                recording_id=recording_id,
                audio_path=str(destination.resolve()),
                split=str(recording["split"]),
                caption=build_caption(recording, profile_id),
                profile_id=profile_id,
                start_seconds=round(start / sample_rate, 6),
                duration_seconds=segment_seconds,
                lyrics_path=resolve_optional("lyrics_path"),
                midi_path=resolve_optional("midi_path"),
                alignment_path=resolve_optional("alignment_path"),
            ))
    if not records:
        raise DatasetValidationError("No usable segments were produced")
    write_jsonl(output_dir / "segments.jsonl", [record.to_dict() for record in records])
    report = {
        "dataset_name": manifest["dataset_name"],
        "version": manifest["version"],
        "profile_id": profile_id,
        "sample_rate": sample_rate,
        "channels": channels,
        "segment_seconds": segment_seconds,
        "overlap_seconds": overlap_seconds,
        "recording_count": len(manifest["recordings"]),
        "segment_count": len(records),
        "split_counts": {split: sum(record.split == split for record in records) for split in ("train", "validation", "test")},
        "rejected": rejected,
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    (output_dir / "preparation-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class AudioSegmentDataset(Dataset):
    def __init__(self, index_path: Path, split: str, sample_rate: int, channels: int = 1) -> None:
        self.records = [record for record in read_jsonl(index_path) if record["split"] == split]
        self.sample_rate = sample_rate
        self.channels = channels
        if not self.records:
            raise DatasetValidationError(f"No {split} records in {index_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        waveform = load_audio(Path(record["audio_path"]), self.sample_rate, self.channels)
        return {"waveform": torch.from_numpy(waveform), "caption": record["caption"], "record": record}


class TokenDataset(Dataset):
    def __init__(self, index_path: Path, split: str) -> None:
        self.records = [record for record in read_jsonl(index_path) if record["split"] == split]
        if not self.records:
            raise DatasetValidationError(f"No {split} token records in {index_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        payload = torch.load(record["token_path"], map_location="cpu", weights_only=True)
        return {"codes": payload["codes"].long(), "text_ids": payload["text_ids"].long(), "record": record}
