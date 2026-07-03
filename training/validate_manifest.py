from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.culture_profiles import CULTURE_PROFILES  # noqa: E402


def validate(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for field in ("dataset_name", "version", "profile_id", "community_reviewers", "recordings"):
        if not data.get(field):
            errors.append(f"Missing required field: {field}")
    profile_id = data.get("profile_id")
    if profile_id and profile_id not in CULTURE_PROFILES:
        errors.append(f"Unknown culture profile: {profile_id}")
    reviewers = data.get("community_reviewers") or []
    if not isinstance(reviewers, list) or not reviewers:
        errors.append("At least one community reviewer is required")
    for index, reviewer in enumerate(reviewers):
        for field in ("name", "role", "community_affiliation"):
            if not reviewer.get(field):
                errors.append(f"Reviewer {index} is missing {field}")
    recordings = data.get("recordings") or []
    if not isinstance(recordings, list) or not recordings:
        errors.append("At least one recording is required")
    seen_ids: set[str] = set()
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for index, recording in enumerate(recordings):
        prefix = f"Recording {index}"
        recording_id = str(recording.get("id") or "").strip()
        if not recording_id:
            errors.append(f"{prefix} is missing id")
        elif recording_id in seen_ids:
            errors.append(f"Duplicate recording id: {recording_id}")
        seen_ids.add(recording_id)
        for field in ("path", "license", "region"):
            if not str(recording.get(field) or "").strip():
                errors.append(f"{prefix} is missing {field}")
        if recording.get("ml_training_consent") is not True:
            errors.append(f"{prefix} does not have explicit ML training consent")
        if recording.get("withdrawn") is not False:
            errors.append(f"{prefix} is withdrawn or withdrawal state is missing")
        if not recording.get("performers"):
            errors.append(f"{prefix} is missing performers")
        if not recording.get("languages"):
            errors.append(f"{prefix} is missing languages")
        split = recording.get("split")
        if split not in split_counts:
            errors.append(f"{prefix} has invalid or missing split")
        else:
            split_counts[split] += 1
    if len(recordings) >= 10:
        if split_counts["validation"] == 0:
            errors.append("A validation split is required for datasets with 10 or more recordings")
        if split_counts["test"] == 0:
            errors.append("A test split is required for datasets with 10 or more recordings")
    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python training/validate_manifest.py <manifest.json>")
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Manifest not found: {path}")
        return 2
    try:
        errors = validate(path)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Could not read manifest: {exc}")
        return 2
    if errors:
        print("Manifest validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Manifest validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
