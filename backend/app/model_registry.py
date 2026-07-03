from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationModelSelection:
    model_id: str
    source: str
    culture_profile_id: str | None
    fine_tuned_for_profile: bool


def culture_model_map() -> dict[str, str]:
    raw = os.getenv("CULTURE_MODEL_MAP", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("CULTURE_MODEL_MAP must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("CULTURE_MODEL_MAP must be a JSON object")
    return {str(key): str(value).strip() for key, value in parsed.items() if str(value).strip()}


def select_generation_model(culture_profile_id: str | None = None) -> GenerationModelSelection:
    mapping = culture_model_map()
    if culture_profile_id and culture_profile_id in mapping:
        return GenerationModelSelection(
            model_id=mapping[culture_profile_id],
            source="culture-model-map",
            culture_profile_id=culture_profile_id,
            fine_tuned_for_profile=True,
        )
    base_model = os.getenv("MUSICGEN_MODEL", "").strip()
    if not base_model:
        raise RuntimeError("MUSICGEN_MODEL is required")
    return GenerationModelSelection(
        model_id=base_model,
        source="base-model",
        culture_profile_id=culture_profile_id,
        fine_tuned_for_profile=False,
    )


def registry_status() -> dict[str, Any]:
    mapping = culture_model_map()
    return {
        "base_model": os.getenv("MUSICGEN_MODEL", "").strip() or None,
        "culture_models": mapping,
        "culture_model_count": len(mapping),
        "note": (
            "Mapped culture models are treated as fine-tuned only when the deployment administrator explicitly "
            "registers them. Built-in culture profiles otherwise provide transparent prompt conditioning, not a claim "
            "of a trained culturally specific model."
        ),
    }
