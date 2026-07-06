from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from .models import ModelConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    config: ModelConfig,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    step: int = 0,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": config.to_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": int(step),
        "epoch": int(epoch),
        "metrics": metrics or {},
        "metadata": metadata or {},
        "torch_version": torch.__version__,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def export_bundle(
    output_dir: Path,
    *,
    model_type: str,
    checkpoint_path: Path,
    tokenizer_path: Path | None,
    dataset_report_path: Path | None,
    model_card: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_destination = output_dir / "model.pt"
    shutil.copy2(checkpoint_path, checkpoint_destination)
    files = {"model": checkpoint_destination.name}
    if tokenizer_path is not None:
        tokenizer_destination = output_dir / "tokenizer.json"
        shutil.copy2(tokenizer_path, tokenizer_destination)
        files["tokenizer"] = tokenizer_destination.name
    if dataset_report_path is not None:
        report_destination = output_dir / "dataset-report.json"
        shutil.copy2(dataset_report_path, report_destination)
        files["dataset_report"] = report_destination.name
    manifest = {
        "format": "BeatMaster Model Bundle",
        "format_version": 1,
        "model_type": model_type,
        "files": files,
        "model_card": model_card,
    }
    (output_dir / "bundle.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    checksums = {path.name: sha256_file(path) for path in output_dir.iterdir() if path.is_file()}
    (output_dir / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    return {"bundle": str(output_dir.resolve()), "checksums": checksums, "manifest": manifest}
