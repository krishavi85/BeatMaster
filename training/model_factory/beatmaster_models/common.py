import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .models import ModelConfig


def load_yaml(path: Path):
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    return value


def load_model_config(path: Path):
    raw = load_yaml(path)
    return ModelConfig.from_dict(raw.get("model", {})), raw


def resolve_device(value="auto"):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def make_optimizer(model, settings):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.get("learning_rate", 3e-4)),
        betas=(float(settings.get("beta1", 0.9)), float(settings.get("beta2", 0.95))),
        weight_decay=float(settings.get("weight_decay", 0.01)),
    )


def make_scheduler(optimizer, total_steps, warmup_steps):
    def scale(step):
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
