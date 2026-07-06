from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from beatmaster_models.checkpoints import save_checkpoint
from beatmaster_models.common import count_parameters, load_model_config, make_optimizer, make_scheduler, resolve_device
from beatmaster_models.data import AudioSegmentDataset
from beatmaster_models.losses import codec_loss
from beatmaster_models.models import BeatMasterAudioCodec


def collate(batch):
    lengths = [item["waveform"].shape[-1] for item in batch]
    target = min(lengths)
    waveform = torch.stack([item["waveform"][..., :target] for item in batch])
    return waveform


def evaluate(model, loader, device, maximum_batches=20):
    model.eval()
    totals = {"loss": 0.0, "waveform": 0.0, "spectral": 0.0, "commitment": 0.0}
    count = 0
    with torch.inference_mode():
        for waveform in loader:
            waveform = waveform.to(device)
            losses = codec_loss(model(waveform), waveform)
            for key in totals:
                totals[key] += float(losses[key])
            count += 1
            if count >= maximum_batches:
                break
    return {key: value / max(1, count) for key, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the BeatMaster neural audio codec from scratch")
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    model_config, config = load_model_config(args.config)
    training = config.get("training", {})
    data_config = config.get("data", {})
    output_dir = Path(training.get("output_dir", "runs/codec"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(training.get("device", "auto")))
    train_set = AudioSegmentDataset(Path(data_config["index"]), "train", model_config.sample_rate, model_config.audio_channels)
    validation_set = AudioSegmentDataset(Path(data_config["index"]), "validation", model_config.sample_rate, model_config.audio_channels)
    loader = DataLoader(train_set, batch_size=int(training.get("batch_size", 4)), shuffle=True, num_workers=int(training.get("workers", 2)), collate_fn=collate, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_set, batch_size=int(training.get("batch_size", 4)), shuffle=False, num_workers=int(training.get("workers", 2)), collate_fn=collate)
    model = BeatMasterAudioCodec(model_config).to(device)
    optimizer = make_optimizer(model, training)
    epochs = int(training.get("epochs", 100))
    accumulation = int(training.get("gradient_accumulation", 1))
    total_steps = max(1, epochs * len(loader) // accumulation)
    scheduler = make_scheduler(optimizer, total_steps, int(training.get("warmup_steps", 1000)))
    start_epoch = 0
    step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        if payload.get("optimizer"):
            optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scheduler"):
            scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        step = int(payload.get("step", 0))
    use_amp = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Training BeatMaster codec with {count_parameters(model):,} parameters on {device}")
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        started = time.time()
        for batch_index, waveform in enumerate(loader):
            waveform = waveform.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                losses = codec_loss(model(waveform), waveform)
                loss = losses["loss"] / accumulation
            scaler.scale(loss).backward()
            if (batch_index + 1) % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
            running += float(losses["loss"])
            if batch_index % int(training.get("log_every", 20)) == 0:
                print(f"epoch={epoch} batch={batch_index}/{len(loader)} loss={float(losses['loss']):.5f} lr={scheduler.get_last_lr()[0]:.3e}")
        validation = evaluate(model, validation_loader, device, int(training.get("validation_batches", 20)))
        metrics = {"train_loss": running / max(1, len(loader)), **{f"validation_{key}": value for key, value in validation.items()}, "epoch_seconds": time.time() - started}
        checkpoint = output_dir / f"codec_epoch_{epoch:04d}.pt"
        save_checkpoint(checkpoint, model=model, config=model_config, optimizer=optimizer, scheduler=scheduler, step=step, epoch=epoch, metrics=metrics, metadata={"model_type": "audio_codec", "source": "from_scratch"})
        save_checkpoint(output_dir / "codec_latest.pt", model=model, config=model_config, optimizer=optimizer, scheduler=scheduler, step=step, epoch=epoch, metrics=metrics, metadata={"model_type": "audio_codec", "source": "from_scratch"})
        print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
