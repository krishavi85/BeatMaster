import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from beatmaster_models.checkpoints import save_checkpoint
from beatmaster_models.common import load_model_config, make_optimizer, make_scheduler, resolve_device
from beatmaster_models.data import TokenDataset
from beatmaster_models.losses import music_loss
from beatmaster_models.models import BeatMasterMusicLM
from beatmaster_models.tokenizer import BeatMasterTokenizer


def collate(batch):
    frames = min(item["codes"].shape[-1] for item in batch)
    codes = torch.stack([item["codes"][..., :frames] for item in batch])
    length = max(item["text_ids"].numel() for item in batch)
    text_ids = torch.zeros((len(batch), length), dtype=torch.long)
    text_mask = torch.zeros((len(batch), length), dtype=torch.bool)
    for index, item in enumerate(batch):
        size = item["text_ids"].numel()
        text_ids[index, :size] = item["text_ids"]
        text_mask[index, :size] = True
    return text_ids, text_mask, codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    model_config, config = load_model_config(args.config)
    settings = config["training"]
    data = config["data"]
    tokenizer = BeatMasterTokenizer.load(Path(data["tokenizer"]))
    model_config.text_vocab_size = len(tokenizer.vocabulary)
    device = resolve_device(str(settings.get("device", "auto")))
    dataset = TokenDataset(Path(data["index"]), "train")
    loader = DataLoader(dataset, batch_size=int(settings.get("batch_size", 2)), shuffle=True, collate_fn=collate)
    model = BeatMasterMusicLM(model_config).to(device)
    optimizer = make_optimizer(model, settings)
    epochs = int(settings.get("epochs", 50))
    scheduler = make_scheduler(optimizer, max(1, epochs * len(loader)), int(settings.get("warmup_steps", 1000)))
    output = Path(settings.get("output_dir", "runs/music_lm"))
    output.mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for text_ids, text_mask, codes in loader:
            text_ids, text_mask, codes = text_ids.to(device), text_mask.to(device), codes.to(device)
            loss = music_loss(model(text_ids, codes, text_mask), codes)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            running += float(loss)
        metrics = {"train_loss": running / max(1, len(loader))}
        save_checkpoint(output / "music_lm_latest.pt", model=model, config=model_config, optimizer=optimizer, scheduler=scheduler, step=step, epoch=epoch, metrics=metrics, metadata={"model_type": "music_lm", "source": "from_scratch"})
        print(epoch, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
