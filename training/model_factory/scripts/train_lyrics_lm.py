import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from beatmaster_models.checkpoints import save_checkpoint
from beatmaster_models.common import load_model_config, make_optimizer, resolve_device
from beatmaster_models.data import read_jsonl
from beatmaster_models.losses import text_loss
from beatmaster_models.models import BeatMasterLyricsLM
from beatmaster_models.tokenizer import BeatMasterTokenizer


class TextDataset(Dataset):
    def __init__(self, index, tokenizer, maximum_length):
        self.items = []
        seen = set()
        for record in read_jsonl(index):
            path = record.get("lyrics_path")
            if record.get("split") != "train" or not path or path in seen or not Path(path).exists():
                continue
            seen.add(path)
            text = "[PROMPT] " + str(record.get("caption") or "song") + " [LYRICS] " + Path(path).read_text(encoding="utf-8")
            ids = tokenizer.encode(text, maximum_length=maximum_length)
            if len(ids) > 3:
                self.items.append(torch.tensor(ids, dtype=torch.long))
        if not self.items:
            raise RuntimeError("No training lyric files were found")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    output = torch.zeros((len(batch), max(item.numel() for item in batch)), dtype=torch.long)
    for index, item in enumerate(batch):
        output[index, : item.numel()] = item
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    model_config, config = load_model_config(args.config)
    settings, data = config["training"], config["data"]
    tokenizer = BeatMasterTokenizer.load(Path(data["tokenizer"]))
    model_config.text_vocab_size = len(tokenizer.vocabulary)
    device = resolve_device(str(settings.get("device", "auto")))
    loader = DataLoader(TextDataset(Path(data["index"]), tokenizer, model_config.text_max_length), batch_size=int(settings.get("batch_size", 4)), shuffle=True, collate_fn=collate)
    model = BeatMasterLyricsLM(model_config).to(device)
    optimizer = make_optimizer(model, settings)
    output = Path(settings.get("output_dir", "runs/lyrics_lm"))
    output.mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(int(settings.get("epochs", 30))):
        running = 0.0
        model.train()
        for tokens in loader:
            tokens = tokens.to(device)
            loss = text_loss(model(tokens), tokens, tokenizer.pad_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss)
            step += 1
        metrics = {"train_loss": running / len(loader)}
        save_checkpoint(output / "lyrics_lm_latest.pt", model=model, config=model_config, optimizer=optimizer, step=step, epoch=epoch, metrics=metrics, metadata={"model_type": "lyrics_lm", "source": "from_scratch"})
        print(epoch, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
