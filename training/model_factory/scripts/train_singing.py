import argparse
import json
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from beatmaster_models.checkpoints import save_checkpoint
from beatmaster_models.common import load_model_config, make_optimizer, resolve_device
from beatmaster_models.data import load_audio, read_jsonl
from beatmaster_models.losses import singing_loss
from beatmaster_models.models import BeatMasterSingingModel


class SingingDataset(Dataset):
    def __init__(self, index, config):
        self.config = config
        self.records = [record for record in read_jsonl(index) if record.get("split") == "train" and record.get("alignment_path")]
        if not self.records:
            raise RuntimeError("No training records contain alignment_path")
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=config.sample_rate, n_fft=1024, hop_length=320, win_length=1024, n_mels=config.n_mels, f_min=40.0, f_max=config.sample_rate / 2)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        alignment = json.loads(Path(record["alignment_path"]).read_text(encoding="utf-8"))
        lyric_ids = torch.tensor(alignment["lyric_frame_ids"], dtype=torch.long)
        pitches = torch.tensor(alignment["midi_pitch"], dtype=torch.long)
        frames = min(lyric_ids.numel(), pitches.numel())
        lyric_ids, pitches = lyric_ids[:frames], pitches[:frames]
        waveform = torch.from_numpy(load_audio(Path(record["audio_path"]), self.config.sample_rate, 1))
        target_samples = frames * 320
        waveform = waveform[..., :target_samples]
        if waveform.shape[-1] < target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[-1]))
        mel = torch.log(self.mel(waveform).clamp_min(1e-5))
        return lyric_ids, pitches, mel, waveform


def collate(batch):
    frames = min(item[0].numel() for item in batch)
    lyric_ids = torch.stack([item[0][:frames] for item in batch])
    pitches = torch.stack([item[1][:frames] for item in batch])
    mel = torch.stack([item[2][..., :frames].squeeze(0) for item in batch])
    samples = frames * 320
    waveform = torch.stack([item[3][..., :samples] for item in batch])
    return lyric_ids, pitches, mel, waveform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    model_config, config = load_model_config(args.config)
    settings, data = config["training"], config["data"]
    device = resolve_device(str(settings.get("device", "auto")))
    dataset = SingingDataset(Path(data["index"]), model_config)
    loader = DataLoader(dataset, batch_size=int(settings.get("batch_size", 2)), shuffle=True, collate_fn=collate)
    model = BeatMasterSingingModel(model_config).to(device)
    optimizer = make_optimizer(model, settings)
    output = Path(settings.get("output_dir", "runs/singing"))
    output.mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(int(settings.get("epochs", 100))):
        running = 0.0
        model.train()
        for lyric_ids, pitches, target_mel, target_waveform in loader:
            lyric_ids, pitches = lyric_ids.to(device), pitches.to(device)
            target_mel, target_waveform = target_mel.to(device), target_waveform.to(device)
            losses = singing_loss(model(lyric_ids, pitches), target_mel, target_waveform)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(losses["loss"])
            step += 1
        metrics = {"train_loss": running / len(loader)}
        save_checkpoint(output / "singing_latest.pt", model=model, config=model_config, optimizer=optimizer, step=step, epoch=epoch, metrics=metrics, metadata={"model_type": "singing", "source": "from_scratch", "requires_consented_voice_data": True})
        print(epoch, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
