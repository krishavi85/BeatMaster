import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from beatmaster_models.data import AudioSegmentDataset, TokenDataset
from beatmaster_models.losses import music_loss
from beatmaster_models.models import BeatMasterAudioCodec, BeatMasterMusicLM, ModelConfig


def audio_collate(batch):
    length = min(item["waveform"].shape[-1] for item in batch)
    return torch.stack([item["waveform"][..., :length] for item in batch])


def token_collate(batch):
    frames = min(item["codes"].shape[-1] for item in batch)
    codes = torch.stack([item["codes"][..., :frames] for item in batch])
    text_length = max(item["text_ids"].numel() for item in batch)
    text = torch.zeros((len(batch), text_length), dtype=torch.long)
    mask = torch.zeros((len(batch), text_length), dtype=torch.bool)
    for index, item in enumerate(batch):
        size = item["text_ids"].numel()
        text[index, :size] = item["text_ids"]
        mask[index, :size] = True
    return text, mask, codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--audio-index", type=Path, required=True)
    parser.add_argument("--music", type=Path)
    parser.add_argument("--token-index", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    codec_payload = torch.load(args.codec, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(codec_payload["config"])
    codec = BeatMasterAudioCodec(config).to(device)
    codec.load_state_dict(codec_payload["model"])
    codec.eval()
    audio_loader = DataLoader(AudioSegmentDataset(args.audio_index, "test", config.sample_rate, config.audio_channels), batch_size=2, collate_fn=audio_collate)
    l1_total = 0.0
    snr_total = 0.0
    code_usage = [set() for _ in range(config.codec_codebooks)]
    count = 0
    with torch.inference_mode():
        for waveform in audio_loader:
            waveform = waveform.to(device)
            output = codec(waveform)
            reconstruction = output["waveform"]
            l1_total += float(torch.nn.functional.l1_loss(reconstruction, waveform))
            noise = (waveform - reconstruction).square().mean().clamp_min(1e-9)
            signal = waveform.square().mean().clamp_min(1e-9)
            snr_total += float(10.0 * torch.log10(signal / noise))
            for book in range(config.codec_codebooks):
                code_usage[book].update(output["codes"][:, book].detach().cpu().reshape(-1).tolist())
            count += 1
            if count >= args.batches:
                break
    report = {
        "device": str(device),
        "codec": {
            "waveform_l1": l1_total / max(1, count),
            "snr_db": snr_total / max(1, count),
            "codebook_utilization": [len(values) / config.codec_bins for values in code_usage],
            "evaluated_batches": count,
        },
    }
    if args.music and args.token_index:
        music_payload = torch.load(args.music, map_location=device, weights_only=False)
        music_config = ModelConfig.from_dict(music_payload["config"])
        music = BeatMasterMusicLM(music_config).to(device)
        music.load_state_dict(music_payload["model"])
        music.eval()
        token_loader = DataLoader(TokenDataset(args.token_index, "test"), batch_size=2, collate_fn=token_collate)
        total = 0.0
        batches = 0
        with torch.inference_mode():
            for text, mask, codes in token_loader:
                text, mask, codes = text.to(device), mask.to(device), codes.to(device)
                total += float(music_loss(music(text, codes, mask), codes))
                batches += 1
                if batches >= args.batches:
                    break
        cross_entropy = total / max(1, batches)
        report["music_lm"] = {"cross_entropy": cross_entropy, "perplexity": math.exp(min(20.0, cross_entropy)), "evaluated_batches": batches}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
