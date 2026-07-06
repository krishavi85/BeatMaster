import argparse
from pathlib import Path

import torch

from beatmaster_models.data import load_audio, read_jsonl, write_jsonl
from beatmaster_models.models import BeatMasterAudioCodec, ModelConfig
from beatmaster_models.tokenizer import BeatMasterTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("codec_checkpoint", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    payload = torch.load(args.codec_checkpoint, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(payload["config"])
    codec = BeatMasterAudioCodec(config).to(device)
    codec.load_state_dict(payload["model"])
    codec.eval()
    tokenizer = BeatMasterTokenizer.load(args.tokenizer)
    records = read_jsonl(args.index)
    token_dir = args.output / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    output_records = []
    with torch.inference_mode():
        for number, record in enumerate(records, start=1):
            waveform = load_audio(Path(record["audio_path"]), config.sample_rate, config.audio_channels)
            codes, _ = codec.encode(torch.from_numpy(waveform).unsqueeze(0).to(device))
            text_ids = tokenizer.encode(record["caption"], maximum_length=config.text_max_length)
            token_path = token_dir / (record["id"] + ".pt")
            torch.save({"codes": codes[0].cpu(), "text_ids": torch.tensor(text_ids), "caption": record["caption"]}, token_path)
            output_records.append({**record, "token_path": str(token_path.resolve()), "codec_frames": int(codes.shape[-1])})
            if number % 50 == 0:
                print("Prepared", number, "of", len(records))
    write_jsonl(args.output / "tokens.jsonl", output_records)
    print("Saved", len(output_records), "records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
