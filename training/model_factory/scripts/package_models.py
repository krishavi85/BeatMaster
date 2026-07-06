import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint(path, expected_type):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in payload or "config" not in payload:
        raise RuntimeError(f"Invalid checkpoint: {path}")
    actual = (payload.get("metadata") or {}).get("model_type")
    if actual and actual != expected_type:
        raise RuntimeError(f"Expected {expected_type}, found {actual}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--music", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--name", default="BeatMaster Music v1")
    parser.add_argument("--license", required=True)
    args = parser.parse_args()
    codec = read_checkpoint(args.codec, "audio_codec")
    music = read_checkpoint(args.music, "music_lm")
    for field in ("codec_codebooks", "codec_bins", "codec_latent", "sample_rate"):
        if codec["config"].get(field) != music["config"].get(field):
            raise RuntimeError(f"Checkpoint mismatch: {field}")
    codec_dir = args.output / "codec"
    music_dir = args.output / "music"
    codec_dir.mkdir(parents=True, exist_ok=True)
    music_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.codec, codec_dir / "model.pt")
    shutil.copy2(args.music, music_dir / "model.pt")
    shutil.copy2(args.tokenizer, music_dir / "tokenizer.json")
    manifest = {
        "format": "BeatMaster Runtime Model Family",
        "version": 1,
        "name": args.name,
        "license": args.license,
        "training_source": "from_scratch",
        "pretrained_weights_bundled": False,
        "sample_rate": codec["config"]["sample_rate"],
        "files": {"codec": "codec/model.pt", "music": "music/model.pt", "tokenizer": "music/tokenizer.json"},
    }
    (args.output / "model-family.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files = [path for path in args.output.rglob("*") if path.is_file()]
    checksums = {str(path.relative_to(args.output)): sha256(path) for path in files}
    (args.output / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    print(f"Packaged {args.name} in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
