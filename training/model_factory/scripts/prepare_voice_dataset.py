import argparse
import json
from pathlib import Path

from beatmaster_models.data import prepare_dataset, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Prepare only recordings with explicit voice-model consent")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = []
    for recording in manifest.get("recordings", []):
        if not recording.get("alignment_path"):
            continue
        if recording.get("voice_model_consent") is not True:
            raise RuntimeError(f"Voice-model consent is missing for {recording.get('id')}")
        selected.append(recording)
    if not selected:
        raise RuntimeError("No aligned recordings with explicit voice-model consent were found")
    filtered = dict(manifest)
    filtered["recordings"] = selected
    filtered_manifest = args.output.parent / (args.output.name + "-voice-manifest.json")
    filtered_manifest.parent.mkdir(parents=True, exist_ok=True)
    filtered_manifest.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
    report = prepare_dataset(filtered_manifest, args.output)
    records = read_jsonl(args.output / "segments.jsonl")
    consented = []
    by_id = {str(item["id"]): item for item in selected}
    for record in records:
        source = by_id[record["recording_id"]]
        record["voice_model_consent"] = True
        record["voice_commercial_use_allowed"] = bool(source.get("voice_commercial_use_allowed", False))
        record["performers"] = source.get("performers", [])
        consented.append(record)
    write_jsonl(args.output / "segments.jsonl", consented)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
