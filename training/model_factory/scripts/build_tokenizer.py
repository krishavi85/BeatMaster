from __future__ import annotations

import argparse
from pathlib import Path

from beatmaster_models.data import read_jsonl
from beatmaster_models.tokenizer import BeatMasterTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the BeatMaster multilingual vocabulary")
    parser.add_argument("index", type=Path, help="Prepared segments.jsonl or another JSONL corpus index")
    parser.add_argument("output", type=Path)
    parser.add_argument("--vocabulary-size", type=int, default=16000)
    parser.add_argument("--minimum-frequency", type=int, default=2)
    args = parser.parse_args()
    records = read_jsonl(args.index)
    texts: list[str] = []
    for record in records:
        caption = str(record.get("caption") or "").strip()
        if caption:
            texts.append(caption)
        lyrics_path = record.get("lyrics_path")
        if lyrics_path and Path(lyrics_path).exists():
            texts.append(Path(lyrics_path).read_text(encoding="utf-8"))
        text = str(record.get("text") or "").strip()
        if text:
            texts.append(text)
    if not texts:
        raise RuntimeError("No caption or lyric text was found")
    tokenizer = BeatMasterTokenizer.train(texts, args.vocabulary_size, args.minimum_frequency)
    tokenizer.save(args.output)
    print(f"Saved {len(tokenizer.vocabulary)} tokens to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
