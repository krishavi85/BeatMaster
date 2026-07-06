from __future__ import annotations

import argparse
from pathlib import Path

from beatmaster_models.data import prepare_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate, normalize and segment a consented BeatMaster audio dataset")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--channels", type=int, default=1, choices=(1, 2))
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--overlap-seconds", type=float, default=1.0)
    args = parser.parse_args()
    report = prepare_dataset(
        args.manifest,
        args.output,
        sample_rate=args.sample_rate,
        channels=args.channels,
        segment_seconds=args.segment_seconds,
        overlap_seconds=args.overlap_seconds,
    )
    print(f"Prepared {report['segment_count']} segments in {args.output}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
