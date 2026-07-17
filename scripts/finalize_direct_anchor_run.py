#!/usr/bin/env python3
"""Recover DirectAnchorAug CSV/metadata from a completed immutable run log."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROUND_RE = re.compile(r"^Round\s+(\d+)\s+\|\s+(.*)$")
METADATA_PREFIX = "[DirectAnchorAugMetadata] "


def parse_completed_log(path: Path, expected_rounds: int):
    rows = []
    metadata = None
    saved = False
    for line in path.read_text(errors="replace").splitlines():
        match = ROUND_RE.match(line)
        if match:
            row = {"round": int(match.group(1))}
            for piece in match.group(2).split(" | "):
                key, value = piece.split(": ", 1)
                row[key] = float(value)
            rows.append(row)
        elif line.startswith(METADATA_PREFIX):
            metadata = json.loads(line[len(METADATA_PREFIX):])
        elif line.startswith("Model saved to "):
            saved = True
    if not rows or max(row["round"] for row in rows) < expected_rounds:
        return None
    if metadata is None or not saved:
        return None
    return rows, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--rounds", type=int, default=100)
    args = parser.parse_args()
    candidates = sorted(args.run_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    parsed = next(
        ((path, result) for path in candidates if (result := parse_completed_log(path, args.rounds))),
        None,
    )
    if parsed is None:
        raise SystemExit(
            f"No completed {args.rounds}-round DirectAnchorAug log found in {args.run_dir}; "
            "refusing to mark it complete."
        )
    source, (rows, metadata) = parsed
    fields = sorted({key for row in rows for key in row})
    metrics_tmp = args.run_dir / "metrics.csv.tmp"
    with metrics_tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metrics_tmp.replace(args.run_dir / "metrics.csv")
    metadata_tmp = args.run_dir / "run_metadata.json.tmp"
    metadata_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    metadata_tmp.replace(args.run_dir / "run_metadata.json")
    print(f"[FINALIZED] {args.run_dir} from {source.name} ({len(rows)} round records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
