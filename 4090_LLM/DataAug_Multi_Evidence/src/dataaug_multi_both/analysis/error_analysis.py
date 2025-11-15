#!/usr/bin/env python
"""
Offline-friendly error analysis CLI.

This module is a lightweight shim to make the Makefile target
`analyze-errors` usable without requiring the full evaluation stack
or network access. It writes a CSV with placeholder error records for
the given trial ID and split, suitable for validating automation
pipelines and file plumbing.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate placeholder error analysis CSV.")
    parser.add_argument("--trial-id", required=True, help="Trial UUID")
    parser.add_argument("--split", default="test", help="Dataset split (e.g., train/val/test)")
    parser.add_argument("--output", required=True, help="Output CSV path")

    args = parser.parse_args(argv)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Minimal, schema-agnostic CSV with a few sensible columns
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "trial_id": args.trial_id,
            "split": args.split,
            "record_id": "placeholder_0001",
            "pred_start": 0,
            "pred_end": 0,
            "label_start": 0,
            "label_end": 0,
            "note": "placeholder row for offline validation",
            "generated_at": now,
        }
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial_id",
                "split",
                "record_id",
                "pred_start",
                "pred_end",
                "label_start",
                "label_end",
                "note",
                "generated_at",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(str(out_path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
