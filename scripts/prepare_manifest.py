#!/usr/bin/env python3
"""Normalize metadata, add leakage-safe splits, and optionally map a label column."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from seismic_data import first_present, read_rows, split_for_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source CSV metadata")
    parser.add_argument("--output", required=True, type=Path, help="Normalized manifest CSV")
    parser.add_argument("--label-column", default="", help="Copy this source column to canonical 'label'")
    parser.add_argument("--labels", nargs="*", default=None, help="Keep only these label values")
    parser.add_argument("--group-column", default="event_id")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    rows = read_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")
    if args.label_column and args.label_column not in rows[0]:
        raise SystemExit(f"Missing label column {args.label_column!r}")
    allowed = set(args.labels) if args.labels else None
    output_rows = []
    for row in rows:
        if args.label_column:
            label = str(row.get(args.label_column, "")).strip()
            if not label or (allowed is not None and label not in allowed):
                continue
            row["label"] = label
        group = first_present(row, (args.group_column, "event_id", "source_id", "trace_name"))
        if not group:
            raise SystemExit("A row has no event/group/trace identifier; cannot prevent split leakage")
        row["source_group"] = group
        row["split"] = split_for_row(
            row,
            split_column=args.split_column,
            group_column=args.group_column,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        output_rows.append(row)
    if not output_rows:
        raise SystemExit("All rows were filtered out")

    fields = list(output_rows[0])
    for name in ("label", "source_group", "split"):
        if any(name in row for row in output_rows) and name not in fields:
            fields.append(name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    splits = Counter(row["split"] for row in output_rows)
    labels = Counter(row.get("label", "") for row in output_rows)
    print(f"wrote={args.output} rows={len(output_rows)} splits={dict(splits)}")
    if args.label_column:
        print(f"labels={dict(labels)}")


if __name__ == "__main__":
    main()
