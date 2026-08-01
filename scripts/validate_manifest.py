#!/usr/bin/env python3
"""Validate manifest schema, split isolation, labels, and optional waveform reads."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from seismic_data import (
    PHASE_ALIASES,
    P_ALIASES,
    S_ALIASES,
    WaveformDataset,
    collect_labels,
    first_present,
    float_or_nan,
    read_rows,
    split_for_row,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["classification", "picking"])
    parser.add_argument("--waveform-h5", type=Path, default=None)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--group-column", default="source_group")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--window-length", type=int, default=10240)
    parser.add_argument("--check-waveforms", type=int, default=0)
    args = parser.parse_args()

    rows = read_rows(args.metadata)
    if not rows:
        raise SystemExit("Manifest is empty")
    errors: list[str] = []
    warnings: list[str] = []
    split_counts: Counter[str] = Counter()
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    sample_rates: Counter[str] = Counter()
    for row in rows:
        try:
            split = split_for_row(
                row,
                split_column=args.split_column,
                group_column=args.group_column,
                seed=args.seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        split_counts[split] += 1
        group = first_present(row, (args.group_column, "event_id", "source_id", "trace_name"))
        groups_by_split[split].add(group)
        sample_rate = first_present(row, ("sampling_rate", "trace_sampling_rate_hz", "sample_rate"))
        if sample_rate:
            sample_rates[sample_rate] += 1

    names = sorted(groups_by_split)
    leakage = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                leakage.append({"splits": [left, right], "group_count": len(overlap)})
    if leakage:
        errors.append(f"Group leakage detected: {leakage}")

    report: dict[str, object] = {
        "rows": len(rows),
        "splits": dict(split_counts),
        "unique_groups": {name: len(values) for name, values in groups_by_split.items()},
        "sampling_rates": dict(sample_rates.most_common(10)),
    }
    class_names: list[str] = []
    if args.task == "classification":
        try:
            class_names = collect_labels(rows, args.label_column)
        except ValueError as exc:
            errors.append(str(exc))
        report["labels"] = dict(Counter(row.get(args.label_column, "") for row in rows))
        if len(class_names) < 2:
            errors.append("Classification requires at least two non-empty classes")
    else:
        coverage = Counter()
        for row in rows:
            for phase, aliases in PHASE_ALIASES.items():
                if math.isfinite(float_or_nan(first_present(row, aliases))):
                    coverage[phase] += 1
            if math.isfinite(float_or_nan(first_present(row, P_ALIASES))):
                coverage["P_fallback"] += 1
            if math.isfinite(float_or_nan(first_present(row, S_ALIASES))):
                coverage["S_fallback"] += 1
        report["phase_label_coverage"] = dict(coverage)
        if not coverage:
            errors.append("No phase sample columns were found")

    invalid_rates = []
    non_100_rates = []
    for rate in sample_rates:
        try:
            numeric_rate = float(rate)
        except ValueError:
            invalid_rates.append(rate)
            continue
        if not math.isfinite(numeric_rate) or numeric_rate <= 0:
            invalid_rates.append(rate)
        elif abs(numeric_rate - 100.0) > 1e-6:
            non_100_rates.append(rate)
    if invalid_rates:
        errors.append(f"Invalid sampling-rate values: {invalid_rates[:10]}")
    if non_100_rates:
        warnings.append("Non-100 Hz rows exist; resample before using the supplied pretrained checkpoints")

    if args.check_waveforms > 0 and not errors:
        checked = {}
        nonfinite_total = 0
        for split in split_counts:
            dataset = WaveformDataset(
                args.metadata,
                split=split,
                task=args.task,
                window_length=args.window_length,
                waveform_h5=args.waveform_h5,
                label_column=args.label_column,
                class_names=class_names,
                split_column=args.split_column,
                group_column=args.group_column,
                seed=args.seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                max_samples=args.check_waveforms,
            )
            shapes = []
            for index in range(min(len(dataset), args.check_waveforms)):
                waveform = dataset[index]["waveform"]
                shapes.append(list(waveform.shape))
                nonfinite_total += int((~waveform.isfinite()).sum().item())
            checked[split] = shapes
            dataset.close()
        report["checked_waveform_shapes"] = checked
        report["checked_nonfinite_values"] = nonfinite_total
        if nonfinite_total:
            errors.append(f"Checked waveforms contain {nonfinite_total} non-finite values")

    report["warnings"] = warnings
    report["errors"] = errors
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
