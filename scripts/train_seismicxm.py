#!/usr/bin/env python3
"""Fine-tune SeismicXM for event classification or five-channel phase picking."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seismic_data import WaveformDataset, collect_labels, read_rows
from train_common import (
    classification_metrics,
    file_sha256,
    git_revision,
    load_state_dict,
    parameter_counts,
    phase_metrics,
    select_device,
    set_seed,
    write_json,
)


def phase_loss(probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1).mean()


def configure_trainable(model: nn.Module, task: str, strategy: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = strategy == "all"
    if strategy == "all":
        return
    head_name = "decoder_event_type" if task == "classification" else "decoder_phase"
    for name, parameter in model.named_parameters():
        if name.startswith(head_name):
            parameter.requires_grad = True
    if strategy == "head-last-block":
        layers = getattr(getattr(model, "transformer", None), "layers", None)
        if layers is None or len(layers) == 0:
            raise ValueError("This SeismicXM variant has no transformer.layers for head-last-block")
        for parameter in layers[-1].parameters():
            parameter.requires_grad = True


def enter_train_mode(model: nn.Module, task: str, strategy: str) -> None:
    if strategy == "all":
        model.train()
        return
    model.eval()
    head_name = "decoder_event_type" if task == "classification" else "decoder_phase"
    getattr(model, head_name).train()
    if strategy == "head-last-block":
        model.transformer.layers[-1].train()


def make_loader(dataset: WaveformDataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    device: torch.device,
    class_count: int,
    tolerance: int,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    targets: list[int] = []
    predictions: list[int] = []
    true_positions: list[list[int]] = []
    predicted_positions: list[list[int]] = []
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        waveforms = batch["waveform"].to(device)
        target = batch["target"].to(device)
        phase, _, event_type, _, _ = model(waveforms)
        if task == "classification":
            loss = criterion(event_type, target)
            predicted = event_type.argmax(dim=1)
            targets.extend(target.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
        else:
            loss = phase_loss(phase, target)
            truth = batch["phase_positions"].cpu().tolist()
            predicted = phase[:, 1:, :].argmax(dim=2).cpu().tolist()
            true_positions.extend(truth)
            predicted_positions.extend(predicted)
        batch_size = waveforms.shape[0]
        total_loss += float(loss.cpu()) * batch_size
        total_examples += batch_size
    result: dict[str, object] = {"loss": total_loss / max(1, total_examples), "examples": total_examples}
    if task == "classification":
        result.update(classification_metrics(targets, predictions, class_count))
    else:
        result["phases"] = phase_metrics(true_positions, predicted_positions, tolerance)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["classification", "picking"])
    parser.add_argument("--seismicxm-repo", required=True, type=Path)
    parser.add_argument("--variant", choices=["middle", "tinny", "rnn"], default="middle")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--waveform-h5", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--classes", nargs="*", default=None, help="Explicit classification label order")
    parser.add_argument("--keep-classification-head", action="store_true")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--group-column", default="source_group")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--window-length", type=int, default=10240)
    parser.add_argument("--pre-pick-samples", type=int, default=2048)
    parser.add_argument("--augment-shift", type=int, default=256)
    parser.add_argument("--gaussian-sigma", type=float, default=20.0)
    parser.add_argument("--missing-channels", choices=["zero", "replicate", "error"], default="zero")
    parser.add_argument("--no-regional-ps-fallback", action="store_true")
    parser.add_argument("--trainable", choices=["head", "head-last-block", "all"], default="head")
    parser.add_argument("--class-balance", choices=["none", "weighted"], default="weighted")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pick-tolerance", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.window_length % 64:
        raise SystemExit("SeismicXM window length must be divisible by 64")
    set_seed(args.seed)
    device = select_device(args.device)
    rows = read_rows(args.metadata)
    class_names = []
    if args.task == "classification":
        class_names = args.classes or collect_labels(rows, args.label_column)
        if len(class_names) < 2:
            raise SystemExit("At least two classes are required")
        if args.keep_classification_head and (len(class_names) != 8 or args.classes is None):
            raise SystemExit("Keeping the pretrained head requires an explicit eight-class --classes order")

    common = dict(
        metadata_csv=args.metadata,
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
        pre_pick_samples=args.pre_pick_samples,
        gaussian_sigma=args.gaussian_sigma,
        missing_channels=args.missing_channels,
        regional_ps_fallback=not args.no_regional_ps_fallback,
    )
    train_data = WaveformDataset(
        split="train", augment_shift=args.augment_shift, max_samples=args.max_train_samples, **common
    )
    val_data = WaveformDataset(split="val", max_samples=args.max_val_samples, **common)
    if args.task == "classification":
        train_label_counts = Counter(str(row.get(args.label_column, "")).strip() for row in train_data.rows)
        missing_train_classes = [name for name in class_names if train_label_counts[name] == 0]
        if missing_train_classes:
            raise SystemExit(f"Classes absent from the selected training split: {missing_train_classes}")
    train_loader = make_loader(train_data, args.batch_size, args.num_workers, True)
    val_loader = make_loader(val_data, args.batch_size, args.num_workers, False)

    sys.path.insert(0, str(args.seismicxm_repo.resolve()))
    module = __import__(f"seismicxm.{args.variant}", fromlist=["SeismicXM", "EQLargeCNN"])
    model_class = getattr(module, "SeismicXM", None) or getattr(module, "EQLargeCNN", None)
    if model_class is None:
        raise SystemExit(f"No supported SeismicXM model class in seismicxm.{args.variant}")
    model = model_class()
    model.load_state_dict(load_state_dict(args.checkpoint), strict=True)
    if args.task == "classification" and not args.keep_classification_head:
        model.decoder_event_type = nn.Linear(model.n_feature, len(class_names))
    configure_trainable(model, args.task, args.trainable)
    counts = parameter_counts(model)
    if counts["trainable"] == 0:
        raise SystemExit("No trainable parameters were selected")
    model.to(device)

    sample = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        outputs = model(sample["waveform"].to(device))
    print(
        json.dumps(
            {
                "device": str(device),
                "train_examples": len(train_data),
                "val_examples": len(val_data),
                "parameters": counts,
                "phase_shape": list(outputs[0].shape),
                "classification_shape": list(outputs[2].shape),
                "classes": class_names,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    class_weights = None
    if args.task == "classification" and args.class_balance == "weighted":
        class_weights = torch.tensor(
            [len(train_data) / (len(class_names) * train_label_counts[name]) for name in class_names],
            dtype=torch.float32,
            device=device,
        )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        enter_train_mode(model, args.task, args.trainable)
        running_loss = 0.0
        examples = 0
        for batch in train_loader:
            waveforms = batch["waveform"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            phase, _, event_type, _, _ = model(waveforms)
            loss = criterion(event_type, target) if args.task == "classification" else phase_loss(phase, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * waveforms.shape[0]
            examples += waveforms.shape[0]
        validation = evaluate(
            model, val_loader, args.task, device, len(class_names), args.pick_tolerance
        )
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, examples),
            "validation": validation,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if float(validation["loss"]) < best_loss:
            best_loss = float(validation["loss"])
            stale_epochs = 0
            torch.save(model.state_dict(), args.output_dir / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    torch.save(model.state_dict(), args.output_dir / "last.pt")
    summary = {
        "task": args.task,
        "variant": args.variant,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "metadata_sha256": file_sha256(args.metadata),
        "upstream_revision": git_revision(args.seismicxm_repo.resolve()),
        "classes": class_names,
        "parameters": counts,
        "best_validation_loss": best_loss,
        "frozen_modules_kept_in_eval": args.trainable != "all",
        "history": history,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(args.output_dir / "run.json", summary)


if __name__ == "__main__":
    main()
