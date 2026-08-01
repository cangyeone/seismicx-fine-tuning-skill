#!/usr/bin/env python3
"""Fine-tune the compact PNSN Pg/Sg/Pn/Sn picker from a waveform manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from seismic_data import WaveformDataset
from train_common import (
    file_sha256,
    git_revision,
    load_state_dict,
    parameter_counts,
    phase_metrics,
    select_device,
    set_seed,
    write_json,
)


VALID_MODULES = {"encoder", "rnns", "decoder"}


def phase_loss(probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1).mean()


def trainable_modules(model: torch.nn.Module, value: str) -> list[str]:
    selected = {item.strip() for item in value.split(",") if item.strip()}
    if selected == {"all"}:
        selected = set(VALID_MODULES)
    unknown = selected - VALID_MODULES
    if unknown or not selected:
        raise ValueError(f"--trainable must select from {sorted(VALID_MODULES)} or all")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name in selected:
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad = True
    return sorted(selected)


def enter_train_mode(model: torch.nn.Module, selected: list[str]) -> None:
    model.eval()
    for name in selected:
        getattr(model, name).train()


@torch.no_grad()
def evaluate(model, loader, device, tolerance):
    model.eval()
    total_loss = 0.0
    examples = 0
    truth = []
    predictions = []
    for batch in loader:
        waveform = batch["waveform"].to(device)
        target = batch["target"].to(device)
        probability = model(waveform)
        loss = phase_loss(probability, target)
        total_loss += float(loss.cpu()) * waveform.shape[0]
        examples += waveform.shape[0]
        truth.extend(batch["phase_positions"].cpu().tolist())
        predictions.extend(probability[:, 1:, :].argmax(dim=2).cpu().tolist())
    return {
        "loss": total_loss / max(1, examples),
        "examples": examples,
        "phases": phase_metrics(truth, predictions, tolerance),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pnsn-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--waveform-h5", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--group-column", default="source_group")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--window-length", type=int, default=5120)
    parser.add_argument("--pre-pick-samples", type=int, default=1024)
    parser.add_argument("--augment-shift", type=int, default=128)
    parser.add_argument("--gaussian-sigma", type=float, default=20.0)
    parser.add_argument("--missing-channels", choices=["zero", "replicate", "error"], default="zero")
    parser.add_argument("--no-regional-ps-fallback", action="store_true")
    parser.add_argument("--trainable", default="decoder", help="encoder,rnns,decoder, a comma list, or all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pick-tolerance", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.window_length % 128:
        raise SystemExit("PNSN window length must be divisible by 128")
    set_seed(args.seed)
    device = select_device(args.device)
    common = dict(
        metadata_csv=args.metadata,
        task="picking",
        window_length=args.window_length,
        waveform_h5=args.waveform_h5,
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
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    sys.path.insert(0, str(args.pnsn_repo.resolve()))
    from models.BRNNPNSN import BRNN

    model = BRNN()
    model.load_state_dict(load_state_dict(args.checkpoint), strict=True)
    selected = trainable_modules(model, args.trainable)
    counts = parameter_counts(model)
    model.to(device)
    sample = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        output = model(sample["waveform"].to(device))
    print(
        json.dumps(
            {
                "device": str(device),
                "train_examples": len(train_data),
                "val_examples": len(val_data),
                "trainable_modules": selected,
                "parameters": counts,
                "output_shape": list(output.shape),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

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
        enter_train_mode(model, selected)
        running_loss = 0.0
        examples = 0
        for batch in train_loader:
            waveform = batch["waveform"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            probability = model(waveform)
            loss = phase_loss(probability, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * waveform.shape[0]
            examples += waveform.shape[0]
        validation = evaluate(model, val_loader, device, args.pick_tolerance)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, examples),
            "validation": validation,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            stale_epochs = 0
            torch.save(model.state_dict(), args.output_dir / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    torch.save(model.state_dict(), args.output_dir / "last.pt")
    write_json(
        args.output_dir / "run.json",
        {
            "task": "picking",
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_checkpoint_sha256": file_sha256(args.checkpoint),
            "metadata_sha256": file_sha256(args.metadata),
            "upstream_revision": git_revision(args.pnsn_repo.resolve()),
            "trainable_modules": selected,
            "parameters": counts,
            "best_validation_loss": best_loss,
            "frozen_modules_kept_in_eval": len(selected) != len(VALID_MODULES),
            "history": history,
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    )


if __name__ == "__main__":
    main()
