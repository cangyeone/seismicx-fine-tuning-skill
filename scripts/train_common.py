#!/usr/bin/env python3
"""Small training helpers shared by the SeismicXM and PNSN entrypoints."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_state_dict(value: Any) -> dict[str, torch.Tensor]:
    if isinstance(value, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = value.get(key)
            if isinstance(nested, dict):
                value = nested
                break
    if not isinstance(value, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    result = {}
    for key, tensor in value.items():
        clean = key.removeprefix("module.")
        result[clean] = tensor
    return result


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    return unwrap_state_dict(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def classification_metrics(targets: list[int], predictions: list[int], class_count: int) -> dict[str, Any]:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    precision_values = []
    recall_values = []
    f1_values = []
    per_class = []
    for class_id in range(class_count):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum() - tp)
        fn = int(matrix[class_id, :].sum() - tp)
        support = int(matrix[class_id, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        denominator = 2 * tp + fp + fn
        f1 = (2 * tp / denominator) if denominator else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_class.append({"precision": precision, "recall": recall, "f1": f1, "support": support})
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def phase_metrics(
    true_positions: list[list[int]],
    predicted_positions: list[list[int]],
    tolerance: int,
) -> dict[str, Any]:
    names = ("Pg", "Sg", "Pn", "Sn")
    output: dict[str, Any] = {}
    for phase_index, name in enumerate(names):
        errors = []
        matched = 0
        available = 0
        for truth, prediction in zip(true_positions, predicted_positions):
            if truth[phase_index] < 0:
                continue
            available += 1
            error = abs(prediction[phase_index] - truth[phase_index])
            errors.append(error)
            matched += int(error <= tolerance)
        output[name] = {
            "labeled": available,
            "recall_at_tolerance": matched / available if available else None,
            "mae_samples": float(np.mean(errors)) if errors else None,
        }
    return output


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
