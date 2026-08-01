#!/usr/bin/env python3
"""Shared manifest, waveform, windowing, and label utilities."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


PHASE_NAMES = ("background", "Pg", "Sg", "Pn", "Sn")
PHASE_ALIASES = {
    "Pg": ("pg_sample", "Pg_sample", "trace_Pg_arrival_sample"),
    "Sg": ("sg_sample", "Sg_sample", "trace_Sg_arrival_sample"),
    "Pn": ("pn_sample", "Pn_sample", "trace_Pn_arrival_sample"),
    "Sn": ("sn_sample", "Sn_sample", "trace_Sn_arrival_sample"),
}
P_ALIASES = ("p_sample", "P_sample", "trace_P_arrival_sample", "trace_p_arrival_sample")
S_ALIASES = ("s_sample", "S_sample", "trace_S_arrival_sample", "trace_s_arrival_sample")
TRACE_RE = re.compile(r"^(?P<group>[^$]+)\$(?P<idx>\d+)(?P<slices>.*)$")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def first_present(row: dict[str, str], names: Sequence[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def float_or_nan(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def stable_unit_interval(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assigned_split(value: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Split ratios must satisfy 0 < train < 1 and train + val < 1")
    number = stable_unit_interval(value, seed)
    if number < train_ratio:
        return "train"
    if number < train_ratio + val_ratio:
        return "val"
    return "test"


def split_for_row(
    row: dict[str, str],
    *,
    split_column: str,
    group_column: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> str:
    explicit = str(row.get(split_column, "")).strip().lower()
    aliases = {"validation": "val", "valid": "val", "dev": "val"}
    explicit = aliases.get(explicit, explicit)
    if explicit in {"train", "val", "test"}:
        return explicit
    group = first_present(row, (group_column, "event_id", "source_id", "trace_name"))
    if not group:
        raise ValueError("Cannot assign a leakage-safe split: no group/event/trace identifier")
    return assigned_split(group, seed, train_ratio, val_ratio)


def parse_trace_name(trace_name: str) -> tuple[str, int | None, int | None]:
    match = TRACE_RE.match(str(trace_name))
    if not match:
        return str(trace_name), None, None
    width = None
    dims = [part.strip() for part in match.group("slices").split(",") if part.strip()]
    if dims:
        width_match = re.search(r":(\d+)", dims[-1])
        if width_match:
            width = int(width_match.group(1))
    return match.group("group"), int(match.group("idx")), width


def collect_labels(rows: Iterable[dict[str, str]], label_column: str) -> list[str]:
    labels = sorted({str(row.get(label_column, "")).strip() for row in rows if row.get(label_column, "") != ""})
    if not labels:
        raise ValueError(f"No labels found in column {label_column!r}")
    return labels


class WaveformDataset(Dataset):
    """Read SeismicX bucketed HDF5 or per-row NPY/NPZ waveforms from a CSV manifest."""

    def __init__(
        self,
        metadata_csv: Path,
        *,
        split: str,
        task: str,
        window_length: int,
        waveform_h5: Path | None = None,
        label_column: str = "label",
        class_names: Sequence[str] | None = None,
        split_column: str = "split",
        group_column: str = "event_id",
        seed: int = 20260801,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        pre_pick_samples: int | None = None,
        augment_shift: int = 0,
        gaussian_sigma: float = 20.0,
        normalize: str = "std-max",
        missing_channels: str = "zero",
        regional_ps_fallback: bool = True,
        max_samples: int = 0,
    ) -> None:
        if task not in {"classification", "picking"}:
            raise ValueError("task must be classification or picking")
        if missing_channels not in {"zero", "replicate", "error"}:
            raise ValueError("missing_channels must be zero, replicate, or error")
        self.metadata_csv = metadata_csv.resolve()
        self.waveform_h5 = waveform_h5.resolve() if waveform_h5 else None
        self.task = task
        self.window_length = window_length
        self.label_column = label_column
        self.class_to_id = {name: index for index, name in enumerate(class_names or [])}
        self.pre_pick_samples = pre_pick_samples if pre_pick_samples is not None else window_length // 5
        self.augment_shift = augment_shift if split == "train" else 0
        self.gaussian_sigma = gaussian_sigma
        self.normalize = normalize
        self.missing_channels = missing_channels
        self.regional_ps_fallback = regional_ps_fallback
        self.seed = seed
        self.split = split
        self._h5_cache: OrderedDict[Path, h5py.File] = OrderedDict()

        rows = read_rows(self.metadata_csv)
        self.rows = [
            row
            for row in rows
            if split_for_row(
                row,
                split_column=split_column,
                group_column=group_column,
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
            )
            == split
        ]
        if max_samples > 0:
            stable_rows = sorted(
                self.rows,
                key=lambda row: stable_unit_interval(
                    first_present(row, (group_column, "event_id", "trace_name")), seed + 17
                ),
            )
            if task == "classification" and self.class_to_id:
                by_class = {
                    name: [row for row in stable_rows if str(row.get(label_column, "")).strip() == name]
                    for name in self.class_to_id
                }
                selected = []
                offset = 0
                while len(selected) < max_samples:
                    added = False
                    for name in self.class_to_id:
                        if offset < len(by_class[name]):
                            selected.append(by_class[name][offset])
                            added = True
                            if len(selected) == max_samples:
                                break
                    if not added:
                        break
                    offset += 1
                self.rows = selected
            else:
                self.rows = stable_rows[:max_samples]
        if not self.rows:
            raise ValueError(f"No {split!r} rows were selected from {metadata_csv}")

    def __len__(self) -> int:
        return len(self.rows)

    def _get_h5(self, path: Path) -> h5py.File:
        handle = self._h5_cache.get(path)
        if handle is not None:
            self._h5_cache.move_to_end(path)
            return handle
        handle = h5py.File(path, "r")
        self._h5_cache[path] = handle
        while len(self._h5_cache) > 4:
            _, old = self._h5_cache.popitem(last=False)
            old.close()
        return handle

    def _resolve_waveform_path(self, row: dict[str, str]) -> Path | None:
        raw = first_present(row, ("waveform_path", "hdf5_path", "h5_path"))
        if raw:
            path = Path(raw).expanduser()
            return path.resolve() if path.is_absolute() else (self.metadata_csv.parent / path).resolve()
        return self.waveform_h5

    def _read_hdf5(self, path: Path, row: dict[str, str]) -> np.ndarray:
        handle = self._get_h5(path)
        direct_key = first_present(row, ("hdf5_key", "waveform_key"))
        if direct_key:
            dataset = handle[direct_key]
            return np.asarray(dataset[()], dtype=np.float32)

        trace_name = first_present(row, ("trace_name", "trace_id"))
        if not trace_name:
            raise ValueError("HDF5 rows require trace_name or hdf5_key")
        group, index, width = parse_trace_name(trace_name)
        root = handle["data"] if "data" in handle else handle
        if group not in root:
            raise KeyError(f"Waveform group {group!r} is absent from {path}")
        dataset = root[group]
        array = dataset[()] if index is None else dataset[index]
        array = np.asarray(array, dtype=np.float32)
        if width and array.ndim >= 1:
            array = array[..., :width]
        return array

    def _read_waveform(self, row: dict[str, str]) -> np.ndarray:
        path = self._resolve_waveform_path(row)
        if path is None:
            raise ValueError("No waveform_h5 argument or waveform_path column was provided")
        suffix = path.suffix.lower()
        if suffix in {".h5", ".hdf5"}:
            array = self._read_hdf5(path, row)
        elif suffix == ".npy":
            array = np.load(path, mmap_mode="r")
        elif suffix == ".npz":
            archive = np.load(path)
            key = first_present(row, ("waveform_key",), "waveform")
            if key not in archive:
                key = next(iter(archive.files))
            array = archive[key]
        else:
            raise ValueError(f"Unsupported waveform format: {path}")
        return self._channels_first(np.asarray(array, dtype=np.float32))

    def _channels_first(self, array: np.ndarray) -> np.ndarray:
        array = np.squeeze(array)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError(f"Expected a 1D/2D waveform, received shape {array.shape}")
        if array.shape[0] > 8 and array.shape[1] <= 8:
            array = array.T
        channels, samples = array.shape
        if samples < channels:
            array = array.T
            channels, samples = array.shape
        if channels == 3:
            return array
        if channels > 3:
            return array[:3]
        if self.missing_channels == "error":
            raise ValueError(f"Expected three components, received {channels}")
        if self.missing_channels == "replicate" and channels == 1:
            return np.repeat(array, 3, axis=0)
        output = np.zeros((3, samples), dtype=np.float32)
        output[:channels] = array
        return output

    def _phase_positions(self, row: dict[str, str]) -> dict[str, float]:
        positions = {
            phase: float_or_nan(first_present(row, aliases))
            for phase, aliases in PHASE_ALIASES.items()
        }
        if self.regional_ps_fallback:
            p_value = float_or_nan(first_present(row, P_ALIASES))
            s_value = float_or_nan(first_present(row, S_ALIASES))
            if not math.isfinite(positions["Pg"]) and math.isfinite(p_value):
                positions["Pg"] = p_value
            if not math.isfinite(positions["Sg"]) and math.isfinite(s_value):
                positions["Sg"] = s_value
        return positions

    def _window(self, array: np.ndarray, positions: dict[str, float], index: int) -> tuple[np.ndarray, dict[str, int]]:
        _, width = array.shape
        length = self.window_length
        finite = [int(value) for value in positions.values() if math.isfinite(value) and 0 <= value < width]
        if width <= length:
            start = 0
        elif finite:
            start = min(finite) - self.pre_pick_samples
            if self.augment_shift:
                rng = np.random.default_rng(self.seed + index)
                start += int(rng.integers(-self.augment_shift, self.augment_shift + 1))
            start = int(np.clip(start, 0, width - length))
        else:
            start = (width - length) // 2
        segment = np.zeros((3, length), dtype=np.float32)
        valid = min(length, max(0, width - start))
        segment[:, :valid] = array[:, start : start + valid]
        relative = {
            phase: (int(value) - start if math.isfinite(value) and 0 <= int(value) - start < length else -1)
            for phase, value in positions.items()
        }
        return self._normalize(segment, valid), relative

    def _normalize(self, waveform: np.ndarray, valid: int) -> np.ndarray:
        if self.normalize == "none":
            return waveform
        output = np.zeros_like(waveform)
        if valid <= 0:
            return output
        data = waveform[:, :valid]
        data = data - data.mean(axis=1, keepdims=True)
        if self.normalize == "std-max":
            data = data / (data.std(axis=1, keepdims=True) + 1e-6)
        data = data / (np.max(np.abs(data), axis=1, keepdims=True) + 1e-6)
        output[:, :valid] = data
        return output

    def _phase_target(self, positions: dict[str, int]) -> np.ndarray:
        target = np.zeros((5, self.window_length), dtype=np.float32)
        samples = np.arange(self.window_length, dtype=np.float32)
        for channel, phase in enumerate(PHASE_NAMES[1:], start=1):
            position = positions[phase]
            if position >= 0:
                target[channel] = np.exp(-0.5 * ((samples - position) / self.gaussian_sigma) ** 2)
        target[0] = np.clip(1.0 - target[1:].sum(axis=0), 0.0, 1.0)
        target /= np.maximum(target.sum(axis=0, keepdims=True), 1e-7)
        return target

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        waveform = self._read_waveform(row)
        positions = self._phase_positions(row)
        waveform, relative = self._window(waveform, positions, index)
        item: dict[str, object] = {
            "waveform": torch.from_numpy(waveform),
            "trace_name": first_present(row, ("trace_name", "trace_id"), str(index)),
        }
        if self.task == "classification":
            label = str(row.get(self.label_column, "")).strip()
            if label not in self.class_to_id:
                raise ValueError(f"Unknown label {label!r}; known labels: {list(self.class_to_id)}")
            item["target"] = torch.tensor(self.class_to_id[label], dtype=torch.long)
        else:
            item["target"] = torch.from_numpy(self._phase_target(relative))
            item["phase_positions"] = torch.tensor(
                [relative[phase] for phase in PHASE_NAMES[1:]], dtype=torch.long
            )
        return item

    def close(self) -> None:
        for handle in self._h5_cache.values():
            handle.close()
        self._h5_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
