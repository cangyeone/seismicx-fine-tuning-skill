#!/usr/bin/env python3
"""Download explicit Seismic-AI-Data files with a large-download guard."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


API = "https://www.modelscope.cn/api/v1/datasets/{dataset}/repo/tree"


def list_directory(dataset: str, root: str, revision: str) -> list[dict[str, object]]:
    query = urllib.parse.urlencode({"Revision": revision, "Root": root})
    with urllib.request.urlopen(f"{API.format(dataset=dataset)}?{query}") as response:
        payload = json.load(response)
    if payload.get("Code") != 200:
        raise RuntimeError(payload.get("Message", "ModelScope API error"))
    return payload["Data"]["Files"]


def resolve_file(dataset: str, path: str, revision: str) -> dict[str, object]:
    item_path = Path(path)
    root = "" if item_path.parent == Path(".") else item_path.parent.as_posix()
    for item in list_directory(dataset, root, revision):
        if item.get("Path") == path and item.get("Type") == "blob":
            return item
    raise FileNotFoundError(f"Dataset file not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cangyeone/Seismic-AI-Data")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--paths", nargs="+", required=True, help="Exact repository-relative paths")
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--large-threshold-gb", type=float, default=5.0)
    parser.add_argument("--allow-large", action="store_true")
    args = parser.parse_args()

    files = [resolve_file(args.dataset, path, args.revision) for path in args.paths]
    total = sum(int(item.get("Size", 0)) for item in files)
    for item in files:
        print(f"{item['Path']}: {int(item.get('Size', 0)) / 1024**3:.3f} GiB")
    print(f"total: {total / 1024**3:.3f} GiB")
    if total > args.large_threshold_gb * 1024**3 and not args.allow_large:
        raise SystemExit("Large download blocked. Confirm disk space and rerun with --allow-large.")
    if not shutil.which("modelscope"):
        raise SystemExit("ModelScope CLI is required: pip install modelscope")
    args.local_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "modelscope",
        "download",
        "--dataset",
        args.dataset,
        *args.paths,
        "--revision",
        args.revision,
        "--local_dir",
        str(args.local_dir),
    ]
    print("+", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
