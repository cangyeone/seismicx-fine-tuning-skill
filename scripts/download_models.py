#!/usr/bin/env python3
"""Download or copy known SeismicXM/PNSN checkpoints and verify their bytes."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SEISMICXM_REVISION = "2d8077c62b6600e94d71a512b704b6fd6902f91d"
MODELS = {
    "pnsn-v3": {
        "filename": "pnsn.v3.pt",
        "bundled": SKILL_ROOT / "assets/models/pnsn.v3.pt",
        "sha256": "9f626e5fff4e9390c88e43c2f6230802496163b5d6eefee05e1b6ac7ccebf9e8",
        "size": 1844958,
    },
    "seismicxm-middle": {
        "filename": "seismicxm.middle.pt",
        "url": f"https://github.com/cangyeone/seismicxm/raw/{SEISMICXM_REVISION}/ckpt/seismicxm.middle.pt",
        "sha256": "3f051c950ef26110f922c30a3c7d11ed5a73b84bb98e99c2b7943a2a0908b174",
        "size": 207709060,
    },
    "seismicxm-classification": {
        "filename": "seismicxm.middle.classification.pt",
        "url": f"https://github.com/cangyeone/seismicxm/raw/{SEISMICXM_REVISION}/ckpt/seismicxm.middle.classification.pt",
        "sha256": "d91b746a727932c701f09e7e18714f4828eac9aa5e0b285cc585eaa6dd90fddf",
        "size": 207709060,
    },
    "seismicxm-tiny": {
        "filename": "seismicxm.tinny.pt",
        "url": f"https://github.com/cangyeone/seismicxm/raw/{SEISMICXM_REVISION}/ckpt/seismicxm.tinny.pt",
        "sha256": "5139a755a1db98c7483d0e95bedbeb5e2185bb5a8ebd770ab3d3fca6b669c9f9",
        "size": 34377834,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(url: str, output: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "seismicx-fine-tuning-skill/1"})
    with urllib.request.urlopen(request) as response, output.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List model keys without downloading")
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.list or not args.model:
        for name, item in MODELS.items():
            print(f"{name:31} {item['size'] / 1024**2:8.1f} MiB  {item['filename']}")
        if not args.model:
            return

    item = MODELS[args.model]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / str(item["filename"])
    if output.exists() and not args.force:
        if item.get("sha256") and sha256(output) != item["sha256"]:
            raise SystemExit(f"Existing file failed checksum: {output}; pass --force after reviewing it")
        print(f"already present: {output}")
        return
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    if "bundled" in item:
        source = Path(item["bundled"])
        if not source.is_file():
            raise SystemExit(f"Bundled checkpoint is missing: {source}")
        shutil.copy2(source, temporary)
    else:
        download_url(str(item["url"]), temporary)
    actual_size = temporary.stat().st_size
    expected_size = int(item["size"])
    if actual_size != expected_size:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Size mismatch for {args.model}: expected {expected_size}, got {actual_size}")
    digest = sha256(temporary)
    if item.get("sha256") and digest != item["sha256"]:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Checksum mismatch for {args.model}")
    temporary.replace(output)
    print(f"saved={output} bytes={actual_size} sha256={digest}")


if __name__ == "__main__":
    main()
