#!/usr/bin/env python3
"""Clone pinned upstream training repositories without downloading Git LFS weights."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPOSITORIES = {
    "seismicxm": (
        "https://github.com/cangyeone/seismicxm.git",
        "2d8077c62b6600e94d71a512b704b6fd6902f91d",
    ),
    "pnsn": (
        "https://github.com/cangyeone/pnsn_training_demo.git",
        "32d317c37aa8129c938a241083ceec7a77108386",
    ),
}


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Directory that will contain upstream repositories")
    parser.add_argument("--components", nargs="+", choices=sorted(REPOSITORIES), default=sorted(REPOSITORIES))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    clone_env = os.environ.copy()
    clone_env["GIT_LFS_SKIP_SMUDGE"] = "1"
    for name in args.components:
        url, revision = REPOSITORIES[name]
        target = args.root / name
        if not target.exists():
            run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(target)], env=clone_env)
            run(["git", "checkout", revision], cwd=target, env=clone_env)
        elif (target / ".git").is_dir():
            current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
            if current != revision:
                raise SystemExit(
                    f"{target} is at {current}, expected {revision}. Use a new directory or review/update it manually."
                )
        else:
            raise SystemExit(f"Refusing to overwrite non-git path: {target}")
        print(f"ready {name}: {target} @ {revision}")


if __name__ == "__main__":
    main()
