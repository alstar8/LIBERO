#!/usr/bin/env python3
"""Download LIBERO demos and the LeRobot Pi0.5 dataset into this clone.

Official HDF5 suites go to datasets/{libero_spatial,libero_object,libero_goal,libero_10,libero_90}.
The LeRobot v3 dataset used by Pi0.5 training/eval goes to datasets/lerobot.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
HF_CACHE = ROOT / ".hf_cache"

HDF5_REPO = "yifengzhu-hf/LIBERO-datasets"
LEROBOT_REPO = "lerobot/libero"


def main() -> None:
    DATASETS.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))

    print(f"HDF5 demos -> {DATASETS}  ({HDF5_REPO})")
    snapshot_download(
        repo_id=HDF5_REPO,
        repo_type="dataset",
        local_dir=str(DATASETS),
        cache_dir=str(HF_CACHE),
    )

    lerobot_dir = DATASETS / "lerobot"
    print(f"LeRobot Pi0.5 dataset -> {lerobot_dir}  ({LEROBOT_REPO})")
    snapshot_download(
        repo_id=LEROBOT_REPO,
        repo_type="dataset",
        local_dir=str(lerobot_dir),
        cache_dir=str(HF_CACHE),
    )

    print("done")
    for name in (
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
        "libero_90",
        "lerobot",
    ):
        path = DATASETS / name
        n = sum(1 for _ in path.rglob("*") if _.is_file()) if path.exists() else 0
        print(f"  {name}: {n} files")


if __name__ == "__main__":
    main()
