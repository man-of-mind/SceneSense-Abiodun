#!/usr/bin/env python3
"""Shared recovery-v2 plumbing: paths, hashing, create-only writes, dataset view."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

ABIODUN = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
PKG = ABIODUN / "pole_lraspp_multimodal_fusion"
IMPL_V1 = PKG / "object_head_pilot_v1" / "fasterrcnn_radar_roi_v1"
IMPL_V2 = PKG / "object_head_pilot_v1" / "fasterrcnn_recovery_v2"
SRC_EXPERIMENT = ABIODUN / "experiments" / "route_b_fasterrcnn_radar_roi_v1" / "20260826_224720"
WARM_START = SRC_EXPERIMENT / "checkpoints" / "fasterrcnn_radar_roi_v1" / "epoch_012.pt"
WARM_START_SHA = "7d3e1b414a892713fe848cfc81266ae4c321109453f0b60ac93efe30d8a1ef13"
DATASET_DIR = SRC_EXPERIMENT / "dataset"
CLASSES = ("vehicle", "person")
EXPECTED_SPLITS = {"train": 6600, "val": 3588, "test": 0}

for _path in (str(IMPL_V2), str(IMPL_V1), str(PKG), str(PKG.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_warm_start() -> str:
    got = sha256(WARM_START)
    if got != WARM_START_SHA:
        raise SystemExit(f"warm-start SHA mismatch: {got} != {WARM_START_SHA}")
    return got


def write_json_create(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def save_checkpoint_create(path: Path, payload: object) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.link(temporary, path)          # create-only: fails if `path` already exists
    finally:
        temporary.unlink(missing_ok=True)


def load_split_rows() -> Dict[str, List[Dict[str, str]]]:
    from pole_lraspp_multimodal_fusion.common import read_manifest
    rows = read_manifest(DATASET_DIR / "manifest.csv")
    split = {key: [row for row in rows if row.get("split") == key] for key in EXPECTED_SPLITS}
    counts = {key: len(value) for key, value in split.items()}
    if counts != EXPECTED_SPLITS:
        raise SystemExit(f"unexpected split counts {counts}, expected {EXPECTED_SPLITS}")
    return split


def notify(title: str, message: str, sink: Path | None = None) -> None:
    import subprocess
    import time
    if sink is not None and not sink.exists():
        write_json_create(sink, {"title": title, "message": message, "time_unix": time.time()})
    for command in (["notify-send", title, message], ["wall", f"{title}: {message}"]):
        try:
            subprocess.run(command, check=False, timeout=10,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            pass
