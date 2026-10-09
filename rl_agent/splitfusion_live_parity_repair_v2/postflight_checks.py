"""Additional exact-population checks for owned sources and full GT timing."""
import hashlib
import json
from pathlib import Path

from .gt_timing import SCHEMA

STAGES = frozenset(("avo_scoring_and_sidecar", "complete_source_build", "prepare_complete_target",
                    "publish_complete_target", "archive_auxiliary_sources"))


def verify_gt_timing(directory, *, expected_frames):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("GT timing archive missing/unsafe")
    if {p.name for p in directory.iterdir()} != {"MANIFEST.json", "stages.jsonl"}:
        raise ValueError("GT timing file population mismatch")
    if any(p.is_symlink() for p in directory.iterdir()):
        raise ValueError("GT timing archive contains symlink")
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    raw = (directory / "stages.jsonl").read_bytes()
    if (manifest.get("schema") != SCHEMA or manifest.get("dropped") != 0
            or manifest.get("sha256") != {"stages.jsonl": hashlib.sha256(raw).hexdigest()}
            or manifest.get("clock") != "CLOCK_MONOTONIC_RAW"
            or manifest.get("work_and_readiness_unchanged") is not True
            or manifest.get("flush_after_worker_drain") is not True):
        raise ValueError("GT timing archive failed integrity/coverage")
    rows = [json.loads(line) for line in raw.splitlines()]
    expected = {(frame, stage) for frame in expected_frames for stage in STAGES}
    seen = set()
    for row in rows:
        key = (row.get("frame_id"), row.get("stage"))
        if key in seen or key not in expected or row.get("completed") is not True:
            raise ValueError("foreign, duplicate, or failed GT stage")
        seen.add(key)
        if (any(type(row.get(k)) is not int or row[k] < 0 for k in
                ("start_raw_ns", "end_raw_ns", "wall_ns", "thread_cpu_ns"))
                or row["end_raw_ns"]-row["start_raw_ns"] != row["wall_ns"]):
            raise ValueError("invalid GT stage timing")
    if seen != expected or manifest.get("records") != len(rows) or len(rows) > manifest.get("bound", -1):
        raise ValueError("GT stage population incomplete")
    return {"passed": True, "frames": len(set(expected_frames)), "stages": len(rows),
            "latency_qualification_claim": False}
