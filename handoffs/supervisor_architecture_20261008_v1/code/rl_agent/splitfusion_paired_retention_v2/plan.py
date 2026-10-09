"""Pre-launch sampling plan (v2): which policy-decision ordinals are retained, fixed before launch.

The sample is a pure function of the planned run length (planned transmitted frames, the registered
decision period in tensors and the target record count). It never reads latency, quality, action or
outcome. The plan document is written create-only before launch; its SHA-256 is the binding that the
observer manifest and the paired-replay admission must both carry.

Ordinal = 0-based count of POLICY_DECISION materializations observed on the single owned UE frame
worker, across all controller sessions of the route (decision_seq resets per session, so it is NOT the
ordinal; the ordinal is recorded beside session_uuid/decision_seq/tensor_seq/frame_id).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

SCHEMA = "splitfusion.paired_retention.sampling_plan.v2"
RULE = "CENTERED_STRIDE_ON_POLICY_DECISION_ORDINAL_V1"
INPUT_SHAPE = (1, 7, 448, 768)
INPUT_BYTES = 1 * 7 * 448 * 768 * 4            # 9,633,792 B per prepared FP32 tensor
MAX_WIRE_BYTES = 2 * 1024 * 1024               # per-decision SFD4 wire bound (mode-11 wires are ~50-120 KB)
HOST_BYTES_CEILING = 1_342_177_280             # 1.25 GiB hard ceiling on retained host references
STORAGE_BYTES_CEILING = 2 * 1024 ** 3          # 2 GiB hard ceiling on the retention archive
MAX_RECORDS_CEILING = 128
PERSISTENCE_MODES = ("POST_DRAIN", "BACKGROUND_SPILL")
INDEPENDENT_OF = ("latency", "deadline_outcome", "quality", "q_perc", "action", "payload_bytes",
                  "gt_availability", "scene_content")


class PlanError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sample_ordinals(planned_policy_decisions: int, target_records: int) -> tuple[int, ...]:
    """Evenly spaced, centred ordinals over [0, planned_policy_decisions)."""
    if (type(planned_policy_decisions) is not int or type(target_records) is not int
            or not 1 <= target_records <= MAX_RECORDS_CEILING
            or planned_policy_decisions < target_records):
        raise PlanError("planned decisions must be an int >= target_records in [1, %d]" % MAX_RECORDS_CEILING)
    stride = planned_policy_decisions // target_records
    offset = stride // 2
    ordinals = tuple(offset + i * stride for i in range(target_records))
    assert ordinals[-1] < planned_policy_decisions
    return ordinals


def build_plan(*, plan_id: str, planned_frames: int, decision_period_tensors: int = 2,
               target_records: int = 100, persistence: str = "POST_DRAIN",
               max_host_bytes: Optional[int] = None, max_storage_bytes: int = 1_342_177_280,
               min_free_disk_bytes: int = 4 * 1024 ** 3, spill_queue_records: int = 4,
               spill_lag_decisions: int = 2, require_cuda_transfer: bool = True) -> dict:
    """Build (not write) the plan document from run-length inputs only."""
    if not isinstance(plan_id, str) or not plan_id or len(plan_id) > 200:
        raise PlanError("explicit plan_id required")
    if type(planned_frames) is not int or not 1 <= planned_frames <= 100_000:
        raise PlanError("planned_frames must be a positive int")
    if type(decision_period_tensors) is not int or not 1 <= decision_period_tensors <= 16:
        raise PlanError("decision_period_tensors must be a small positive int")
    if persistence not in PERSISTENCE_MODES:
        raise PlanError("unknown persistence mode")
    planned_decisions = planned_frames // decision_period_tensors
    ordinals = sample_ordinals(planned_decisions, target_records)
    if max_host_bytes is None:
        if persistence == "POST_DRAIN":     # every sampled record stays referenced until drain
            max_host_bytes = HOST_BYTES_CEILING
        else:                               # bounded window: lag + writer queue + the one being written
            held = spill_queue_records + spill_lag_decisions + 1
            max_host_bytes = min(HOST_BYTES_CEILING, held * (INPUT_BYTES + MAX_WIRE_BYTES))
    for name, value, low, high in (
            ("max_host_bytes", max_host_bytes, INPUT_BYTES + 1, HOST_BYTES_CEILING),
            ("max_storage_bytes", max_storage_bytes, INPUT_BYTES + 1, STORAGE_BYTES_CEILING),
            ("min_free_disk_bytes", min_free_disk_bytes, 0, 1 << 50),
            ("spill_queue_records", spill_queue_records, 1, 32),
            ("spill_lag_decisions", spill_lag_decisions, 1, 16)):
        if type(value) is not int or not low <= value <= high:
            raise PlanError(f"{name} outside [{low}, {high}]")
    if type(require_cuda_transfer) is not bool:
        raise PlanError("require_cuda_transfer must be bool")
    return {
        "schema": SCHEMA, "plan_id": plan_id, "rule": RULE,
        "selection_inputs": {"planned_frames": planned_frames,
                             "decision_period_tensors": decision_period_tensors,
                             "target_records": target_records},
        "planned_policy_decisions": planned_decisions,
        "stride": planned_decisions // target_records,
        "offset": (planned_decisions // target_records) // 2,
        "ordinals": list(ordinals),
        "independent_of": list(INDEPENDENT_OF),
        "shorter_run": "PLANNED_ORDINALS_AT_OR_BEYOND_OBSERVED_COUNT_ARE_NOT_REACHED",
        "longer_run": "ORDINALS_BEYOND_PLANNED_DECISIONS_ARE_SKIPPED_BEYOND_PLAN_AND_COUNTED_NEVER_RESAMPLED",
        "bounds": {"max_records": target_records, "max_host_bytes": max_host_bytes,
                   "max_storage_bytes": max_storage_bytes, "min_free_disk_bytes": min_free_disk_bytes,
                   "input_bytes": INPUT_BYTES, "max_wire_bytes": MAX_WIRE_BYTES},
        "persistence": {"mode": persistence, "spill_queue_records": spill_queue_records,
                        "spill_lag_decisions": spill_lag_decisions},
        "require_cuda_transfer": require_cuda_transfer,
        "claim_scope": "QUALITY_DIAGNOSTIC__HOST_MEMORY_PRESSURE__NOT_AN_UNINSTRUMENTED_TIMING_RUN",
        "live_launch_authorized_by_this_file": False,
    }


def validate_plan(plan: dict) -> dict:
    """Refuse any plan whose ordinals/bounds are not exactly the rule applied to its own inputs."""
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA or plan.get("rule") != RULE:
        raise PlanError("foreign sampling plan")
    inputs = plan.get("selection_inputs", {})
    bounds, persistence = plan.get("bounds", {}), plan.get("persistence", {})
    try:
        rebuilt = build_plan(plan_id=plan["plan_id"], planned_frames=inputs["planned_frames"],
            decision_period_tensors=inputs["decision_period_tensors"],
            target_records=inputs["target_records"], persistence=persistence["mode"],
            max_host_bytes=bounds["max_host_bytes"], max_storage_bytes=bounds["max_storage_bytes"],
            min_free_disk_bytes=bounds["min_free_disk_bytes"],
            spill_queue_records=persistence["spill_queue_records"],
            spill_lag_decisions=persistence["spill_lag_decisions"],
            require_cuda_transfer=plan["require_cuda_transfer"])
    except (KeyError, TypeError) as exc:
        raise PlanError(f"incomplete sampling plan: {exc}") from exc
    if canonical(rebuilt) != canonical(plan):
        raise PlanError("sampling plan is not the registered rule applied to its own inputs")
    return plan


def write_plan(path: Path, plan: dict) -> str:
    """Create-only pre-launch write; returns the plan SHA-256 (digest of the exact file bytes)."""
    validate_plan(plan)
    path = Path(path)
    raw = canonical(plan)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_bytes(raw)


def load_plan(path: Path, expected_sha256: str) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise PlanError("unsafe plan path")
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise PlanError("plan file digest differs from the pre-launch binding")
    plan = json.loads(raw)
    if canonical(plan) != raw:
        raise PlanError("plan file is not canonical")
    return validate_plan(plan)


def plan_sha256(plan: dict) -> str:
    return sha256_bytes(canonical(validate_plan(plan)))


@dataclass(frozen=True)
class RetentionConfigV2:
    """Explicit opt-in. The default instance is DISABLED and yields a no-op observer."""
    enabled: bool = False
    plan: Optional[dict] = None
    plan_sha256: Optional[str] = None

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise PlanError("enabled must be bool")
        if not self.enabled:
            if self.plan is not None or self.plan_sha256 is not None:
                raise PlanError("a disabled config must not carry a plan")
            return
        if self.plan is None or plan_sha256(self.plan) != self.plan_sha256:
            raise PlanError("enabled retention requires a validated plan and its pre-launch digest")

    @classmethod
    def from_plan_file(cls, path: Path, expected_sha256: str) -> "RetentionConfigV2":
        return cls(enabled=True, plan=load_plan(path, expected_sha256), plan_sha256=expected_sha256)
