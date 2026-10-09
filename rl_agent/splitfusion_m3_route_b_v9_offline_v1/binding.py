"""Fail-closed binding of an offline assessment to one declared V9 prospective."""
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path

from rl_agent.splitfusion_m3_route_b_v9 import contract as V9
from rl_agent.splitfusion_m3_route_b_v9 import host as V9_HOST

SCHEMA = "splitfusion.m3_route_b_v9_offline_v1.binding.v1"
DECLARED = (V9.ID, V9.LAUNCHER_REVISION, V9.RUNTIME_REVISION)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Binding:
    registration_id: str
    launcher_revision: str
    runtime_revision: str
    registration_file_sha256: str
    registration_content_sha256: str
    PROGRESS_COVERAGE_MIN: float
    SCHEDULE: dict
    CONTROLLER_MODULE: str
    TRACE_DIR: str
    POSE: dict
    MAP_CLASS: dict
    MAP_JOURNAL_DIR: str
    source_identities: dict


def resolve(arm_dir):
    """Verify the prospective and prerequisite first, then derive expectations.

    The version map is deliberately singular.  A differently declared revision
    is rejected rather than interpreted with V9 or accepted as a compatible
    alias.  ``verify_prospective`` retains the complete inventory, file hash,
    registration canonical-hash and semantic contract checks.
    """
    arm = Path(arm_dir).resolve(strict=True)
    record = V9_HOST.verify_prospective(arm)
    V9_HOST.verify_prerequisite(arm, record)
    registration_path = arm / "registration.json"
    registration = json.loads(registration_path.read_bytes())
    declared = (registration.get("id"), registration.get("launcher_revision"),
                registration.get("runtime_revision"))
    if declared != DECLARED:
        raise ValueError("unsupported declared assessment revision: " + repr(declared))
    if (record.get("launcher_revision"), record.get("runtime_revision")) != DECLARED[1:]:
        raise ValueError("prospective/registration revision binding differs")

    channel = registration.get("channel") or {}
    route = registration.get("route") or {}
    progress = ((route.get("completion") or {}).get("progress_evidence") or {})
    evidence = registration.get("evidence_additions") or {}
    journal = evidence.get("map_journal") or {}
    gt_gc = registration.get("gt_gc") or {}
    values = {
        "schedule": channel.get("schedule"),
        "controller": channel.get("controller_module"),
        "coverage": progress.get("minimum_ordered_coverage"),
        "trace_dir": gt_gc.get("trace_dir"),
        "pose": (registration.get("pose_repair") or {}).get("rule"),
        "map_class": registration.get("map_class"),
        "map_journal_dir": journal.get("directory"),
    }
    registered_map_rule = {key: values["map_class"][key] for key in V9.MAP_CLASS}
    contract_schedule = V9.channel_registration(Path(record["worktree"]))["schedule"]
    if (values != {"schedule": contract_schedule, "controller": V9.CONTROLLER_MODULE,
                   "coverage": V9.PROGRESS_COVERAGE_MIN, "trace_dir": V9.TRACE_DIR,
                   "pose": V9.POSE, "map_class": V9.map_class_registration(),
                   "map_journal_dir": V9.MAP_JOURNAL_DIR}):
        raise ValueError("verified registration assessment fields differ from its declared V9 contract")
    source_identities = {
        "declared_contract": V9.__name__,
        "verifier": V9_HOST.__name__,
        "contract_file_sha256": sha(V9.__file__),
        "verifier_file_sha256": sha(V9_HOST.__file__),
        "prospective_file_sha256": sha(arm / "PROSPECTIVE.json"),
        "registration_file_sha256": sha(registration_path),
        "prerequisite_file_sha256": sha(arm / "PREREQUISITE.json"),
        "runtime_inventory_entries": len(record["runtime_sources"]),
        "exact_gt_source_entries": len(registration["exact_gt_sources"]),
    }
    return Binding(*declared, source_identities["registration_file_sha256"],
                   registration["registration_sha256"], float(values["coverage"]),
                   copy.deepcopy(values["schedule"]), values["controller"], values["trace_dir"],
                   copy.deepcopy(values["pose"]), copy.deepcopy(registered_map_rule),
                   values["map_journal_dir"], source_identities)


def document(binding):
    return dict(schema=SCHEMA, registration_id=binding.registration_id,
                launcher_revision=binding.launcher_revision, runtime_revision=binding.runtime_revision,
                registration_file_sha256=binding.registration_file_sha256,
                registration_content_sha256=binding.registration_content_sha256,
                expectations=dict(schedule=binding.SCHEDULE, controller_module=binding.CONTROLLER_MODULE,
                                  progress_coverage_min=binding.PROGRESS_COVERAGE_MIN,
                                  trace_dir=binding.TRACE_DIR, pose=binding.POSE,
                                  map_class=binding.MAP_CLASS, map_journal_dir=binding.MAP_JOURNAL_DIR),
                sources=binding.source_identities)
