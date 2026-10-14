"""UE-child retention loader bound to the v19 contract.

``load_observer`` is a versioned copy of ``splitfusion_m3_route_b_v3.retention.load_observer`` reading the v19
contract: the v3 function compares the plan file with the v3 plan (id ``..._V3_P3000_STRIDE15``) and would refuse the
registered, unchanged v4 plan (``..._V4_P3000_STRIDE15``).  The observer class and the post-drain finalizer are the
unchanged v3 implementations (they read only RETENTION_DIR / RETENTION_RESULT_FILE, identical in v3 and v19).
"""
from pathlib import Path

from rl_agent.splitfusion_paired_retention_v2 import plan as RP
from rl_agent.splitfusion_m3_route_b_v3.retention import AttemptRetentionObserver, finalize_after_child  # noqa: F401

from . import contract as C


def load_observer(output, registration):
    """Bind the sealed registration's pre-launch plan file (digest-checked) to an enabled observer."""
    section = registration.get("paired_retention") or {}
    if section != C.retention_registration(section.get("plan_sha256")):
        raise ValueError("registration retention section differs from the contract")
    config = RP.RetentionConfigV2.from_plan_file(Path(output) / C.RETENTION_PLAN_FILE, section["plan_sha256"])
    if RP.canonical(config.plan) != RP.canonical(C.retention_plan(registration["transmitted_budget"])):
        raise ValueError("retention plan is not the registered rule for this budget")
    return AttemptRetentionObserver(config)
