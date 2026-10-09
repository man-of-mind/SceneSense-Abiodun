"""Bind the outcome-aware assessment to a verified v13 registration (no older revision's globals supply anything)."""
from __future__ import annotations

import json
from pathlib import Path


def resolve(root):
    from rl_agent.splitfusion_static_modes_v13 import contract as C
    from rl_agent.splitfusion_static_modes_v13 import host as H
    from rl_agent import ue_target_snr_cell_runtime_v1 as base
    root = Path(root)
    record = H.verify_prospective(root)
    H.verify_prerequisite(root, record)
    reg = json.loads((root / "registration.json").read_bytes())
    if (reg.get("id"), reg.get("launcher_revision"), reg.get("runtime_revision")) != (
            C.ID, C.LAUNCHER_REVISION, C.RUNTIME_REVISION):
        raise ValueError("assessment refuses a registration of another revision")
    config = json.loads((Path(record["worktree"]) / C.LIVE_CONFIG).read_bytes())
    _seq, prefix, frozen = base.prepare_sequence(campaign=config, profile_id=C.ORIGINAL_PROFILE)
    if frozen.get("trace_sha256") != reg["channel"]["schedule"]["trace_sha256"]:
        raise ValueError("retained generator no longer reproduces the registered FAVORABLE_STABLE trace")
    return dict(external_fixed_action=reg["external_fixed_action"], budget=int(reg["transmitted_budget"]),
                trace_prefix=[(int(s), float(t)) for s, t in prefix], arm=reg["static_arm"],
                binding=dict(registration_id=reg["id"], static_arm=reg["static_arm"],
                             registration_file_sha256=C.sha(root / "registration.json"),
                             registration_content_sha256=reg["registration_sha256"],
                             runtime_revision=reg["runtime_revision"]))
