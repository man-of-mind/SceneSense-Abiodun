"""Candidate-only GT trampoline identities: run the tilt-aware AVO inside the real GT service, on the V7 runtime.

NOT a launch identity: there is no host, runner, child, edge, registration or startup proof here.  A live capture
would need a new prospective revision that registers these sources and substitutions (see REPORT).  Everything not
named below is the V7 contract (``from ...v7.contract import *``); the V7 package itself is imported, never edited
or rebound.

  seam (this package) -> service_entry (this package; versioned copy of V7's, reads THIS contract)
    -> worker_entry (this package): prep installs ``switch`` (tilt-aware AVO), then the V7 worker trampoline
       (FREEZE lifetime repair) -> V4 trampoline -> V3 worker; score: the V7 trampoline unchanged.
"""
from pathlib import Path

from rl_agent.splitfusion_m3_route_b_v7 import contract as V7
from rl_agent.splitfusion_m3_route_b_v7.contract import *  # noqa: F401,F403

from . import geometry as G

PACKAGE = "rl_agent/splitfusion_avo_tilt_v1"
MODULE = "rl_agent.splitfusion_avo_tilt_v1"
ID = "M3_1_AVO_TILT_CANDIDATE_V1_OFFLINE_ONLY"
SERVICE_ENTRY = MODULE + ".service_entry"
WORKER_ENTRY = MODULE + ".worker_entry"
GC_SEAM = MODULE + ".gc_seams"
V7_WORKER_ENTRY = V7.WORKER_ENTRY
AVO_BINDING_RECORD = "avo_tilt_binding.json"
SWITCH = MODULE + ".switch"
SUBSTITUTION = dict(target="rl_agent.splitfusion_numeric_gt_producer_v1.avo:numeric_avo",
                    replacement=MODULE + ".quality:numeric_avo", process="GT prep worker only")


def avo_registration():
    return dict(geometry_version=G.GEOMETRY_VERSION, switch=SWITCH, substitution=dict(SUBSTITUTION), roles=["prep"],
                upright_boundary_deg=G.UPRIGHT_TOLERANCE_DEG, ground_rule_envelope_m=G.core.GROUND_REJECT_MARGIN_M,
                binding_record=f"{TRACE_DIR}/{AVO_BINDING_RECORD}")


def gc_registration(arm, budget, producer="v2"):
    value = V7.gc_registration(arm, budget, producer)
    value["trampolines"] = dict(service=SERVICE_ENTRY, worker=WORKER_ENTRY)
    value["implementations"] = dict(value["implementations"], service=SERVICE_ENTRY, v7_worker=V7_WORKER_ENTRY)
    value["avo_geometry"] = avo_registration()
    value["seam"] = GC_SEAM
    return value


def own_sources(root):
    root = Path(root).resolve(strict=True)
    return {str(p.relative_to(root)): sha(p) for p in sorted((root / PACKAGE).glob("*.py"))
            if not p.name.startswith("test_")}


def exact_sources(root):
    result = dict(V7.exact_sources(root))
    result.update(own_sources(root))
    return dict(sorted(result.items()))
