"""Candidate-only GT trampoline identities: the exact storage-accounting optimisation inside the real GT service, on
the V8 runtime (tilt-aware AVO + V7 FREEZE lifetime repair).

NOT a launch identity: there is no host, runner, child, edge, registration or startup proof here.  A live capture
would need a new prospective revision that registers these sources and substitutions.  Everything not named below is
the V8 contract (``from ...v8.contract import *``); the V8 package itself is imported, never edited or rebound.

  seam (this package) -> service_entry (this package; versioned copy of V8's, reads THIS contract)
    -> worker_entry (this package): prep installs ``switch`` (storage accounting), then the UNCHANGED V8 worker
       trampoline (tilt-aware AVO switch -> V7 FREEZE lifetime repair -> V4 trampoline -> V3 worker);
       score: the V8 trampoline unchanged (nothing installed here).
"""
from pathlib import Path

from rl_agent.splitfusion_m3_route_b_v8 import contract as V8
from rl_agent.splitfusion_m3_route_b_v8.contract import *  # noqa: F401,F403

from . import accounting as A, switch as SW

PACKAGE = "rl_agent/splitfusion_gt_storage_accounting_v1"
MODULE = "rl_agent.splitfusion_gt_storage_accounting_v1"
ID = "M3_1_GT_STORAGE_ACCOUNTING_CANDIDATE_V1_OFFLINE_ONLY"
SERVICE_ENTRY = MODULE + ".service_entry"
WORKER_ENTRY = MODULE + ".worker_entry"
GC_SEAM = MODULE + ".gc_seams"
V8_WORKER_ENTRY = V8.WORKER_ENTRY
ACCOUNTING_BINDING_RECORD = "storage_accounting_binding.json"
ACCOUNTING_SWITCH = MODULE + ".switch"
ACCOUNTING_SUBSTITUTION = dict(SW.SUBSTITUTION)
ACCOUNTING_VERSION = A.VERSION


def accounting_registration():
    return dict(version=ACCOUNTING_VERSION, switch=ACCOUNTING_SWITCH, substitution=dict(ACCOUNTING_SUBSTITUTION),
                roles=["prep"], value_contract="returns exactly the original payload_storage_bytes integer",
                cache=None, reservation_bytes_unchanged=True, binding_record=f"{TRACE_DIR}/{ACCOUNTING_BINDING_RECORD}")


def gc_registration(arm, budget, producer="v2"):
    value = V8.gc_registration(arm, budget, producer)
    value["trampolines"] = dict(service=SERVICE_ENTRY, worker=WORKER_ENTRY)
    value["implementations"] = dict(value["implementations"], service=SERVICE_ENTRY, v8_worker=V8_WORKER_ENTRY)
    value["storage_accounting"] = accounting_registration()
    value["seam"] = GC_SEAM
    return value


def own_sources(root):
    root = Path(root).resolve(strict=True)
    return {str(p.relative_to(root)): sha(p) for p in sorted((root / PACKAGE).glob("*.py"))
            if not p.name.startswith("test_")}


def exact_sources(root):
    result = dict(V8.exact_sources(root))
    result.update(own_sources(root))
    return dict(sorted(result.items()))
