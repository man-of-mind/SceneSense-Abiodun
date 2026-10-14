"""Candidate-only identities for the GT geometry-refusal containment (OFFLINE ONLY; not a launch identity).

Correction 01 (review_v1 blocker 1): the GT stack is the CURRENT V15/V16 one.  Everything not named below is the
V16 contract (``from ...v16.contract import *``: limits, records, accounting V3.v2 constants, ...); the GT registration
is V16's ``gc_registration`` (incl. ``storage_accounting``) with only the trampolines, implementations, seam and the
added ``geometry_containment`` section changed.  Layering, prep role only (score role: the V8 worker trampoline,
exactly as V16):

  seam (this package) -> service_entry (this package; code-identical to the V16/avo_tilt one, reads THIS contract)
    -> worker_entry (this package), prep:
         1. storage accounting V3.v2: ONE bounded AccountingSession + V16 AccountingTimer, capability refusal (= V16)
         2. tilt-aware AVO switch (= V8)
         3. FREEZE lifetime repair install (= V7)
         4. containment install (this package: PrepCore -> PrepCoreContained, engine adopted as ContainedEngine)
         5. V4 worker trampoline -> V3 worker (capacity, P3 pose, GC policy, tracing; unchanged)

No older module is edited or rebound; a live capture would need a new prospective revision registering these sources
(see HANDOFF_02.md).
"""
from pathlib import Path

from rl_agent.splitfusion_m3_route_b_v16 import contract as V16
from rl_agent.splitfusion_m3_route_b_v16.contract import *  # noqa: F401,F403

from . import classify as CL

PACKAGE = "rl_agent/splitfusion_gt_geometry_containment_v1"
MODULE = "rl_agent.splitfusion_gt_geometry_containment_v1"
ID = "M3_GT_GEOMETRY_REFUSAL_CONTAINMENT_V1_OFFLINE_ONLY"
SERVICE_ENTRY = MODULE + ".service_entry"
WORKER_ENTRY = MODULE + ".worker_entry"
GC_SEAM = MODULE + ".gc_seams"
BASE_WORKER_ENTRY = V16.WORKER_ENTRY            # the composed stack this candidate reproduces, plus containment
CONTAINMENT_BINDING_RECORD = "geometry_containment_binding.json"
DIAGNOSTICS_DIR = "geometry_refusal_v1"


def containment_registration():
    return dict(version=CL.VERSION, statuses=sorted(CL.REASONS), reasons=dict(CL.REASONS),
                transport="existing BIND refusal: prep Refused('DEFERRAL_REFUSED_' + reason) -> service "
                          "outcome 'NO_GT_DEFERRAL_REFUSED_' + reason -> edge GROUND_TRUTH_UNAVAILABLE",
                plan_validation="identical to a normal frame's bind (owner bind_plan checks, identity/anchor, "
                                "payload-independent admission context and live-context/source-camera pose)",
                engine="DeferredProducerEngine adopted as ContainedEngine (prep worker only)",
                prep_core="PrepCoreLifetime subclass PrepCoreContained (prep worker only)",
                base_stack=BASE_WORKER_ENTRY,
                diagnostics=f"{TRACE_DIR}/{DIAGNOSTICS_DIR}",
                binding_record=f"{TRACE_DIR}/{CONTAINMENT_BINDING_RECORD}")


def gc_registration(arm, budget, producer="v2"):
    value = V16.gc_registration(arm, budget, producer)
    value["trampolines"] = dict(service=SERVICE_ENTRY, worker=WORKER_ENTRY)
    value["implementations"] = dict(value["implementations"], service=SERVICE_ENTRY, base_worker=BASE_WORKER_ENTRY)
    value["geometry_containment"] = containment_registration()
    value["seam"] = GC_SEAM
    return value


def own_sources(root):
    root = Path(root).resolve(strict=True)
    return {str(p.relative_to(root)): sha(p) for p in sorted((root / PACKAGE).glob("*.py"))
            if not p.name.startswith("test_")}


def exact_sources(root):
    result = dict(V16.exact_sources(root))
    result.update(own_sources(root))
    return dict(sorted(result.items()))
