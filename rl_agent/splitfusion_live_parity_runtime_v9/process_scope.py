"""Prospective E3 classification of exactly the genuinely closed worker pair."""
from contextlib import contextmanager
import json
from pathlib import Path
from rl_agent.splitfusion_live_gt_optimization_v7 import process_scope as V7
from rl_agent.splitfusion_live_gt_placement_trial_v1 import contract as T
from rl_agent.splitfusion_live_gt_handoff_v1 import contract as H
from .bindings import namespace, specialize
from . import contract as C
from .placement_evidence import verify_trial_scope
from .worker_observer import FACTORY

_bound = namespace(V7, T=T, H=H)
specialize(V7.verify_registered_children, _bound, [
    ("require(canonical(dict(schema=\"splitfusion.gt_scoring_before_archive.v1\",\n        sources=expected_bootstrap_sources)) == expected_binding, \"bootstrap source binding differs\")",
     "require(expected_bootstrap_sources == T.sources() and expected_binding == H.binding(), \"independent bootstrap/wire source bindings differ\")")])
verify_registered_children = _bound["verify_registered_children"]
classify_inventory = V7.classify_inventory


@contextmanager
def registered_inventory_scope(output):
    from rl_agent.splitfusion_single_gpu_profiler_v1 import threadcap_v1 as TC
    from .host_capture import verify_prospective
    output = Path(output)
    original = TC.ProcessEnvironmentInventory.summary
    def summary(owner):
        retained = original(owner)
        record = verify_prospective(output)
        registration = json.loads((output / "registration.json").read_bytes())
        cells = list((output / "capture/cells").iterdir())
        if len(cells) != 1:
            raise ValueError("one complete trial cell required for E3 classification")
        cell = cells[0]
        scope = json.loads((cell / "GT_WORKER_SCOPE.json").read_bytes())
        timeline = json.loads((cell / "CPU_GT_PROCESS_TIMELINE.json").read_bytes())
        placement = verify_trial_scope(cell, mode=record["trial"]["mode"])
        children = verify_registered_children(registration, scope, timeline, owner._rows,
            baseline=owner.baseline, expected_worker_sources=record["worker_sources"],
            expected_bootstrap_sources=record["bootstrap_sources"], expected_binding=record["handoff_binding"],
            expected_factory=FACTORY, expected_bootstrap_module=T.BOOTSTRAP)
        classified = classify_inventory(owner._rows, baseline=owner.baseline,
                                         children=children, inventory_errors=owner.errors)
        return dict(retained, scope_ok=classified["observed_other_processes_baseline"],
            scope_violations=classified["scope_violations"],
            original_unclassified_scope=retained, registered_cpu_children=children,
            registered_placement=placement, registered_classification=classified,
            coverage=classified["coverage"], all_processes_verified=classified["all_processes_verified"],
            correction="PROSPECTIVE_EXACT_NATIVE_PID_PAIR_NOT_PROCESS_NAME_EXCEPTION")
    TC.ProcessEnvironmentInventory.summary = summary
    try:
        yield
    finally:
        TC.ProcessEnvironmentInventory.summary = original
