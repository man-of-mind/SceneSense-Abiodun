"""All retained native/quality gates plus explicit main-thread placement proof."""
from rl_agent.splitfusion_live_parity_runtime_v8 import postflight_checks as V8
from . import contract as C
from .bindings import namespace, specialize
from .placement_evidence import original_scheduling_reference, verify_trial_scope

_bound = namespace(V8, C=C, original_scheduling_reference=original_scheduling_reference,
                   verify_trial_scope=verify_trial_scope)
specialize(V8.verify_worker_scope, _bound, [
    ("from rl_agent.splitfusion_live_gt_handoff_v1 import contract as H", "from .contract import WORKER as H"),
    ("schedule = V7._child_context(ready.get(\"process_context\"), pid)",
     "schedule = original_scheduling_reference(ready.get(\"process_context\"), pid)"),
    ("require(V7._child_context(closed.get(\"process_context\"), pid) == schedule, \"GT CLOSE scheduling changed\")",
     "require(original_scheduling_reference(closed.get(\"process_context\"), pid) == schedule, \"GT CLOSE original scheduling reference changed\")"),
    ("require(V7._child_context(record.get(\"process_context\"), pid) == schedule, \"GT request scheduling changed\")",
     "require(original_scheduling_reference(record.get(\"process_context\"), pid) == schedule, \"GT request original scheduling reference changed\")"),
    ("rl_agent.splitfusion_live_parity_runtime_v8.worker_observer.ObservedClient",
     "rl_agent.splitfusion_live_parity_runtime_v9.worker_observer.ObservedClient")])
specialize(V8.verify_handoff_scope, _bound, [
    ("from rl_agent.splitfusion_live_gt_handoff_v1 import scoring_packet as P, contract as H",
     "from rl_agent.splitfusion_live_gt_handoff_v1 import scoring_packet as P\n    from .contract import WORKER as H")])
specialize(V8.postflight, _bound, [
    ("operations = dict(\n        optimization=", "operations = dict(\n        placement=lambda:verify_trial_scope(cell, mode=record[\"trial\"][\"mode\"]),\n        optimization=")])
globals().update({name: value for name, value in _bound.items() if not name.startswith("__")})
