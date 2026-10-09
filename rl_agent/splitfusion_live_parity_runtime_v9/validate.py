"""Retained source-bound regressions plus the matched-worker integration."""
from rl_agent.splitfusion_live_parity_runtime_v8 import validate as V8
from . import contract as C
from .bindings import namespace

GROUPS = dict(V8.GROUPS,
    integration_v9={"directory": "rl_agent/splitfusion_live_parity_runtime_v9"},
    placement_trial={"directory": "rl_agent/splitfusion_live_gt_placement_trial_v1"},
    lifecycle_and_process_scope={"names": ["rl_agent.splitfusion_live_gt_optimization_v7.test_lifecycle",
        "rl_agent.splitfusion_live_parity_runtime_v9.test_integration.CombinedTests"]})
main = namespace(V8, C=C, GROUPS=GROUPS, __file__=__file__)["main"]

if __name__ == "__main__":
    raise SystemExit(main())
