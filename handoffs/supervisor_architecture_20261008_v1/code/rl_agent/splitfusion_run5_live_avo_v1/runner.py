"""Explicit new Run-5 live evaluation using retained lifecycle/instrumentation.

The old Run-4 preflight is an infrastructure prerequisite only. Its actor is
not used to make Run-5 decisions. The new binding is independently checked,
recorded, replayed and included in the controller lineage.
"""
import hashlib
import json
from pathlib import Path


def lineage():
    from .audit_actor import load_registered_actor
    from .quality_adapter import live_metrics_contract
    actor = load_registered_actor()
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(Path(__file__).parent.glob('*.py'))}
    value = dict(actor_boundary_sha256=actor.boundary_sha256,
                 evaluator=live_metrics_contract(), source_files=files,
                 controller='RewardHoldControllerV2', engine='Run4DecisionEngineV2',
                 state='ORIGINAL_RUN5_22D_QUALITY_AWARE', fallback='mode11_q9800_a71')
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def install_parent():
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as R
    from .audit_actor import load_registered_actor
    from .quality_adapter import live_metrics_contract
    original = R.offline_preflight
    digest = lineage()
    R.controller_lineage_sha256 = lambda: digest

    def preflight(*args, **kwargs):
        config, cells, legacy = original(*args, **kwargs)
        actor = load_registered_actor()
        return config, cells, dict(
            schema='splitfusion.run5_live_avo.preflight.v1',
            legacy_infrastructure_preflight=legacy,
            actor_boundary_sha256=actor.boundary_sha256,
            controller_lineage_sha256=digest,
            state_dim=22, live_metric_contract=live_metrics_contract(),
            live_command_executed=False, cuda_initialized=False,
            note='Legacy Run4 schema/packet names identify reused infrastructure, not the selected actor.')

    R.offline_preflight = preflight


def main():
    install_parent()
    from rl_agent.splitfusion_single_gpu_profiler_v1 import runner_cpu_partition_v1 as R
    return R.main()


if __name__ == '__main__':
    raise SystemExit(main())
