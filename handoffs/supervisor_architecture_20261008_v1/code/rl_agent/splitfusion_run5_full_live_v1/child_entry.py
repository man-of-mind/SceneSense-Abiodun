"""Full Run-5 process-local wiring around the retained cap/partition child (versioned copy of
``splitfusion_run5_live_avo_v1.child_entry``; installs the full Run-5 policy seams and actor export)."""
import sys
from pathlib import Path

EDGE_MODULE = 'rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_cpu_partition_v1_run5_live_avo'


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1 as BASE
    from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1_threadcap as CAP
    from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1_cpu_partition_v1 as PART
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_child_nobuild_v2 as NB
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt
    from . import runtime_policy
    attempt = BASE.attempt_dir_from_argv(args)
    if attempt is None:
        raise ValueError('An owned attempt directory is required')
    handles = []
    gt = live_gt.install_ue_gt_seams(attempt_dir=Path(attempt))
    original = NB.install_run4_seams_nobuild

    def install(campaign, *, attempt_dir, **kwargs):
        seams = original(campaign, attempt_dir=attempt_dir, **kwargs)
        handles.append(runtime_policy.install_policy_seams(
            attempt_dir=Path(attempt_dir), export_dir=Path(__file__).with_name('actor_export')))  # full Run-5 export
        return seams

    NB.install_run4_seams_nobuild = install
    ordering = CAP.install_ordering_route

    def route(*args, **kwargs):
        # PART sets its legacy module before calling this hook. Change only
        # the entry point; serial ordering and the READY partition gate remain.
        CAP.ORDERING_EDGE_MODULE = '-m ' + EDGE_MODULE
        return ordering(*args, **kwargs)

    CAP.install_ordering_route = route
    original_argv = sys.argv
    try:
        # The retained partition entry takes no argv parameter and delegates
        # to the cap child, which reads sys.argv. Preserve the explicit API
        # instead of silently launching with unrelated parent/test arguments.
        sys.argv = [original_argv[0], *args]
        return PART.main()
    finally:
        sys.argv = original_argv
        for handle in reversed(handles):
            handle.close()
        if gt is not None and hasattr(gt, 'close'):
            gt.close()
        NB.install_run4_seams_nobuild = original
        CAP.install_ordering_route = ordering


if __name__ == '__main__':
    raise SystemExit(main())
