#!/usr/bin/env python3
"""UE child entry point for the thread-cap experiment (both arms).

The module name keeps ``splitfusion_single_gpu_profiler_v1.child_v1`` as a
substring on purpose: the pinned sampler classifies the UE process by that
substring of its command line, and the instrumentation must stay unchanged.

Runs the unchanged profiled child (``child_v1.main``) with the same argv.
Before it, using only the standard library:

1. verifies the parent handed this process the registered arm and that the
   arm's variables are present at ``exec`` (so numerical libraries read them at
   initialization);
2. installs the ``subprocess.Popen`` restore so every process this child
   launches gets the baseline values of the cap variables;
3. records the pre-import settings snapshot.

After the child returns (or raises), it records the post-run snapshot with the
effective PyTorch/OpenMP/MKL/OpenCV thread counts and the loaded pools.  Both
snapshots are written create-only under ``<attempt-dir>/threadcap/`` and are
identical in structure in both arms.  The child's return code is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from . import threadcap_v1 as TC


def install_bundle_seams(arm: str, attempt_dir: Path) -> None:
    """Arms whose registered bundle is ``optimized`` get the radar bundle around
    the profiler's UE seams: packing before them, builder/tracker after them.
    Production arms record a production snapshot at the same point."""
    from . import radar_bundle_v1 as RB
    from . import seams_ue_v1 as SU

    evidence = attempt_dir / "radar_bundle"
    original = SU.install_ue_seams
    bundle = TC.BUNDLES[arm]
    ordering = TC.ORDERINGS[arm]

    def install_ue_seams(rec: Any, *, attempt_dir: Path, policy_identity: Any) -> Any:
        record = {}
        if bundle == RB.OPTIMIZED:
            from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as live_base

            record = RB.install_packing_short_circuit(live_base)
        out = original(rec, attempt_dir=attempt_dir, policy_identity=policy_identity)
        if bundle == RB.OPTIMIZED:
            from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
            from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U

            RB.install_builder_and_tracker(pinned=pinned, phase6_ue=U, evidence_dir=evidence,
                                           record=record)
        else:
            RB.record_production(evidence)
        rec.note("radar_bundle", bundle=bundle)
        if arm in TC.ORDERING_ARMS:
            install_ordering_route(arm, ordering, rec)
        return out

    install_ue_seams.__wrapped__ = original  # type: ignore[attr-defined]
    SU.install_ue_seams = install_ue_seams


ORDERING_EDGE_MODULE = "-m rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_ordering"


def install_ordering_route(arm: str, ordering: str, rec: Any) -> None:
    """Both ordering arms run the ordering edge entry; the registered ordering is
    handed to the container as a file in the edge scratch's profiler leaf (the
    leaf the profiler's seed seam creates), written right after that seed."""
    from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
    from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as D

    from . import edge_ordering_v1 as OR
    from . import seams_ue_v1 as SU

    D.DIRECT_EDGE_MODULE = ORDERING_EDGE_MODULE
    inner_seed = pinned.seed_cell_edge_state

    def seed(campaign: Any, edge_scratch: Path) -> None:
        inner_seed(campaign, edge_scratch)
        path = OR.write_ordering_config(Path(edge_scratch) / SU.EDGE_SCRATCH_LEAF, ordering, arm)
        rec.note("edge_ordering_config", ordering=ordering, path=str(path))

    pinned.seed_cell_edge_state = seed


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - live
    args = list(sys.argv[1:] if argv is None else argv)
    arm, restore = TC.child_arm()
    TC.install_child_environment_restore(restore)
    from . import child_v1 as CH             # stdlib-only module; numerical imports happen later

    attempt_dir = CH.attempt_dir_from_argv(args)
    TC.require(attempt_dir is not None, "--attempt-dir is required")
    evidence = Path(attempt_dir) / TC.EVIDENCE_LEAF
    install_bundle_seams(arm, Path(attempt_dir))
    TC.write_snapshot(evidence, TC.settings_snapshot("pre", arm=arm))
    try:
        return CH.main(args)
    finally:
        TC.write_snapshot(evidence, TC.settings_snapshot("post", arm=arm))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
