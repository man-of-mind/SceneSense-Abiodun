#!/usr/bin/env python3
"""Edge-container entry point: the profiled Phase-6 edge, plus the ordering seam.

Runs ``edge_profiled_v1.main`` unchanged (same argv, same profiler); after its
seams are installed, the ordering seam wraps the profiled processor/evaluator
according to ``<edge scratch>/profiler/edge_ordering.json`` written by the UE
child for the registered arm (absent = serial control).

The module name keeps ``splitfusion_single_gpu_profiler_v1.edge_profiled_v1``
as a substring on purpose: the pinned sampler classifies the edge process by
that substring of its command line.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from . import edge_ordering_v1 as OR
from . import edge_profiled_v1 as EP
from . import recorder_v1 as RV


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - live
    args = list(sys.argv[1:] if argv is None else argv)
    leaf = EP.profile_path(args).parent
    ordering = OR.read_ordering(leaf / OR.ORDERING_FILE)
    original_install = EP.install_edge_seams

    def install_edge_seams(rec):  # noqa: ANN001 - same contract as the original
        notes = original_install(rec)
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP

        notes["ordering"] = OR.install(edge_module=E, prewarm_module=PW, map_protocol=MP,
                                       ordering=ordering, counters_dir=leaf, recorder_fn=RV.active)
        rec.note("edge_ordering", **notes["ordering"])
        return notes

    EP.install_edge_seams = install_edge_seams
    return EP.main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
