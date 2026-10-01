"""Run-4B child: install the UE/edge seams, then run the unchanged no-build child.

The Phase-6 child (``phase6_live_child_nobuild_v2`` over ``phase6_live_child_v2``)
owns the route, bounded collector, cycle budget, map, edge launch, GT writers,
GT/prediction preservation and the transmitted-budget stop.  This entry only
selects the Run-4B UE seams and the GT-free edge module before delegating.
"""

from __future__ import annotations

import sys
from typing import Sequence

EDGE_MODULE = "-m rl_agent.splitfusion_run4b_single_host_v1.edge_runtime_v1"


def install() -> dict:
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
        phase6_live_child_v2 as C,
    )

    from . import ue_v1 as UE

    seams = UE.install_ue_seams()
    C.RUN4_EDGE_MODULE = EDGE_MODULE
    return {**seams, "edge_module": EDGE_MODULE}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - live
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
        phase6_live_child_nobuild_v2 as NB,
    )

    install()
    return NB.main(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
