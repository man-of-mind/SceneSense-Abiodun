#!/usr/bin/env python3
"""Phase 6 child entry point with the no-build edge launch (addendum 3).

Runs the unchanged ``phase6_live_child_v2`` exactly. The only difference is one
process-local seam, added right after its ``install_run4_seams``:
``adapter_direct_v1``'s call to the shared legacy launcher is routed through
``phase6_edge_launch_v2.launch_no_build``. That starts the admitted image
without build or pull, and writes ``run4_phase6/edge_image_launch.json`` into
the attempt directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from . import phase6_edge_launch_v2 as EL
from . import phase6_live_child_v2 as C

EVIDENCE_RELPATH = Path("run4_phase6") / "edge_image_launch.json"
_ORIGINAL_INSTALL = C.install_run4_seams


def install_run4_seams_nobuild(campaign: Any, *, attempt_dir: Path, **kwargs: Any) -> Any:
    from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as D

    seams = _ORIGINAL_INSTALL(campaign, attempt_dir=attempt_dir, **kwargs)
    EL.install_adapter_launch_seam(D, campaign, Path(attempt_dir) / EVIDENCE_RELPATH)
    return seams


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - live
    C.install_run4_seams = install_run4_seams_nobuild
    return C.main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
