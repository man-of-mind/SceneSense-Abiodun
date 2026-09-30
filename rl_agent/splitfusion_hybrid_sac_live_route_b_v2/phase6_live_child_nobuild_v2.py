#!/usr/bin/env python3
"""Phase 6 child entry point: no-build edge launch plus addendum-6 seams.

Runs ``phase6_live_child_v2`` with process-local seams added right after its
``install_run4_seams``:

* addendum 3: ``adapter_direct_v1``'s call to the shared legacy launcher is
  routed through ``phase6_edge_launch_v2.launch_no_build``. That starts the
  admitted image without build or pull, and writes
  ``run4_phase6/edge_image_launch.json``;
* addendum 6: the runtime's :class:`CycleBudgetV2` receives the frame budget
  and an optional ``--stop-after-decisions`` cap, so the run stops only at a
  closed k_min decision cycle;
* addendum 6: UE-side GT writes are recorded to
  ``run4_phase6/gt_handoff_ue.jsonl``;
* addendum 6: the edge GT scratch directory is preserved, with a verified
  manifest, before teardown deletes it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from . import phase6_edge_launch_v2 as EL
from . import phase6_gt_handoff_v2 as GH
from . import phase6_gt_priority_v2 as GP
from . import phase6_live_child_v2 as C

EVIDENCE_RELPATH = Path("run4_phase6") / "edge_image_launch.json"
GT_WRITES_RELPATH = Path("run4_phase6") / "gt_handoff_ue.jsonl"
GT_SCRATCH_RELPATH = Path("run4_phase6") / "gt_scratch_preserved"
_ORIGINAL_INSTALL = C.install_run4_seams
_DECISION_CAP: dict[str, Optional[int]] = {"value": None}


def install_run4_seams_nobuild(campaign: Any, *, attempt_dir: Path, **kwargs: Any) -> Any:
    from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
    from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as D
    from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as Q

    from . import phase6_ue_runtime_v2 as U

    seams = _ORIGINAL_INSTALL(campaign, attempt_dir=attempt_dir, **kwargs)
    EL.install_adapter_launch_seam(D, campaign, Path(attempt_dir) / EVIDENCE_RELPATH)

    budget = kwargs.get("transmitted_budget")
    factory = pinned.LivePilotCellRuntime
    ticket_log = GP.GtTicketLogV2()          # addendum 7: object-GT ticket timeline

    def budgeted_factory(**factory_kwargs: Any) -> Any:
        runtime = factory(**factory_kwargs)
        runtime.cycle_budget = U.CycleBudgetV2(frame_budget=budget,
                                               decision_cap=_DECISION_CAP["value"])
        runtime.gt_log = ticket_log
        return runtime

    pinned.LivePilotCellRuntime = budgeted_factory

    recorder = GH.GtWriteRecorderV2(Path(attempt_dir) / GT_WRITES_RELPATH,
                                    ticket_log=ticket_log)
    Q.write_object_ground_truth, Q.write_semantic_ground_truth = recorder.wrap(
        Q.write_object_ground_truth, Q.write_semantic_ground_truth)

    inner_stop = pinned.stop_tail

    def stop_tail_preserving_gt() -> bool:
        stopped = inner_stop()
        scratch = D._ENDPOINT.get("edge_scratch")
        target = Path(attempt_dir) / GT_SCRATCH_RELPATH
        preserved = False
        if target.exists():                        # already preserved (create-only)
            preserved = bool(GH._load(target.parent / (target.name + ".manifest.json"),
                                      {}).get("verified"))
        elif scratch:
            manifest = GH.preserve_directory(Path(scratch) / pinned.EDGE_EVIDENCE_LEAF,
                                             target)
            preserved = bool(manifest["verified"])
        # The temporary source is deleted by the caller only after this returns.
        return bool(stopped and preserved)

    pinned.stop_tail = stop_tail_preserving_gt
    return seams


def split_argv(argv: Sequence[str]) -> tuple[list[str], Optional[int]]:
    """Strip ``--stop-after-decisions N`` (the unchanged child does not know it)."""
    remaining, cap, items = [], None, list(argv)
    index = 0
    while index < len(items):
        if items[index] == "--stop-after-decisions":
            cap = int(items[index + 1])
            if cap < 1:
                raise SystemExit("--stop-after-decisions must be >= 1")
            index += 2
            continue
        remaining.append(items[index])
        index += 1
    return remaining, cap


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - live
    remaining, cap = split_argv(sys.argv[1:] if argv is None else argv)
    _DECISION_CAP["value"] = cap
    C.install_run4_seams = install_run4_seams_nobuild
    return C.main(remaining)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
