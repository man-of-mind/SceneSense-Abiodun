#!/usr/bin/env python3
"""One live Route-B cell's CARLA-side work, in its own process.

The CARLA Python client can abort the interpreter on shutdown
(``libc++abi: terminating due to uncaught exception``). When the cell ran
in-process that abort killed the parent's teardown blocks, leaking the gNB,
UE, CARLA server and 5G core onto a shared host and destroying the
measurement. The qualified 288-cell supervisor avoids this by running its
CARLA client out of process, and this module restores that property here.

The parent owns the radio, the CARLA *server* process group, the edge
container, the target-SNR actuator, teardown and evidence -- none of which
touch the CARLA client API. This child owns everything that does: the map
install process, the instrumented collector and the qualified
``run_route_b``. A client-side abort is then just a non-zero exit code the
parent handles, and the collector has already persisted its rows and summary
at the 300th transmitted frame, before any unwinding begins.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from . import diagnostic_common as common
from .diagnostic_common import DiagnosticError, require
from .live_capture import RouteBudgetReached, install_live_wrappers


CHILD_RESULT_NAME = "child_result.json"


def _write_result(artifacts: Path, payload: dict[str, Any]) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    path = artifacts / CHILD_RESULT_NAME
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=1, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    from rl_agent import ue_route_b_split_cell_adapter_v1 as adapter

    artifacts = Path(args.artifacts_dir).resolve()
    campaign = common.load_json(Path(args.campaign_json).resolve(strict=True))
    cell = common.load_json(Path(args.cell_json).resolve(strict=True))
    attempt_dir = Path(args.attempt_dir).resolve(strict=True)
    evidence_dir = Path(args.edge_evidence_dir).resolve(strict=True)
    temporary = Path(args.temporary_dir).resolve(strict=True)

    row = adapter.action_row(campaign, int(cell["action_id"]))
    require(
        str(row["profile_id"]) == str(cell["profile_id"])
        and str(row["model_family"]).casefold() == str(cell["model_family"]).casefold()
        and str(row["entropy_coder"]) == "zstd",
        "child cell/catalog identity mismatch",
    )

    install_live_wrappers(
        adapter,
        transmitted_budget=int(args.transmitted_budget),
        safety_timeout_s=float(args.safety_timeout_s),
        artifacts_dir=artifacts,
    )

    result: dict[str, Any] = {
        "schema": "scenesense.splitfusion_timing_diagnostic_live_child.v1",
        "cell_id": str(cell["cell_id"]),
        "action_id": int(cell["action_id"]),
        "transmitted_budget": int(args.transmitted_budget),
        "safety_timeout_s": float(args.safety_timeout_s),
        "started_at_unix_s": time.time(),
        "map_process_started": False,
        "route_accepted_by_campaign_gate": None,
        "route_detail": {},
        "stop_reason": "",
        "error": "",
        "map_process_stopped": None,
    }
    map_process: Any = None
    try:
        map_process = adapter.start_map_process(
            campaign, temporary_dir=temporary, action_id=str(cell["action_id"]),
            carla_host="127.0.0.1", carla_port=int(args.carla_port),
            api_port=int(args.map_api_port), udp_port=int(args.spatial_map_port),
            feedback_port=int(args.feedback_port),
        )
        result["map_process_started"] = True
        print(f"[child {cell['cell_id']}] map install path ready", flush=True)
        print(
            f"[child {cell['cell_id']}] live capture: "
            f"{args.transmitted_budget} transmitted frames or "
            f"{float(args.safety_timeout_s):.0f} s",
            flush=True,
        )
        try:
            route_ok, route_detail, collector = adapter.run_route_b(
                campaign=campaign, cell=cell, row=row,
                binding={"dispatcher": "phase13_sfd1_v2"},
                attempt_dir=attempt_dir, carla_host="127.0.0.1",
                carla_port=int(args.carla_port), map_api_port=int(args.map_api_port),
                feedback_port=int(args.feedback_port), edge_evidence_dir=evidence_dir,
                maximum_loop_sim_s=float(args.safety_timeout_s),
            )
        except RouteBudgetReached as reached:
            # The sentinel is the intended terminal, not a failure. It can only
            # surface here if run_route_b re-raised it.
            route_ok, route_detail, collector = False, {"error": str(reached)}, None
        result["route_accepted_by_campaign_gate"] = bool(route_ok)
        result["route_detail"] = {
            key: route_detail.get(key)
            for key in (
                "route_runner_returncode", "density_status", "route_completed",
                "route_abort_reason", "error", "route_summary_identity_ok",
                "route_summary_identity_mismatches",
            )
        }
        require(collector is not None, "the route never entered its drive loop")
        persisted = collector.persist_artifacts()
        result["artifacts"] = persisted
        result["stop_reason"] = str(collector.diagnostic_stop_reason)
        result["transmitted_frames"] = int(collector.sent)
        result["failures"] = list(collector.failures)
        result["cleanup_ok"] = bool(collector.cleanup_ok)
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        _write_result(artifacts, {**result, "finished_at_unix_s": time.time()})
        raise
    finally:
        if map_process is not None:
            result["map_process_stopped"] = adapter.stop_process(map_process)
        result["finished_at_unix_s"] = time.time()
        _write_result(artifacts, result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-json", required=True)
    parser.add_argument("--cell-json", required=True)
    parser.add_argument("--attempt-dir", required=True)
    parser.add_argument("--temporary-dir", required=True)
    parser.add_argument("--edge-evidence-dir", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--carla-port", type=int, required=True)
    parser.add_argument("--map-api-port", type=int, required=True)
    parser.add_argument("--spatial-map-port", type=int, required=True)
    parser.add_argument("--feedback-port", type=int, required=True)
    parser.add_argument("--transmitted-budget", type=int, required=True)
    parser.add_argument("--safety-timeout-s", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return run(args)
    except DiagnosticError as exc:
        print(f"live cell child contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
