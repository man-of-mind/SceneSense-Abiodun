#!/usr/bin/env python3
"""One isolated, bounded scene-descriptor sensor-profiling cell.

The parent owns the fresh OAI and CARLA-server lifecycles.  This child owns
the direct map, direct edge, target-SNR actuator, CARLA client and collector.
It composes three additive seams without editing any hash-pinned production
source: direct edge-to-map, sensor profiling, and the proven non-discarding
exact-transmission budget wrapper.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from rl_agent.splitfusion_quality_feedback_probe_v1.live_cell_child import (
    build_bounded_collector_class,
)


class SceneDescriptorLiveError(RuntimeError):
    """The bounded scene-descriptor cell violated its contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SceneDescriptorLiveError(message)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(dict(payload), sort_keys=True, indent=1, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
    from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as direct
    from rl_agent.splitfusion_supervisor_analysis_v1 import (
        profiled_direct_adapter_v1 as profiled,
    )

    campaign_path = Path(args.campaign_json).resolve(strict=True)
    cell_path = Path(args.cell_json).resolve(strict=True)
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    attempt_dir = Path(args.attempt_dir).resolve(strict=True)
    temporary_dir = Path(args.temporary_dir).resolve(strict=True)
    artifacts_dir = Path(args.artifacts_dir).resolve(strict=True)
    result_path = artifacts_dir / "child_result.json"

    row = pinned.action_row(campaign, int(cell["action_id"]))
    require(
        str(row["profile_id"]) == str(cell["profile_id"])
        and str(row["model_family"]).casefold()
        == str(cell["model_family"]).casefold()
        and str(row["entropy_coder"]) == "zstd",
        "cell/action catalog identity mismatch",
    )

    direct._ENDPOINT["cell_id"] = str(cell["cell_id"])
    direct._ENDPOINT["attempt_dir"] = str(attempt_dir)
    direct._ENDPOINT["config_relpath"] = str(
        campaign["runtime"]["direct_edge_config_relpath"]
    )
    direct.install_direct_seams(campaign)
    profiled.install_sensor_profile_seams(campaign)
    pinned.PassiveSplitCollector = build_bounded_collector_class(
        pinned.PassiveSplitCollector,
        transmitted_budget=int(args.transmitted_budget),
        safety_timeout_s=float(args.safety_timeout_s),
        artifacts_dir=artifacts_dir,
    )

    result: dict[str, Any] = {
        "schema": "scenesense.scene_descriptor_live_child.v1",
        "cell_id": str(cell["cell_id"]),
        "action_id": int(cell["action_id"]),
        "profile_id": str(cell["profile_id"]),
        "network_profile_id": str(cell["network_profile_id"]),
        "scene_descriptor_mode": str(
            campaign["_sensor_preparation_diagnostic"]["scene_descriptor_mode"]
        ),
        "transmitted_budget": int(args.transmitted_budget),
        "started_at_unix_s": time.time(),
        "map_started": False,
        "edge_started": False,
        "target_snr_started": False,
        "full_route_completion_claimed": False,
        "error": "",
        "cleanup": {},
    }
    map_process: Any = None
    edge_scratch: Path | None = None
    target_process: Any = None
    target_output = Path()
    target_stop = Path()
    collector: Any = None
    return_code = 2
    try:
        map_process = pinned.start_map_process(
            campaign,
            temporary_dir=temporary_dir,
            action_id=str(cell["action_id"]),
            carla_host="127.0.0.1",
            carla_port=int(args.carla_port),
            api_port=int(args.map_api_port),
            udp_port=int(args.spatial_map_port),
            feedback_port=int(args.feedback_port),
        )
        result["map_started"] = True
        edge_scratch = pinned.start_live_edge(campaign, cell, temporary_dir)
        result["edge_started"] = True
        edge_evidence_dir = edge_scratch / pinned.EDGE_EVIDENCE_LEAF
        require(edge_evidence_dir.is_dir(), "edge evidence mount is missing")

        isolated_campaign = temporary_dir / "scene_descriptor_campaign.yaml"
        target_start = temporary_dir / "scene_descriptor_target_start"
        campaign["_target_start_file"] = str(target_start)
        isolated_campaign.write_text(
            yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8"
        )
        target_process, target_output, target_stop = pinned.start_target_snr(
            campaign,
            campaign_path=isolated_campaign,
            profile_id=str(cell["network_profile_id"]),
            temporary_dir=temporary_dir,
            start_file=target_start,
        )
        time.sleep(1.0)
        require(
            target_process.poll() is None,
            "target-SNR actuator exited during startup",
        )
        result["target_snr_started"] = True

        route_ok, route_detail, collector = pinned.run_route_b(
            campaign=campaign,
            cell=cell,
            row=row,
            binding={"dispatcher": "phase13_sfd1_v2"},
            attempt_dir=attempt_dir,
            carla_host="127.0.0.1",
            carla_port=int(args.carla_port),
            map_api_port=int(args.map_api_port),
            feedback_port=int(args.feedback_port),
            edge_evidence_dir=edge_evidence_dir,
            maximum_loop_sim_s=float(args.safety_timeout_s),
        )
        result["full_route_completed"] = bool(route_ok)
        result["route_detail"] = {
            key: route_detail.get(key)
            for key in (
                "route_runner_returncode",
                "density_status",
                "route_completed",
                "route_abort_reason",
                "error",
            )
        }
        require(collector is not None, "route never entered the collector")
        result["collector"] = collector.probe_summary()
        require(
            int(collector.sent) == int(args.transmitted_budget),
            f"only {int(collector.sent)}/{int(args.transmitted_budget)} "
            "transmitted frames completed",
        )
        require(
            str(collector.probe_stop_reason) == "TRANSMITTED_BUDGET_REACHED",
            f"unexpected stop reason: {collector.probe_stop_reason!r}",
        )
        require(bool(collector.cleanup_ok), "collector drain did not finish cleanly")
        require(not collector.failures, f"collector failures: {collector.failures}")
        return_code = 0
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return_code = 2
    finally:
        cleanup = result["cleanup"]
        if target_process is not None:
            try:
                cleanup["target_snr_restored"] = bool(
                    pinned.stop_target_snr(
                        target_process,
                        target_output,
                        target_stop,
                        attempt_dir / "radio_trace.csv",
                    )
                )
            except BaseException as exc:
                cleanup["target_snr_error"] = f"{type(exc).__name__}: {exc}"
        else:
            cleanup["target_snr_restored"] = False
        cleanup["map_stopped"] = bool(pinned.stop_process(map_process))
        cleanup["edge_stopped"] = bool(pinned.stop_live_edge(edge_scratch))
        result["finished_at_unix_s"] = time.time()
        if return_code == 0 and not all(
            bool(cleanup.get(name))
            for name in ("target_snr_restored", "map_stopped", "edge_stopped")
        ):
            result["error"] = "one or more child-owned resources failed cleanup"
            return_code = 2
        result["return_code"] = int(return_code)
        _atomic_json(result_path, result)
    return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-json", required=True)
    parser.add_argument("--cell-json", required=True)
    parser.add_argument("--attempt-dir", required=True)
    parser.add_argument("--temporary-dir", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--map-api-port", type=int, default=35001)
    parser.add_argument("--spatial-map-port", type=int, default=39310)
    parser.add_argument("--feedback-port", type=int, default=39401)
    parser.add_argument("--transmitted-budget", type=int, default=520)
    parser.add_argument("--safety-timeout-s", type=float, default=90.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
