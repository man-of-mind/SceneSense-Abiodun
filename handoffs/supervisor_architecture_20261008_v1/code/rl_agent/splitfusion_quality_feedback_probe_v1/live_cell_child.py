#!/usr/bin/env python3
"""One bounded live quality-feedback cell, isolated from its lifecycle parent.

The parent owns the fresh OAI radio and CARLA server process group.  This
child owns the direct map process, direct edge container, target-SNR actuator
and CARLA client.  Isolating the client matters because CARLA can abort the
interpreter during actor destruction; such an abort must not bypass the
parent's radio/CARLA teardown.

This is deliberately not a Route-B completion run.  The collector stops after
exactly ``--transmitted-budget`` successful feature transmissions.  Unlike the
older timing diagnostic, none of the semantic/object evaluation queues is
discarded: the quality evaluator and compact quality ACK are the measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


class LiveProbeError(RuntimeError):
    """A bounded-cell contract was violated."""


class RouteBudgetReached(RuntimeError):
    """Intentional exception used to stop before a full Route-B loop."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LiveProbeError(message)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(dict(payload), sort_keys=True, indent=1, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_bounded_collector_class(
    base: type,
    *,
    transmitted_budget: int,
    safety_timeout_s: float,
    artifacts_dir: Path,
) -> type:
    """Return a non-discarding exact-transmission-budget collector.

    The superclass still owns all production and evaluation queues.  In
    particular, semantic GT, object GT, exact record retrieval and quality ACK
    handling are allowed to drain in ``super().finish()``.  The only behavior
    changed here is the route terminal and additive durable reporting.
    """

    class BoundedQualityCollector(base):  # type: ignore[misc, valid-type]
        _transmitted_budget = int(transmitted_budget)
        _safety_timeout_s = float(safety_timeout_s)
        _artifacts_dir = Path(artifacts_dir)

        def __init__(self, **keywords: Any) -> None:
            super().__init__(**keywords)
            self.probe_stop_reason = ""
            self.probe_started_wall_ns = 0
            self.probe_budget_reached_wall_ns = 0
            self.probe_finished_wall_ns = 0
            self.probe_ticks_observed = 0
            self._probe_started_monotonic: float | None = None
            self._probe_stop_requested = False
            self._probe_finish_called = False
            self._probe_lock = threading.Lock()

        def _request_probe_stop(self, reason: str) -> None:
            if not self._probe_stop_requested:
                self._probe_stop_requested = True
                self.probe_stop_reason = str(reason)

        def _write_budget_marker(self) -> None:
            marker = self._artifacts_dir / "budget_reached.json"
            if marker.exists():
                return
            _atomic_json(
                marker,
                {
                    "schema": "scenesense.quality_feedback_probe_budget.v1",
                    "transmitted_budget": self._transmitted_budget,
                    "transmitted_frames": int(self.sent),
                    "budget_reached_wall_ns": int(
                        self.probe_budget_reached_wall_ns
                    ),
                    "quality_and_evaluation_drain_complete": False,
                },
            )

        def on_world_tick(self, frame_id: int, route_tick: int) -> None:
            self.probe_ticks_observed += 1
            if self._probe_started_monotonic is None:
                self._probe_started_monotonic = time.monotonic()
                self.probe_started_wall_ns = time.time_ns()
            if self._probe_stop_requested:
                raise RouteBudgetReached(self.probe_stop_reason)
            if int(self.sent) >= self._transmitted_budget:
                self._request_probe_stop("TRANSMITTED_BUDGET_REACHED")
                raise RouteBudgetReached(self.probe_stop_reason)
            if (
                self._probe_started_monotonic is not None
                and time.monotonic() - self._probe_started_monotonic
                >= self._safety_timeout_s
            ):
                self._request_probe_stop("SAFETY_TIMEOUT_EXPIRED")
                raise RouteBudgetReached(self.probe_stop_reason)
            super().on_world_tick(frame_id, route_tick)

        def _process_token(self, token: Mapping[str, Any]) -> None:
            # One worker owns this method, but checking both before and after
            # the qualified implementation makes the exact upper bound clear.
            if int(self.sent) >= self._transmitted_budget:
                self.transport_counters.bump("quality_probe_skipped_after_budget")
                return
            before = int(self.sent)
            super()._process_token(token)
            if int(self.sent) > before and int(self.sent) >= self._transmitted_budget:
                require(
                    int(self.sent) == self._transmitted_budget,
                    "transmission budget was overshot",
                )
                self.probe_budget_reached_wall_ns = time.time_ns()
                self._request_probe_stop("TRANSMITTED_BUDGET_REACHED")
                # This create-only marker survives even if CARLA aborts later.
                self._write_budget_marker()

        def probe_summary(self) -> dict[str, Any]:
            return {
                "schema": "scenesense.quality_feedback_probe_collector.v1",
                "transmitted_budget": self._transmitted_budget,
                "safety_timeout_s": self._safety_timeout_s,
                "transmitted_frames": int(self.sent),
                "reached_budget": int(self.sent) == self._transmitted_budget,
                "stop_reason": self.probe_stop_reason,
                "ticks_observed": int(self.probe_ticks_observed),
                "started_wall_ns": int(self.probe_started_wall_ns),
                "budget_reached_wall_ns": int(
                    self.probe_budget_reached_wall_ns
                ),
                "finished_wall_ns": int(self.probe_finished_wall_ns),
                "dropped_preparation_opportunities": int(self.dropped),
                "transport_counters": self.transport_counters.snapshot(),
                "evaluation_queues_discarded": False,
                "quality_and_evaluation_drain_complete": bool(
                    self._probe_finish_called
                ),
                "collector_cleanup_ok": bool(self.cleanup_ok),
                "collector_failures": list(self.failures),
            }

        def _persist_probe_summary(self) -> None:
            with self._probe_lock:
                _atomic_json(
                    self._artifacts_dir / "collector_summary.json",
                    self.probe_summary(),
                )

        def finish(self) -> bool:
            cleanup_ok = False
            try:
                cleanup_ok = bool(super().finish())
                return cleanup_ok
            finally:
                self._probe_finish_called = True
                self.probe_finished_wall_ns = time.time_ns()
                try:
                    # Standard per-frame and exact perception outputs are
                    # useful joins for this probe and are written only after
                    # every qualified drain has completed.
                    if not (self.attempt_dir / "per_frame_metrics.csv").exists():
                        self.write_per_frame()
                    if not (self.attempt_dir / "perception_metrics.csv").exists():
                        self.write_perception()
                except Exception as exc:
                    self.failures.append(
                        "bounded probe evidence: "
                        f"{type(exc).__name__}: {exc}"
                    )
                try:
                    self._persist_probe_summary()
                except Exception as exc:
                    self.failures.append(
                        "bounded probe summary: "
                        f"{type(exc).__name__}: {exc}"
                    )

    return BoundedQualityCollector


def _write_child_result(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_json(path, payload)


def run(args: argparse.Namespace) -> int:
    from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
    from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1

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

    adapter_quality_v1.install_quality_seams(
        campaign, cell=cell, attempt_dir=attempt_dir
    )
    pinned.PassiveSplitCollector = build_bounded_collector_class(
        pinned.PassiveSplitCollector,
        transmitted_budget=int(args.transmitted_budget),
        safety_timeout_s=float(args.safety_timeout_s),
        artifacts_dir=artifacts_dir,
    )

    result: dict[str, Any] = {
        "schema": "scenesense.quality_feedback_probe_live_child.v1",
        "cell_id": str(cell["cell_id"]),
        "action_id": int(cell["action_id"]),
        "profile_id": str(cell["profile_id"]),
        "network_profile_id": str(cell["network_profile_id"]),
        "transmitted_budget": int(args.transmitted_budget),
        "started_at_unix_s": time.time(),
        "map_started": False,
        "edge_started": False,
        "target_snr_started": False,
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

        isolated_campaign = temporary_dir / "quality_probe_campaign.yaml"
        target_start = temporary_dir / "quality_probe_target_start"
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
        result["full_route_completion_claimed"] = False
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
        require(
            bool(collector.cleanup_ok),
            "collector/quality feedback drain did not finish cleanly",
        )
        require(
            not collector.failures,
            f"collector reported failures: {collector.failures}",
        )
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
        try:
            cleanup["quality_evidence"] = (
                adapter_quality_v1.preserve_quality_edge_evidence()
            )
        except BaseException as exc:
            cleanup["quality_evidence_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
        cleanup["edge_stopped"] = bool(pinned.stop_live_edge(edge_scratch))
        result["finished_at_unix_s"] = time.time()
        result["return_code"] = int(return_code)
        if return_code == 0 and not all(
            bool(cleanup.get(name))
            for name in (
                "target_snr_restored",
                "map_stopped",
                "quality_evidence",
                "edge_stopped",
            )
        ):
            result["error"] = "one or more child-owned resources failed cleanup"
            return_code = 2
            result["return_code"] = return_code
        _write_child_result(result_path, result)
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
    parser.add_argument("--transmitted-budget", type=int, default=300)
    parser.add_argument("--safety-timeout-s", type=float, default=90.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
