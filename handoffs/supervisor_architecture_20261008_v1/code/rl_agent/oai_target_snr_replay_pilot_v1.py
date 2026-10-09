#!/usr/bin/env python3
"""Run one bounded OAI replay of four fixed target-SNR trace excerpts.

The pilot starts one clean single-UE OAI epoch, measures at most two upper
mapping anchors, then replays exactly 100 saved samples from each of the four
v2 Gaussian-Markov profiles.  CARLA is forbidden.  Target SNR is converted to
an RFsim command through measured monotone piecewise-linear interpolation; it
is not passed to RFsim directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent import ue_n2_oai_ul_calibration_smoke as n2  # noqa: E402
from rl_agent import ue_n3_oai_ul_command_calibration_v1 as calibration  # noqa: E402
from rl_agent import ue_n3e_fallback_snr_floor_v1 as fallback  # noqa: E402


DEFAULT_CONFIG = ROOT / "rl_agent/configs/oai_target_snr_replay_pilot_v1.json"
PROFILE_ORDER = (
    "FAVORABLE_STABLE",
    "MID_VARIABLE",
    "ADVERSE_STABLE",
    "FADE_RECOVERY",
)
VERDICTS = {
    "REPLAY_INTEGRATION_PASS",
    "REPLAY_TIMING_PASS_MAPPING_REFINEMENT_REQUIRED",
    "UPPER_MAPPING_UNRESOLVED",
    "REPLAY_INTEGRATION_FAIL",
}

INTERVAL_FIELDS = (
    "profile_id",
    "trace_id",
    "trace_step_index",
    "target_snr_db",
    "mapped_rfsim_command_db",
    "scheduled_monotonic_ns",
    "interval_end_monotonic_ns",
    "command_send_monotonic_ns",
    "command_send_wall_ns",
    "command_ack_monotonic_ns",
    "command_ack_wall_ns",
    "command_latency_ms",
    "command_timing_status",
    "achieved_pusch_snr_count",
    "achieved_pusch_snr_median_db",
    "achieved_pusch_snr_p05_db",
    "achieved_pusch_snr_p95_db",
    "target_minus_achieved_error_db",
    "mcs_median",
    "application_sequence",
    "application_message_count",
    "application_send_ok_count",
    "application_acked_count",
    "application_delivery_ack_result",
    "application_ack_latency_ms",
    "application_ack_latencies_ms",
    "ue_pdu_session_status",
)


class PilotFailure(RuntimeError):
    """A bounded replay prerequisite, runtime, or cleanup failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotFailure(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise PilotFailure(f"path escapes repository root: {value}") from exc
    return path


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    os.replace(temporary, path)


def round_to_granularity(value: float, granularity: float) -> float:
    require(granularity > 0.0, "command granularity must be positive")
    return round(float(value) / granularity) * granularity


def correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta)) / denominator


def load_trace_excerpts(
    path: Path, profiles: Sequence[str], samples_per_profile: int
) -> dict[str, list[dict[str, Any]]]:
    excerpts = {profile: [] for profile in profiles}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            profile = str(row["profile_id"])
            if profile not in excerpts:
                continue
            step = int(row["step_index"])
            if step >= samples_per_profile:
                continue
            excerpts[profile].append(
                {
                    "profile_id": profile,
                    "trace_id": str(row["trace_id"]),
                    "trace_step_index": step,
                    "target_snr_db": float(row["target_snr_db"]),
                }
            )
    for profile, rows in excerpts.items():
        rows.sort(key=lambda row: int(row["trace_step_index"]))
        require(len(rows) == samples_per_profile,
                f"{profile}: expected {samples_per_profile} trace rows, found {len(rows)}")
        require([int(row["trace_step_index"]) for row in rows]
                == list(range(samples_per_profile)),
                f"{profile}: trace excerpt is not the sample-zero prefix")
        require(len({str(row["trace_id"]) for row in rows}) == 1,
                f"{profile}: trace ID changed inside excerpt")
    return excerpts


def validate_mapping(anchors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in anchors),
        key=lambda row: float(row["achieved_median_pusch_snr_db"]),
    )
    require(len(ordered) >= 2, "mapping requires at least two measured anchors")
    achieved = [float(row["achieved_median_pusch_snr_db"]) for row in ordered]
    commands = [float(row["noise_power_db"]) for row in ordered]
    require(all(achieved[index] < achieved[index + 1] for index in range(len(achieved) - 1)),
            "measured achieved-SNR anchors are not strictly increasing")
    require(all(commands[index] > commands[index + 1] for index in range(len(commands) - 1)),
            "RFsim commands are not monotone with achieved SNR")
    return ordered


def inverse_interpolate(
    target_snr_db: float,
    anchors: Sequence[Mapping[str, Any]],
    granularity_db: float,
) -> float:
    target = float(target_snr_db)
    ordered = validate_mapping(anchors)
    lower = float(ordered[0]["achieved_median_pusch_snr_db"])
    upper = float(ordered[-1]["achieved_median_pusch_snr_db"])
    require(lower <= target <= upper,
            f"target {target:.3f} dB is outside measured mapping [{lower:.3f}, {upper:.3f}]")
    for left, right in zip(ordered[:-1], ordered[1:]):
        left_snr = float(left["achieved_median_pusch_snr_db"])
        right_snr = float(right["achieved_median_pusch_snr_db"])
        if left_snr <= target <= right_snr:
            fraction = (target - left_snr) / (right_snr - left_snr)
            command = float(left["noise_power_db"]) + fraction * (
                float(right["noise_power_db"]) - float(left["noise_power_db"])
            )
            return round_to_granularity(command, granularity_db)
    raise PilotFailure(f"no measured interpolation segment brackets {target:.3f} dB")


class ReplayPilot(fallback.FallbackRunRunner):
    """One attached OAI session containing calibration and replay."""

    def __init__(self, pilot_config_path: Path, output_dir: Path) -> None:
        self.pilot_config_path = pilot_config_path.resolve()
        self.pilot = load_json(self.pilot_config_path)
        paths = self.pilot["paths"]
        base_config = repo_path(str(paths["oai_base_config"]))
        upper = self.pilot["upper_anchor"]
        super().__init__(
            base_config,
            output_dir,
            run_index=1,
            command_db=float(upper["first_command_db"]),
            condition_id="TARGET_SNR_REPLAY_PILOT",
            repetition_index=1,
        )
        profiles = tuple(str(value) for value in self.pilot["profile_order"])
        require(profiles == PROFILE_ORDER, "profile order must preserve the accepted v2 order")
        replay = self.pilot["replay"]
        self.excerpts = load_trace_excerpts(
            repo_path(str(paths["network_profile_traces"])),
            profiles,
            int(replay["samples_per_profile"]),
        )
        self.max_replay_target = max(
            float(row["target_snr_db"])
            for rows in self.excerpts.values()
            for row in rows
        )
        self.config["run"]["client_total_messages"] = int(replay["client_total_messages"])
        self.config["run"]["responder_duration_s"] = float(replay["responder_duration_s"])
        self.anchor_rows: list[dict[str, Any]] = []
        self.interval_rows: list[dict[str, Any]] = []
        self.mapping_anchors: list[dict[str, Any]] = [
            {**dict(row), "source": "EXISTING_MEASURED_ANCHOR"}
            for row in self.pilot["existing_measured_anchors"]
        ]
        self.last_health_check_ns = 0
        self.cached_session_status = "ATTACHED_CURRENT_RNTI_PDU_ACTIVE"

    def cleanup(self, *, strict: bool = False) -> list[str]:
        extra_errors: list[str] = []
        if self.live_mcs is not None:
            try:
                self.live_mcs.stop()
            except Exception as exc:
                extra_errors.append(f"live MCS stop: {exc}")
            self.live_mcs = None
        errors = [*extra_errors, *super().cleanup(strict=False)]
        if self.ext_dn_pid is not None:
            try:
                if self.namespace_udp_busy():
                    errors.append("ext-DN ACK responder UDP port remains busy")
            except Exception as exc:
                errors.append(f"ext-DN UDP cleanup gate: {exc}")
        errors = list(dict.fromkeys(errors))
        report_path = self.output_dir / "cleanup_report.json"
        report = load_json(report_path) if report_path.exists() else {}
        report.update({
            "clean": not errors and report.get("clean") is not False,
            "errors": errors,
            "ext_dn_ack_port_busy": any("UDP port remains busy" in value for value in errors),
            "checked_at": n2.utc_now(),
        })
        n2.atomic_json(report_path, report)
        if strict and not report["clean"]:
            raise PilotFailure("cleanup failed: " + "; ".join(errors))
        return errors

    def health_status(self, *, force: bool = False) -> str:
        now = time.monotonic_ns()
        for name in ("gnb", "ue"):
            process = next((item for item in self.processes if item.name == name), None)
            require(process is not None and process.process.poll() is None,
                    f"RAN process exited during pilot: {name}")
        require(self.live_csv is not None and self.live_csv.process.poll() is None,
                "live PUSCH collector exited during pilot")
        require(self.live_mcs is not None and self.live_mcs.process.poll() is None,
                "live MCS collector exited during pilot")
        if not force and now - self.last_health_check_ns < 1_000_000_000:
            return self.cached_session_status
        self.last_health_check_ns = now
        if self.tunnel_ip() != self.ue_ip:
            self.disconnect_reason = "UE_TUNNEL_IDENTITY_LOST"
        elif self.current_rnti is not None and any(
            value != self.current_rnti for value in self.observed_rntis()
        ):
            self.disconnect_reason = "RNTI_CHANGED"
        self.cached_session_status = (
            "DISCONNECTED:" + self.disconnect_reason
            if self.disconnect_reason else "ATTACHED_CURRENT_RNTI_PDU_ACTIVE"
        )
        require(self.disconnect_reason is None, self.cached_session_status)
        return self.cached_session_status

    def wait_until(self, deadline_ns: int) -> None:
        while True:
            remaining = deadline_ns - time.monotonic_ns()
            if remaining <= 0:
                return
            self.health_status()
            time.sleep(min(remaining / 1e9, 0.02))

    def send_noise_command(
        self,
        model_index: int,
        command_db: float,
        *,
        purpose: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        require(self.telnet is not None, "control session unavailable")
        target = fallback.format_command_db(command_db)
        command = f"channelmod modify {model_index} noise_power_dB {target}"
        sent_mono, sent_wall, ack_mono, ack_wall, response = self.telnet.command(command)
        self.validate_modify_response(response, target)
        row = {
            "purpose": purpose,
            "rfsim_command": command,
            "commanded_noise_power_db": float(command_db),
            "send_monotonic_ns": sent_mono,
            "send_wall_ns": sent_wall,
            "ack_monotonic_ns": ack_mono,
            "ack_wall_ns": ack_wall,
            "command_latency_ms": (ack_mono - sent_mono) / 1e6,
            "status": "ACK_VALIDATED",
            "control_session_id": self.control_session_id,
            "ran_epoch_id": self.ran_epoch_id,
            **dict(extra or {}),
        }
        self.command_rows.append(row)
        self.restored = math.isclose(float(command_db), -50.0, abs_tol=1e-9)
        return row

    def measure_upper_attempt(self, model_index: int, command_db: float, attempt: int) -> dict[str, Any]:
        upper = self.pilot["upper_anchor"]
        command = self.send_noise_command(
            model_index,
            command_db,
            purpose="UPPER_ANCHOR_ATTEMPT",
            extra={"upper_anchor_attempt": attempt},
        )
        start_ns = int(command["ack_monotonic_ns"])
        end_ns = start_ns + int(float(upper["measurement_duration_s"]) * 1e9)
        self.wait_until(end_ns)
        require(self.current_rnti is not None, "current RNTI unavailable for upper anchor")
        tail = calibration.summarize_tail(
            self.live_csv.snapshot() if self.live_csv else [],
            self.live_mcs.snapshot() if self.live_mcs else [],
            start_ns=start_ns,
            end_ns=end_ns,
            expected_rnti=int(self.current_rnti),
            minimum_pusch=int(upper["minimum_pusch_samples"]),
            minimum_mcs=int(upper["minimum_mcs_samples"]),
            required_mcs_table=int(self.config["analysis"]["scheduler_required_mcs_table"]),
            required_force_mcs=int(self.config["analysis"]["scheduler_required_force_ul_mcs"]),
        )
        row = {
            "attempt": attempt,
            "noise_power_db": float(command_db),
            "measurement_start_monotonic_ns": start_ns,
            "measurement_end_monotonic_ns": end_ns,
            "command_latency_ms": command["command_latency_ms"],
            "telemetry_status": tail["status"],
            "pusch_samples": tail["pusch_samples"],
            "achieved_median_pusch_snr_db": tail["achieved_pusch_snr_db_median"],
            "achieved_pusch_snr_p05_db": tail["achieved_pusch_snr_db_p05"],
            "achieved_pusch_snr_p95_db": tail["achieved_pusch_snr_db_p95"],
            "mcs_median": tail["final_mcs_median"],
        }
        self.anchor_rows.append(row)
        return row

    def upper_attempt_resolves_mapping(self, row: Mapping[str, Any]) -> bool:
        achieved = row.get("achieved_median_pusch_snr_db")
        if row.get("telemetry_status") != "TAIL_ACCEPTED" or achieved is None:
            return False
        desired = float(self.pilot["upper_anchor"]["desired_achieved_snr_db"])
        tolerance = float(self.pilot["upper_anchor"]["maximum_absolute_error_db"])
        return (
            abs(float(achieved) - desired) <= tolerance
            and float(achieved) >= self.max_replay_target
        )

    def select_second_upper_command(self, first: Mapping[str, Any]) -> float | None:
        achieved = first.get("achieved_median_pusch_snr_db")
        if achieved is None:
            return None
        reference_command = -10.0
        reference_snr = 19.5
        first_command = float(first["noise_power_db"])
        slope = (float(achieved) - reference_snr) / (first_command - reference_command)
        if not math.isfinite(slope) or slope >= -0.05:
            return None
        desired = float(self.pilot["upper_anchor"]["desired_achieved_snr_db"])
        candidate = first_command + (desired - float(achieved)) / slope
        granularity = float(self.pilot["replay"]["command_granularity_db"])
        candidate = round_to_granularity(candidate, granularity)
        if math.isclose(candidate, first_command, abs_tol=1e-9):
            candidate += -granularity if float(achieved) < desired else granularity
        if not math.isfinite(candidate) or candidate > -10.0 or candidate < -25.0:
            return None
        return candidate

    def establish_upper_mapping(self, model_index: int) -> bool:
        upper = self.pilot["upper_anchor"]
        first = self.measure_upper_attempt(
            model_index, float(upper["first_command_db"]), 1
        )
        attempts = [first]
        if not self.upper_attempt_resolves_mapping(first):
            second_command = self.select_second_upper_command(first)
            if second_command is not None and int(upper["maximum_attempts"]) >= 2:
                attempts.append(self.measure_upper_attempt(model_index, second_command, 2))
        for row in attempts:
            if row.get("telemetry_status") == "TAIL_ACCEPTED" \
                    and row.get("achieved_median_pusch_snr_db") is not None:
                self.mapping_anchors.append({
                    "noise_power_db": float(row["noise_power_db"]),
                    "achieved_median_pusch_snr_db": float(row["achieved_median_pusch_snr_db"]),
                    "source": f"PILOT_MEASURED_UPPER_ATTEMPT_{row['attempt']}",
                })
        try:
            self.mapping_anchors = validate_mapping(self.mapping_anchors)
        except PilotFailure:
            return False
        return (
            bool(attempts)
            and self.upper_attempt_resolves_mapping(attempts[-1])
            and float(self.mapping_anchors[-1]["achieved_median_pusch_snr_db"])
            >= self.max_replay_target
        )

    def replay_profile(self, model_index: int, profile_id: str) -> None:
        replay = self.pilot["replay"]
        period_ns = int(float(replay["sample_period_s"]) * 1e9)
        clean = float(self.config["run"]["clean_commanded_noise_power_db"])
        self.send_noise_command(model_index, clean, purpose="PROFILE_STABLE_GAP",
                                extra={"profile_id": profile_id})
        self.wait_until(time.monotonic_ns() + int(float(replay["stable_gap_s"]) * 1e9))
        anchor = time.monotonic_ns() + int(float(replay["profile_lead_s"]) * 1e9)
        granularity = float(replay["command_granularity_db"])
        for trace_row in self.excerpts[profile_id]:
            step = int(trace_row["trace_step_index"])
            scheduled = anchor + step * period_ns
            interval_end = scheduled + period_ns
            self.wait_until(scheduled)
            target_snr = float(trace_row["target_snr_db"])
            mapped = inverse_interpolate(target_snr, self.mapping_anchors, granularity)
            before_send = time.monotonic_ns()
            base = {
                **trace_row,
                "mapped_rfsim_command_db": mapped,
                "scheduled_monotonic_ns": scheduled,
                "interval_end_monotonic_ns": interval_end,
                "ue_pdu_session_status": self.cached_session_status,
            }
            if before_send >= interval_end:
                self.interval_rows.append({
                    **base,
                    "command_timing_status": "SKIPPED",
                })
                continue
            command = self.send_noise_command(
                model_index,
                mapped,
                purpose="TARGET_SNR_REPLAY",
                extra={
                    "profile_id": profile_id,
                    "trace_id": trace_row["trace_id"],
                    "trace_step_index": step,
                    "scheduled_monotonic_ns": scheduled,
                    "interval_end_monotonic_ns": interval_end,
                },
            )
            timing = "ON_TIME" if int(command["ack_monotonic_ns"]) < interval_end else "LATE"
            self.interval_rows.append({
                **base,
                "command_send_monotonic_ns": command["send_monotonic_ns"],
                "command_send_wall_ns": command["send_wall_ns"],
                "command_ack_monotonic_ns": command["ack_monotonic_ns"],
                "command_ack_wall_ns": command["ack_wall_ns"],
                "command_latency_ms": command["command_latency_ms"],
                "command_timing_status": timing,
            })
        self.wait_until(anchor + len(self.excerpts[profile_id]) * period_ns)

    def read_client_rows_if_complete(self) -> list[dict[str, Any]]:
        if self.client is not None and self.client.process.poll() is None:
            try:
                self.client.process.wait(timeout=20.0)
            except subprocess.TimeoutExpired as exc:
                raise PilotFailure("ACK client exceeded the bounded pilot") from exc
        require(self.client is not None and self.client.process.returncode == 0,
                f"ACK client exited rc={None if self.client is None else self.client.process.returncode}")
        return self.read_client_rows()

    def associate_observations(
        self,
        pusch_snapshot: Sequence[tuple[int, int, str]],
        mcs_snapshot: Sequence[tuple[int, int, str]],
        client_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        require(self.current_rnti is not None, "current RNTI unavailable for association")
        pusch = [
            row for row in (calibration.parse_live_pusch(item) for item in pusch_snapshot)
            if row is not None and int(row["rnti"]) == int(self.current_rnti)
        ]
        mcs = [
            row for row in (calibration.parse_live_mcs(item) for item in mcs_snapshot)
            if row is not None and int(row["rnti"]) == int(self.current_rnti)
        ]
        for row in self.interval_rows:
            scheduled = int(row["scheduled_monotonic_ns"])
            end = int(row["interval_end_monotonic_ns"])
            applied = row.get("command_timing_status") in {"ON_TIME", "LATE"}
            start = int(row["command_ack_monotonic_ns"]) if applied else end
            radio_rows = [item for item in pusch if start <= int(item["mono_ns"]) < end]
            radio_values = [float(item["snr_db"]) for item in radio_rows]
            mcs_rows = [item for item in mcs if start <= int(item["mono_ns"]) < end]
            mcs_values = [float(item["final_mcs"]) for item in mcs_rows]
            achieved = statistics.median(radio_values) if radio_values else None
            application = [
                item for item in client_rows
                if scheduled <= int(item["send_monotonic_ns"]) < end
            ]
            acked = [item for item in application if item["acked"] == "1"]
            latencies = [float(item["ack_latency_ms"]) for item in acked]
            send_ok = sum(item["send_ok"] == "1" for item in application)
            row.update({
                "achieved_pusch_snr_count": len(radio_values),
                "achieved_pusch_snr_median_db": achieved if achieved is not None else "",
                "achieved_pusch_snr_p05_db": n2.percentile(radio_values, 0.05) or "",
                "achieved_pusch_snr_p95_db": n2.percentile(radio_values, 0.95) or "",
                "target_minus_achieved_error_db": (
                    float(row["target_snr_db"]) - achieved if achieved is not None else ""
                ),
                "mcs_median": statistics.median(mcs_values) if mcs_values else "",
                "application_sequence": ";".join(str(item["seq"]) for item in application),
                "application_message_count": len(application),
                "application_send_ok_count": send_ok,
                "application_acked_count": len(acked),
                "application_delivery_ack_result": (
                    "ACKED" if application and len(acked) == len(application)
                    else "NOT_ACKED" if application else "NO_SCHEDULED_APPLICATION_MESSAGE"
                ),
                "application_ack_latency_ms": statistics.median(latencies) if latencies else "",
                "application_ack_latencies_ms": ";".join(f"{value:.6f}" for value in latencies),
            })

    def summarize_profiles(self) -> list[dict[str, Any]]:
        summaries = []
        for profile in PROFILE_ORDER:
            rows = [row for row in self.interval_rows if row["profile_id"] == profile]
            applied = [row for row in rows if row["command_timing_status"] in {"ON_TIME", "LATE"}]
            late = [row for row in rows if row["command_timing_status"] == "LATE"]
            skipped = [row for row in rows if row["command_timing_status"] == "SKIPPED"]
            command_latencies = [float(row["command_latency_ms"]) for row in applied]
            valid = [row for row in rows if row.get("achieved_pusch_snr_median_db") != ""]
            targets = [float(row["target_snr_db"]) for row in valid]
            achieved = [float(row["achieved_pusch_snr_median_db"]) for row in valid]
            errors = [abs(target - observed) for target, observed in zip(targets, achieved)]
            app_attempted = sum(int(row.get("application_message_count", 0)) for row in rows)
            app_acked = sum(int(row.get("application_acked_count", 0)) for row in rows)
            app_latencies = [
                float(value)
                for row in rows
                for value in str(row.get("application_ack_latencies_ms", "")).split(";")
                if value
            ]
            summaries.append({
                "profile_id": profile,
                "trace_id": rows[0]["trace_id"] if rows else "",
                "commands_expected": len(rows),
                "commands_applied": len(applied),
                "commands_late": len(late),
                "commands_skipped": len(skipped),
                "command_ack_latency_ms_p50": n2.percentile(command_latencies, 0.50),
                "command_ack_latency_ms_p95": n2.percentile(command_latencies, 0.95),
                "command_ack_latency_ms_max": max(command_latencies) if command_latencies else None,
                "valid_achieved_snr_intervals": len(valid),
                "target_achieved_correlation": correlation(targets, achieved),
                "snr_tracking_mae_db": statistics.fmean(errors) if errors else None,
                "snr_tracking_p95_absolute_error_db": n2.percentile(errors, 0.95),
                "command_to_first_observed_effect_latency": "UNAVAILABLE_STOCK_TRACER_LIMITATION",
                "application_messages": app_attempted,
                "application_acked": app_acked,
                "application_delivery_ratio": app_acked / app_attempted if app_attempted else 0.0,
                "application_rtt_ms_p50": n2.percentile(app_latencies, 0.50),
                "application_rtt_ms_p95": n2.percentile(app_latencies, 0.95),
                "application_rtt_ms_max": max(app_latencies) if app_latencies else None,
                "ue_or_pdu_disconnected": self.disconnect_reason is not None,
                "outage_status": (
                    "NO_APPLICATION_ACK_OUTAGE_OBSERVED"
                    if app_attempted and app_acked == app_attempted
                    else "APPLICATION_ACK_GAPS_OBSERVED"
                ),
            })
        return summaries

    def determine_verdict(
        self, summaries: Sequence[Mapping[str, Any]], cleanup_clean: bool
    ) -> tuple[str, dict[str, Any]]:
        criteria = self.pilot["pass_criteria"]
        expected = sum(int(row["commands_expected"]) for row in summaries)
        applied = sum(int(row["commands_applied"]) for row in summaries)
        command_latencies = [
            float(row["command_latency_ms"])
            for row in self.interval_rows
            if row.get("command_timing_status") in {"ON_TIME", "LATE"}
        ]
        valid = sum(int(row["valid_achieved_snr_intervals"]) for row in summaries)
        app_attempted = sum(int(row["application_messages"]) for row in summaries)
        app_acked = sum(int(row["application_acked"]) for row in summaries)
        app_latencies = [
            float(value)
            for row in self.interval_rows
            for value in str(row.get("application_ack_latencies_ms", "")).split(";")
            if value
        ]
        errors = [
            abs(float(row["target_minus_achieved_error_db"]))
            for row in self.interval_rows
            if row.get("target_minus_achieved_error_db") != ""
        ]
        command_p95 = n2.percentile(command_latencies, 0.95)
        app_p95 = n2.percentile(app_latencies, 0.95)
        command_ratio = applied / expected if expected else 0.0
        app_ratio = app_acked / app_attempted if app_attempted else 0.0
        valid_ratio = valid / expected if expected else 0.0
        timing_service_pass = all((
            expected == 400,
            command_ratio >= float(criteria["minimum_command_apply_ratio"]),
            command_p95 is not None
            and command_p95 < float(criteria["maximum_command_ack_p95_ms"]),
            self.disconnect_reason is None,
            app_ratio >= float(criteria["minimum_application_delivery_ratio"]),
            app_p95 is not None
            and app_p95 < float(criteria["maximum_application_ack_p95_ms"]),
            valid_ratio >= float(criteria["minimum_valid_achieved_interval_ratio"]),
            self.restored,
            cleanup_clean,
        ))
        mapping_mae = statistics.fmean(errors) if errors else None
        if not timing_service_pass:
            verdict = "REPLAY_INTEGRATION_FAIL"
        elif mapping_mae is None or mapping_mae > float(criteria["maximum_tracking_mae_db"]):
            verdict = "REPLAY_TIMING_PASS_MAPPING_REFINEMENT_REQUIRED"
        else:
            verdict = "REPLAY_INTEGRATION_PASS"
        return verdict, {
            "commands_expected": expected,
            "commands_applied": applied,
            "command_apply_ratio": command_ratio,
            "commands_late": sum(int(row["commands_late"]) for row in summaries),
            "commands_skipped": sum(int(row["commands_skipped"]) for row in summaries),
            "burst_catch_up_used": False,
            "command_ack_latency_ms_p50": n2.percentile(command_latencies, 0.50),
            "command_ack_latency_ms_p95": command_p95,
            "command_ack_latency_ms_max": max(command_latencies) if command_latencies else None,
            "valid_achieved_snr_intervals": valid,
            "valid_achieved_snr_interval_ratio": valid_ratio,
            "snr_tracking_mae_db": mapping_mae,
            "snr_tracking_p95_absolute_error_db": n2.percentile(errors, 0.95),
            "application_messages": app_attempted,
            "application_acked": app_acked,
            "application_delivery_ratio": app_ratio,
            "application_rtt_ms_p50": n2.percentile(app_latencies, 0.50),
            "application_rtt_ms_p95": app_p95,
            "application_rtt_ms_max": max(app_latencies) if app_latencies else None,
            "ue_or_pdu_disconnected": self.disconnect_reason is not None,
            "disconnect_reason": self.disconnect_reason,
            "clean_restore_verified": self.restored,
            "cleanup_clean": cleanup_clean,
            "timing_and_service_pass": timing_service_pass,
            "mapping_quality_target_mae_db": float(criteria["maximum_tracking_mae_db"]),
            "command_to_first_observed_effect_latency": "UNAVAILABLE_STOCK_TRACER_LIMITATION",
        }

    def write_mapping_and_anchor_tables(self) -> None:
        write_csv(
            self.output_dir / "upper_anchor_attempts.csv",
            (
                "attempt", "noise_power_db", "measurement_start_monotonic_ns",
                "measurement_end_monotonic_ns", "command_latency_ms", "telemetry_status",
                "pusch_samples", "achieved_median_pusch_snr_db",
                "achieved_pusch_snr_p05_db", "achieved_pusch_snr_p95_db", "mcs_median",
            ),
            self.anchor_rows,
        )
        write_csv(
            self.output_dir / "target_to_rfsim_mapping.csv",
            ("achieved_median_pusch_snr_db", "noise_power_db", "source"),
            self.mapping_anchors,
        )

    def write_command_table(self) -> None:
        fields: list[str] = []
        for row in self.command_rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        write_csv(self.output_dir / "command_log.csv", fields or ["status"], self.command_rows)

    def report_markdown(self, summary: Mapping[str, Any]) -> str:
        verdict = str(summary["verdict"])

        def metric(value: Any, digits: int = 3) -> str:
            return "n/a" if value is None else f"{float(value):.{digits}f}"

        lines = [
            "# OAI target-SNR replay integration pilot", "",
            f"**Verdict:** `{verdict}`", "",
            "This was one short integration pilot using saved target-SNR design traces. The target values are not measured achieved-OAI traces and were converted through the measured command mapping; RFsim did not receive target PUSCH SNR directly.", "",
            "## Upper-anchor attempts", "",
            "| Attempt | RFsim noise command (dB) | Achieved median PUSCH SNR (dB) | PUSCH samples | Status |",
            "|---:|---:|---:|---:|---|",
        ]
        for row in self.anchor_rows:
            achieved = row.get("achieved_median_pusch_snr_db")
            lines.append(
                f"| {row['attempt']} | {float(row['noise_power_db']):.2f} | "
                f"{'n/a' if achieved is None else f'{float(achieved):.2f}'} | "
                f"{row['pusch_samples']} | {row['telemetry_status']} |"
            )
        lines += ["", "## Piecewise target-to-RFsim mapping", "",
                  "| Achieved-SNR anchor (dB) | RFsim noise command (dB) | Source |",
                  "|---:|---:|---|"]
        for row in self.mapping_anchors:
            lines.append(
                f"| {float(row['achieved_median_pusch_snr_db']):.2f} | "
                f"{float(row['noise_power_db']):.2f} | {row['source']} |"
            )
        profiles = summary.get("profiles", [])
        if profiles:
            lines += ["", "## Replay result by profile", "",
                      "| Profile | Applied/expected | Late | Skipped | Valid SNR intervals | Correlation | MAE (dB) | Delivery | RTT p95 (ms) |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for row in profiles:
                corr = row.get("target_achieved_correlation")
                mae = row.get("snr_tracking_mae_db")
                lines.append(
                    f"| `{row['profile_id']}` | {row['commands_applied']}/{row['commands_expected']} | "
                    f"{row['commands_late']} | {row['commands_skipped']} | "
                    f"{row['valid_achieved_snr_intervals']} | "
                    f"{metric(corr)} | "
                    f"{metric(mae)} | "
                    f"{100 * float(row['application_delivery_ratio']):.2f}% | "
                    f"{metric(row['application_rtt_ms_p95'])} |"
                )
        overall = summary.get("overall")
        if overall:
            lines += [
                "", "## Overall", "",
                f"- Commands applied: {overall['commands_applied']} / {overall['commands_expected']} ({100 * float(overall['command_apply_ratio']):.2f}%)",
                f"- Commands late/skipped: {overall['commands_late']} / {overall['commands_skipped']}",
                f"- Command ACK p50/p95/max: {metric(overall['command_ack_latency_ms_p50'])} / {metric(overall['command_ack_latency_ms_p95'])} / {metric(overall['command_ack_latency_ms_max'])} ms",
                f"- Valid achieved-SNR intervals: {overall['valid_achieved_snr_intervals']} / {overall['commands_expected']}",
                f"- Tracking MAE / p95 absolute error: {metric(overall['snr_tracking_mae_db'])} / {metric(overall['snr_tracking_p95_absolute_error_db'])} dB",
                f"- Application delivery: {overall['application_acked']} / {overall['application_messages']} ({100 * float(overall['application_delivery_ratio']):.2f}%)",
                f"- Application RTT p50/p95/max: {metric(overall['application_rtt_ms_p50'])} / {metric(overall['application_rtt_ms_p95'])} / {metric(overall['application_rtt_ms_max'])} ms",
                f"- UE/PDU disconnection: {overall['ue_or_pdu_disconnected']}",
                f"- Clean `noise_power_dB=-50` restore: {overall['clean_restore_verified']}",
                f"- Cleanup clean: {overall['cleanup_clean']}",
                "- Command-to-first-observed-effect latency: unavailable with the stock tracer; no causal value is claimed.",
            ]
        if summary.get("error"):
            lines += ["", "## Failure", "", f"`{summary['error']}`", ""]
        return "\n".join(lines) + "\n"

    def execute(self) -> int:
        n2.atomic_json(self.output_dir / "resolved_pilot_config.json", self.pilot)
        n2.atomic_json(self.output_dir / "input_hashes.json", {
            "pilot_config_sha256": n2.sha256(self.pilot_config_path),
            **{
                key + "_sha256": n2.sha256(repo_path(str(self.pilot["paths"][key])))
                for key in (
                    "oai_base_config", "network_profile_config",
                    "network_profile_traces", "network_profile_summary",
                )
            },
        })
        summary: dict[str, Any] = {
            "verdict": "REPLAY_INTEGRATION_FAIL",
            "ran_epoch_id": self.ran_epoch_id,
            "control_session_id": self.control_session_id,
            "trace_semantics": "TARGET_SNR_DESIGN_NOT_MEASURED_ACHIEVED_OAI_SNR",
            "rfsim_actuation_semantics": "MEASURED_PIECEWISE_MAPPING_NOT_DIRECT_TARGET_SNR",
        }
        cleanup_errors: list[str] = []
        normal_terminal = False
        try:
            self.preflight()
            gnb_config, ue_config = self.materialize_configs()
            self.start_ran(gnb_config, ue_config)
            self.wait_attach()
            n2.wait_tcp(int(self.config["actuator"]["telnet_port"]), 15)
            self.start_telemetry()
            time.sleep(2.0)
            model_index = self.open_and_validate_telnet()
            self.start_probe()
            self.establish_clean_lead()
            resolved = self.establish_upper_mapping(model_index)
            self.write_mapping_and_anchor_tables()
            if not resolved:
                self.restore(model_index)
                write_csv(
                    self.output_dir / "replay_intervals.csv",
                    INTERVAL_FIELDS,
                    self.interval_rows,
                )
                summary.update({
                    "verdict": "UPPER_MAPPING_UNRESOLVED",
                    "max_replay_target_snr_db": self.max_replay_target,
                    "upper_anchor_attempts": self.anchor_rows,
                })
                normal_terminal = True
            else:
                for profile in PROFILE_ORDER:
                    self.replay_profile(model_index, profile)
                self.restore(model_index)
                recovery_end = time.monotonic_ns() + int(
                    float(self.pilot["replay"]["clean_recovery_s"]) * 1e9
                )
                self.wait_until(recovery_end)
                self.health_status(force=True)
                pusch_snapshot = self.live_csv.snapshot() if self.live_csv else []
                mcs_snapshot = self.live_mcs.snapshot() if self.live_mcs else []
                client_rows = self.read_client_rows_if_complete()
                self.associate_observations(pusch_snapshot, mcs_snapshot, client_rows)
                write_csv(self.output_dir / "replay_intervals.csv", INTERVAL_FIELDS, self.interval_rows)
                profiles = self.summarize_profiles()
                profile_fields: list[str] = []
                for row in profiles:
                    for key in row:
                        if key not in profile_fields:
                            profile_fields.append(key)
                write_csv(
                    self.output_dir / "profile_summary.csv",
                    profile_fields or ["profile_id"],
                    profiles,
                )
                summary["profiles"] = profiles
                normal_terminal = True
            self.write_command_table()
        except (Exception, KeyboardInterrupt) as exc:
            self.best_effort_restore()
            summary["error"] = f"{type(exc).__name__}: {exc}"
            try:
                self.write_mapping_and_anchor_tables()
                self.write_command_table()
                write_csv(self.output_dir / "replay_intervals.csv", INTERVAL_FIELDS, self.interval_rows)
            except Exception as write_exc:
                summary["artifact_write_error"] = f"{type(write_exc).__name__}: {write_exc}"
        finally:
            cleanup_errors = self.cleanup(strict=False)

        cleanup = load_json(self.output_dir / "cleanup_report.json")
        cleanup_clean = bool(cleanup.get("clean")) and not cleanup_errors
        if normal_terminal and summary["verdict"] != "UPPER_MAPPING_UNRESOLVED":
            verdict, overall = self.determine_verdict(summary.get("profiles", []), cleanup_clean)
            summary["verdict"] = verdict
            summary["overall"] = overall
        elif summary["verdict"] == "UPPER_MAPPING_UNRESOLVED" and not (
            self.restored and cleanup_clean
        ):
            summary["verdict"] = "REPLAY_INTEGRATION_FAIL"
            summary["error"] = "upper mapping unresolved and clean restore/cleanup failed"
        summary.update({
            "upper_anchor_attempts": self.anchor_rows,
            "mapping_anchors": self.mapping_anchors,
            "clean_restore_verified": self.restored,
            "cleanup_clean": cleanup_clean,
            "cleanup_errors": cleanup_errors,
        })
        require(summary["verdict"] in VERDICTS, f"unknown verdict {summary['verdict']}")
        n2.atomic_json(self.output_dir / "pilot_summary.json", summary)
        n2.atomic_text(self.output_dir / "REPORT.md", self.report_markdown(summary))
        print(json.dumps({
            "verdict": summary["verdict"],
            "output_dir": str(self.output_dir),
            "clean_restore_verified": self.restored,
            "cleanup_clean": cleanup_clean,
        }, sort_keys=True))
        return 1 if summary["verdict"] == "REPLAY_INTEGRATION_FAIL" else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runner = ReplayPilot(Path(args.config), Path(args.output_dir))
    previous_handlers: dict[signal.Signals, Any] = {}

    def terminate(signum: int, _frame: Any) -> None:
        raise PilotFailure(f"received termination signal {signal.Signals(signum).name}")

    for caught in (signal.SIGTERM, signal.SIGHUP):
        previous_handlers[caught] = signal.getsignal(caught)
        signal.signal(caught, terminate)
    try:
        return runner.execute()
    finally:
        for caught, previous in previous_handlers.items():
            signal.signal(caught, previous)


if __name__ == "__main__":
    raise SystemExit(main())
