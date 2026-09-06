#!/usr/bin/env python3
"""Phase-15 start-gated use of the immutable Phase-14 target-SNR runtime.

The Phase-14 runtime remains byte-identical because it is part of the frozen
calibration provenance.  This narrow live-pilot adapter reuses its generator,
mapping, actuator, response validation and restoration logic, adding only the
Route-B first-capture gate and required radio identity columns.
"""

from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent import ue_target_snr_cell_runtime_v1 as base


FIELDS = (
    "radio_profile_id", "radio_profile_sha256", "bandwidth_mhz", "prb",
    "downlink_slots", "uplink_slots", *base.FIELDS,
)


def write_row(handle: Any, writer: csv.DictWriter, row: Mapping[str, Any]) -> None:
    writer.writerow({field: row.get(field, "") for field in FIELDS})
    handle.flush()


def run(args: argparse.Namespace) -> int:
    campaign = base.load_yaml(args.campaign.resolve(strict=True))
    network = campaign["network"]
    base.require(int(network["sample_period_ms"]) == 100, "runtime requires a 100-ms period")
    base.require(network["catch_up_policy"] == "SKIP_OBSOLETE_NEVER_BURST", "catch-up policy drift")
    base.require(float(network["clean_restore_noise_power_db"]) == -50.0, "clean restore must be -50 dB")
    baseline = network["radio_baseline"]
    base.require(str(baseline["profile_id"]) == "OAI_N78_100MHZ_273PRB_4D5U_V1", "radio profile drift")
    radio = base.load_json(base.repo_path(str(baseline["profile_config"])))["radio"]
    sequence, prefix, frozen = base.prepare_sequence(campaign=campaign, profile_id=args.profile_id)
    replay = base.load_replay_module()
    mapping = base.load_mapping(base.repo_path(str(network["mapping_csv"])), replay)
    oai = base.load_oai_module()
    oai_config = base.load_json(args.oai_config.resolve(strict=True))
    session, model_index = base.open_actuator(oai_config, oai)
    stop_event = threading.Event()

    def stop_handler(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    output = args.output.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    period_ns, step, restored, error = 100_000_000, 0, False, ""
    try:
        with output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
            writer.writeheader()
            handle.flush()
            while (
                not stop_event.is_set()
                and not (args.stop_file and args.stop_file.exists())
                and not args.start_file.exists()
            ):
                stop_event.wait(0.02)
            if stop_event.is_set() or (args.stop_file and args.stop_file.exists()):
                return 0
            anchor = time.monotonic_ns()
            while not stop_event.is_set() and not (args.stop_file and args.stop_file.exists()):
                scheduled = anchor + step * period_ns
                interval_end = scheduled + period_ns
                while not stop_event.is_set():
                    remaining = scheduled - time.monotonic_ns()
                    if remaining <= 0:
                        break
                    stop_event.wait(min(remaining / 1e9, 0.02))
                _state, target_snr = prefix[step] if step < len(prefix) else sequence.next_sample()
                mapped = replay.inverse_interpolate(float(target_snr), mapping, float(args.command_granularity_db))
                command_value = f"{mapped:.12g}"
                base_row = {
                    "radio_profile_id": baseline["profile_id"],
                    "radio_profile_sha256": baseline["profile_config_sha256"],
                    "bandwidth_mhz": int(radio["bandwidth_mhz"]), "prb": int(radio["prb"]),
                    "downlink_slots": int(radio["tdd"]["downlink_slots"]), "uplink_slots": int(radio["tdd"]["uplink_slots"]),
                    "profile_id": args.profile_id, "trace_id": frozen["trace_id"], "seed": int(frozen["seed"]),
                    "step_index": step, "target_snr_db": float(target_snr), "mapped_rfsim_command_db": mapped,
                    "mapped_rfsim_command": f"channelmod modify {model_index} noise_power_dB {command_value}",
                    "achieved_snr_db": "", "scheduled_monotonic_ns": scheduled, "interval_end_monotonic_ns": interval_end,
                }
                if time.monotonic_ns() >= interval_end:
                    write_row(handle, writer, {**base_row, "command_timing_status": "SKIP_OBSOLETE_NEVER_BURST"})
                    step += 1
                    continue
                sent_mono, sent_wall, ack_mono, ack_wall, response = session.command(base_row["mapped_rfsim_command"])
                base.validate_modify(response, command_value, oai)
                write_row(handle, writer, {
                    **base_row, "command_send_monotonic_ns": sent_mono, "command_ack_monotonic_ns": ack_mono,
                    "command_send_wall_ns": sent_wall, "command_ack_wall_ns": ack_wall,
                    "command_latency_ms": (ack_mono - sent_mono) / 1e6,
                    "command_timing_status": "ACK_ON_TIME" if ack_mono < interval_end else "ACK_LATE",
                })
                step += 1
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        restore_error = ""
        try:
            _sm, _sw, _am, _aw, response = session.command(f"channelmod modify {model_index} noise_power_dB -50")
            base.validate_modify(response, "-50", oai)
            _sm, _sw, _am, _aw, state = session.command("channelmod show current")
            row = oai.parse_channel_models(state).get(oai_config["actuator"]["channel_model_name"], {})
            restored = math.isclose(float(row.get("noise_power_db", math.nan)), -50.0, abs_tol=1e-6)
            if not restored and not error:
                error = f"post-restore state mismatch: {row}"
        except Exception as exc:
            restore_error = f"{type(exc).__name__}: {exc}"
            if not error:
                error = "clean restore failed: " + restore_error
        finally:
            session.close()
        base.atomic_json(output.with_suffix(output.suffix + ".summary.json"), {
            "schema": "splitfusion_live_pilot_target_snr_runtime.v1", "profile_id": args.profile_id,
            "trace_id": frozen["trace_id"], "seed": int(frozen["seed"]), "rows": step,
            "same_sequence_continued_after_prefix": step > len(prefix), "clean_restore_noise_power_db": -50.0,
            "clean_restore_verified": restored, "clean_restore_error": restore_error, "error": error,
        })
    base.require(restored, "RFsim noise_power_dB=-50 restore was not verified")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--start-file", type=Path, required=True)
    parser.add_argument("--oai-config", type=Path, default=base.DEFAULT_OAI)
    parser.add_argument("--command-granularity-db", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except base.RuntimeContractError as exc:
        print(f"live pilot target-SNR runtime contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
