#!/usr/bin/env python3
"""Phase 2C: one bounded, radio-only qualification of the live UE telemetry.

Authorized scope: exactly one create-only run of 300 scheduled decision
snapshots at the intended 10-Hz Route-B rate; OAI/RFsim/T-tracer only; no
CARLA, CUDA, FCOS or actor inference.  The durable ``record`` client keeps
writing ``ue.raw`` beside the live readers so the online values can be
compared with an independent post-run replay.

The radio lifecycle is inherited unchanged from the production-domain queue
capture runner (pinned launcher attach, ``multi`` relay + ``record``,
telnet channel actuator, UDP probe, target-channel primer, production sender
and receivers, RF restore, RAN/core teardown, cold checks).  Only the
scheduled decision loop, the live provider and the post-run comparison are
new.  The gates in :data:`GATES` are registered in version control before
collection and are never relaxed afterwards.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "scenesense.run4_live_v2.phase2c_telemetry_qualification.v1"
DECISIONS = 300
PERIOD_NS = 100_000_000
DECISION_LEAD_NS = 5_000_000        # snapshot 5 ms before the frame opens
WARMUP_DECISIONS = 10               # first 1 s excluded from gate 2 only
CELL_ID = "favorable_stable__perm0"
PROFILE_ID = "FAVORABLE_STABLE"
OUTPUT_PARENT_RELPATH = "rl_agent/experiments/splitfusion_hybrid_sac_live_route_b_v2"
PRODUCTION_CONFIG_RELPATH = "rl_agent/ue_production_queue_capture_v1/config_v1.json"

GATES: Mapping[str, Mapping[str, Any]] = {
    "G1_READERS_ALIVE_SCHEMAS_AND_PINS": {
        "rule": "all three live readers STREAMING at every decision; zero "
                "malformed headers; radio_binding.verify and emitter pins match "
                "before and after"},
    "G2_FRESH_COVERAGE": {
        "rule": "after the first 10 decisions, >= 95% of decisions have a valid "
                "UL MCS and a complete RLC backlog, each <= 100 ms old at action "
                "open", "min_fraction": 0.95,
        "denominator": DECISIONS - WARMUP_DECISIONS},
    "G3_ZERO_CAUSALITY_VIOLATIONS": {
        "rule": "zero future/negative-age, cross-UE, cross-session, partial-tick "
                "or action-contaminated samples in any decision"},
    "G4_RAW_REPLAY_AGREEMENT": {
        "rule": "every selected online DCI/RLC value and identity equals the "
                "post-run replay of ue.raw under round-0/NDI-toggle semantics"},
    "G5_EXPLICIT_FALLBACK": {
        "rule": "every non-admitted decision raises ExternalFallbackRequired with "
                "typed missing/stale evidence; no invented zero"},
    "G6_SNAPSHOT_LATENCY": {"rule": "provider snapshot latency P99 <= 1 ms",
                            "p99_max_ns": 1_000_000},
    "G7_CLOCK_BRIDGE_RESIDUAL": {
        "rule": "out-of-sample PDCP anchor residual P95 <= 5 us and max <= 1 ms",
        "p95_max_ns": 5_000, "max_ns": 1_000_000},
    "G8_BOUNDED_CACHE": {"rule": "cache capacities constant; sizes never exceed "
                         "capacity"},
    "G9_COEXISTENCE": {"rule": "record and all live readers coexist through the "
                       "decision window without disconnect or unexpected EOF"},
    "G10_RESTORE_AND_COLD": {
        "rule": "RF channel model restored to its initial read-back; core stopped; "
                "host application-cold"},
}

# Event-emission sources observed at registration.  These are the OAI working
# tree as found (the RLC/PDCP T-event edits are uncommitted in the submodule);
# this task neither authored nor modified them.
EMITTER_PINS: Mapping[str, str] = {
    "OAI/openairinterface5g/openair2/LAYER2/NR_MAC_UE/nr_ue_procedures.c":
        "9173eae0e2d032d4524b4e4be84206d2ebadf32778821c32b66b3c82c598cc70",
    "OAI/openairinterface5g/openair2/LAYER2/NR_MAC_UE/nr_ue_scheduler.c":
        "97707d2dfcb212428f690d33f872fe36f43ce4fba8407fb594129c3fcf1aa66c",
    "OAI/openairinterface5g/openair2/LAYER2/nr_pdcp/nr_pdcp_oai_api.c":
        "3bdc6185626f3d77d0d81dcdc4304f990110e57933502dc9432ffacbff90a215",
    "OAI/openairinterface5g/openair2/LAYER2/nr_rlc/nr_rlc_oai_api.c":
        "6783eaaae1eefd4ac1c60c0f83ea4f149c688f7a56002866fd061a636327498e",
    PRODUCTION_CONFIG_RELPATH:
        "df49469d11f7b9d40e3fdfdb247104e0342bc1ffb289251efd64076fc8678569",
}
EMITTER_PIN_OWNERSHIP = (
    "OBSERVED_OAI_WORKING_TREE_AT_REGISTRATION__NOT_AUTHORED_OR_MODIFIED_BY_"
    "THIS_TASK")


class QualificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")).hexdigest()


def gates_sha256() -> str:
    return canonical_sha256(GATES)


def verify_emitter_pins(repo_root: Path = ROOT) -> dict[str, str]:
    observed = {}
    for relpath, expected in EMITTER_PINS.items():
        actual = sha256_file(repo_root / relpath)
        require(actual == expected, f"emitter/config pin drifted: {relpath}")
        observed[relpath] = actual
    return observed


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def summary(values: Sequence[float]) -> dict[str, Any]:
    return {"n": len(values), "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95), "p99": percentile(values, 0.99),
            "max": max(values) if values else None}


# ---------------------------------------------------------------------------
# Post-run raw replay comparison (pure; unit tested)
# ---------------------------------------------------------------------------


def parse_grant_identity(identity: str) -> dict[str, Any]:
    """``sess8:seq:HH:MM:SS.ffffff:frame.slot:hHARQ`` -> fields."""
    head, harq = identity.rsplit(":h", 1)
    rest, frame_slot = head.rsplit(":", 1)
    _session, _seq, tod = rest.split(":", 2)
    frame, slot = frame_slot.split(".")
    return {"tod": tod, "dci_frame": int(frame), "dci_slot": int(slot),
            "harq_pid": int(harq)}


def parse_rlc_source(source: str) -> dict[str, Any]:
    fields = dict(item.split("=", 1) for item in source.split(":")[1:]
                  if "=" in item)
    # tod contains ':' so re-split around the tod= marker.
    tod = source.split(":tod=", 1)[1].rsplit(":lcids=", 1)[0]
    frame, slot = fields["tick"].split(".")
    return {"frame": int(frame), "slot": int(slot), "tod": tod,
            "lcids": int(source.rsplit(":lcids=", 1)[1])}


def raw_dci_index(rows: Sequence[Mapping[str, str]]) -> dict[tuple, dict[str, int]]:
    index: dict[tuple, dict[str, int]] = {}
    for row in rows:
        key = (row["time"], int(row["rnti"]), int(row["dci_frame"]),
               int(row["dci_slot"]), int(row["harq_pid"]), int(row["direction"]))
        index.setdefault(key, {name: int(row[name]) for name in (
            "mcs", "mcs_table", "round", "ndi", "direction", "rnti")})
    return index


def raw_rlc_ticks(rows: Sequence[Mapping[str, str]], *, rnti: int,
                  ue_id: int) -> dict[tuple, dict[str, int]]:
    """Independent replay of the tick rule over the post-run raw CSV."""
    ticks: dict[tuple, dict[str, int]] = {}
    key = None
    values: dict[int, int] = {}
    last_tod = ""
    for row in rows:
        if (int(row["rnti"]), int(row["ue_id"])) != (rnti, ue_id):
            continue
        current = (int(row["frame"]), int(row["slot"]))
        if key is not None and current != key:
            ticks[(key[0], key[1], last_tod)] = {
                "backlog_bytes": sum(values.values()), "lcids": len(values)}
            values = {}
        key = current
        values[int(row["lcid"])] = int(row["bytes_in_buffer"])
        last_tod = row["time"]
    return ticks            # the final, unproven group is intentionally dropped


def compare_with_raw(decisions: Sequence[Mapping[str, Any]], *,
                     dci_rows: Sequence[Mapping[str, str]],
                     rlc_rows: Sequence[Mapping[str, str]],
                     rnti: int, ue_id: int) -> dict[str, Any]:
    dci_index = raw_dci_index(dci_rows)
    ticks = raw_rlc_ticks(rlc_rows, rnti=rnti, ue_id=ue_id)
    mismatches: list[dict[str, Any]] = []
    matched_dci = matched_rlc = 0
    for row in decisions:
        mcs = row.get("mcs")
        if mcs is not None:
            grant = parse_grant_identity(mcs["grant_identity"])
            key = (grant["tod"], rnti, grant["dci_frame"], grant["dci_slot"],
                   grant["harq_pid"], 1)
            raw = dci_index.get(key)
            expected = {"mcs": mcs["value"], "mcs_table": 0, "round": 0,
                        "ndi": mcs["ndi"], "direction": 1, "rnti": rnti}
            if raw != expected:
                mismatches.append({"decision": row["decision_seq"], "slot": "dci",
                                   "online": expected, "raw": raw})
            else:
                matched_dci += 1
        backlog = row.get("backlog")
        if backlog is not None:
            tick = parse_rlc_source(backlog["source"])
            raw = ticks.get((tick["frame"], tick["slot"], tick["tod"]))
            expected = {"backlog_bytes": backlog["value"], "lcids": tick["lcids"]}
            if raw != expected:
                mismatches.append({"decision": row["decision_seq"], "slot": "rlc",
                                   "online": expected, "raw": raw})
            else:
                matched_rlc += 1
    eligible_raw = sum(1 for key, value in dci_index.items()
                       if value["direction"] == 1 and value["mcs_table"] == 0
                       and value["round"] == 0 and value["rnti"] == rnti)
    ndi_counts = {str(ndi): sum(1 for v in dci_index.values()
                                if v["direction"] == 1 and v["round"] == 0
                                and v["mcs_table"] == 0 and v["ndi"] == ndi)
                  for ndi in (0, 1)}
    return {"matched_dci": matched_dci, "matched_rlc": matched_rlc,
            "mismatches": mismatches, "raw_eligible_round0_ul_grants": eligible_raw,
            "raw_eligible_round0_ul_grants_by_ndi": ndi_counts,
            "raw_complete_rlc_ticks": len(ticks)}


def evaluate_gates(*, decisions: Sequence[Mapping[str, Any]],
                   comparison: Mapping[str, Any],
                   run: Mapping[str, Any]) -> dict[str, Any]:
    """Pure gate evaluation over the recorded evidence."""
    results: dict[str, Any] = {}
    alive_all = all(row["readers_alive"] for row in decisions)
    results["G1_READERS_ALIVE_SCHEMAS_AND_PINS"] = (
        alive_all and run["malformed_headers"] == 0
        and run["pins_before_ok"] and run["pins_after_ok"])
    scored = [row for row in decisions if row["decision_seq"] >= WARMUP_DECISIONS]
    fresh = [row for row in scored if row["admitted"]]
    fraction = len(fresh) / len(scored) if scored else 0.0
    results["G2_FRESH_COVERAGE"] = (
        len(scored) == GATES["G2_FRESH_COVERAGE"]["denominator"]
        and fraction >= GATES["G2_FRESH_COVERAGE"]["min_fraction"])
    violations = sum(int(row["violations"]) for row in decisions)
    results["G3_ZERO_CAUSALITY_VIOLATIONS"] = violations == 0
    admitted = [row for row in decisions if row["admitted"]]
    results["G4_RAW_REPLAY_AGREEMENT"] = (
        not comparison["mismatches"]
        and comparison["matched_dci"] == sum(1 for r in decisions if r.get("mcs"))
        and comparison["matched_rlc"] == sum(1 for r in decisions if r.get("backlog"))
        and bool(admitted))
    fallbacks_ok = all(
        row["admitted"] or (row["fallback_raised"] and row["reasons"]
                            and not row["invented_zero"])
        for row in decisions)
    results["G5_EXPLICIT_FALLBACK"] = fallbacks_ok
    latency = summary([row["snapshot_latency_ns"] for row in decisions])
    results["G6_SNAPSHOT_LATENCY"] = (
        latency["p99"] is not None
        and latency["p99"] <= GATES["G6_SNAPSHOT_LATENCY"]["p99_max_ns"])
    residual = run["bridge_residual_abs_ns"]
    results["G7_CLOCK_BRIDGE_RESIDUAL"] = (
        residual["n"] > 0
        and residual["p95"] <= GATES["G7_CLOCK_BRIDGE_RESIDUAL"]["p95_max_ns"]
        and residual["max"] <= GATES["G7_CLOCK_BRIDGE_RESIDUAL"]["max_ns"])
    results["G8_BOUNDED_CACHE"] = all(
        row["cache"]["dci"] <= row["cache"]["dci_capacity"]
        and row["cache"]["rlc"] <= row["cache"]["rlc_capacity"]
        and row["cache"]["dci_capacity"] == decisions[0]["cache"]["dci_capacity"]
        and row["cache"]["rlc_capacity"] == decisions[0]["cache"]["rlc_capacity"]
        for row in decisions)
    results["G9_COEXISTENCE"] = (
        run["record_alive_after_window"] and alive_all
        and run["unexpected_reader_eof"] == 0)
    results["G10_RESTORE_AND_COLD"] = (
        run["rf_restored"] and run["channel_state_matches_initial"]
        and run["core_stopped"] and run["final_cold"])
    return {
        "gates": results,
        "passed": all(results.values()) and len(results) == len(GATES),
        "fresh_fraction_after_warmup": fraction,
        "fallback_fraction_all": sum(1 for r in decisions if not r["admitted"])
                                  / len(decisions),
        "snapshot_latency_ns": latency,
        "causality_violations": violations,
    }


# ---------------------------------------------------------------------------
# Live runner (inherits the production-capture lifecycle unchanged)
# ---------------------------------------------------------------------------


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def build_runner_class():
    from rl_agent.ue_mcs_backlog_calibration_v1 import contract as V3C
    from rl_agent.ue_mcs_backlog_calibration_v1 import runner as V3R
    from rl_agent.ue_mcs_backlog_near_capacity_v1 import radio_binding as RB
    from rl_agent.ue_production_queue_capture_v1 import contract as C
    from rl_agent.ue_production_queue_capture_v1 import payload_schedule as PS
    from rl_agent.ue_production_queue_capture_v1 import runner as PQR
    from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract

    from . import ue_telemetry_provider_v2 as T

    class TelemetryQualificationRunner(PQR.Runner):
        def __init__(self, output_dir: Path, initial_inventory) -> None:
            super().__init__(ROOT / PRODUCTION_CONFIG_RELPATH, output_dir,
                             initial_inventory=initial_inventory)
            self.provider: T.UeTelemetryProviderV2 | None = None
            self.readers: list[T.LiveEventReaderV2] = []
            self.audit: T.AuditWriterV2 | None = None

        # -- preflight -----------------------------------------------------
        def preflight_qualification(self) -> dict[str, Any]:
            RB.assert_no_forbidden_env(os.environ)
            require(os.environ.get("CUDA_VISIBLE_DEVICES", None) == "",
                    "run with CUDA_VISIBLE_DEVICES= so no CUDA can start")
            privileged = self._run_external(["sudo", "-n", "true"],
                                            timeout_name="process_probe")
            require(privileged.returncode == 0, "passwordless sudo failed")
            self.assert_cold_ran("phase2c preflight")
            core = self._container_states()
            require(not any(v.startswith("true") for v in core.values()),
                    f"qualification requires a cold core, found {core}")
            carla = self._run_external(["pgrep", "-af", "CarlaUE4|CarlaUnreal"],
                                       timeout_name="process_probe",
                                       stderr=subprocess.DEVNULL)
            require(not carla.stdout.strip(), "CARLA is running")
            record = {"utc": _utc(), "core_before": core, "carla_absent": True,
                      "loadavg": Path("/proc/loadavg").read_text().strip(),
                      "cuda_or_model_started": False, "gates_sha256": gates_sha256(),
                      "emitter_pins": verify_emitter_pins(),
                      "emitter_pin_ownership": EMITTER_PIN_OWNERSHIP,
                      "radio_binding": RB.verify("before_preflight", ROOT)}
            _write_json(self.out("preflight.json"), record)
            return record

        def materialize_one_schedule(self, cell) -> None:
            ports = self.config["traffic"]["ports"]
            connection = PS._open_authority(ROOT)
            try:
                frames = PS.build_cell_schedule(cell, repo_root=ROOT,
                                                connection=connection)
            finally:
                connection.close()
            document = PS.schedule_document(cell, frames)
            for row in document["frames"]:
                row["port"] = int(ports[row["tier"]])
            document["ports"] = dict(ports)
            document["schedule_sha256"] = C.canonical_sha256(document["frames"])
            self.schedules[cell.cell_id] = document

        # -- provider --------------------------------------------------------
        def start_provider(self, cell_dir: Path) -> None:
            bridge = T.CausalClockBridgeV2()
            self.provider = T.UeTelemetryProviderV2(bridge=bridge)
            troot = ROOT / self.config["paths"]["t_tracer_dir"]
            msgs = ROOT / self.config["paths"]["t_messages"]
            relay = int(self.config["telemetry"]["ue_relay_port"])
            handlers = {"NRUE_MAC_DCI_GRANT": self.provider.on_dci,
                        "NRUE_MAC_RLC_BUFFER_STATUS": self.provider.on_rlc,
                        "NR_PDCP_TX_SDU": self.provider.on_pdcp}
            for event, handler in handlers.items():
                reader = T.LiveEventReaderV2(event, handler, self.provider)
                reader.start(T.csv_reader_argv(troot, msgs, relay, event), cwd=ROOT)
                self.readers.append(reader)
            self.audit = T.AuditWriterV2(self.readers, cell_dir / "telemetry_live")
            self.audit.start()
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                if self.provider.snapshot().all_readers_alive:
                    return
                time.sleep(0.05)
            raise QualificationError("live readers did not reach STREAMING")

        def bind_observed_ue(self) -> dict[str, Any]:
            assert self.provider is not None
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not self.provider.unbound_ue_candidates:
                time.sleep(0.05)
            candidates = sorted(self.provider.unbound_ue_candidates)
            require(len(candidates) == 1,
                    f"expected exactly one UE in RLC telemetry, saw {candidates}")
            rnti, ue_id = candidates[0]
            self.provider.bind_ue(rnti=rnti, oai_ue_id=ue_id)
            return {"rnti": rnti, "oai_ue_id": ue_id,
                    "ue_label": self.provider.ue_label}

        def stop_provider(self) -> dict[str, Any]:
            eofs = 0
            for reader in self.readers:
                eofs += reader.counters["unexpected_eof"]
            for reader in self.readers:
                reader.stop()
            if self.audit is not None:
                self.audit.stop()
            return {
                "unexpected_reader_eof_before_stop": eofs,
                "reader_counters": {r.event: dict(r.counters) for r in self.readers},
                "provider_counters": dict(self.provider.counters)
                if self.provider else {},
                "bridge_counters": dict(self.provider.bridge.counters)
                if self.provider else {},
            }

        # -- decision loop -----------------------------------------------------
        def decision_loop(self, epoch_mono_ns: int) -> list[dict[str, Any]]:
            provider = self.provider
            assert provider is not None
            rows: list[dict[str, Any]] = []
            for k in range(DECISIONS):
                due = epoch_mono_ns + k * PERIOD_NS - DECISION_LEAD_NS
                now = time.monotonic_ns()
                if now < due:
                    time.sleep((due - now) / 1e9)
                started_mono = time.monotonic_ns()
                identity = contract.DecisionIdentityV1(
                    session_uuid=provider.session_uuid,
                    ue_id=provider.ue_label, decision_seq=k)
                snapshot, boundary, snap_latency = T.open_decision(provider, identity)
                host = T.read_host_clock_pair()
                planned_open_raw = (epoch_mono_ns + k * PERIOD_NS
                                    + (host.raw_ns - host.mono_ns))
                assembly_start = T.raw_now_ns()
                row: dict[str, Any] = {
                    "decision_seq": k, "due_monotonic_ns": due,
                    "lateness_ns": started_mono - due,
                    "snapshot_latency_ns": snap_latency,
                    "snapshot_seq": snapshot.seq,
                    "readers_alive": snapshot.all_readers_alive,
                    "bridge_warm": snapshot.bridge_warm,
                    "bridge_generation": snapshot.bridge_generation,
                    "cache": provider.cache_sizes(),
                    "state_commit_raw_ns": boundary.state_commit_timestamp_ns,
                    "action_open_raw_ns": boundary.action_open_timestamp_ns,
                    "planned_frame_open_raw_ns": planned_open_raw,
                    "host_mono_to_raw_ns": host.raw_ns - host.mono_ns,
                    "mcs": None, "backlog": None, "violations": 0,
                    "invented_zero": False,
                }
                try:
                    evidence = T.assemble_radio_evidence(
                        snapshot, identity=identity, boundary=boundary,
                        payload_enqueue_timestamp_ns=planned_open_raw)
                except contract.MetadataError as exc:
                    row.update({"admitted": False, "fallback_raised": True,
                                "reasons": [f"SCHEDULE_OVERRUN:{exc}"]})
                    rows.append(row)
                    continue
                row["assembly_latency_ns"] = T.raw_now_ns() - assembly_start
                try:
                    T.require_admitted(evidence)
                    row["fallback_raised"] = False
                except contract.ExternalFallbackRequired:
                    row["fallback_raised"] = True
                row["admitted"] = evidence.admitted
                row["reasons"] = list(evidence.fallback_reasons)
                prior = evidence.prior_ul_mcs
                if prior.observation.metadata.valid:
                    meta = prior.observation.metadata
                    row["mcs"] = {
                        "value": prior.observation.value, "ndi": prior.new_data_indicator,
                        "grant_identity": prior.grant_identity,
                        "source_raw_ns": meta.source_timestamp_ns,
                        "available_raw_ns": meta.available_timestamp_ns,
                        "session": meta.identity.session_uuid,
                        "ue": meta.identity.ue_id, "age_ns": evidence.mcs_age_ns}
                elif prior.observation.value is not None:
                    row["invented_zero"] = True
                backlog = evidence.pre_action_rlc_backlog
                if backlog.metadata.valid:
                    meta = backlog.metadata
                    row["backlog"] = {
                        "value": backlog.value, "source": meta.source,
                        "source_raw_ns": meta.source_timestamp_ns,
                        "available_raw_ns": meta.available_timestamp_ns,
                        "session": meta.identity.session_uuid,
                        "ue": meta.identity.ue_id, "age_ns": evidence.backlog_age_ns}
                elif backlog.value is not None:
                    row["invented_zero"] = True
                for slot in ("mcs", "backlog"):
                    item = row[slot]
                    if item is None:
                        continue
                    if (item["source_raw_ns"] > item["available_raw_ns"]
                            or item["available_raw_ns"]
                            >= boundary.state_commit_timestamp_ns
                            or item["age_ns"] is None or item["age_ns"] < 0
                            or item["session"] != provider.session_uuid
                            or item["ue"] != provider.ue_label):
                        row["violations"] += 1
                rows.append(row)
            return rows

        # -- one qualification cell ---------------------------------------------
        def run_qualification(self) -> dict[str, Any]:
            cell = next(c for c in C.planned_cells() if c.cell_id == CELL_ID)
            profiles = {p.profile_id: p
                        for p in V3C.resolve_profiles(ROOT, C.FRAMES_PER_CELL)}
            profile = profiles[PROFILE_ID]
            self.materialize_one_schedule(cell)
            cell_tag = f"qualification__{cell.cell_id}"
            cell_dir = self.output_dir / "cells" / cell_tag
            cell_dir.mkdir(parents=True, exist_ok=False)
            command_log: list[dict[str, Any]] = []
            notes_at_start = len(self.notes)
            record: dict[str, Any] = {"cell_tag": cell_tag, "status": "FAILED",
                                      "started_utc": _utc(), **cell.to_json()}
            decisions: list[dict[str, Any]] = []
            run: dict[str, Any] = {}
            try:
                self.verify_identities("before_cell")
                self.assert_cold_ran(f"cell {cell_tag}")
                record["radio_attach"] = self.start_ran_via_launcher(cell_tag, cell_dir)
                record["edge_context"] = self.bind_edge_context()
                record["radio_path"] = self.verify_radio_path(cell_dir)
                self.start_telemetry(cell_tag)
                self.start_provider(cell_dir)
                self.open_telnet(cell_dir)
                initial_models = V3R.n2.parse_channel_models(
                    (cell_dir / "channel_state_initial.txt").read_text())
                record["noise_before_cell_db"] = self.read_back_noise()
                record["udp_probe"] = self.udp_probe(cell_tag, cell_dir)
                record["ue_binding"] = self.bind_observed_ue()
                granularity = float(self.config["actuator"]["command_granularity_db"])
                command, clamped = V3R.inverse_interpolate(
                    profile.samples[0]["target_snr_db"], self.anchors)
                require(not clamped, "profile prime would clamp")
                command = V3R.round_to_granularity(command, granularity)
                self.send_noise(command, reason="PROFILE_PRIME_HOLD",
                                log=command_log, profile_id=profile.profile_id,
                                target_snr_db=profile.samples[0]["target_snr_db"])
                require(abs(self.read_back_noise() - command) <= 1e-6,
                        "profile prime read-back mismatch")
                time.sleep(float(self.config["campaign"]["warmup_s"]))
                sessions = self.launch_traffic(cell_tag=cell_tag, cell=cell,
                                               cell_dir=cell_dir)
                epoch = int(sessions["epoch"]["shared_epoch_monotonic_ns"])
                decisions = self.decision_loop(epoch)
                record_proc = [p for p in self.processes if p.name == "ue_record"]
                run["record_alive_after_window"] = bool(
                    record_proc and record_proc[0].process.poll() is None)
                run["readers_alive_after_window"] = (
                    self.provider.snapshot().all_readers_alive)
                run["bridge_residual_abs_ns"] = summary(
                    [abs(v) for v in self.provider.bridge.residuals_ns])
                run["cache_after_window"] = self.provider.cache_sizes()
                self.finish_traffic(sessions)
                record["traffic"] = self.audit_traffic(sessions, cell, cell_dir)
                record["sender_csv"] = str(sessions["sender_csv"])
                record["restored"] = self.restore_clean(cell_dir, command_log)
                _, _, _, _, after = self.telnet.command("channelmod show current")
                (cell_dir / "channel_state_after_restore.txt").write_text(after)
                after_models = V3R.n2.parse_channel_models(after)
                run["channel_state_matches_initial"] = after_models == initial_models
                record["status"] = "COLLECTED_PENDING_TEARDOWN"
            except Exception as exc:  # noqa: BLE001 - preserve evidence, clean up
                record["failure"] = f"{type(exc).__name__}: {exc}"
                if self.telnet is not None:
                    try:
                        record["restored"] = self.restore_clean(cell_dir, command_log)
                    except Exception as inner:  # noqa: BLE001
                        record["restore_failure"] = f"{type(inner).__name__}: {inner}"
            finally:
                try:
                    run["provider_stop"] = self.stop_provider()
                except Exception as exc:  # noqa: BLE001
                    run["provider_stop"] = {"error": f"{type(exc).__name__}: {exc}"}
                _write_json(cell_dir / "command_log.json", command_log)
                _write_json(cell_dir / "decisions.json", decisions)
                ran_notes = self.teardown_ran()
                record["lifecycle_notes"] = list(self.notes[notes_at_start:]) + ran_notes
                try:
                    self.extract_ttracer(cell_tag, cell_dir)
                    record["ttracer"] = self.audit_ttracer_nonempty(cell_dir)
                except Exception as exc:  # noqa: BLE001
                    record["ttracer_failure"] = f"{type(exc).__name__}: {exc}"
                record["core_teardown"] = self.stop_core()
                record["finished_utc"] = _utc()
                _write_json(cell_dir / "cell_record.json", record)
            return {"record": record, "decisions": decisions, "run": run,
                    "cell_dir": cell_dir}

        def run(self) -> int:
            status, failure = "FAILED", None
            result: dict[str, Any] = {}
            evaluation: dict[str, Any] = {}
            comparison: dict[str, Any] = {}
            try:
                self.verify_identities("before_preflight")
                self.preflight_qualification()
                result = self.run_qualification()
            except Exception as exc:  # noqa: BLE001
                failure = f"{type(exc).__name__}: {exc}"
            finally:
                final_cold = self.final_cold_with_core()
                try:
                    self.verify_identities("final_sealing")
                    pins_after = RB.verify("final_sealing", ROOT)
                    verify_emitter_pins()
                    pins_after_ok = True
                except Exception as exc:  # noqa: BLE001
                    pins_after, pins_after_ok = {"error": str(exc)}, False
            record = result.get("record", {})
            decisions = result.get("decisions", [])
            run = result.get("run", {})
            if failure is None:
                failure = record.get("failure")
            if decisions and "cell_dir" in result and "ttracer" in record:
                csv_root = result["cell_dir"] / "ttracer" / "ue" / "csv"
                binding = record["ue_binding"]

                def rows(name):
                    with (csv_root / f"{name}.csv").open(newline="") as handle:
                        return list(csv.DictReader(handle))
                comparison = compare_with_raw(
                    decisions, dci_rows=rows("NRUE_MAC_DCI_GRANT"),
                    rlc_rows=rows("NRUE_MAC_RLC_BUFFER_STATUS"),
                    rnti=binding["rnti"], ue_id=binding["oai_ue_id"])
                contamination = self._contamination(decisions, Path(record["sender_csv"]))
                for seq in contamination["contaminated_decisions"]:
                    decisions[seq]["violations"] += 1
                reader_counters = run.get("provider_stop", {}).get("reader_counters", {})
                run.update({
                    "malformed_headers": sum(c.get("malformed_header", 0)
                                             for c in reader_counters.values()),
                    "unexpected_reader_eof": run.get("provider_stop", {}).get(
                        "unexpected_reader_eof_before_stop", 1),
                    "pins_before_ok": True, "pins_after_ok": pins_after_ok,
                    "rf_restored": bool(record.get("restored")),
                    "core_stopped": bool(record.get("core_teardown", {}).get("stopped")),
                    "final_cold": bool(final_cold.get("cold")),
                    "contamination": contamination,
                })
                run.setdefault("channel_state_matches_initial", False)
                run.setdefault("record_alive_after_window", False)
                evaluation = evaluate_gates(decisions=decisions,
                                            comparison=comparison, run=run)
                evaluation["ages_ns"] = {
                    "mcs_source_to_available": summary(
                        [r["mcs"]["available_raw_ns"] - r["mcs"]["source_raw_ns"]
                         for r in decisions if r["mcs"]]),
                    "backlog_source_to_available": summary(
                        [r["backlog"]["available_raw_ns"] - r["backlog"]["source_raw_ns"]
                         for r in decisions if r["backlog"]]),
                    "mcs_decision_age": summary(
                        [r["mcs"]["age_ns"] for r in decisions if r["mcs"]]),
                    "backlog_decision_age": summary(
                        [r["backlog"]["age_ns"] for r in decisions if r["backlog"]]),
                }
                if failure is None and evaluation["passed"]:
                    status = "PASSED"
            document = {
                "schema": SCHEMA, "status": status, "failure": failure,
                "gates_sha256": gates_sha256(), "gates": GATES,
                "evaluation": evaluation, "raw_comparison": comparison,
                "run": run, "final_cold_state": final_cold,
                "pins_after": pins_after, "notes": list(self.notes),
                "claim_boundary": (
                    "RADIO_ONLY_TELEMETRY_QUALIFICATION; NO CARLA, CUDA, FCOS OR "
                    "ACTOR INFERENCE; NOT A POLICY OR PERCEPTION RESULT"),
                "created_utc": _utc(),
            }
            _write_json(self.out("QUALIFICATION_RESULT.json"), document)
            inventory = [
                {"path": str(p.relative_to(self.output_dir)), "sha256": sha256_file(p)}
                for p in sorted(self.output_dir.rglob("*"))
                if p.is_file() and p.name != "TERMINAL.json"]
            _write_json(self.out("TERMINAL.json"), {
                "schema": SCHEMA, "status": status,
                "result_sha256": sha256_file(self.out("QUALIFICATION_RESULT.json")),
                "inventory_sha256": canonical_sha256(inventory),
                "files": len(inventory), "created_utc": _utc()})
            return 0 if status == "PASSED" else 1

        def _contamination(self, decisions, sender_csv: Path) -> dict[str, Any]:
            with sender_csv.open(newline="") as handle:
                frames = {int(r["frame_index"]): r for r in csv.DictReader(handle)}
            contaminated = []
            for row in decisions:
                frame = frames.get(row["decision_seq"])
                if frame is None or not row["admitted"]:
                    continue
                first_send_raw = (int(frame["first_send_monotonic_ns"])
                                  + row["host_mono_to_raw_ns"])
                row["first_send_raw_ns"] = first_send_raw
                if any(row[s] and row[s]["available_raw_ns"] >= first_send_raw
                       for s in ("mcs", "backlog")) or (
                        row["action_open_raw_ns"] >= first_send_raw):
                    contaminated.append(row["decision_seq"])
            return {"contaminated_decisions": contaminated,
                    "checked_frames": len(frames)}

    return TelemetryQualificationRunner


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_dir).resolve()
    require(output.parent == (ROOT / OUTPUT_PARENT_RELPATH).resolve(),
            f"output must be a direct child of {OUTPUT_PARENT_RELPATH}")
    require(not output.exists(), "output directory is create-only")
    from rl_agent.ue_production_queue_capture_v1 import config as CFG
    inventory = CFG.source_inventory(ROOT)
    verify_emitter_pins()
    output.mkdir(parents=False, exist_ok=False)
    _write_json(output / "AUTHORIZATION.json", {
        "schema": SCHEMA, "authorization":
            "Explicit user prompt 2026-09-29: exactly one short create-only "
            "radio-only Phase-2C qualification; 300 decisions at 10 Hz; no "
            "CARLA/CUDA/FCOS/actor.",
        "gates_sha256": gates_sha256(), "created_utc": _utc()})
    runner = build_runner_class()(output, inventory)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
