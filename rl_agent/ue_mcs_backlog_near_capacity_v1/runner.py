"""Live runner for Run 4, bound to OAI_N78_100MHZ_273PRB_4D5U_V1.

What is inherited from the qualified Run-3 runner: telemetry start, telnet
actuation, noise command/read-back, traffic launch, per-cell T-tracer
extraction, and the managed-process bookkeeping.

What is replaced, and why:

* **RAN bring-up.** Run 3 materialised a 106 PRB config and spawned the
  softmodems itself. Run 4 calls the hash-bound
  ``run_splitfusion_oai_100mhz_4d5u_v1.sh`` instead, so the radio identity is
  the launcher's, not a config block this package could drift.
* **Teardown.** The launcher leaves the RAN running by design, and its
  softmodems are root-owned and detached, so they are *not* in
  ``self.processes``. Inheriting Run-3 teardown unchanged would leak a gNB and
  a UE between cells.
* **Failure policy.** Run 3 recorded clamps, skips, restore results, teardown
  notes and extraction failures without letting any of them fail the run. Every
  one is a hard failure here.
* **Radio path proof.** Run 3 proved routing. Run 4 additionally sends real
  datagrams and requires both receiver arrival and nonzero UE PDCP evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.ue_mcs_backlog_calibration_v1 import runner as V3R
from rl_agent.ue_mcs_backlog_calibration_v1.runner import RunFailure, require, utc_now
from rl_agent.ue_mcs_backlog_near_capacity_v1 import analysis_spec as S
from rl_agent.ue_mcs_backlog_near_capacity_v1 import authorization as AUTH
from rl_agent.ue_mcs_backlog_near_capacity_v1 import capacity_qualification as CQ
from rl_agent.ue_mcs_backlog_near_capacity_v1 import contract as C
from rl_agent.ue_mcs_backlog_near_capacity_v1 import protected_evidence as PE
from rl_agent.ue_mcs_backlog_near_capacity_v1 import radio_binding as RB

n2 = V3R.n2
ROOT = V3R.ROOT
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_v1.json"

STATUS_OK = "UE_MCS_BACKLOG_NEAR_CAPACITY_CAPTURED"
STATUS_PARTIAL = "UE_MCS_BACKLOG_NEAR_CAPACITY_PARTIAL"
STATUS_FAILED = "UE_MCS_BACKLOG_NEAR_CAPACITY_FAILED"

VERIFY_STAGES = ("before_preflight", "before_scientific_cells", "final_sealing")

#: NR_PDCP_TX_SDU field list, as the extractor defines it.
PDCP_FIELDS = ("time", "mono_sec", "mono_nsec", "ue_id", "rb_id", "sdu_bytes")

DCI_GRANT_FIELDS = V3R.C.DCI_GRANT_HEADER
RLC_BUFFER_FIELDS = V3R.C.RLC_BUFFER_HEADER

def eligible_primer_grants(
    rows: Sequence[tuple[int, int, str]], *, after_receipt_ns: int,
) -> list[dict[str, int]]:
    """Parse UE-visible table-0 round-0 grants received after the primer."""
    result: list[dict[str, int]] = []
    fields = DCI_GRANT_FIELDS
    for _wall_ns, receipt_ns, line in rows:
        if receipt_ns <= after_receipt_ns:
            continue
        values = line.split(",")
        if len(values) != len(fields):
            continue
        row = dict(zip(fields, values))
        try:
            if (
                row["direction"] == V3R.C.UL_DIRECTION
                and int(row["mcs_table"]) == 0
                and int(row["round"]) == V3R.C.NEW_DATA_HARQ_ROUND
                and int(row["ndi"]) in (0, 1)
            ):
                result.append({
                    "receipt_monotonic_ns": receipt_ns,
                    "mcs": int(row["mcs"]),
                    "mcs_table": int(row["mcs_table"]),
                    "round": int(row["round"]),
                    "ndi": int(row["ndi"]),
                })
        except (KeyError, ValueError):
            continue
    return result


def trailing_zero_rlc_ticks(
    rows: Sequence[tuple[int, int, str]], *, after_receipt_ns: int,
) -> tuple[int, list[dict[str, Any]]]:
    """Return trailing complete zero-backlog UE MAC ticks.

    Rows sharing the frame/slot/tracer-time identity are one per-LCID tick.
    The newest group is excluded because its remaining LCIDs may still be in
    the live CSV pipe.
    """
    fields = RLC_BUFFER_FIELDS
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    for _wall_ns, receipt_ns, line in rows:
        if receipt_ns <= after_receipt_ns:
            continue
        values = line.split(",")
        if len(values) != len(fields):
            continue
        row = dict(zip(fields, values))
        try:
            key = (row["time"], row["frame"], row["slot"])
            if key not in grouped:
                grouped[key] = {
                    "receipt_monotonic_ns": receipt_ns,
                    "total_bytes": 0,
                    "rows": 0,
                }
                order.append(key)
            grouped[key]["receipt_monotonic_ns"] = max(
                grouped[key]["receipt_monotonic_ns"], receipt_ns
            )
            grouped[key]["total_bytes"] += int(row["bytes_in_buffer"])
            grouped[key]["rows"] += 1
        except (KeyError, ValueError):
            continue
    complete = [grouped[key] for key in order[:-1]]
    trailing = 0
    for tick in reversed(complete):
        if tick["total_bytes"] != 0:
            break
        trailing += 1
    return trailing, complete


class Runner(V3R.Runner):
    """Run-3 instrumentation, Run-4 radio, campaign plan and failure policy."""

    def __init__(self, config_path: Path, output_dir: Path) -> None:
        # Do not call the Run-3 constructor: it unconditionally reads the
        # legacy ``existing_measured_anchors`` key before this subclass can
        # replace it.  Initialise the inherited lifecycle state explicitly
        # from the target-radio config, then bind only the reviewed Phase-14a
        # 273PRB/4D5U anchors.  This avoids adding a second alias key whose two
        # copies could drift silently.
        self.config_path = config_path
        self.config = json.loads(config_path.read_text())
        self.output_dir = output_dir
        self.processes: list[n2.ManagedProcess] = []
        self.telnet = None
        self.live_pusch = None
        self.model_index: int | None = None
        self.ue_ip: str | None = None
        self.anchors = RB.anchors_for_interpolation()
        self.edge_host: str | None = None
        self.edge_pid: int | None = None
        self.aborted = False
        self.notes: list[str] = []
        self.radio_state_dir: Path | None = None
        self.verifications: list[dict[str, Any]] = []
        self.tiers: tuple[C.LoadTier, ...] = ()

    # -- identity ------------------------------------------------------
    def verify_identities(self, stage: str) -> dict[str, Any]:
        """Verify radio binding and protected evidence. Raises on drift."""
        require(stage in VERIFY_STAGES, f"unregistered verification stage {stage!r}")
        report = {
            "stage": stage, "utc": utc_now(),
            "radio_binding": RB.verify(stage, ROOT),
            "protected_evidence": PE.require_unchanged(stage, ROOT),
        }
        self.verifications.append(report)
        self.out(f"verification/{stage}.json").write_text(
            json.dumps(report, indent=2) + "\n")
        return report

    # -- cold gate with the correct ports -------------------------------
    def assert_cold_ran(self, stage: str) -> None:
        for name in ("nr-softmodem", "nr-uesoftmodem"):
            found = subprocess.run(["sudo", "-n", "pgrep", "-a", "-x", name],
                                   text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
            require(not (found.returncode == 0 and found.stdout.strip()),
                    f"{stage}: cold-RAN gate failed: {found.stdout.strip()}")
        stale = n2.oai_tunnel_interfaces()
        require(not stale, f"{stage}: stale UE tunnel(s) {stale}")
        tel = self.config["telemetry"]
        ports = [4043, self.config["actuator"]["telnet_port"],
                 tel["gnb_port"], tel["ue_port"],
                 tel["gnb_relay_port"], tel["ue_relay_port"]]
        busy = [p for p in ports if not n2.port_is_free(int(p))]
        require(not busy, f"{stage}: ports not free: {busy}")

    def preflight(self) -> dict[str, Any]:
        RB.assert_no_forbidden_env(os.environ)
        snapshot = super().preflight()
        snapshot["radio_profile_id"] = RB.RADIO_PROFILE_ID
        snapshot["forbidden_env_absent"] = True
        self.out("preflight.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        return snapshot

    # -- RAN via the qualified launcher ---------------------------------
    def start_ran_via_launcher(self, cell_tag: str, cell_dir: Path) -> dict[str, Any]:
        """Attach the 273PRB/4D5U radio through the hash-bound launcher."""
        RB.assert_no_forbidden_env(os.environ)
        launcher = ROOT / self.config["paths"]["launcher"]
        state_dir = (ROOT / self.config["paths"]["radio_state_root"]
                     / f"{self.output_dir.name}__{cell_tag}")
        PE.assert_outside_protected_run(state_dir, ROOT)
        require(not state_dir.exists(),
                f"radio state dir already exists (create-only): {state_dir}")

        argv = ["bash", str(launcher), "--execute", RB.EXECUTION_TOKEN,
                "--output", str(state_dir)]
        log = self.out(f"cells/{cell_tag}/logs/launcher.log")
        env = {k: v for k, v in os.environ.items() if k not in RB.FORBIDDEN_ENV}
        completed = subprocess.run(argv, cwd=str(ROOT), text=True, env=env,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        log.write_text(completed.stdout or "")
        require(completed.returncode == 0,
                f"launcher failed ({completed.returncode}): "
                f"{(completed.stdout or '')[-800:]}")
        require("SPLITFUSION_OAI_100MHZ_4D5U_ATTACHED" in (completed.stdout or ""),
                "launcher did not report the attach token")

        self.radio_state_dir = state_dir
        materialization = json.loads(
            (state_dir / "radio_materialization.json").read_text())
        self.ue_ip = self.config["radio"]["ue_static_ip"]
        record = {
            "launcher": self.config["paths"]["launcher"],
            "launcher_sha256": RB.PINS["launcher"]["sha256"],
            "execution_token": RB.EXECUTION_TOKEN,
            "command": " ".join(shlex.quote(a) for a in argv),
            "radio_state_dir": str(state_dir),
            "effective_gnb_path": materialization.get("effective_gnb_path"),
            "effective_ue_path": materialization.get("effective_ue_path"),
            "radio_profile_id": RB.RADIO_PROFILE_ID,
            "ue_ip": self.ue_ip,
        }
        (cell_dir / "radio_attach.json").write_text(
            json.dumps(record, indent=2) + "\n")
        return record

    def teardown_ran(self) -> list[str]:
        """Stop our processes *and* the launcher's detached, root-owned RAN."""
        notes = super().teardown_ran()
        for name in ("nr-uesoftmodem", "nr-softmodem"):
            found = subprocess.run(["sudo", "-n", "pgrep", "-x", name], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if found.returncode != 0 or not found.stdout.strip():
                continue
            subprocess.run(["sudo", "-n", "pkill", "-INT", "-x", name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                again = subprocess.run(["sudo", "-n", "pgrep", "-x", name],
                                       text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)
                if again.returncode != 0 or not again.stdout.strip():
                    break
                time.sleep(0.5)
            else:
                # OAI's UE has been observed to ignore SIGINT after a clean
                # 300-frame capture.  Give it the ordinary graceful process
                # termination signal before declaring lifecycle failure and
                # resorting to SIGKILL.  A successful SIGTERM is not a
                # teardown note; SIGKILL remains a hard evidence-gate failure.
                subprocess.run(["sudo", "-n", "pkill", "-TERM", "-x", name],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                term_deadline = time.monotonic() + 10.0
                while time.monotonic() < term_deadline:
                    again = subprocess.run(
                        ["sudo", "-n", "pgrep", "-x", name],
                        text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    if again.returncode != 0 or not again.stdout.strip():
                        break
                    time.sleep(0.5)
                else:
                    subprocess.run(
                        ["sudo", "-n", "pkill", "-KILL", "-x", name],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    notes.append(f"{name} required SIGKILL")
        stale = n2.oai_tunnel_interfaces()
        if stale:
            notes.append(f"stale UE tunnel(s) after teardown: {stale}")
        self.radio_state_dir = None
        return notes

    # -- pre-scientific UDP probe ---------------------------------------
    def udp_probe(self, cell_tag: str, cell_dir: Path) -> dict[str, Any]:
        """Send real datagrams and require arrival AND UE PDCP evidence.

        Routing proofs show where a packet *would* go. This proves one actually
        crossed the radio: a receiver inside the ext-DN namespace must see it,
        and the UE's own PDCP TX path must have produced SDU events while it was
        in flight. Either alone is insufficient -- arrival without PDCP evidence
        would be satisfied by a host-local shortcut.
        """
        traffic = self.config["traffic"]
        assert self.ue_ip is not None and self.edge_host is not None
        port = int(traffic["udp_probe_port"])
        count = int(traffic["udp_probe_datagrams"])
        size = int(traffic["udp_probe_payload_bytes"])
        timeout = float(traffic["udp_probe_timeout_s"])

        received_path = cell_dir / "udp_probe_received.json"
        sink = self.spawn(
            f"udp_probe_sink_{cell_tag}",
            ["sudo", "-n", "nsenter", "-t", str(self.edge_pid), "-n",
             sys.executable, "-c", _PROBE_SINK_SOURCE,
             str(port), str(count), f"{timeout:.3f}", str(received_path)],
            f"cells/{cell_tag}/logs/udp_probe_sink.log", root_owned=True)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not (
                cell_dir / "udp_probe_ready").exists():
            require(sink.process.poll() is None, "UDP probe sink exited early")
            time.sleep(0.1)

        # Live PDCP evidence, counted only while the probe is in flight.
        troot = ROOT / self.config["paths"]["t_tracer_dir"]
        msgs = ROOT / self.config["paths"]["t_messages"]
        live = n2.LiveCsv(
            [str(troot / "csv"), "-d", str(msgs), "-ip", "127.0.0.1",
             "-p", str(self.config["telemetry"]["ue_relay_port"]),
             "-f", "-s", ",", "-t", "time",
             "NR_PDCP_TX_SDU", *PDCP_FIELDS],
            cell_dir / "udp_probe_pdcp.csv")
        try:
            time.sleep(0.5)
            before = live.count()
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sender.bind((self.ue_ip, 0))
                payload = b"SCENESENSE_RUN4_UDP_PROBE".ljust(size, b"\0")
                for _ in range(count):
                    sender.sendto(payload, (self.edge_host, port))
                    time.sleep(0.05)
            finally:
                sender.close()
            try:
                sink.process.wait(timeout=timeout + 5.0)
            except subprocess.TimeoutExpired:
                pass
            time.sleep(1.0)
            after = live.count()
        finally:
            live.stop()

        received = 0
        if received_path.is_file():
            received = int(json.loads(received_path.read_text())["received"])
        pdcp_events = after - before

        outcome = {
            "ue_ip": self.ue_ip, "edge_host": self.edge_host, "port": port,
            "datagrams_sent": count, "datagrams_received": received,
            "pdcp_tx_sdu_events_during_probe": pdcp_events,
            "min_received_required": int(traffic["udp_probe_min_received"]),
            "pdcp_evidence_required": bool(
                traffic["udp_probe_requires_pdcp_evidence"]),
            "arrival_ok": received >= int(traffic["udp_probe_min_received"]),
            "pdcp_ok": pdcp_events > 0,
        }
        outcome["probe_verified"] = outcome["arrival_ok"] and outcome["pdcp_ok"]
        (cell_dir / "udp_probe.json").write_text(json.dumps(outcome, indent=2) + "\n")
        require(outcome["arrival_ok"],
                f"UDP probe: only {received}/{count} datagrams reached the ext-DN "
                f"receiver; traffic is not crossing the radio")
        require(outcome["pdcp_ok"],
                "UDP probe: zero NR_PDCP_TX_SDU events while datagrams were in "
                "flight; the traffic did not traverse UE PDCP/RLC")
        return outcome

    def target_channel_primer(self, cell_tag: str, cell_dir: Path) -> dict[str, Any]:
        """Create one fresh target-channel MCS, then prove the tiny queue drained.

        This executes after the profile command/warm-up and after all measured
        receivers are ready, but immediately before the tagged sender opens its
        first decision. The primer uses a separate port and is not a decision.
        """
        assert self.ue_ip is not None and self.edge_host is not None
        config = self.config["traffic"]["target_channel_primer"]
        port = int(config["port"])
        count = int(config["datagrams"])
        size = int(config["payload_bytes"])
        timeout_s = float(config["timeout_s"])
        required_zero = int(config["zero_rlc_ticks_required"])
        quiet_ns = int(float(config["ingress_quiet_ms"]) * 1e6)

        troot = ROOT / self.config["paths"]["t_tracer_dir"]
        messages = ROOT / self.config["paths"]["t_messages"]
        relay = int(self.config["telemetry"]["ue_relay_port"])
        lives = {
            "dci": n2.LiveCsv(
                [str(troot / "csv"), "-d", str(messages), "-ip", "127.0.0.1",
                 "-p", str(relay), "-f", "-s", ",", "-t", "time",
                 "NRUE_MAC_DCI_GRANT", *DCI_GRANT_FIELDS],
                cell_dir / "ttracer/ue/primer_dci_live.csv"),
            "rlc": n2.LiveCsv(
                [str(troot / "csv"), "-d", str(messages), "-ip", "127.0.0.1",
                 "-p", str(relay), "-f", "-s", ",", "-t", "time",
                 "NRUE_MAC_RLC_BUFFER_STATUS", *RLC_BUFFER_FIELDS],
                cell_dir / "ttracer/ue/primer_rlc_live.csv"),
            "pdcp": n2.LiveCsv(
                [str(troot / "csv"), "-d", str(messages), "-ip", "127.0.0.1",
                 "-p", str(relay), "-f", "-s", ",", "-t", "time",
                 "NR_PDCP_TX_SDU", *PDCP_FIELDS],
                cell_dir / "ttracer/ue/primer_pdcp_live.csv"),
        }
        ready_path = cell_dir / "target_channel_primer_ready.json"
        received_path = cell_dir / "target_channel_primer_received.json"
        sink = self.spawn(
            f"target_channel_primer_sink_{cell_tag}",
            ["sudo", "-n", "nsenter", "-t", str(self.edge_pid), "-n",
             sys.executable, "-c", _PRIMER_SINK_SOURCE,
             str(port), str(count), f"{timeout_s:.3f}", str(ready_path),
             str(received_path)],
            f"cells/{cell_tag}/logs/target_channel_primer_sink.log",
            root_owned=True,
        )
        outcome: dict[str, Any] = {
            "role": "TARGET_CHANNEL_STATE_PRIMER_NOT_A_SCIENTIFIC_DECISION",
            "port": port, "datagrams_sent": count, "payload_bytes": size,
        }
        try:
            time.sleep(0.25)
            require(all(live.process.poll() is None for live in lives.values()),
                    "a target-channel primer live extractor exited")
            ready_deadline = time.monotonic() + 5.0
            while time.monotonic() < ready_deadline and not ready_path.is_file():
                require(sink.process.poll() is None, "primer sink exited before READY")
                time.sleep(0.02)
            require(ready_path.is_file(), "target-channel primer sink did not become ready")

            send_start = time.monotonic_ns()
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sender.bind((self.ue_ip, 0))
                payload = b"SCENESENSE_RUN4_TARGET_CHANNEL_PRIMER".ljust(size, b"\0")
                for _ in range(count):
                    sender.sendto(payload, (self.edge_host, port))
                    time.sleep(0.01)
            finally:
                sender.close()
            send_end = time.monotonic_ns()

            try:
                sink.process.wait(timeout=timeout_s + 2.0)
            except subprocess.TimeoutExpired:
                raise RunFailure("target-channel primer sink timed out")
            received = 0
            if received_path.is_file():
                received = int(json.loads(received_path.read_text())["received"])
            require(received == count,
                    f"target-channel primer received {received}/{count} datagrams")

            deadline = time.monotonic() + timeout_s
            grants: list[dict[str, int]] = []
            pdcp_rows: list[tuple[int, int, str]] = []
            zero_run = 0
            retained_ticks: list[dict[str, Any]] = []
            latest_pdcp_receipt = send_start
            while time.monotonic() < deadline:
                grants = eligible_primer_grants(
                    lives["dci"].snapshot(), after_receipt_ns=send_start)
                pdcp_rows = [
                    row for row in lives["pdcp"].snapshot()
                    if row[1] > send_start
                ]
                if pdcp_rows:
                    latest_pdcp_receipt = max(row[1] for row in pdcp_rows)
                # A zero tick received before the primer reached UE PDCP says
                # nothing about whether the primer itself has drained. Count
                # only complete RLC ticks observed after the latest primer
                # ingress row. Receipt time is an early live guard only; the
                # post-extraction join applies the authoritative source-clock
                # ordering gate.
                zero_run, retained_ticks = trailing_zero_rlc_ticks(
                    lives["rlc"].snapshot(),
                    after_receipt_ns=latest_pdcp_receipt)
                quiet = time.monotonic_ns() - latest_pdcp_receipt >= quiet_ns
                if grants and pdcp_rows and zero_run >= required_zero and quiet:
                    break
                time.sleep(0.01)
            require(grants, "primer produced no fresh UE-decoded round-0 UL grant")
            require(pdcp_rows, "primer produced no UE PDCP ingress evidence")
            require(zero_run >= required_zero,
                    f"primer queue did not drain: {zero_run} trailing zero ticks")
            require(time.monotonic_ns() - latest_pdcp_receipt >= quiet_ns,
                    "primer queue has not observed its registered ingress-quiet interval")
            latest = max(grants, key=lambda row: row["receipt_monotonic_ns"])
            completed = time.monotonic_ns()
            outcome.update({
                "send_start_monotonic_ns": send_start,
                "send_end_monotonic_ns": send_end,
                "datagrams_received": received,
                "pdcp_rows_after_send": len(pdcp_rows),
                "latest_pdcp_receipt_monotonic_ns": latest_pdcp_receipt,
                "fresh_round0_grants": len(grants),
                "latest_grant": latest,
                "trailing_zero_rlc_ticks": zero_run,
                "zero_rlc_ticks_required": required_zero,
                "ingress_quiet_ms_required": float(config["ingress_quiet_ms"]),
                "retained_rlc_ticks": retained_ticks[-10:],
                "completed_monotonic_ns": completed,
                "qualified_for_sender_start": True,
            })
            (cell_dir / "target_channel_primer.json").write_text(
                json.dumps(outcome, indent=2) + "\n")
            return outcome
        finally:
            for live in lives.values():
                live.stop()

    def launch_traffic(self, *, cell_tag: str, cell_id: str,
                       blocks: Sequence[Mapping[str, Any]], cell_dir: Path,
                       total_frames: int) -> dict[str, Any]:
        """Prepare receivers, prime target-channel MCS, then start decisions."""
        traffic = self.config["traffic"]
        assert self.ue_ip is not None and self.edge_host is not None
        duration = (
            total_frames / C.FPS + float(traffic["receiver_tail_s"])
            + float(traffic["target_channel_primer"]["timeout_s"])
        )
        receivers: list[dict[str, Any]] = []
        for block in blocks:
            tag = f"block{block['block_index']}_{block['tier']}"
            events = cell_dir / f"receiver_{tag}_events.jsonl"
            summary = cell_dir / f"receiver_{tag}_summary.json"
            ready = cell_dir / f"receiver_{tag}_ready.json"
            process = self.spawn(
                f"receiver_{cell_tag}_{tag}",
                ["sudo", "-n", "nsenter", "-t", str(self.edge_pid), "-n",
                 sys.executable, str(self.path(C.PRODUCTION_RECEIVER_RELPATH)),
                 "--bind-host", "0.0.0.0", "--port", str(block["port"]),
                 "--events-jsonl", str(events), "--summary-json", str(summary),
                 "--ready-json", str(ready), "--duration-s", f"{duration:.3f}",
                 "--expected-frames", str(block["frames"]),
                 "--expected-chunks-per-frame", str(block["chunks_per_frame"]),
                 "--max-chunks-per-frame",
                 str(max(16, int(block["chunks_per_frame"]) * 2)),
                 "--socket-receive-buffer-bytes",
                 str(traffic["receive_buffer_bytes"])],
                f"cells/{cell_tag}/logs/receiver_{tag}.log", root_owned=True)
            receivers.append({
                "process": process, "ready": ready,
                "block_index": int(block["block_index"]),
                "tier": block["tier"], "events": events, "summary": summary,
            })
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if all(item["ready"].is_file() for item in receivers):
                break
            for item in receivers:
                require(item["process"].process.poll() is None,
                        f"receiver {item['tier']} exited before READY")
            time.sleep(0.2)
        require(all(item["ready"].is_file() for item in receivers),
                "not every block receiver reported READY")

        primer = self.target_channel_primer(cell_tag, cell_dir)
        plan_path = cell_dir / "block_plan.json"
        plan_path.write_text(json.dumps(list(blocks), indent=2) + "\n")
        sender_csv = cell_dir / "sender_decisions.csv"
        sender_summary = cell_dir / "sender_summary.json"
        sender = self.spawn(
            f"sender_{cell_tag}",
            [sys.executable, "-m",
             "rl_agent.ue_mcs_backlog_calibration_v1.tagged_sender",
             "--cell-id", cell_id, "--bind-host", self.ue_ip,
             "--remote-host", self.edge_host, "--block-plan", str(plan_path),
             "--payload-seed", str(self.config["campaign"]["payload_seed"]),
             "--chunk-bytes", str(C.CHUNK_BYTES), "--fps", str(C.FPS),
             "--socket-sendbuf", str(traffic["send_buffer_bytes"]),
             "--log-csv", str(sender_csv), "--summary-json", str(sender_summary)],
            f"cells/{cell_tag}/logs/sender.log")
        return {
            "receivers": receivers, "sender": sender, "frames": total_frames,
            "sender_csv": sender_csv, "target_channel_primer": primer,
        }

    def audit_primer_first_decision(
        self, sessions: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Early receipt-time gate; exact source-time age is checked offline."""
        with Path(sessions["sender_csv"]).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            first = next(reader, None)
        require(first is not None, "sender decision CSV is empty")
        first_ns = int(first["decision_monotonic_ns"])
        primer = sessions["target_channel_primer"]
        grant_receipt = int(primer["latest_grant"]["receipt_monotonic_ns"])
        completed = int(primer["completed_monotonic_ns"])
        gap_ms = (first_ns - grant_receipt) / 1e6
        completion_gap_ms = (first_ns - completed) / 1e6
        maximum = float(self.config["traffic"]["target_channel_primer"][
            "max_grant_receipt_to_first_decision_ms"])
        require(completion_gap_ms >= 0.0,
                "first scientific decision preceded primer queue-drain proof")
        require(0.0 <= gap_ms <= maximum,
                f"primer grant receipt-to-first-decision gap {gap_ms:.3f} ms "
                f"is outside [0,{maximum}] ms")
        return {
            "first_decision_monotonic_ns": first_ns,
            "latest_grant_receipt_monotonic_ns": grant_receipt,
            "grant_receipt_to_first_decision_ms": gap_ms,
            "primer_complete_to_first_decision_ms": completion_gap_ms,
            "receipt_time_gate_passed": True,
            "authoritative_source_time_gate":
                "DEFERRED_TO_POST_EXTRACTION_CAUSAL_DECISION_JOIN",
        }

    # -- one cell, with the registered failure policy --------------------
    def run_cell(self, cell: C.Cell, profile: Any) -> dict[str, Any]:
        cell_tag = f"{cell.run_index:02d}__{cell.cell_id}"
        cell_dir = self.output_dir / "cells" / cell_tag
        PE.assert_outside_protected_run(cell_dir, ROOT)
        cell_dir.mkdir(parents=True, exist_ok=False)
        command_log: list[dict[str, Any]] = []
        record: dict[str, Any] = {
            "cell_tag": cell_tag, **cell.to_json(),
            "trace_id": profile.trace_id,
            "registered_trace_sha256": profile.trace_sha256,
            "radio_profile_id": RB.RADIO_PROFILE_ID,
            "started_utc": utc_now(), "status": "FAILED",
        }
        try:
            self.assert_cold_ran(f"cell {cell_tag}")
            record["radio_attach"] = self.start_ran_via_launcher(cell_tag, cell_dir)
            record["ue_ip"] = self.ue_ip
            record["radio_path_check"] = self.verify_radio_path(cell_dir)
            self.start_telemetry(cell_tag)
            self.open_telnet(cell_dir)
            record["noise_before_cell_db"] = self.read_back_noise()
            record["udp_probe"] = self.udp_probe(cell_tag, cell_dir)

            marker = {"clean_cell_marker_utc": utc_now(),
                      "clean_cell_monotonic_ns": time.monotonic_ns(),
                      "ran_rebuilt_from_cold": True,
                      "queue_empty_by_construction": True,
                      "initial_noise_read_back_db": record["noise_before_cell_db"]}
            (cell_dir / "clean_cell_marker.json").write_text(
                json.dumps(marker, indent=2) + "\n")
            record["clean_cell_marker"] = marker

            gran = float(self.config["actuator"]["command_granularity_db"])
            first_cmd, clamped0 = V3R.inverse_interpolate(
                profile.samples[0]["target_snr_db"], self.anchors)
            require(not clamped0,
                    f"profile prime target {profile.samples[0]['target_snr_db']} dB "
                    f"is outside the registered mapping; unregistered clamp")
            first_cmd = V3R.round_to_granularity(first_cmd, gran)
            self.send_noise(first_cmd, reason="PROFILE_PRIME", log=command_log,
                            profile_id=profile.profile_id, step_index=0,
                            target_snr_db=profile.samples[0]["target_snr_db"],
                            clamped=False)
            observed = self.read_back_noise()
            record["profile_prime_command_db"] = first_cmd
            record["profile_prime_read_back_db"] = observed
            require(abs(observed - first_cmd) <= 1e-6,
                    f"profile read-back mismatch: {observed} != {first_cmd}")

            time.sleep(float(self.config["campaign"]["warmup_s"]))

            blocks = [block.to_json() for block in cell.blocks]
            sessions = self.launch_traffic(
                cell_tag=cell_tag, cell_id=cell.cell_id, blocks=blocks,
                cell_dir=cell_dir, total_frames=C.FRAMES_PER_CELL)
            record["target_channel_primer"] = sessions["target_channel_primer"]

            start = time.monotonic_ns()
            period_ns = int(float(self.config["campaign"]["sample_period_s"]) * 1e9)
            last_cmd: float | None = first_cmd
            sent = skipped = clamped = 0
            for sample in profile.samples:
                due = start + sample["step_index"] * period_ns
                now = time.monotonic_ns()
                if now > due + period_ns:
                    skipped += 1
                    continue
                if now < due:
                    time.sleep((due - now) / 1e9)
                command, was_clamped = V3R.inverse_interpolate(
                    sample["target_snr_db"], self.anchors)
                command = V3R.round_to_granularity(command, gran)
                clamped += int(was_clamped)
                if command != last_cmd:
                    self.send_noise(command, reason="PROFILE_REPLAY",
                                    log=command_log,
                                    profile_id=profile.profile_id,
                                    step_index=sample["step_index"],
                                    target_snr_db=sample["target_snr_db"],
                                    clamped=was_clamped)
                    last_cmd = command
                    sent += 1
            record.update({"commands_sent": sent, "commands_skipped": skipped,
                           "targets_clamped": clamped})
            # Run 3 logged these and continued. They are failures here.
            require(clamped == 0,
                    f"{clamped} target(s) clamped outside the registered mapping")
            require(skipped == 0,
                    f"{skipped} actuator command(s) skipped; the 100 ms replay "
                    f"schedule was not met")

            self.finish_traffic(sessions)
            record["primer_first_decision_audit"] = (
                self.audit_primer_first_decision(sessions))
            record["traffic"] = self.audit_traffic(sessions, cell, cell_dir)
            record["noise_after_cell_db"] = self.read_back_noise()
            record["restored"] = self.restore_clean(cell_dir, command_log)
            require(record["restored"],
                    "RF restore/read-back failed; refusing to mark the cell captured")
            record["status"] = "CAPTURED"
        except Exception as exc:  # noqa: BLE001 - evidence must be preserved
            record["failure"] = f"{type(exc).__name__}: {exc}"
            if self.telnet is not None:
                try:
                    record["restored"] = self.restore_clean(cell_dir, command_log)
                except Exception as inner:  # noqa: BLE001
                    record["restore_failure"] = f"{type(inner).__name__}: {inner}"
        finally:
            notes = self.teardown_ran()
            record["teardown_notes"] = notes
            (cell_dir / "command_log.json").write_text(
                json.dumps(command_log, indent=2) + "\n")
            try:
                self.extract_ttracer(cell_tag, cell_dir)
                record["extraction_ok"] = True
            except Exception as exc:  # noqa: BLE001
                record["extraction_ok"] = False
                record["extraction_failure"] = f"{type(exc).__name__}: {exc}"
            if record["status"] == "CAPTURED":
                if notes:
                    record["status"] = "FAILED"
                    record["failure"] = f"teardown notes present: {notes}"
                elif not record.get("extraction_ok"):
                    record["status"] = "FAILED"
                    record["failure"] = record.get(
                        "extraction_failure", "T-tracer extraction failed")
            record["finished_utc"] = utc_now()
            (cell_dir / "cell_record.json").write_text(
                json.dumps(record, indent=2) + "\n")
        return record

    def audit_traffic(self, sessions: Mapping[str, Any], cell: C.Cell,
                      cell_dir: Path) -> dict[str, Any]:
        """Exact sender/receiver/chunk accounting. Any shortfall fails the cell."""
        summary_path = cell_dir / "sender_summary.json"
        require(summary_path.is_file(), "sender wrote no summary")
        sender = json.loads(summary_path.read_text())
        frames = int(sender.get("frames_sent", sender.get("frames", -1)))
        require(frames == C.FRAMES_PER_CELL,
                f"sender emitted {frames} frames, expected {C.FRAMES_PER_CELL}")
        require(int(sender.get("socket_errors", 0)) == 0,
                f"sender reported {sender.get('socket_errors')} socket error(s)")

        receivers = []
        for item in sessions["receivers"]:
            require(item["summary"].is_file(),
                    f"receiver {item['tier']} wrote no summary")
            data = json.loads(item["summary"].read_text())
            receivers.append({"tier": item["tier"],
                              "block_index": item["block_index"], **data})
            require(item["process"].process.poll() == 0
                    or item["process"].process.poll() is None,
                    f"receiver {item['tier']} exited nonzero")

        chunks_sent = int(sender.get("chunks_sent", 0))
        expected_chunks = sum(b.chunks_per_frame * b.frames for b in cell.blocks)
        outcome = {"sender": sender, "receivers": receivers,
                   "expected_chunks": expected_chunks,
                   "chunks_sent": chunks_sent,
                   "chunk_accounting_exact": chunks_sent == expected_chunks}
        (cell_dir / "traffic_audit.json").write_text(
            json.dumps(outcome, indent=2) + "\n")
        require(outcome["chunk_accounting_exact"],
                f"chunk accounting: sent {chunks_sent}, expected {expected_chunks}")
        return outcome

    # -- manifest -------------------------------------------------------
    def manifest(self, status: str, extra: Mapping[str, Any]) -> None:
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                files.append({"relative_path": str(path.relative_to(self.output_dir)),
                              "size_bytes": path.stat().st_size,
                              "sha256": n2.sha256(path)})
        self.out("manifest.json").write_text(json.dumps({
            "schema": self.config["schema"],
            "contract_id": C.CONTRACT_ID,
            "contract_version": C.CONTRACT_VERSION,
            "claim_boundary": C.CLAIM_BOUNDARY,
            "radio_profile_id": RB.RADIO_PROFILE_ID,
            "status": status, "utc": utc_now(),
            "config_sha256": n2.sha256(self.config_path),
            "source_hashes": C.resolved_source_hashes(ROOT),
            "identity_verifications": self.verifications,
            "mcs_max_age_ms": S.MCS_MAX_AGE_MS,
            "clock_bridge_max_residual_p95_us": S.CLOCK_BRIDGE_MAX_RESIDUAL_P95_US,
            **dict(extra), "files": files,
        }, indent=2) + "\n")

    # -- campaign -------------------------------------------------------
    def run(self, *, capacity_result: Mapping[str, Any] | None = None) -> int:
        import signal

        status = STATUS_FAILED
        failure: str | None = None
        cell_records: list[dict[str, Any]] = []
        plan_audit: dict[str, Any] = {}
        cold: dict[str, Any] = {}

        def terminate(signum: int, _frame: Any) -> None:
            self.aborted = True
            raise RunFailure(f"received signal {signum}")

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, terminate)

        try:
            self.verify_identities("before_preflight")

            require(capacity_result is not None,
                    "no capacity-qualification result was supplied; the tiers are "
                    "not frozen in the contract and cannot be guessed")
            require(bool(capacity_result.get("binding_verified")),
                    "capacity result was not opened through its manifest and "
                    "terminal seals")
            capacity = float(capacity_result["adverse_capacity_mbps"])
            require(bool(capacity_result.get("qualified")),
                    f"capacity qualification did not pass its gates: "
                    f"{capacity_result.get('problems')}")
            selection = CQ.select_tiers(capacity, repo_root=ROOT)
            self.tiers = C.load_tiers_from_selection(selection)

            profiles = {p.profile_id: p for p in
                        C.resolve_profiles(ROOT, C.FRAMES_PER_CELL)}
            plan = C.build_cell_plan(
                self.tiers, ports=self.config["traffic"]["ports"],
                seed=int(self.config["campaign"]["cell_order_seed"]))
            plan_audit = C.audit_cell_plan(plan)

            self.preflight()
            require(C.plan_is_registered_design(plan_audit),
                    f"cell plan is not the registered balanced design: {plan_audit}")
            self.out("plan.json").write_text(json.dumps({
                "contract_id": C.CONTRACT_ID,
                "claim_boundary": C.CLAIM_BOUNDARY,
                "radio_profile_id": RB.RADIO_PROFILE_ID,
                "design": {
                    "permutations": [list(o) for o in C.PERMUTATIONS],
                    "partitions": list(C.PARTITIONS),
                    "contrast_profiles": list(C.CONTRAST_PROFILE_IDS),
                    "frames_per_block": C.FRAMES_PER_BLOCK,
                    "frames_per_cell": C.FRAMES_PER_CELL,
                    "expected_cells": C.EXPECTED_CELLS,
                    "expected_decisions": C.EXPECTED_DECISIONS,
                },
                "plan_audit": plan_audit,
                "cells": [c.to_json() for c in plan],
                "tiers": [t.to_json() for t in self.tiers],
                "capacity_qualification": dict(capacity_result),
                "tier_rule": {
                    "target_ratios": dict(CQ.TIER_TARGET_RATIOS),
                    "medium_boundary_tolerance": CQ.MEDIUM_BOUNDARY_TOLERANCE,
                    "authority": C.ACTION_SELECTION_AUTHORITY,
                    "actions_frozen_in_contract": C.ACTIONS_FROZEN_IN_CONTRACT,
                },
                "actuator_mapping": {
                    "source": self.config["actuator"]["mapping_source"],
                    "radio_profile_id": RB.RADIO_PROFILE_ID,
                    "anchors": RB.anchors_for_interpolation(),
                    "legacy_106prb_anchors_reused": False,
                    "qualification_note": RB.MAPPING_QUALIFICATION_NOTE,
                },
                "cell_order_seed": self.config["campaign"]["cell_order_seed"],
                "source_hashes": C.resolved_source_hashes(ROOT),
                "oai_citations": dict(C.OAI_CITATIONS),
            }, indent=2) + "\n")

            self.verify_identities("before_scientific_cells")

            for cell in plan:
                if self.aborted:
                    self.notes.append(f"campaign aborted before {cell.cell_id}")
                    break
                cell_records.append(self.run_cell(cell, profiles[cell.profile_id]))
            require(not self.aborted, "campaign was aborted by signal")

            captured = [r for r in cell_records if r["status"] == "CAPTURED"]
            require(len(captured) == len(plan),
                    f"{len(captured)}/{len(plan)} cells captured")
            status = STATUS_OK
        except Exception as exc:  # noqa: BLE001
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            self.notes.extend(self.teardown_ran())
            cold = self.final_cold_state()
            if not cold.get("cold"):
                failure = failure or f"final state is not cold: {cold}"
                status = STATUS_FAILED
            if self.notes:
                failure = failure or f"teardown notes present: {self.notes}"
                status = STATUS_FAILED
            try:
                self.verify_identities("final_sealing")
            except Exception as exc:  # noqa: BLE001
                failure = failure or f"{type(exc).__name__}: {exc}"
                status = STATUS_FAILED
            if status == STATUS_OK and failure:
                status = STATUS_FAILED
            self.manifest(status, {
                "failure": failure, "notes": self.notes,
                "plan_audit": plan_audit, "cells": cell_records,
                "final_cold_state": cold,
                "capacity_qualification": dict(capacity_result or {}),
                "tiers": [t.to_json() for t in self.tiers],
            })
        print(json.dumps({
            "status": status, "failure": failure,
            "cells_captured": sum(1 for r in cell_records
                                  if r["status"] == "CAPTURED"),
            "cells_planned": C.EXPECTED_CELLS,
            "output_dir": str(self.output_dir),
            "cold": cold.get("cold"),
            "radio_profile_id": RB.RADIO_PROFILE_ID,
        }, indent=2))
        return 0 if status == STATUS_OK else 1


_PRIMER_SINK_SOURCE = """
import json, socket, sys, time
from pathlib import Path
port = int(sys.argv[1])
count = int(sys.argv[2])
timeout = float(sys.argv[3])
ready = Path(sys.argv[4])
out = Path(sys.argv[5])
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", port))
s.settimeout(0.2)
ready.write_text(json.dumps({"ready": True, "port": port}) + "\\n")
received, deadline = 0, time.monotonic() + timeout
while time.monotonic() < deadline and received < count:
    try:
        s.recvfrom(65535)
        received += 1
    except socket.timeout:
        continue
out.write_text(json.dumps({"received": received, "expected": count}) + "\\n")
"""

#: Runs inside the ext-DN namespace. Kept inline so the probe needs no extra
#: file on the edge mount, and writes a ready marker so the sender never races
#: an unbound socket.
_PROBE_SINK_SOURCE = """
import json, socket, sys, time
from pathlib import Path
port, count, timeout, out = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", port))
s.settimeout(0.5)
(out.parent / "udp_probe_ready").write_text("ready\\n")
received, deadline = 0, time.monotonic() + timeout
while time.monotonic() < deadline and received < count:
    try:
        s.recvfrom(65535)
        received += 1
    except socket.timeout:
        continue
out.write_text(json.dumps({"received": received, "expected": count}) + "\\n")
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--capacity-result", type=Path, default=None,
                        help="qualified capacity-stage result JSON")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())

    if args.output_dir is not None:
        output = args.output_dir
        output_root = output.parent
    else:
        output_root = ROOT / config["paths"]["output_root"]
        output = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")

    PE.assert_outside_protected_run(output, ROOT)
    PE.require_unchanged("before_mkdir", ROOT)
    RB.verify("before_mkdir", ROOT)
    # Authorized once, BEFORE the attempt directory exists. Re-checking after
    # mkdir would count this run as a prior attempt and refuse itself.
    authorization = AUTH.require_authorization(
        AUTH.SCIENTIFIC_STAGE, output_root,
        expected_token=config["authorization"]["scientific_stage_token"],
        repo_root=ROOT)

    capacity_result = None
    if args.capacity_result is not None:
        # Lazy import avoids a module cycle: capacity_runner reuses this
        # class's proven radio lifecycle, while this scientific entry point
        # consumes only its cryptographic result verifier.
        from rl_agent.ue_mcs_backlog_near_capacity_v1.capacity_runner import (
            verify_bound_capacity_result,
        )
        capacity_result = verify_bound_capacity_result(args.capacity_result)

    output.mkdir(parents=True, exist_ok=False)
    (output / "lineage.json").write_text(json.dumps(AUTH.lineage_record(
        AUTH.SCIENTIFIC_STAGE, run_id=output.name,
        parent=(capacity_result or {}).get("lineage"),
        authorization=authorization), indent=2) + "\n")
    return Runner(args.config, output).run(capacity_result=capacity_result)


if __name__ == "__main__":
    raise SystemExit(main())
