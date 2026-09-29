#!/usr/bin/env python3
"""Offline tests for the production-domain queue capture package.

No test launches a radio, a container or a live capture.
"""

from __future__ import annotations

import csv
import hashlib
import json
import socket
import struct
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

from rl_agent.ue_production_queue_capture_v1 import config as CFG
from rl_agent.ue_production_queue_capture_v1 import contract as C
from rl_agent.ue_production_queue_capture_v1 import payload_schedule as PS
from rl_agent.ue_production_queue_capture_v1 import production_receiver as PR
from rl_agent.ue_production_queue_capture_v1 import production_sender as PSND


class ContractTests(unittest.TestCase):
    def test_authorities_pinned(self) -> None:
        self.assertTrue(C.verify_authorities()["verified"])

    def test_production_packetization_is_the_live_binding(self) -> None:
        self.assertEqual(C.UDP_CHUNK_BYTES_INCLUDING_HEADER, 12_500)
        self.assertEqual(C.UDP_CHUNK_HEADER_BYTES, 8)
        self.assertEqual(C.UDP_PAYLOAD_BYTES_PER_DATAGRAM, 12_492)
        self.assertIs(C.RETRANSMISSION, False)
        self.assertEqual(C.UDP_CHUNK_HEADER_STRUCT, "!IHH")
        self.assertEqual(struct.Struct(C.UDP_CHUNK_HEADER_STRUCT).size, 8)
        self.assertNotEqual(C.UDP_CHUNK_BYTES_INCLUDING_HEADER,
                            C.HISTORICAL_CHUNK_BYTES_REJECTED)

    def test_ipv4_fragmentation_is_expected_not_assumed_away(self) -> None:
        self.assertTrue(C.IPV4_FRAGMENTATION_EXPECTED)
        self.assertGreater(C.FULL_IPV4_PACKET_BYTES, C.PATH_MTU_BYTES)
        self.assertEqual(C.FRAGMENTS_PER_FULL_DATAGRAM, 9)

    def test_datagram_accounting_boundary(self) -> None:
        self.assertEqual(C.datagram_count(12_492), 1)
        self.assertEqual(C.datagram_count(12_493), 2)
        self.assertEqual(C.udp_application_bytes(12_492), 12_500)
        self.assertEqual(C.udp_application_bytes(12_493), 12_509)
        with self.assertRaises(C.ContractError):
            C.datagram_count(0)

    def test_byte_roles_reproduce_registered_action_arithmetic(self) -> None:
        for spec in C.TIERS:
            self.assertEqual(C.action_id_for(spec.mode_id, spec.q_e4),
                             spec.action_id)
        self.assertEqual([spec.action_id for spec in C.TIERS], [71, 39, 38])

    def test_guard_is_outside_modeled_support_and_marked(self) -> None:
        guard = C.tier_by_name("guard")
        self.assertFalse(guard.inside_run4_modeled_payload_support)
        self.assertGreater(guard.median_total_transmitted_bytes,
                           C.RUN4_MODELED_PAYLOAD_SUPPORT_BYTES[1])
        self.assertTrue(C.tier_by_name("floor")
                        .inside_run4_modeled_payload_support)
        self.assertTrue(C.tier_by_name("knee")
                        .inside_run4_modeled_payload_support)

    def test_cell_plan_is_balanced_and_partitioned(self) -> None:
        cells = C.planned_cells()
        self.assertEqual(len(cells), C.EXPECTED_CELLS)
        self.assertEqual(sum(1 for c in cells if c.partition == C.FIT), 6)
        self.assertEqual(
            sum(1 for c in cells if c.partition == C.VALIDATION), 6)
        for profile in C.PROFILES:
            self.assertEqual(
                sum(1 for c in cells if c.profile_id == profile), 6)
        # disjoint permutations and disjoint scene splits
        fit_orders = {tuple(b.tier for b in c.blocks)
                      for c in cells if c.partition == C.FIT}
        val_orders = {tuple(b.tier for b in c.blocks)
                      for c in cells if c.partition == C.VALIDATION}
        self.assertEqual(fit_orders & val_orders, set())
        self.assertEqual(
            {c.scene_split for c in cells if c.partition == C.FIT},
            {C.FIT_SCENE_SPLIT})
        self.assertEqual(
            {c.scene_split for c in cells if c.partition == C.VALIDATION},
            {C.VALIDATION_SCENE_SPLIT})
        # every tier appears in every block position across the campaign
        for position in range(C.BLOCKS_PER_CELL):
            self.assertEqual(
                {c.blocks[position].tier for c in cells}, set(C.TIER_NAMES))

    def test_cycle_structure(self) -> None:
        cycles = C.primary_cycle_indices()
        self.assertEqual(len(cycles), C.PRIMARY_CYCLES_PER_CELL)
        self.assertEqual(len(cycles) * C.EXPECTED_CELLS,
                         C.EXPECTED_PRIMARY_CYCLES)
        self.assertEqual(C.unclosed_decision_indices(), (448,))
        for start, held, successor in cycles:
            self.assertEqual(held, start + 1)
            self.assertEqual(successor, start + 2)
            self.assertLess(successor, C.FRAMES_PER_CELL)
        # blocks are an even number of frames, so a cycle never straddles a
        # tier change in its own two frames
        self.assertEqual(C.FRAMES_PER_BLOCK % C.DURATION_STEPS, 0)

    def test_queue_conservation_is_exact_and_clamped_at_zero(self) -> None:
        self.assertEqual(C.queue_next_backlog(100, 50, 30), 120)
        self.assertEqual(C.queue_next_backlog(10, 0, 999), 0)
        with self.assertRaises(C.ContractError):
            C.queue_next_backlog(10, 5, 1.5)  # type: ignore[arg-type]

    def test_profile_and_audit_fields_are_never_model_inputs(self) -> None:
        for field in ("profile_id", "target_snr_db", "gnb_mcs",
                      "post_decision_ul_mcs"):
            self.assertIn(field, C.AUDIT_ONLY_NOT_MODEL_INPUT_FIELDS)
            self.assertNotIn(field, C.PRIMARY_MODEL_INPUT_FIELDS)
        self.assertEqual(set(C.PRIMARY_MODEL_INPUT_FIELDS)
                         & set(C.AUDIT_ONLY_NOT_MODEL_INPUT_FIELDS), set())

    def test_latency_replacement_is_disjoint_not_additive(self) -> None:
        self.assertTrue(C.ADDING_BOTH_COMPONENTS_IS_FORBIDDEN)
        self.assertEqual(C.REPLACED_288_COMPONENT,
                         "application_feature_uplink_ms")
        self.assertEqual(C.SHARED_ENDPOINT, "LAST_UDP_SOCKET_HANDOFF")
        self.assertIn(C.SHARED_ENDPOINT, C.UE_ACTION_PATH_BOUNDARY)
        self.assertIn(C.SHARED_ENDPOINT, C.PRODUCTION_TRANSPORT_BOUNDARY)

    def test_anti_bias_rule_is_frozen(self) -> None:
        self.assertTrue(C.INCOMPLETE_BY_DEADLINE_IS_FAILURE)
        self.assertTrue(C.SURVIVOR_ONLY_LATENCY_FITTING_IS_FORBIDDEN)
        self.assertTrue(C.INFRASTRUCTURE_FAULT_IS_EXCLUDED_NOT_CHARGED)
        self.assertEqual(C.REGISTERED_FAILURE_REWARD, -1.0)
        self.assertEqual(C.REWARD_DEADLINE_MS, 170.0)
        self.assertEqual(C.REWARD_LATENCY_WEIGHT, 0.25)
        self.assertEqual(C.BOUNDARY_INVERSION_POLICY, "EXCLUDED_NOT_CLAMPED")
        self.assertEqual(
            set(C.TERMINAL_OUTCOMES) - set(C.POLICY_CHARGEABLE_OUTCOMES),
            {"EXCLUDED_INFRASTRUCTURE_FAULT"})

    def test_gate_set_is_complete(self) -> None:
        self.assertEqual([gate.number for gate in C.GATES],
                         list(range(1, 11)))


class DisjointnessEvidenceTests(unittest.TestCase):
    """The replacement boundary is proven on retained evidence, not asserted."""

    def test_retained_evidence_satisfies_the_identity_exactly(self) -> None:
        root = C.ROOT / C.DISJOINTNESS_EVIDENCE_RELPATH
        self.assertTrue(root.is_dir(), f"missing retained evidence: {root}")
        files = sorted(root.glob("action_*.csv"))
        self.assertGreaterEqual(len(files), 3)
        checked = 0
        for path in files:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    try:
                        uplink = float(row["application_feature_uplink_ms"])
                        send = float(row["ue_send_loop_ms"])
                        post = float(row["post_send_to_reassembly_ms"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    self.assertAlmostEqual(uplink, send + post, places=9)
                    checked += 1
        self.assertGreater(checked, 500)


class ConfigTests(unittest.TestCase):
    def test_config_matches_the_frozen_contract(self) -> None:
        value = CFG.load_config()
        self.assertEqual(value["package_id"], C.PACKAGE_ID)
        self.assertTrue(value["packetization"]["historical_60kb_binding_rejected"])
        self.assertFalse(value["byte_roles"]["perception_endorsement"])
        self.assertEqual(value["byte_roles"]["catalogue_contract_tier"],
                         "EMERGENCY_ONLY")

    def test_effective_runtime_binds_the_273prb_radio(self) -> None:
        runtime = CFG.effective_runtime_config()
        self.assertEqual(runtime["radio"]["profile_id"],
                         "OAI_N78_100MHZ_273PRB_4D5U_V1")
        self.assertEqual(runtime["radio"]["prb"], 273)
        self.assertEqual(set(runtime["traffic"]["ports"]), set(C.TIER_NAMES))

    def test_source_inventory_covers_every_behavioural_file(self) -> None:
        inventory = CFG.source_inventory()
        for relative in CFG.INVENTORY_FILES + CFG.INHERITED_SOURCES:
            self.assertIn(relative, inventory["files"])

    def test_inherited_sources_are_committed_and_clean(self) -> None:
        for relative in CFG.INHERITED_SOURCES:
            completed = subprocess.run(
                ["git", "status", "--porcelain", "--", relative],
                cwd=str(C.ROOT), capture_output=True, text=True, check=True)
            self.assertEqual(completed.stdout.strip(), "",
                             f"inherited source is dirty: {relative}")


class PayloadScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = PS._open_authority(C.ROOT)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()

    def test_schedule_is_deterministic(self) -> None:
        cell = C.planned_cells()[0]
        first = PS.build_cell_schedule(cell, connection=self.connection)
        second = PS.build_cell_schedule(cell, connection=self.connection)
        self.assertEqual([f.to_json() for f in first],
                         [f.to_json() for f in second])

    def test_schedule_is_natural_and_nonconstant(self) -> None:
        for cell in C.planned_cells():
            frames = PS.build_cell_schedule(cell, connection=self.connection)
            self.assertEqual(len(frames), C.FRAMES_PER_CELL)
            payloads = [f.total_transmitted_bytes for f in frames]
            self.assertGreater(len(set(payloads)), 300)
            for frame in frames:
                self.assertEqual(frame.grid_split, cell.scene_split)
                self.assertEqual(C.datagram_count(frame.total_transmitted_bytes),
                                 frame.datagram_count)

    def test_fit_and_validation_scene_rows_are_disjoint(self) -> None:
        fit_rows: set[str] = set()
        validation_rows: set[str] = set()
        for cell in C.planned_cells():
            frames = PS.build_cell_schedule(cell, connection=self.connection)
            target = fit_rows if cell.partition == C.FIT else validation_rows
            target.update(frame.row_sha256 for frame in frames)
        self.assertEqual(fit_rows & validation_rows, set())

    def test_blocks_carry_their_registered_action_identity(self) -> None:
        cell = C.planned_cells()[0]
        frames = PS.build_cell_schedule(cell, connection=self.connection)
        for frame in frames:
            spec = C.tier_by_name(frame.tier)
            self.assertEqual(frame.action_id, spec.action_id)
            self.assertEqual(frame.mode_id, spec.mode_id)
            self.assertEqual(frame.q_e4, spec.q_e4)


class WireTests(unittest.TestCase):
    def test_chunking_matches_the_deployed_header_and_accounting(self) -> None:
        for total in (1, 12_491, 12_492, 12_493, 374_531, 619_825):
            payload = PSND.build_frame_bytes(total, hashlib.sha256(
                str(total).encode()).hexdigest())
            self.assertEqual(len(payload), total)
            chunks = PSND.chunk_frame(payload, message_id=7)
            self.assertEqual(len(chunks), C.datagram_count(total))
            self.assertEqual(
                sum(len(chunk) for chunk in chunks),
                C.udp_application_bytes(total))
            for chunk in chunks:
                self.assertLessEqual(
                    len(chunk), C.UDP_CHUNK_BYTES_INCLUDING_HEADER)
            rebuilt = b"".join(chunk[8:] for chunk in chunks)
            self.assertEqual(rebuilt, payload)
            message_id, index, count = PSND.CHUNK_HEADER.unpack_from(chunks[0])
            self.assertEqual((message_id, index, count),
                             (7, 0, len(chunks)))

    def test_frame_bytes_are_reproducible_from_the_row_digest(self) -> None:
        digest = hashlib.sha256(b"row").hexdigest()
        self.assertEqual(PSND.build_frame_bytes(5_000, digest),
                         PSND.build_frame_bytes(5_000, digest))
        self.assertNotEqual(
            PSND.build_frame_bytes(5_000, digest),
            PSND.build_frame_bytes(5_000, hashlib.sha256(b"other").hexdigest()))


class ReceiverTests(unittest.TestCase):
    def test_incomplete_frame_can_never_be_reported_complete(self) -> None:
        """Drop one datagram of a multi-datagram frame; it must stay incomplete."""
        port = _free_udp_port()
        out = Path(_scratch())
        log = out / "rx.csv"
        summary = out / "rx.json"
        ready = out / "rx_ready.json"
        args = PR.build_parser().parse_args([
            "--cell-id", "unit", "--bind-host", "127.0.0.1",
            "--bind-port", str(port), "--log-csv", str(log),
            "--summary-json", str(summary), "--ready-json", str(ready),
            "--idle-timeout-s", "1.0", "--initial-idle-timeout-s", "8.0",
        ])
        thread = threading.Thread(target=PR.run, args=(args,), daemon=True)
        thread.start()
        deadline = time.monotonic() + 8.0
        while not ready.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(ready.is_file(), "receiver never reported READY")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        complete = PSND.chunk_frame(b"a" * 30_000, message_id=1)
        partial = PSND.chunk_frame(b"b" * 30_000, message_id=2)
        for chunk in complete:
            sock.sendto(chunk, ("127.0.0.1", port))
        for chunk in partial[:-1]:            # deliberately drop the last one
            sock.sendto(chunk, ("127.0.0.1", port))
        sock.close()
        thread.join(timeout=20.0)
        self.assertFalse(thread.is_alive())

        rows = {int(r["message_id"]): r
                for r in csv.DictReader(log.open(newline=""))}
        self.assertEqual(rows[1]["complete"], "True")
        self.assertEqual(rows[2]["complete"], "False")
        self.assertEqual(rows[2]["terminal_reason"], "INCOMPLETE_REASSEMBLY")
        self.assertEqual(rows[2]["complete_reassembly_monotonic_ns"], "")
        self.assertEqual(int(rows[2]["datagrams_received"]),
                         len(partial) - 1)
        value = json.loads(summary.read_text())
        self.assertEqual(value["messages_complete"], 1)
        self.assertEqual(value["messages_incomplete"], 1)
        self.assertEqual(value["malformed_datagrams"], 0)
        # every observed message produces exactly one terminal row
        self.assertEqual(value["messages_observed"], len(rows))

    def test_ip_counter_snapshot_parses(self) -> None:
        counters = PR.read_ip_counters()
        for key in ("ReasmReqds", "ReasmOKs", "FragCreates"):
            self.assertIn(key, counters)
            self.assertIsInstance(counters[key], int)


_SCRATCH: list[str] = []


def _scratch() -> str:
    import tempfile
    path = tempfile.mkdtemp(prefix="pqc_test_")
    _SCRATCH.append(path)
    return path


def _free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class NoLiveLaunchTests(unittest.TestCase):
    def test_runner_import_has_no_side_effects(self) -> None:
        from rl_agent.ue_production_queue_capture_v1 import runner
        self.assertTrue(hasattr(runner, "Runner"))
        self.assertEqual(runner.C.CONTRACT_SHA256, C.CONTRACT_SHA256)

    def test_run_requires_authorization(self) -> None:
        from rl_agent.ue_production_queue_capture_v1 import runner
        parser = runner.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["run", "--output-dir", "/tmp/nope"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TerminalClassificationTests(unittest.TestCase):
    """The anti-bias rule, enforced at the single place outcomes are assigned."""

    def test_incomplete_transport_can_never_be_success(self) -> None:
        from rl_agent.ue_production_queue_capture_v1 import parse as PARSE
        self.assertEqual(
            PARSE.classify_terminal(complete=False, datagrams_received=29,
                                    transport_ns=None),
            "INCOMPLETE_AT_DEADLINE")
        self.assertEqual(
            PARSE.classify_terminal(complete=False, datagrams_received=0,
                                    transport_ns=None),
            "NEVER_COMPLETED")
        for outcome in ("INCOMPLETE_AT_DEADLINE", "NEVER_COMPLETED"):
            self.assertNotIn(outcome, C.SUCCESS_OUTCOMES)
            self.assertIn(outcome, C.POLICY_CHARGEABLE_OUTCOMES)

    def test_late_completion_is_a_failure_not_a_slow_success(self) -> None:
        from rl_agent.ue_production_queue_capture_v1 import parse as PARSE
        self.assertEqual(
            PARSE.classify_terminal(complete=True, datagrams_received=30,
                                    transport_ns=C.REWARD_DEADLINE_NS),
            "COMPLETE_WITHIN_DEADLINE")
        self.assertEqual(
            PARSE.classify_terminal(complete=True, datagrams_received=30,
                                    transport_ns=C.REWARD_DEADLINE_NS + 1),
            "COMPLETE_AFTER_DEADLINE")
        self.assertNotIn("COMPLETE_AFTER_DEADLINE", C.SUCCESS_OUTCOMES)

    def test_boundary_inversion_is_excluded_not_clamped(self) -> None:
        from rl_agent.ue_production_queue_capture_v1 import parse as PARSE
        for value in (0, -1, -190_000):
            self.assertEqual(
                PARSE.classify_terminal(complete=True, datagrams_received=1,
                                        transport_ns=value),
                "EXCLUDED_INFRASTRUCTURE_FAULT")


class RateModelTests(unittest.TestCase):
    def _rows(self, partition: str, rate_bps: float, n: int = 60):
        rows = []
        for index in range(n):
            backlog = 1_000 * (index + 1)
            payload = 12_500
            latency_ns = int((backlog + payload) / rate_bps * 1e9)
            rows.append({
                "partition": partition, "tier_audit_only": "knee",
                "udp_application_bytes": payload,
                "pre_action_rlc_backlog_bytes": backlog,
                "prior_ul_mcs": 20,
                "transport_latency_ns": latency_ns,
                "terminal_outcome": "COMPLETE_WITHIN_DEADLINE",
                "complete": True,
            })
        return rows

    def test_rate_recovers_a_known_rate_and_is_monotone(self) -> None:
        from rl_agent.ue_production_queue_capture_v1 import analysis as A
        model = A.fit_rate_model(self._rows(C.FIT, 20e6))
        self.assertAlmostEqual(model.service_rate_bps(20), 20e6, delta=1e4)
        # more bytes -> not faster; more backlog -> not faster
        base, _ = model.predict_latency_ns(
            backlog=5_000, bytes_on_wire=12_500, mcs=20)
        more_bytes, _ = model.predict_latency_ns(
            backlog=5_000, bytes_on_wire=600_000, mcs=20)
        more_backlog, _ = model.predict_latency_ns(
            backlog=500_000, bytes_on_wire=12_500, mcs=20)
        self.assertGreater(more_bytes, base)
        self.assertGreater(more_backlog, base)

    def test_next_backlog_is_clamped_at_zero(self) -> None:
        from rl_agent.ue_production_queue_capture_v1 import analysis as A
        model = A.fit_rate_model(self._rows(C.FIT, 20e6))
        value, _ = model.predict_next_backlog(
            backlog=10.0, ingress_bytes=0, mcs=20)
        self.assertEqual(value, 0.0)

    def test_fit_ignores_rows_without_a_measurable_rate(self) -> None:
        from rl_agent.ue_production_queue_capture_v1 import analysis as A
        rows = self._rows(C.FIT, 20e6)
        rows.append({
            "partition": C.FIT, "tier_audit_only": "guard",
            "udp_application_bytes": 620_000,
            "pre_action_rlc_backlog_bytes": 1_000,
            "prior_ul_mcs": 20, "transport_latency_ns": None,
            "terminal_outcome": "NEVER_COMPLETED", "complete": False,
        })
        model = A.fit_rate_model(rows)
        self.assertEqual(model.support["fit_rows_used"], len(rows) - 1)
        self.assertFalse(A.usable_for_rate(rows[-1]))


class TransportCompositionTests(unittest.TestCase):
    """The 288 uplink term is replaced, never added."""

    def _model_document(self) -> dict:
        return {
            "schema": "scenesense.production_transport_model.v1",
            "contract_sha256": C.CONTRACT_SHA256,
            "all_gates_passed": True,
            "replaces_288_component": C.REPLACED_288_COMPONENT,
            "adding_both_components_is_forbidden": True,
            "global_median_rate_bps": 20e6,
            "min_bin_support": 20,
            "bins": {"()": {"count": 500, "median_rate_bps": 20e6}},
            "binning": {
                "backlog_edges": [0.0, 1.0, 1e3, 1e4, 1e5, 1e6, 1e7, None],
                "mcs_edges": [0, 4, 8, 12, 16, 20, 24, None],
            },
            "measured_support": {
                "min_udp_application_bytes": 6_000,
                "max_udp_application_bytes": 700_000,
                "min_pre_action_backlog_bytes": 0.0,
                "max_pre_action_backlog_bytes": 5e7,
                "fit_rows_used": 500,
            },
        }

    def test_refuses_foreign_or_ungated_model(self) -> None:
        from rl_agent.ue_production_queue_capture_v1 import transport_model as TM
        document = self._model_document()
        TM.ProductionTransportModelV1(document)          # baseline is accepted
        for key, value in (("all_gates_passed", False),
                           ("adding_both_components_is_forbidden", False),
                           ("contract_sha256", "0" * 64),
                           ("schema", "something.else.v1")):
            broken = dict(document)
            broken[key] = value
            with self.assertRaises(TM.TransportModelError):
                TM.ProductionTransportModelV1(broken)

    def test_refuses_outside_measured_support(self) -> None:
        from rl_agent.ue_production_queue_capture_v1 import transport_model as TM
        model = TM.ProductionTransportModelV1(self._model_document())
        with self.assertRaises(TM.OutOfSupport):
            model.predict(backlog_bytes=0.0, udp_application_bytes=5_000,
                          ul_mcs=20)
        with self.assertRaises(TM.OutOfSupport):
            model.predict(backlog_bytes=1e9, udp_application_bytes=12_500,
                          ul_mcs=20)
        inside = model.predict(backlog_bytes=0.0,
                               udp_application_bytes=12_500, ul_mcs=20)
        self.assertGreater(inside.latency_ns, 0)

    def test_composition_replaces_rather_than_adds(self) -> None:
        from rl_agent.ue_production_queue_capture_v1 import transport_model as TM
        model = TM.ProductionTransportModelV1(self._model_document())
        retained = TM.RetainedSourceRow(
            row_id="r1", profile_label="FAVORABLE_STABLE",
            total_ns=175_000_000, uplink_ns=122_000_000,
            evidence_sha256="a" * 64)
        prediction = model.predict(backlog_bytes=0.0,
                                   udp_application_bytes=12_500, ul_mcs=20)
        composed = TM.compose_latency(
            retained=retained, prediction=prediction,
            measured_send_span_ns=500_000, actor_reserve_ns=1_000_000)
        # the retained uplink is gone from the total
        self.assertEqual(composed.retained_residual_ns,
                         retained.total_ns - retained.uplink_ns)
        self.assertEqual(
            composed.source_total_ns,
            retained.residual_ns + 500_000 + prediction.latency_ns)
        # strictly smaller than naively adding the new term to the old total
        naive_addition = retained.total_ns + prediction.latency_ns
        self.assertLess(composed.source_total_ns, naive_addition)
        self.assertEqual(composed.total_ns,
                         composed.source_total_ns + 1_000_000)

    def test_zero_actor_cost_is_refused(self) -> None:
        from rl_agent.ue_production_queue_capture_v1 import transport_model as TM
        model = TM.ProductionTransportModelV1(self._model_document())
        retained = TM.RetainedSourceRow(
            row_id="r1", profile_label="ADVERSE_STABLE",
            total_ns=196_000_000, uplink_ns=155_000_000,
            evidence_sha256="b" * 64)
        prediction = model.predict(backlog_bytes=0.0,
                                   udp_application_bytes=12_500, ul_mcs=20)
        with self.assertRaises(TM.TransportModelError):
            TM.compose_latency(retained=retained, prediction=prediction,
                               measured_send_span_ns=0, actor_reserve_ns=0)

    def test_retained_uplink_must_lie_inside_its_total(self) -> None:
        from rl_agent.ue_production_queue_capture_v1 import transport_model as TM
        with self.assertRaises(TM.TransportModelError):
            TM.RetainedSourceRow(row_id="r", profile_label="P",
                                 total_ns=100, uplink_ns=100,
                                 evidence_sha256="c" * 64)
