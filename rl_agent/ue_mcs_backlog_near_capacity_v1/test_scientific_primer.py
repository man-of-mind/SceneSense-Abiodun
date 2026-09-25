"""CPU-only tests for the target-channel scientific-state primer."""

from __future__ import annotations

import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from rl_agent.ue_mcs_backlog_calibration_v1 import tagged_sender
from . import contract as C
from . import runner as subject


def csv_line(fields: tuple[str, ...], **updates: object) -> str:
    values = {field: "0" for field in fields}
    values.update({key: str(value) for key, value in updates.items()})
    return ",".join(values[field] for field in fields)


class PrimerParserTests(unittest.TestCase):
    def test_only_fresh_ue_ul_table0_round0_new_data_grants_survive(self) -> None:
        valid = csv_line(
            subject.DCI_GRANT_FIELDS, direction=1, mcs=17, mcs_table=0,
            round=0, ndi=1,
        )
        rows = [
            (0, 99, valid),
            (0, 101, csv_line(subject.DCI_GRANT_FIELDS, direction=0, mcs=18,
                             mcs_table=0, round=0, ndi=1)),
            (0, 102, csv_line(subject.DCI_GRANT_FIELDS, direction=1, mcs=18,
                             mcs_table=1, round=0, ndi=1)),
            (0, 103, csv_line(subject.DCI_GRANT_FIELDS, direction=1, mcs=18,
                             mcs_table=0, round=1, ndi=1)),
            (0, 104, csv_line(subject.DCI_GRANT_FIELDS, direction=1, mcs=18,
                             mcs_table=0, round=0, ndi=2)),
            (0, 105, valid),
            (0, 106, "malformed"),
        ]
        self.assertEqual(subject.eligible_primer_grants(
            rows, after_receipt_ns=100), [{
                "receipt_monotonic_ns": 105,
                "mcs": 17, "mcs_table": 0, "round": 0, "ndi": 1,
            }])

    def test_rlc_ticks_sum_lcids_drop_incomplete_newest_and_count_trailing_zero(self) -> None:
        rows = [
            (0, 101, csv_line(subject.RLC_BUFFER_FIELDS, time="a", frame=1, slot=1,
                             lcid=1, bytes_in_buffer=5)),
            (0, 102, csv_line(subject.RLC_BUFFER_FIELDS, time="a", frame=1, slot=1,
                             lcid=2, bytes_in_buffer=7)),
            (0, 103, csv_line(subject.RLC_BUFFER_FIELDS, time="b", frame=1, slot=2,
                             lcid=1, bytes_in_buffer=0)),
            (0, 104, csv_line(subject.RLC_BUFFER_FIELDS, time="c", frame=1, slot=3,
                             lcid=1, bytes_in_buffer=0)),
            (0, 105, csv_line(subject.RLC_BUFFER_FIELDS, time="d", frame=1, slot=4,
                             lcid=1, bytes_in_buffer=99)),
        ]
        trailing, complete = subject.trailing_zero_rlc_ticks(
            rows, after_receipt_ns=100)
        self.assertEqual([item["total_bytes"] for item in complete], [12, 0, 0])
        self.assertEqual(trailing, 2)

    def test_rows_at_or_before_boundary_are_invisible(self) -> None:
        line = csv_line(subject.RLC_BUFFER_FIELDS, time="a", frame=1, slot=1,
                        lcid=1, bytes_in_buffer=0)
        self.assertEqual(subject.trailing_zero_rlc_ticks(
            [(0, 100, line)], after_receipt_ns=100), (0, []))

    def test_pre_ingress_zero_ticks_cannot_prove_primer_drain(self) -> None:
        before = csv_line(
            subject.RLC_BUFFER_FIELDS, time="a", frame=1, slot=1,
            lcid=1, bytes_in_buffer=0)
        after = csv_line(
            subject.RLC_BUFFER_FIELDS, time="b", frame=1, slot=2,
            lcid=1, bytes_in_buffer=0)
        newest = csv_line(
            subject.RLC_BUFFER_FIELDS, time="c", frame=1, slot=3,
            lcid=1, bytes_in_buffer=0)
        trailing, complete = subject.trailing_zero_rlc_ticks(
            [(0, 101, before), (0, 201, after), (0, 202, newest)],
            after_receipt_ns=200)
        self.assertEqual(trailing, 1)
        self.assertEqual(
            [row["receipt_monotonic_ns"] for row in complete], [201])


class PrimerOrderingTests(unittest.TestCase):
    def test_launch_method_primes_after_receivers_and_before_sender(self) -> None:
        source = inspect.getsource(subject.Runner.launch_traffic)
        receiver_ready = source.index("not every block receiver reported READY")
        primer = source.index("primer = self.target_channel_primer")
        sender = source.index("sender = self.spawn")
        self.assertLess(receiver_ready, primer)
        self.assertLess(primer, sender)

    def test_first_decision_receipt_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sender.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=tagged_sender.FRAME_ROW_FIELDS)
                writer.writeheader()
                row = {field: "0" for field in tagged_sender.FRAME_ROW_FIELDS}
                row["decision_monotonic_ns"] = "150000000"
                writer.writerow(row)
            runner = object.__new__(subject.Runner)
            runner.config = {"traffic": {"target_channel_primer": {
                "max_grant_receipt_to_first_decision_ms": 100.0,
            }}}
            sessions = {
                "sender_csv": path,
                "target_channel_primer": {
                    "latest_grant": {"receipt_monotonic_ns": 100000000},
                    "completed_monotonic_ns": 120000000,
                },
            }
            report = runner.audit_primer_first_decision(sessions)
            self.assertEqual(report["grant_receipt_to_first_decision_ms"], 50.0)
            self.assertTrue(report["receipt_time_gate_passed"])
            self.assertIn("DEFERRED", report["authoritative_source_time_gate"])

    def test_late_primer_receipt_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sender.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=tagged_sender.FRAME_ROW_FIELDS)
                writer.writeheader()
                row = {field: "0" for field in tagged_sender.FRAME_ROW_FIELDS}
                row["decision_monotonic_ns"] = "250000000"
                writer.writerow(row)
            runner = object.__new__(subject.Runner)
            runner.config = {"traffic": {"target_channel_primer": {
                "max_grant_receipt_to_first_decision_ms": 100.0,
            }}}
            sessions = {
                "sender_csv": path,
                "target_channel_primer": {
                    "latest_grant": {"receipt_monotonic_ns": 100000000},
                    "completed_monotonic_ns": 120000000,
                },
            }
            with self.assertRaisesRegex(subject.RunFailure, "outside"):
                runner.audit_primer_first_decision(sessions)

    def test_config_registers_a_separate_tiny_primer(self) -> None:
        config = json.loads(subject.DEFAULT_CONFIG.read_text())
        primer = config["traffic"]["target_channel_primer"]
        self.assertNotIn(primer["port"], config["traffic"]["ports"].values())
        self.assertLessEqual(primer["payload_bytes"], 1200)
        self.assertGreaterEqual(primer["datagrams"], 1)
        self.assertLessEqual(
            primer["max_grant_receipt_to_first_decision_ms"], 100.0)


if __name__ == "__main__":
    unittest.main()
