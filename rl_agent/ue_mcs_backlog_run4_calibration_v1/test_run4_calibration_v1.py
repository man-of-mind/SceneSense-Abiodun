"""Offline tests for the registered Run-4 physical calibration runner."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from . import authorization as AUTH
from . import contract as C
from . import runner as R
from . import tagged_sender as TS


ROOT = Path(__file__).resolve().parents[2]


class ContractTests(unittest.TestCase):
    def test_config_and_inherited_sources_are_exact(self) -> None:
        value = C.load_config()
        self.assertEqual(value["package_id"], C.PACKAGE_ID)
        self.assertTrue(C.verify_inherited_sources()["verified"])
        self.assertEqual(value["design"]["future_epoch_lead_s"],
                         C.FUTURE_EPOCH_LEAD_S)
        self.assertEqual(value["design"]["sender_arm_timeout_s"],
                         C.SENDER_ARM_TIMEOUT_S)

    def test_preregistration_numeric_timeline_matches_config(self) -> None:
        text = (Path(__file__).with_name("PREREGISTRATION.md")
                .read_text(encoding="utf-8"))
        self.assertIn("20 ms into the future", text)
        self.assertIn("100 ms", text)
        self.assertNotIn("at least 3 seconds", text)

    def test_amendment_seals_are_pinned_without_relabeling_refusal(self) -> None:
        root = ROOT / "rl_agent/ue_mcs_backlog_robust_bracket_v1/sealed"
        self.assertEqual(C.sha256_file(root / "PROSPECTIVE_CAPACITY_AMENDMENT.json"),
                         C.AMENDMENT_SHA256)
        value = json.loads((root / "PROSPECTIVE_CAPACITY_AMENDMENT.json").read_text())
        self.assertFalse(value["original_capacity_qualification_overturned"])
        self.assertFalse(value["original_tier_selection_qualified"])
        self.assertEqual(value["preserved_attempt"]["original_disposition"]["status"],
                         "CAPACITY_QUALIFICATION_REFUSED")

    def test_registered_tiers_are_one_mode_and_mtu_safe(self) -> None:
        tiers = C.registered_tiers()
        self.assertEqual([row.action_id for row in tiers], [40, 39, 38])
        self.assertEqual([row.chunks_per_frame for row in tiers], [106, 312, 517])
        self.assertEqual([row.last_chunk_payload_bytes for row in tiers],
                         [237, 1064, 363])
        self.assertLessEqual(C.FULL_IPV4_PACKET_BYTES, C.PATH_MTU_BYTES)

    def test_plan_is_twelve_whole_cells_and_disjoint(self) -> None:
        plan = C.build_cell_plan(ports={"low": 5401, "medium": 5402,
                                        "high": 5403})
        audit = C.audit_cell_plan(plan)
        self.assertTrue(audit["registered_design"], audit)
        self.assertEqual(audit["cells"], 12)
        self.assertEqual(audit["decisions"], 5400)
        self.assertTrue(audit["whole_cell_partition"])
        self.assertTrue(audit["transition_sets_disjoint"])
        for profile in C.CONTRAST_PROFILE_IDS:
            self.assertEqual(audit["partition_counts_per_profile"][profile],
                             {C.FIT: 3, C.VALIDATION: 3})

    def test_cell_shuffle_is_deterministic_but_not_partition_mutating(self) -> None:
        ports = {"low": 1, "medium": 2, "high": 3}
        first = C.build_cell_plan(ports=ports)
        second = C.build_cell_plan(ports=ports)
        self.assertEqual(first, second)
        expected = dict(zip(C.PERMUTATIONS, C.PARTITIONS))
        for cell in first:
            self.assertEqual(cell.partition, expected[cell.sequence])

    def test_effective_config_changes_only_registered_additions(self) -> None:
        effective = C.effective_runtime_config()
        self.assertEqual(effective["radio"]["prb"], 273)
        self.assertEqual(effective["campaign"]["sample_period_s"], 0.1)
        self.assertEqual(effective["paths"]["output_root"],
                         "rl_agent/experiments/ue_mcs_backlog_run4_calibration_v1")


class SenderContractTests(unittest.TestCase):
    def test_chunk_spans_preserve_every_payload_byte(self) -> None:
        for tier in C.registered_tiers():
            spans = TS.chunk_spans(tier.payload_bytes, C.CHUNK_BYTES)
            self.assertEqual(len(spans), tier.chunks_per_frame)
            self.assertEqual(sum(size for _offset, size in spans), tier.payload_bytes)
            self.assertEqual(spans[-1][1], tier.last_chunk_payload_bytes)

    def test_sender_schema_uses_real_summary_names(self) -> None:
        self.assertIn("datagrams_handed_to_socket", TS.FRAME_FIELDS)
        self.assertIn("datagrams_dropped_at_socket", TS.FRAME_FIELDS)
        self.assertNotIn("frames_sent", TS.FRAME_FIELDS)
        self.assertNotIn("chunks_sent", TS.FRAME_FIELDS)

    def test_sender_requires_ready_bound_epoch_contract(self) -> None:
        parser = TS.parse_args
        with self.assertRaises(SystemExit):
            parser(["--cell-id", "x"])
        source = Path(TS.__file__).read_text()
        self.assertIn("--ready-json", source)
        self.assertIn("--epoch-contract", source)
        self.assertNotIn("--epoch-monotonic-ns", source)
        self.assertIn("sender_ready_sha256", source)

    def test_sender_accepts_only_future_ready_bound_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            path = Path(value) / "epoch.json"
            ready_sha = "a" * 64
            epoch = time.monotonic_ns() + 1_000_000_000
            path.write_text(json.dumps({
                "schema": TS.EPOCH_SCHEMA, "cell_id": "cell-x",
                "epoch_monotonic_ns": epoch, "period_ns": 100_000_000,
                "created_monotonic_ns": epoch - 500_000_000,
                "sender_ready_sha256": ready_sha,
            }))
            observed = TS._wait_for_epoch_contract(
                path, cell_id="cell-x", period_ns=100_000_000,
                sender_ready_sha256=ready_sha, timeout_s=0.1)
            self.assertEqual(observed["epoch_monotonic_ns"], epoch)


class AuthorizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output_root = self.root / "campaign"
        self.output = self.output_root / "run_once"
        self.auth = self.root / "grant.json"
        self.now = datetime(2026, 9, 25, 8, 0, tzinfo=timezone.utc)
        self.identity = {"head": "a" * 40, "parent_head": "b" * 40,
                         "package_sources_clean": True, "package_status": []}
        self.inventory = {"repo_files": {}, "inventory_sha256": "c" * 64}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def grant(self, **changes: object) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": AUTH.AUTH_SCHEMA, "stage": AUTH.STAGE, "token": AUTH.TOKEN,
            "grant_id": "12345678-1234-4234-9234-123456789abc",
            "granted_by": "Abiodun",
            "granted_utc": "2026-09-25T07:55:00Z",
            "expires_utc": "2026-09-25T09:00:00Z",
            "execution_head": self.identity["head"],
            "execution_parent_head": self.identity["parent_head"],
            "source_inventory_sha256": self.inventory["inventory_sha256"],
            "config_sha256": C.sha256_file(C.DEFAULT_CONFIG),
            "amendment_sha256": C.AMENDMENT_SHA256,
            "output_path": str(self.output.resolve()),
        }
        value.update(changes)
        self.auth.write_text(json.dumps(value))
        return value

    def consume(self) -> dict[str, object]:
        with mock.patch.object(AUTH, "execution_identity", return_value=self.identity), \
             mock.patch.object(C, "source_inventory", return_value=self.inventory):
            return AUTH.consume_authorization(
                self.auth, self.output, output_root=self.output_root,
                repo_root=ROOT, now=self.now)

    def test_valid_grant_is_durably_consumed_once(self) -> None:
        self.grant()
        value = self.consume()
        marker = Path(str(value["consumption_marker"]))
        self.assertTrue(marker.is_file())
        self.assertEqual(json.loads(marker.read_text())["grant_id"],
                         "12345678-1234-4234-9234-123456789abc")
        with self.assertRaises(AUTH.AuthorizationError):
            self.consume()
        self.assertTrue(marker.is_file())

    def test_expired_future_and_overlong_grants_fail_unspent(self) -> None:
        for changes in (
            {"expires_utc": "2026-09-25T07:59:59Z"},
            {"granted_utc": "2026-09-25T08:01:00Z",
             "expires_utc": "2026-09-25T09:00:00Z"},
            {"granted_utc": "2026-09-25T00:00:00Z",
             "expires_utc": "2026-09-25T07:00:01Z"},
        ):
            self.grant(**changes)
            with self.assertRaises(AUTH.AuthorizationError):
                self.consume()
            self.assertFalse((self.output_root / AUTH.CONSUMPTION_DIRNAME).exists())

    def test_every_binding_is_enforced(self) -> None:
        cases = {
            "execution_head": "d" * 40,
            "execution_parent_head": "e" * 40,
            "source_inventory_sha256": "f" * 64,
            "config_sha256": "0" * 64,
            "amendment_sha256": "1" * 64,
            "output_path": str((self.root / "other").resolve()),
        }
        for field, value in cases.items():
            self.grant(**{field: value})
            with self.assertRaises(AUTH.AuthorizationError, msg=field):
                self.consume()

    def test_unknown_field_and_noncanonical_uuid_fail(self) -> None:
        self.grant(extra="x")
        with self.assertRaises(AUTH.AuthorizationError):
            self.consume()
        self.grant(grant_id="NOT-A-UUID")
        with self.assertRaises(AUTH.AuthorizationError):
            self.consume()

    def test_output_must_be_direct_child_of_registered_root(self) -> None:
        self.grant()
        nested = self.output_root / "nested" / "run"
        value = json.loads(self.auth.read_text())
        value["output_path"] = str(nested.resolve())
        self.auth.write_text(json.dumps(value))
        with mock.patch.object(AUTH, "execution_identity", return_value=self.identity), \
             mock.patch.object(C, "source_inventory", return_value=self.inventory):
            with self.assertRaises(AUTH.AuthorizationError):
                AUTH.consume_authorization(
                    self.auth, nested, output_root=self.output_root,
                    repo_root=ROOT, now=self.now)


class LifecycleAuditTests(unittest.TestCase):
    def test_every_inherited_method_has_the_registered_owner(self) -> None:
        value = R.inherited_lifecycle_audit(R.Runner)
        self.assertTrue(value["verified"])
        self.assertEqual(set(value["methods"]), set(R.EXPECTED_INHERITED_METHODS))

    def test_import_is_offline_and_launch_free(self) -> None:
        source = (
            "import socket,subprocess; "
            "socket.socket=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('socket')); "
            "subprocess.Popen=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('popen')); "
            "import rl_agent.ue_mcs_backlog_run4_calibration_v1.runner"
        )
        completed = subprocess.run([sys.executable, "-c", source], cwd=ROOT,
                                   text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_main_consumes_grant_before_output_and_live_runner(self) -> None:
        source = Path(R.__file__).read_text()
        consume = source.index("consumed = AUTH.consume_authorization")
        mkdir = source.index("args.output_dir.mkdir", consume)
        construct = source.index("return Runner(", mkdir)
        self.assertLess(consume, mkdir)
        self.assertLess(mkdir, construct)

    def test_outer_finally_always_attempts_ran_and_core_cleanup(self) -> None:
        source = inspect_source(R.Runner.run)
        tail = source[source.index("finally:"):]
        self.assertIn("self.teardown_ran()", tail)
        self.assertIn("self.stop_core()", tail)
        self.assertIn("self.final_cold_state()", tail)

    def test_sender_is_armed_before_primer_and_epoch_publication(self) -> None:
        source = inspect_source(R.Runner.launch_traffic)
        spawn = source.index("sender = self.spawn")
        primer = source.index("primer = self.target_channel_primer")
        freshness = source.index("epoch_ns - grant_ns <= max_gap_ns")
        publish = source.index("write_json_create(epoch_path, epoch)")
        self.assertLess(spawn, primer)
        self.assertLess(primer, freshness)
        self.assertLess(freshness, publish)


def inspect_source(value: object) -> str:
    import inspect
    return inspect.getsource(value)


class TrafficAuditTests(unittest.TestCase):
    def make_cell(self) -> C.Cell:
        return C.build_cell_plan(ports={"low": 1, "medium": 2, "high": 3})[0]

    def materialize_sender(self, root: Path, cell: C.Cell,
                           *, drop_at: int | None = None,
                           lag_at: int | None = None
                           ) -> tuple[Path, Path, Path, Path]:
        epoch = 1_000_000_000
        rows = []
        for index in range(C.FRAMES_PER_CELL):
            block = cell.blocks[index // C.FRAMES_PER_BLOCK]
            dropped = 1 if drop_at == index else 0
            handed = block.chunks_per_frame - dropped
            payload_handed = block.payload_bytes if dropped == 0 else 0
            wire = (block.payload_bytes + block.chunks_per_frame * 24
                    if dropped == 0 else 0)
            row = {field: "" for field in TS.FRAME_FIELDS}
            row.update({
                "schema": TS.FRAME_SCHEMA, "cell_id": cell.cell_id,
                "decision_index": index, "block_index": block.block_index,
                "tier": block.tier, "action_id": block.action_id,
                "frame_index_in_block": index % C.FRAMES_PER_BLOCK,
                "epoch_monotonic_ns": epoch, "payload_bytes": block.payload_bytes,
                "chunks_per_frame": block.chunks_per_frame,
                "datagrams_handed_to_socket": handed,
                "datagrams_dropped_at_socket": dropped,
                "application_payload_bytes_handed_to_socket": payload_handed,
                "schedule_lag_ms": 101.0 if lag_at == index else 0.1,
                "bytes_handed_to_socket": wire,
            })
            rows.append(row)
        csv_path = root / "sender.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TS.FRAME_FIELDS))
            writer.writeheader(); writer.writerows(rows)
        ready_path = root / "sender_ready.json"
        epoch_path = root / "shared_epoch.json"
        ready_path.write_text("ready\n")
        epoch_path.write_text("epoch\n")
        summary = {
            "schema": TS.SUMMARY_SCHEMA, "decisions": C.FRAMES_PER_CELL,
            "epoch_monotonic_ns": epoch,
            "sender_ready_sha256": C.sha256_file(ready_path),
            "epoch_contract_sha256": C.sha256_file(epoch_path),
            "datagrams_handed_to_socket": sum(
                int(row["datagrams_handed_to_socket"]) for row in rows),
            "datagrams_dropped_at_socket": sum(
                int(row["datagrams_dropped_at_socket"]) for row in rows),
            "application_payload_bytes_handed_to_socket": sum(
                int(row["application_payload_bytes_handed_to_socket"]) for row in rows),
            "bytes_handed_to_socket": sum(
                int(row["bytes_handed_to_socket"]) for row in rows),
            "max_schedule_lag_ms": max(float(row["schedule_lag_ms"]) for row in rows),
        }
        summary_path = root / "sender.json"
        summary_path.write_text(json.dumps(summary))
        return csv_path, summary_path, ready_path, epoch_path

    def materialize_receivers(self, root: Path, cell: C.Cell) -> list[dict[str, object]]:
        values = []
        for block in cell.blocks:
            path = root / f"receiver_{block.block_index}.json"
            path.write_text(json.dumps({
                "schema": "scenesense.ue_n3_structured_udp_receiver_summary.v1",
                "clean_shutdown": True, "malformed_datagrams": 0,
                "stream_limit_exceeded_datagrams": 0,
                "measurement": {"expected_frames_per_stream": C.FRAMES_PER_BLOCK,
                                "expected_chunks_per_frame": block.chunks_per_frame},
                "streams": [],
            }))
            values.append({"process": SimpleNamespace(
                               process=SimpleNamespace(poll=lambda: 0)),
                           "summary": path, "tier": block.tier,
                           "block_index": block.block_index})
        return values

    def test_exact_real_sender_schema_passes(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value); cell = self.make_cell()
            csv_path, summary, ready, epoch_path = self.materialize_sender(root, cell)
            runner = object.__new__(R.Runner)
            result = runner.audit_traffic({
                "sender_csv": csv_path, "sender_summary": summary,
                "epoch": {"shared_epoch_monotonic_ns": 1_000_000_000},
                "sender_ready": ready, "epoch_contract": epoch_path,
                "receivers": self.materialize_receivers(root, cell),
            }, cell, root)
            self.assertTrue(result["sender_accounting_exact"])

    def test_one_socket_drop_fails_instead_of_changing_offered_load(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value); cell = self.make_cell()
            csv_path, summary, ready, epoch_path = self.materialize_sender(root, cell, drop_at=7)
            runner = object.__new__(R.Runner)
            with self.assertRaisesRegex(R.RunError, "socket loss"):
                runner.audit_traffic({
                    "sender_csv": csv_path, "sender_summary": summary,
                    "epoch": {"shared_epoch_monotonic_ns": 1_000_000_000},
                    "sender_ready": ready, "epoch_contract": epoch_path,
                    "receivers": self.materialize_receivers(root, cell),

                }, cell, root)
    def test_one_period_schedule_lag_fails(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value); cell = self.make_cell()
            csv_path, summary, ready, epoch_path = self.materialize_sender(
                root, cell, lag_at=7)
            runner = object.__new__(R.Runner)
            with self.assertRaisesRegex(R.RunError, "schedule lag"):
                runner.audit_traffic({
                    "sender_csv": csv_path, "sender_summary": summary,
                    "epoch": {"shared_epoch_monotonic_ns": 1_000_000_000},
                    "sender_ready": ready, "epoch_contract": epoch_path,
                    "receivers": self.materialize_receivers(root, cell),
                }, cell, root)


class TracerGateTests(unittest.TestCase):
    def test_empty_required_event_fails(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            runner = object.__new__(R.Runner)
            runner.config = C.effective_runtime_config()
            for source in ("gnb", "ue"):
                raw = root / "ttracer" / source / f"{source}.raw"
                raw.parent.mkdir(parents=True); raw.write_bytes(b"raw")
                for event in runner.config["telemetry"]["events"][source]:
                    path = root / "ttracer" / source / "csv" / f"{event}.csv"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("header\nrow\n")
            target = root / "ttracer" / "ue" / "csv" / "NRUE_MAC_DCI_GRANT.csv"
            target.write_text("header\n")
            with self.assertRaisesRegex(R.RunError, "no evidence row"):
                runner.audit_ttracer_nonempty(root)


if __name__ == "__main__":
    unittest.main()
