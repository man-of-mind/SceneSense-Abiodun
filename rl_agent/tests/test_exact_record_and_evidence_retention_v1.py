#!/usr/bin/env python3
"""Offline contract tests; these never import or start CARLA, OAI or Docker.

Regression cover for two campaign-scale defects found in retry4:

1. The adapter fetched the exact installed map record synchronously from
   _feedback_worker with a single 0.5 s HTTP attempt, converting every error to
   None. One stalled read blocked feedback reception long enough for later ACKs
   to expire and failed cell a17 on a single frame.
2. Raw label maps were retained per cell up to a 1 GiB quota (~1.1 GiB
   observed), which cannot fit 288 cells on the remaining filesystem.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from abiodun.rl_agent import ue_route_b_split_cell_adapter_v1 as adapter


class _StubCounters:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    def bump(self, name: str, amount: int = 1) -> None:
        self.values[name] = self.values.get(name, 0) + int(amount)


class _StubSnapshot:
    class timestamp:  # noqa: N801 - mirrors the CARLA attribute layout
        elapsed_seconds = 12.5


class _StubWorld:
    @staticmethod
    def get_snapshot():
        return _StubSnapshot()


class _RetrievalStub:
    """Carries only the attributes the retrieval methods actually touch."""

    def __init__(self) -> None:
        self.transport_counters = _StubCounters()
        self.gt_lock = threading.Lock()
        self.exact_retrieval_status: dict[int, dict] = {}
        self.exact_retrieval_stop_event = threading.Event()
        self.exact_retrieval_queue = adapter.queue.Queue(
            maxsize=adapter.EXACT_RECORD_QUEUE_MAXSIZE
        )
        self.installed_predictions: dict[int, list] = {}
        self.aligned_gt: dict[int, list] = {}
        self.failures: list[str] = []
        self.world = _StubWorld()
        self.camera = object()
        self.aligned_actor_tracker = object()
        self.stream_id = "stub-stream"
        self.map_api_port = 1
        self.fetch_log: list[tuple[int, float]] = []

    # Bind the real implementations under test.
    _enqueue_exact_retrieval = adapter.PassiveSplitCollector._enqueue_exact_retrieval
    _exact_record_worker = adapter.PassiveSplitCollector._exact_record_worker
    _retrieve_exact_record = adapter.PassiveSplitCollector._retrieve_exact_record

    def _ground_truth(self, **_kwargs):
        return []


class ExactRecordRetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self._matrix = adapter.actor_world_matrix
        self._inverse = adapter.actor_world_inverse_matrix
        adapter.actor_world_matrix = lambda _actor: None
        adapter.actor_world_inverse_matrix = lambda _actor: None

    def tearDown(self) -> None:
        adapter.actor_world_matrix = self._matrix
        adapter.actor_world_inverse_matrix = self._inverse

    def test_first_timeout_then_success_does_not_block_later_acks(self) -> None:
        stub = _RetrievalStub()
        stalled = 12
        follower = 34
        attempts: dict[int, int] = {}

        def fake_fetch(frame_id: int):
            attempts[frame_id] = attempts.get(frame_id, 0) + 1
            stub.fetch_log.append((int(frame_id), time.monotonic()))
            if int(frame_id) == stalled and attempts[frame_id] == 1:
                # Emulate a read that consumes the whole per-attempt timeout.
                time.sleep(adapter.EXACT_RECORD_HTTP_TIMEOUT_S + 0.1)
                return "TRANSIENT_TRANSPORT", None
            return "OK", [{"class_name": "vehicle", "world_x": 1.0}]

        stub._fetch_installed_record = fake_fetch
        worker = threading.Thread(target=stub._exact_record_worker, daemon=True)
        worker.start()
        try:
            started = time.monotonic()
            stub._enqueue_exact_retrieval(stalled)
            stub._enqueue_exact_retrieval(follower)
            enqueue_elapsed = time.monotonic() - started
            # The feedback receiver must not wait on HTTP at all.
            self.assertLess(
                enqueue_elapsed,
                0.1,
                f"enqueue blocked for {enqueue_elapsed:.3f}s",
            )
            deadline = time.monotonic() + 15.0
            while (
                len(stub.installed_predictions) < 2 and time.monotonic() < deadline
            ):
                time.sleep(0.02)
        finally:
            stub.exact_retrieval_stop_event.set()
            stub.exact_retrieval_queue.put_nowait(None)
            worker.join(timeout=5.0)

        self.assertEqual(sorted(stub.installed_predictions), [stalled, follower])
        self.assertEqual(stub.failures, [])
        self.assertEqual(attempts[stalled], 2, "transient fault must be retried")
        self.assertEqual(attempts[follower], 1)
        self.assertEqual(
            stub.exact_retrieval_status[stalled]["status"], "RETRIEVED"
        )
        self.assertEqual(stub.exact_retrieval_status[stalled]["attempts"], 2)
        self.assertEqual(
            stub.transport_counters.values.get(
                "exact_record_retrieval_recovered_after_retry"
            ),
            1,
        )
        self.assertIsNone(
            stub.transport_counters.values.get("exact_record_permanent_failures")
        )

    def test_authoritative_not_found_fails_permanently_without_retry(self) -> None:
        stub = _RetrievalStub()
        attempts = {"count": 0}

        def fake_fetch(_frame_id: int):
            attempts["count"] += 1
            return "NOT_FOUND", None

        stub._fetch_installed_record = fake_fetch
        stub._retrieve_exact_record(99)
        self.assertEqual(attempts["count"], 1, "NOT_FOUND must not be retried")
        self.assertEqual(stub.installed_predictions, {})
        self.assertEqual(stub.exact_retrieval_status[99]["status"], "PERMANENT_FAILURE")
        self.assertEqual(stub.exact_retrieval_status[99]["reason"], "NOT_FOUND")
        self.assertEqual(
            stub.failures,
            ["exact installed map record missing after ACK for frame 99"],
        )
        self.assertEqual(
            stub.transport_counters.values.get("exact_record_permanent_failures"), 1
        )

    def test_persistent_transient_fault_exhausts_and_fails_structurally(self) -> None:
        stub = _RetrievalStub()
        attempts = {"count": 0}

        def fake_fetch(_frame_id: int):
            attempts["count"] += 1
            return "TRANSIENT_TRANSPORT", None

        stub._fetch_installed_record = fake_fetch
        stub._retrieve_exact_record(7)
        self.assertLessEqual(attempts["count"], adapter.EXACT_RECORD_RETRIEVAL_MAX_ATTEMPTS)
        self.assertGreater(attempts["count"], 1)
        self.assertEqual(stub.installed_predictions, {})
        self.assertEqual(stub.exact_retrieval_status[7]["status"], "PERMANENT_FAILURE")
        self.assertEqual(len(stub.failures), 1)

    def test_identity_mismatch_never_stores_predictions(self) -> None:
        stub = _RetrievalStub()
        stub._fetch_installed_record = lambda _frame_id: ("IDENTITY_MISMATCH", None)
        stub._retrieve_exact_record(5)
        self.assertEqual(stub.installed_predictions, {})
        self.assertEqual(stub.exact_retrieval_status[5]["reason"], "IDENTITY_MISMATCH")


class EvidenceRetentionSelectionTest(unittest.TestCase):
    def test_zero_eligible_frames(self) -> None:
        self.assertEqual(adapter.select_evidence_frames([]), [])

    def test_fewer_than_sample_size_keeps_all(self) -> None:
        eligible = [10, 20, 30, 40, 50]
        self.assertEqual(adapter.select_evidence_frames(eligible), eligible)

    def test_exactly_sample_size_keeps_all(self) -> None:
        eligible = list(range(100, 100 + adapter.EVIDENCE_SAMPLE_FRAMES))
        selected = adapter.select_evidence_frames(eligible)
        self.assertEqual(selected, eligible)
        self.assertEqual(len(selected), adapter.EVIDENCE_SAMPLE_FRAMES)

    def test_more_than_sample_size_is_bounded_and_endpoint_inclusive(self) -> None:
        eligible = list(range(500, 500 + 1460))
        selected = adapter.select_evidence_frames(eligible)
        self.assertEqual(len(selected), adapter.EVIDENCE_SAMPLE_FRAMES)
        self.assertEqual(selected[0], eligible[0])
        self.assertEqual(selected[-1], eligible[-1])
        self.assertEqual(selected, sorted(set(selected)))
        self.assertTrue(set(selected).issubset(set(eligible)))

    def test_selection_is_deterministic_and_ignores_ordering(self) -> None:
        eligible = list(range(0, 337, 3))
        first = adapter.select_evidence_frames(eligible)
        shuffled = list(reversed(eligible))
        self.assertEqual(adapter.select_evidence_frames(shuffled), first)
        self.assertEqual(adapter.select_evidence_frames(eligible + eligible), first)

    def test_selection_spacing_is_even(self) -> None:
        selected = adapter.select_evidence_frames(list(range(1000)))
        gaps = {selected[i + 1] - selected[i] for i in range(len(selected) - 1)}
        self.assertLessEqual(max(gaps) - min(gaps), 1)

    def test_frame_id_inventory_hash_is_order_independent_and_bound(self) -> None:
        ids = [3, 1, 2]
        expected = adapter.hashlib.sha256(
            json.dumps([3, 1, 2], separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(adapter.frame_id_list_sha256(ids), expected)
        self.assertNotEqual(
            adapter.frame_id_list_sha256([1, 2, 3]),
            adapter.frame_id_list_sha256([1, 2, 4]),
        )


class EvidencePreservationAccountingTest(unittest.TestCase):
    def _stub(self, installed: list[int], tmp: Path):
        source = tmp / "edge"
        source.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for frame_id in installed:
            name = f"mask_{frame_id}.npy"
            (source / name).write_bytes(b"x" * 32)
            manifest[frame_id] = {
                "evidence_name": name,
                "bytes": 32,
                "hash_verified": True,
            }

        class _Stub:
            preserve_evidence = adapter.PassiveSplitCollector.preserve_evidence

            def __init__(self) -> None:
                self.gt_lock = threading.Lock()
                self.evidence_manifest = manifest
                self.ack_installed_frames = set(installed)
                self.edge_evidence_dir = source
                self.stream_id = "stub-stream"
                self.cell = {"cell_id": "a00__stub", "action_id": 0}

        return _Stub()

    def _run(self, count: int) -> dict:
        installed = list(range(200, 200 + count))
        with tempfile.TemporaryDirectory() as scratch:
            tmp = Path(scratch)
            stub = self._stub(installed, tmp)
            result = stub.preserve_evidence(tmp / "preserved")
            written = json.loads(
                (tmp / "preserved" / "segmentation_evidence_manifest.json").read_text()
            )
            preserved_npy = sorted(p.name for p in (tmp / "preserved").glob("*.npy"))
        expected = adapter.select_evidence_frames(installed)
        self.assertEqual(result["selected_frame_ids"], expected)
        self.assertEqual(len(preserved_npy), len(expected))
        self.assertEqual(result["preserved_masks"], len(expected))
        self.assertEqual(result["evaluated_masks"], count)
        self.assertEqual(result["hash_verified_masks"], count)
        self.assertEqual(result["eligible_frames"], count)
        self.assertEqual(
            result["sampling_elided_masks"], max(0, count - len(expected))
        )
        self.assertEqual(
            result["eligible_frame_ids_sha256"],
            adapter.frame_id_list_sha256(installed),
        )
        self.assertEqual(
            result["selected_frame_ids_sha256"],
            adapter.frame_id_list_sha256(expected),
        )
        self.assertEqual(
            result["selection_rule_version"], adapter.EVIDENCE_SELECTION_RULE_VERSION
        )
        self.assertTrue(
            written[
                "nonselected_masks_are_intentionally_elided_by_registered_sampling_policy"
            ]
        )
        # Every evaluated mask keeps complete manifest metadata either way.
        self.assertEqual(len(written["masks"]), count)
        statuses = {row["preservation_status"] for row in written["masks"]}
        if count > len(expected):
            self.assertIn("ELIDED_BY_REGISTERED_SAMPLING_POLICY", statuses)
        if count:
            self.assertIn("PRESERVED", statuses)
        return result

    def test_zero_eligible(self) -> None:
        result = self._run(0)
        self.assertEqual(result["preserved_masks"], 0)
        self.assertEqual(result["selected_frames"], 0)

    def test_fewer_than_sixteen(self) -> None:
        result = self._run(9)
        self.assertEqual(result["preserved_masks"], 9)

    def test_exactly_sixteen(self) -> None:
        result = self._run(16)
        self.assertEqual(result["preserved_masks"], 16)

    def test_more_than_sixteen_is_bounded(self) -> None:
        result = self._run(1460)
        self.assertEqual(result["preserved_masks"], adapter.EVIDENCE_SAMPLE_FRAMES)
        self.assertEqual(result["sampling_elided_masks"], 1460 - 16)


if __name__ == "__main__":
    unittest.main()


class RouteOutcomeClassificationTest(unittest.TestCase):
    """Retry5 cell a58 returned rc=1 on a completed, intervention-free route and
    the evidence that could classify it was deleted with the temp directory."""

    @staticmethod
    def _summary(**overrides):
        summary = {
            "loops_completed": 1,
            "watchdog_aborted_any": False,
            "intervention_count_total": 0,
            "cleanup_succeeded": True,
            "collision_incident_count_total": 1,
            "collision_count_total": 3,
            "collision_incidents": [
                {"other_actor_type": "vehicle.audi.tt", "frame_id": 4242}
            ],
            "walker_brake_ticks_total": 70,
        }
        summary.update(overrides)
        return summary

    def _classify(self, *, rc=1, density_status="FAIL", completed=True, summary_ok=True, **over):
        return adapter.classify_route_outcome(
            route_rc=rc,
            density_status=density_status,
            route_completed=completed,
            interventions_permitted=True,
            summary=self._summary(**over),
            summary_ok=summary_ok,
        )

    def test_collision_only_rc1_is_a_measured_accepted_outcome(self) -> None:
        outcome = self._classify()
        self.assertEqual(outcome["classification"], adapter.ROUTE_OUTCOME_COLLISION_ONLY)
        self.assertIn(outcome["classification"], adapter.ROUTE_ACCEPTABLE_OUTCOMES)
        self.assertEqual(outcome["fatal_reasons"], [])
        # The collision must be reported, never hidden or zeroed.
        self.assertEqual(outcome["collision_incident_count"], 1)
        self.assertEqual(outcome["collision_count"], 3)
        self.assertEqual(len(outcome["collision_incidents"]), 1)
        self.assertEqual(outcome["walker_brake_ticks_total"], 70)

    def test_route_runner_cleanup_failure_remains_fatal(self) -> None:
        outcome = self._classify(cleanup_succeeded=False)
        self.assertEqual(outcome["classification"], adapter.ROUTE_OUTCOME_FATAL)
        self.assertNotIn(outcome["classification"], adapter.ROUTE_ACCEPTABLE_OUTCOMES)
        self.assertTrue(
            any("cleanup_succeeded=false" in reason for reason in outcome["fatal_reasons"])
        )

    def test_incomplete_route_watchdog_and_intervention_remain_fatal(self) -> None:
        for kwargs, needle in (
            ({"completed": False}, "did not complete"),
            ({"loops_completed": 0}, "did not complete"),
            ({"watchdog_aborted_any": True}, "watchdog"),
            ({"intervention_count_total": 2}, "interventions occurred"),
        ):
            with self.subTest(kwargs=kwargs):
                outcome = self._classify(**kwargs)
                self.assertEqual(outcome["classification"], adapter.ROUTE_OUTCOME_FATAL)
                self.assertTrue(
                    any(needle in reason for reason in outcome["fatal_reasons"]),
                    outcome["fatal_reasons"],
                )

    def test_unexplained_nonzero_return_code_remains_fatal(self) -> None:
        outcome = self._classify(collision_incident_count_total=0)
        self.assertEqual(outcome["classification"], adapter.ROUTE_OUTCOME_FATAL)
        self.assertTrue(
            any("unexplained route runner returncode" in r for r in outcome["fatal_reasons"])
        )

    def test_missing_summary_is_fatal_and_never_collision_only(self) -> None:
        outcome = self._classify(summary_ok=False)
        self.assertEqual(outcome["classification"], adapter.ROUTE_OUTCOME_FATAL)
        self.assertTrue(any("summary missing" in r for r in outcome["fatal_reasons"]))

    def test_clean_and_permitted_intervention_paths_are_unchanged(self) -> None:
        self.assertEqual(
            self._classify(rc=0, density_status="PASS")["classification"],
            adapter.ROUTE_OUTCOME_PASS,
        )
        self.assertEqual(
            self._classify(density_status="INTERVENED")["classification"],
            adapter.ROUTE_OUTCOME_INTERVENED,
        )


class RouteSummaryRetentionTest(unittest.TestCase):
    def test_summary_is_atomically_retained_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            source = root / "tmp" / adapter.ROUTE_SUMMARY_NAME
            source.parent.mkdir(parents=True)
            payload = {"route_id": "unit-test-route", "status": "FAIL", "x": "y" * 4096}
            source.write_text(json.dumps(payload), encoding="utf-8")
            attempt = root / "attempt"
            attempt.mkdir()
            result = adapter.persist_route_summary(source, attempt)
            self.assertTrue(result["persisted"])
            durable = attempt / adapter.ROUTE_SUMMARY_NAME
            self.assertEqual(result["sha256"], adapter.sha256_file(durable))
            self.assertEqual(result["bytes"], durable.stat().st_size)
            # Whole JSON retained, not a tail.
            self.assertEqual(json.loads(durable.read_text()), payload)
            # No staging file survives.
            self.assertEqual([p.name for p in attempt.glob(".*partial")], [])

    def test_absent_source_reports_a_reason_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            attempt = Path(scratch)
            result = adapter.persist_route_summary(attempt / "missing.json", attempt)
            self.assertFalse(result["persisted"])
            self.assertEqual(result["reason"], "ROUTE_SUMMARY_NOT_WRITTEN_BY_ROUTE_RUNNER")
            self.assertFalse((attempt / adapter.ROUTE_SUMMARY_NAME).exists())

    def test_identity_mismatch_is_detected(self) -> None:
        expected = {"route_id": "r", "density": "traffic_50_50", "seed": 31, "loops": 1}
        self.assertEqual(
            adapter.validate_route_summary_identity(dict(expected), expected), []
        )
        mismatches = adapter.validate_route_summary_identity(
            {**expected, "seed": 32}, expected
        )
        self.assertEqual(len(mismatches), 1)
        self.assertIn("seed", mismatches[0])

    def test_route_summary_is_a_registered_output(self) -> None:
        self.assertIn(adapter.ROUTE_SUMMARY_NAME, adapter.EXPECTED_OUTPUTS)


class NoBrakingEventOutcomeTest(unittest.TestCase):
    """retry7 cell a61 completed a clean uneventful loop; the density runner
    returned PEDESTRIAN_BRAKING_NOT_EXERCISED and exited 1."""

    @staticmethod
    def _summary(**overrides):
        summary = {
            "loops_completed": 1,
            "watchdog_aborted_any": False,
            "intervention_count_total": 0,
            "cleanup_succeeded": True,
            "collision_incident_count_total": 0,
            "collision_count_total": 0,
            "collision_incidents": [],
            "walker_brake_ticks_total": 0,
            "driven_distance_m_median": 1251.6,
            "planned_route_length_m": 1268.68,
        }
        summary.update(overrides)
        return summary

    def _classify(self, *, status=adapter.ROUTE_STATUS_NO_BRAKING, completed=True,
                  summary_ok=True, **over):
        return adapter.classify_route_outcome(
            route_rc=1, density_status=status, route_completed=completed,
            interventions_permitted=True, summary=self._summary(**over),
            summary_ok=summary_ok,
        )

    def test_uneventful_completed_route_is_a_measured_outcome(self) -> None:
        outcome = self._classify()
        self.assertEqual(outcome["classification"], adapter.ROUTE_OUTCOME_NO_BRAKING)
        self.assertIn(outcome["classification"], adapter.ROUTE_ACCEPTABLE_OUTCOMES)
        self.assertEqual(outcome["fatal_reasons"], [])

    def test_it_never_claims_braking_coverage_or_safety_success(self) -> None:
        outcome = self._classify()
        self.assertFalse(outcome["pedestrian_braking_exercised"])
        self.assertEqual(outcome["walker_brake_ticks_total"], 0)
        self.assertTrue(
            outcome["does_not_claim_pedestrian_braking_coverage_or_safety_success"]
        )
        self.assertTrue(
            outcome["no_braking_event_is_measured_outcome_not_braking_coverage"]
        )

    def test_braking_ticks_collisions_and_interventions_are_always_preserved(self) -> None:
        outcome = self._classify(
            status="FAIL", walker_brake_ticks_total=31,
            intervention_count_total=0, collision_incident_count_total=3,
            collision_count_total=7,
        )
        self.assertEqual(outcome["walker_brake_ticks_total"], 31)
        self.assertTrue(outcome["pedestrian_braking_exercised"])
        self.assertEqual(outcome["collision_incident_count"], 3)
        self.assertEqual(outcome["collision_count"], 7)
        self.assertEqual(outcome["intervention_count_total"], 0)

    def test_every_required_proof_is_enforced(self) -> None:
        for kwargs, needle in (
            ({"completed": False}, "did not complete"),
            ({"loops_completed": 2}, "did not complete"),
            ({"watchdog_aborted_any": True}, "watchdog"),
            ({"intervention_count_total": 1}, "interventions occurred"),
            ({"cleanup_succeeded": False}, "cleanup_succeeded=false"),
            ({"summary_ok": False}, "summary missing"),
        ):
            with self.subTest(kwargs=kwargs):
                outcome = self._classify(**kwargs)
                self.assertEqual(
                    outcome["classification"], adapter.ROUTE_OUTCOME_FATAL, kwargs
                )
                self.assertTrue(
                    any(needle in r for r in outcome["fatal_reasons"]),
                    outcome["fatal_reasons"],
                )

    def test_collision_during_no_braking_status_is_not_silently_accepted(self) -> None:
        outcome = self._classify(collision_incident_count_total=2)
        self.assertEqual(outcome["classification"], adapter.ROUTE_OUTCOME_FATAL)
        self.assertEqual(outcome["collision_incident_count"], 2)
