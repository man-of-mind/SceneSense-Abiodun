"""Retained-evidence audit tests: real join gates plus synthetic adversarial joins.

Reads the retained capture read-only.  CPU only; no CARLA/OAI/RFsim/Docker.
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rl_agent.splitfusion_hybrid_sac_run5_v1 import retained_snr_audit as A
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_state_contract as C

PACKAGE = A.REPO_ROOT / A.PACKAGE_DIR


def _session(log, tag="99__synthetic__perm0") -> A.CellSession:
    return A.CellSession(
        cell_id="synthetic", cell_tag=tag, partition="FIT",
        session_uuid="33333333-3333-4333-8333-333333333333", log=tuple(log),
        window=(min(e["send_monotonic_ns"] for e in log),
                max(e["ack_monotonic_ns"] for e in log)))


def _entry(send, ack, target=10.0, status="ACK", clamped=False, **extra):
    entry = {"send_monotonic_ns": send, "ack_monotonic_ns": ack, "status": status,
             "profile_id": "FAVORABLE_STABLE", "step_index": 1,
             "commanded_noise_power_db": -11.0, "reason": "PROFILE_REPLAY", **extra}
    if target is not None:
        entry.update(target_snr_db=target, clamped=clamped)
    return entry


class RetainedJoinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result, cls.join_bytes = A.run_once()

    def test_all_join_gates_pass_on_2700_decisions(self) -> None:
        join = self.result["join"]
        self.assertEqual(join["counts"]["decisions"], 2700)
        self.assertEqual(join["counts"]["valid"], 2700)
        self.assertEqual(join["counts"]["coverage"], 1.0)
        self.assertTrue(all(join["gates"].values()), join["gates"])
        self.assertEqual(len(join["sessions"]), 12)

    def test_joined_table_has_no_future_or_leaky_columns(self) -> None:
        rows = list(csv.DictReader(io.StringIO(self.join_bytes.decode())))
        self.assertEqual(len(rows), 2700)
        for forbidden in ("profile", "step", "noise", "trace", "gnb", "pusch", "markov"):
            self.assertFalse(any(forbidden in name for name in rows[0]), forbidden)
        self.assertFalse(set(rows[0]) & set(C.RFSIM_LOG_FIELDS_NEVER_EXPOSED))
        for row in rows:
            self.assertLess(int(row["snr_source_ack_monotonic_ns"]),
                            int(row["frame_open_monotonic_ns"]))
            self.assertEqual(row["valid"], "True")

    def test_baseline_reproduction_is_exact(self) -> None:
        reproduction = self.result["baseline_reproduction"]
        self.assertTrue(reproduction["pooled_equal"])
        self.assertTrue(reproduction["folds_equal"])
        self.assertEqual(reproduction["protocol"],
                         "GROUPED_LEAVE_ONE_WHOLE_FIT_CELL_OUT_CROSS_VALIDATION")

    def test_committed_artifacts_match_a_fresh_run(self) -> None:
        committed = json.loads((PACKAGE / A.AUDIT_FILENAME).read_text())
        committed.pop("determinism")
        committed["join"]["gates"].pop("DETERMINISTIC_OUTPUT")
        self.assertEqual(A.canonical_sha256(committed), A.canonical_sha256(self.result))
        self.assertEqual((PACKAGE / A.JOIN_FILENAME).read_bytes(), self.join_bytes)

    def test_every_source_is_hashed(self) -> None:
        sources = self.result["sources"]
        self.assertEqual(sum(1 for k in sources if k.endswith("command_log.json")), 12)
        for key in ("decisions.csv", "EVALUATION_V2.json", "CAUSAL_JOIN_REPORT.json",
                    "transport_model_v2.json", "model_v2.py", "run5_state_contract.py"):
            self.assertTrue(any(k.endswith(key) for k in sources), key)
        for value in sources.values():
            self.assertEqual(len(value["sha256"]), 64)

    def test_comparison_uses_fit_cells_only_and_reports_six_folds(self) -> None:
        comparison = self.result["comparison"]
        self.assertEqual(len(comparison["folds"]), 6)
        self.assertTrue(all(f["held_out_cell"].split("__")[-1] in
                            {"perm0", "perm1", "perm2"} for f in comparison["folds"]))
        with self.assertRaises(A.AuditError):
            A.grouped_comparison([{"partition": "VALIDATION", "cell_id": "x"}])
        self.assertEqual(self.result["validation_descriptive"]["status"],
                         "DESCRIPTIVE_ONLY__NOT_USED_FOR_ANY_DECISION")

    def test_verdict_follows_the_preregistered_rule(self) -> None:
        questions = self.result["comparison"]["questions"]
        identified = [q for q, v in questions.items() if v.get("direct_effect_identified")]
        self.assertEqual(identified, self.result["identified_questions"])
        for q, v in questions.items():
            expected = (v["pooled_snr"] < v["pooled_baseline"]
                        and v["held_out_cells_improved"] >= A.FOLD_MAJORITY)
            self.assertEqual(v["improves"], expected, q)
        if not identified:
            self.assertEqual(self.result["verdict"],
                             "SNR_DIRECT_EFFECT_NOT_IDENTIFIED_BY_RETAINED_EVIDENCE")

    def test_outcome_analysis_is_blocked_when_a_join_gate_fails(self) -> None:
        real = A.join_snr

        def failing(ledger):
            joined, summary = real(ledger)
            summary["gates"]["ZERO_FUTURE_JOINS"] = False
            return joined, summary

        with mock.patch.object(A, "join_snr", failing):
            result, _ = A.run_once()
        self.assertEqual(result["verdict"], "BLOCKED")
        self.assertNotIn("comparison", result)

    def test_outputs_are_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / A.AUDIT_FILENAME).write_text("{}")
            with self.assertRaises(A.AuditError):
                A.run(Path(tmp))


class SyntheticJoinTest(unittest.TestCase):
    """Adversarial joins on synthetic logs through the production provider."""

    def observe(self, log, open_ns):
        session = _session(log)
        observation = A.build_provider(session).observe(
            A.retained_boundary(session, 0, open_ns))
        return observation, A.reference_join(session.log, open_ns)

    def test_ack_exactly_at_action_open_is_not_joined(self) -> None:
        log = [_entry(0, 100, 10.0), _entry(900, 1_000, 20.0)]
        observation, reference = self.observe(log, 1_000)
        self.assertEqual(observation.value_db, 10.0)
        self.assertEqual(reference["entry"]["target_snr_db"], 10.0)
        observation, _ = self.observe(log, 1_001)
        self.assertEqual(observation.value_db, 20.0)

    def test_in_flight_command_is_flagged_and_ignored(self) -> None:
        log = [_entry(0, 100, 10.0), _entry(990, 1_050, 20.0)]
        observation, reference = self.observe(log, 1_000)
        self.assertEqual(observation.value_db, 10.0)
        self.assertTrue(observation.newer_command_in_flight)
        self.assertTrue(reference["in_flight"])

    def test_clamped_or_restore_latest_is_missing(self) -> None:
        for latest in (_entry(500, 600, 12.0, clamped=True),
                       _entry(500, 600, None, reason="RESTORE_CLEAN"),
                       _entry(500, 600, 12.0, status="ERROR")):
            observation, _ = self.observe([_entry(0, 100, 10.0), latest], 1_000)
            self.assertFalse(observation.valid)
            self.assertIsNone(observation.value_db)

    def test_sibling_session_commands_cannot_enter_a_provider(self) -> None:
        session = _session([_entry(0, 100, 10.0)])
        provider = A.build_provider(session)
        foreign = C.RfsimSnrCommandRecordV1.from_log_entry(
            _entry(200, 300, 30.0), session_uuid="44444444-4444-4444-8444-444444444444",
            command_seq=1, clock_domain=A.RETAINED_CLOCK_DOMAIN)
        with self.assertRaises(C.MetadataError):
            provider.ingest(foreign)


if __name__ == "__main__":
    unittest.main()
