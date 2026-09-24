from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rl_agent.splitfusion_hybrid_sac_run4_v1 import dynamic_mcs_sequence as subject


def _grant(timestamp: int, mcs: int, row: int) -> subject.RawUeGrantV1:
    payload = {
        "dci_frame": row % 1024,
        "dci_slot": 2,
        "harq_pid": row % 16,
        "mcs_index": mcs,
        "rnti": subject.EXPECTED_UE_RNTI,
        "sched_frame": row % 1024,
        "sched_slot": 8,
        "source_row_number": row,
        "source_timestamp_ns": timestamp,
    }
    return subject.RawUeGrantV1(
        source_identity_sha256=subject._canonical_sha256(payload), **payload
    )


class DynamicMcsSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = subject.load_dynamic_mcs_evidence(
            maximum_age_ns=100_000_000
        )

    def test_real_source_binding_and_verifier_are_exact(self) -> None:
        evidence = self.evidence
        self.assertEqual(
            evidence.source_manifest_sha256,
            subject.EXPECTED_SOURCE_MANIFEST_SHA256,
        )
        self.assertEqual(
            evidence.source_config_sha256, subject.EXPECTED_SOURCE_CONFIG_SHA256
        )
        self.assertEqual(evidence.verifier.eligible_ue_rows, 49_466)
        self.assertEqual(evidence.verifier.uniquely_joined_rows, 49_448)
        self.assertEqual(evidence.verifier.unmatched_rows, 18)
        self.assertEqual(evidence.verifier.ambiguous_rows, 0)
        self.assertEqual(evidence.verifier.mcs_mismatch_rows, 0)
        self.assertAlmostEqual(
            evidence.verifier.unique_join_fraction, 49_448 / 49_466
        )
        self.assertEqual(
            evidence.source_binding_sha256,
            "4d64c31522e4563ea9637fa6512dff18f7cef17892a4c9149ef22701d2b0bfb7",
        )
        self.assertEqual(
            evidence.evidence_binding_sha256,
            "bda0d2e661763c4f7fcd7a5c5f9d65bc590f7aa3ae6f3ae88b8f3577b8aaa946",
        )

    def test_frozen_split_is_contiguous_disjoint_and_complete(self) -> None:
        evidence = self.evidence
        self.assertEqual(len(evidence.segments), 4)
        for profile in subject.EXPECTED_PROFILE_TRACE_IDS:
            fit = evidence.segment(profile, subject.SequencePartition.FIT)
            validation = evidence.segment(
                profile, subject.SequencePartition.INTERNAL_VALIDATION
            )
            self.assertEqual(len(fit.observations), subject.FIT_DECISION_COUNT)
            self.assertEqual(
                len(validation.observations),
                subject.INTERNAL_VALIDATION_DECISION_COUNT,
            )
            self.assertEqual(
                validation.observations[0].decision_ordinal,
                fit.observations[-1].decision_ordinal + 1,
            )
            self.assertEqual(
                validation.observations[0].decision_timestamp_ns
                - fit.observations[-1].decision_timestamp_ns,
                subject.DECISION_PERIOD_NS,
            )
            fit_sources = {
                item.source_identity_sha256 for item in fit.observations
            }
            validation_sources = {
                item.source_identity_sha256 for item in validation.observations
            }
            self.assertFalse(fit_sources & validation_sources)
            self.assertEqual(len(fit.transitions), len(fit.observations) - 1)
            self.assertEqual(
                len(validation.transitions), len(validation.observations) - 1
            )

    def test_segment_plan_is_frozen_from_manifest_without_reading_csv(self) -> None:
        root = Path(__file__).resolve().parents[2]
        manifest = json.loads(
            (
                root
                / subject.SOURCE_RUN_RELATIVE_PATH
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        with mock.patch.object(
            subject, "_read_exact_csv", side_effect=AssertionError("CSV opened")
        ):
            plans = subject.predeclare_segment_plans(manifest)
        self.assertEqual(
            [(item.partition, item.decision_count) for item in plans],
            [
                (subject.SequencePartition.FIT, 174),
                (subject.SequencePartition.INTERNAL_VALIDATION, 75),
                (subject.SequencePartition.FIT, 174),
                (subject.SequencePartition.INTERNAL_VALIDATION, 75),
            ],
        )

    def test_every_real_observation_is_strictly_prior_and_fresh(self) -> None:
        for segment in self.evidence.segments:
            for observation in segment.observations:
                self.assertIs(observation.validity, subject.McsValidity.VALID)
                self.assertLess(
                    observation.source_timestamp_ns,
                    observation.decision_timestamp_ns,
                )
                self.assertLessEqual(observation.age_ns, 100_000_000)

    def test_dynamic_profiles_retain_time_varying_mcs(self) -> None:
        expected = {
            "MID_VARIABLE": set(range(9, 28)),
            "FADE_RECOVERY": {
                9, 10, 11, 13, 15, 16, 17, 18,
                19, 20, 21, 22, 24, 25, 26, 27,
            },
        }
        for profile in subject.EXPECTED_PROFILE_TRACE_IDS:
            values = {
                observation.mcs_index
                for partition in subject.SequencePartition
                for observation in self.evidence.segment(
                    profile, partition
                ).observations
            }
            self.assertEqual(values, expected[profile])
            self.assertGreater(max(values), min(values))

    def test_policy_surface_contains_only_mcs_not_hidden_identity(self) -> None:
        observation = self.evidence.segments[0].observations[0]
        self.assertEqual(
            set(observation.policy_feature_dict()), {"prior_ul_mcs_index"}
        )
        text = json.dumps(observation.policy_feature_dict(), sort_keys=True)
        for forbidden in (
            "MID_VARIABLE", "FADE_RECOVERY", "trace", "profile", "backlog",
            "payload", "action",
        ):
            self.assertNotIn(forbidden, text)
        self.assertNotIn("profile", {field.name for field in dataclasses.fields(
            subject.DynamicMcsTransitionV1
        )})

    def test_latest_strictly_prior_rejects_equal_timestamp(self) -> None:
        grants = (_grant(100, 4, 1), _grant(200, 9, 2))
        selected = subject.select_latest_strictly_prior_mcs(
            grants,
            decision_ordinal=1,
            decision_timestamp_ns=200,
            maximum_age_ns=1_000,
        )
        self.assertEqual(selected.mcs_index, 4)
        self.assertEqual(selected.source_timestamp_ns, 100)

    def test_missing_and_stale_are_explicit_and_never_numeric_zero(self) -> None:
        missing = subject.select_latest_strictly_prior_mcs(
            (_grant(200, 0, 1),),
            decision_ordinal=1,
            decision_timestamp_ns=100,
            maximum_age_ns=50,
        )
        stale = subject.select_latest_strictly_prior_mcs(
            (_grant(100, 0, 1),),
            decision_ordinal=2,
            decision_timestamp_ns=200,
            maximum_age_ns=50,
        )
        self.assertIs(missing.validity, subject.McsValidity.MISSING)
        self.assertIs(stale.validity, subject.McsValidity.STALE)
        self.assertIsNone(missing.mcs_index)
        self.assertIsNone(stale.mcs_index)
        self.assertEqual(stale.age_ns, 100)
        for observation in (missing, stale):
            with self.assertRaises(subject.CausalSelectionError):
                observation.policy_feature_dict()

    def test_no_forward_fill_across_stale_interval(self) -> None:
        grant = _grant(100, 11, 1)
        first = subject.select_latest_strictly_prior_mcs(
            (grant,),
            decision_ordinal=1,
            decision_timestamp_ns=150,
            maximum_age_ns=60,
        )
        second = subject.select_latest_strictly_prior_mcs(
            (grant,),
            decision_ordinal=2,
            decision_timestamp_ns=250,
            maximum_age_ns=60,
        )
        self.assertEqual(first.mcs_index, 11)
        self.assertIsNone(second.mcs_index)
        self.assertIs(second.validity, subject.McsValidity.STALE)

    def test_terminal_transition_requires_reset(self) -> None:
        segment = self.evidence.segments[0]
        current, successor = segment.transition_at(0).policy_values()
        self.assertEqual(current, segment.observations[0].mcs_index)
        self.assertEqual(successor, segment.observations[1].mcs_index)
        with self.assertRaises(subject.SegmentBoundaryResetRequired):
            segment.transition_at(len(segment.observations) - 1)

    def test_production_stays_fail_closed(self) -> None:
        self.assertIsNone(
            subject.REGISTERED_COMPOSITE_DYNAMIC_MCS_BINDING_SHA256
        )
        with self.assertRaisesRegex(
            subject.ProductionDynamicMcsUnavailable, "no reviewed composite"
        ):
            subject.require_production_dynamic_mcs_binding(
                self.evidence,
                composite_verifier_manifest_sha256="a" * 64,
            )

    def test_any_required_hash_drift_fails_before_csv_parse(self) -> None:
        real = subject._sha256_file

        def poisoned(path: Path) -> str:
            if path.as_posix().endswith(str(subject.UE_DCI_RELATIVE_PATH)):
                return "0" * 64
            return real(path)

        with mock.patch.object(subject, "_sha256_file", side_effect=poisoned):
            with mock.patch.object(
                subject, "_read_exact_csv", side_effect=AssertionError("parsed")
            ):
                with self.assertRaises(subject.EvidenceHashMismatch):
                    subject.load_dynamic_mcs_evidence(
                        maximum_age_ns=100_000_000
                    )

    def test_missing_source_root_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(subject.EvidenceHashMismatch):
                subject.load_dynamic_mcs_evidence(
                    maximum_age_ns=100_000_000,
                    repository_root=Path(directory),
                )

    def test_import_performs_no_evidence_io_or_runtime_launch(self) -> None:
        module = "rl_agent.splitfusion_hybrid_sac_run4_v1.dynamic_mcs_sequence"
        code = f'''\
import pathlib, subprocess, socket
real_open = pathlib.Path.open
def guarded(self, *args, **kwargs):
    if "ue_snr_bridge_qualification_v1/20260923_220340" in self.as_posix():
        raise AssertionError("evidence read during import")
    return real_open(self, *args, **kwargs)
pathlib.Path.open = guarded
subprocess.Popen = lambda *a, **k: (_ for _ in ()).throw(AssertionError("process"))
socket.socket = lambda *a, **k: (_ for _ in ()).throw(AssertionError("socket"))
__import__("{module}")
'''
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
