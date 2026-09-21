"""Focused CPU-only tests for the fixed-panel actor evaluator."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from .empirical_contextual_contract import (
    DIRECT_QUALITY_COMPONENT,
    PILOT_UTILITY_SPEC,
    fixed_stage_latency_ms,
)
from .empirical_contextual_fit_validation_evaluator import (
    EVALUATION_CHECKPOINT_UPDATES,
    MCS_ROUNDING_RULE_SHA256,
    FitValidationActorEvaluatorV1,
    _initial_actor,
    _round_panel_mcs,
    aggregate_evaluation_rows,
    load_actor_for_evaluation,
)
from .empirical_contextual_fit_validation_panel import (
    REGISTERED_FIT_VALIDATION_PANEL_SHA256,
)
from .payload_network_surrogate import UDP_PAYLOAD_CAPACITY_BYTES


class FitValidationEvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.campaign = cls.root / (
            "experiments/splitfusion_hybrid_sac_preliminary_baseline_v1/"
            "20260921_train_split_5000x3_v1"
        )
        cls.python_rng = random.getstate()
        cls.torch_rng = torch.get_rng_state().clone()
        cls.cuda_before = torch.cuda.is_initialized()
        cls.evaluator = FitValidationActorEvaluatorV1(project_root=cls.root)
        cls.actor, cls.actor_identity = _initial_actor(17)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.evaluator.close()

    def test_panel_is_complete_and_identity_join_is_exact(self) -> None:
        panel = self.evaluator.panel
        self.assertEqual(panel.canonical_sha256(), REGISTERED_FIT_VALIDATION_PANEL_SHA256)
        self.assertEqual(len(panel.entries), 340)
        self.assertEqual(len({row.scene_sample_id for row in panel.entries}), 85)
        self.assertEqual(
            {(row.scene_sample_id, row.network_profile) for row in panel.entries},
            {
                (row.scene_sample_id, profile)
                for row in panel.entries[::4]
                for profile in panel.profile_order
            },
        )
        for entry in panel.entries:
            context, radio = self.evaluator._join_entry(entry)
            self.assertEqual(context.sample_id, entry.scene_sample_id)
            self.assertEqual(radio.row_sha256, entry.radio_row_sha256)

    def test_half_integer_mcs_is_identity_deterministic_not_ambient_rng(self) -> None:
        half_entries = []
        for entry in self.evaluator.panel.entries:
            _context, radio = self.evaluator._join_entry(entry)
            if radio.mcs_median - math.floor(radio.mcs_median) == 0.5:
                half_entries.append((entry, radio))
        self.assertTrue(half_entries)
        before_python = random.getstate()
        before_torch = torch.get_rng_state().clone()
        first = [_round_panel_mcs(*pair) for pair in half_entries]
        random.seed(991177)
        torch.manual_seed(881166)
        second = [_round_panel_mcs(*pair) for pair in half_entries]
        random.setstate(before_python)
        torch.set_rng_state(before_torch)
        self.assertEqual(first, second)
        self.assertEqual(len(MCS_ROUNDING_RULE_SHA256), 64)
        self.assertTrue(
            all(value in (math.floor(row.mcs_median), math.ceil(row.mcs_median))
                for (value, _status), (_entry, row) in zip(first, half_entries))
        )

    def test_update_zero_actor_and_entry_are_bit_deterministic(self) -> None:
        actor_2, identity_2 = _initial_actor(17)
        self.assertEqual(self.actor_identity, identity_2)
        for name, value in self.actor.state_dict().items():
            self.assertTrue(torch.equal(value, actor_2.state_dict()[name]))
        entry = self.evaluator.panel.entries[17]
        first = self.evaluator.evaluate_entry(
            actor=self.actor, actor_identity=self.actor_identity, entry=entry
        )
        second = self.evaluator.evaluate_entry(
            actor=actor_2, actor_identity=identity_2, entry=entry
        )
        self.assertEqual(first, second)

    def test_reward_matches_exact_d1_quality_network_and_utility(self) -> None:
        entry = self.evaluator.panel.entries[123]
        row = self.evaluator.evaluate_entry(
            actor=self.actor, actor_identity=self.actor_identity, entry=entry
        )
        query = self.evaluator.environment._surface.query_fit_q_e4(
            row.sample_id, row.executed_mode_id, row.executed_q_e4
        )
        component = query.policy.component(DIRECT_QUALITY_COMPONENT)
        self.assertTrue(component.valid)
        self.assertEqual(row.q_perc, float(component.value))
        payload = float(query.policy.payload.total_transmitted_bytes)
        datagrams = math.ceil(payload / UDP_PAYLOAD_CAPACITY_BYTES)
        prediction = self.evaluator.environment._prediction_session.predict(
            network_profile=row.network_profile,
            payload_bytes=payload,
            datagram_count=datagrams,
        )
        latency = prediction.conditional_retained_survivor_latency_model()
        probability = (
            prediction.p_complete_reassembly_given_sent
            * prediction.p_edge_admission_given_reassembled
        )
        expected = PILOT_UTILITY_SPEC.expected_utility(
            p_edge_admission_given_sent=probability,
            q_perc=float(component.value),
            latency_proxy_ms=fixed_stage_latency_ms() + latency.p50_ms,
        )
        self.assertEqual(row.total_transmitted_bytes, payload)
        self.assertEqual(row.datagram_count, datagrams)
        self.assertEqual(row.p_edge_admission_given_sent, probability)
        self.assertEqual(row.latency_proxy_ms, fixed_stage_latency_ms() + latency.p50_ms)
        self.assertEqual(row.reward, expected)

    def test_full_actor_panel_coverage_and_aggregates(self) -> None:
        rows = self.evaluator.evaluate_actor(
            actor=self.actor, actor_identity=self.actor_identity
        )
        self.assertEqual(len(rows), 340)
        self.assertEqual(tuple(row.panel_index for row in rows), tuple(range(340)))
        aggregates = aggregate_evaluation_rows(rows)
        self.assertEqual(len(aggregates), 5)
        self.assertEqual(
            sorted(row["panel_row_count"] for row in aggregates),
            [85, 85, 85, 85, 340],
        )

    def test_numbered_checkpoint_identity_and_training_files_are_read_only(self) -> None:
        if not self.campaign.exists():
            self.skipTest("completed preliminary campaign is not present")
        seed_dir = self.campaign / "seed_17"
        watched = (
            seed_dir / "config.json",
            seed_dir / "bindings.json",
            seed_dir / "report.json",
            seed_dir / "metrics.csv",
            seed_dir / "checkpoints/checkpoint_000500.pt",
        )
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in watched}
        actor, identity = load_actor_for_evaluation(
            campaign_directory=self.campaign, seed=17, update_index=500
        )
        self.assertEqual(identity.seed, 17)
        self.assertEqual(identity.update_index, 500)
        self.assertEqual(len(identity.actor_state_sha256), 64)
        self.assertEqual(len(identity.checkpoint_canonical_sha256), 64)
        self.assertEqual(len(identity.checkpoint_file_sha256 or ""), 64)
        row = self.evaluator.evaluate_entry(
            actor=actor, actor_identity=identity, entry=self.evaluator.panel.entries[0]
        )
        self.assertEqual(row.checkpoint_canonical_sha256, identity.checkpoint_canonical_sha256)
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in watched}
        self.assertEqual(before, after)

    def test_no_global_rng_or_cuda_side_effect(self) -> None:
        self.assertEqual(self.python_rng, random.getstate())
        self.assertTrue(torch.equal(self.torch_rng, torch.get_rng_state()))
        if not self.cuda_before:
            self.assertFalse(torch.cuda.is_initialized())

    def test_registered_checkpoint_cadence(self) -> None:
        self.assertEqual(
            EVALUATION_CHECKPOINT_UPDATES,
            (0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000),
        )


class EvaluatorImportAuditTest(unittest.TestCase):
    def test_import_opens_no_experiment_and_initializes_no_cuda(self) -> None:
        code = r'''
import sys
opened = []
def audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        value = args[0].decode() if isinstance(args[0], bytes) else args[0]
        if "/experiments/" in value:
            opened.append(value)
    if event in ("socket.__new__", "subprocess.Popen"):
        raise RuntimeError(event)
sys.addaudithook(audit)
import torch
before = torch.cuda.is_initialized()
import rl_agent.splitfusion_hybrid_sac_v1.empirical_contextual_fit_validation_evaluator
print(json.dumps({"before": before, "after": torch.cuda.is_initialized(), "opened": opened}))
'''
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", "import json\n" + code],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result, {"before": False, "after": False, "opened": []})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
