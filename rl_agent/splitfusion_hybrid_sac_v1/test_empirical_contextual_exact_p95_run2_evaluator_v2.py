"""Focused synthetic tests for the isolated Run-2-v2 evaluator."""

from __future__ import annotations

import inspect
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from . import empirical_contextual_exact_p95_run2_evaluator_v2 as evaluator
from .empirical_contextual_smoke_runner import EmpiricalSmokeConfigV1


def _row(
    source: str,
    seed: int,
    update: int,
    panel_index: int,
    *,
    quality: float,
    admission: float,
    p95: float,
) -> evaluator.ExactP95Run2EvaluationRowV2:
    base, shaped, emitted = evaluator._score_v2(
        p_admit=admission, quality=quality, latency_p95_ms=p95
    )
    profile = evaluator.NETWORK_PROFILE_ORDER[panel_index % 4]
    d1 = admission * (quality - 0.25 * ((p95 - 20.0) / 200.0)) + (
        1.0 - admission
    ) * -1.0
    actor_source = source in (evaluator.SOURCE_V2, evaluator.SOURCE_RUN1)
    return evaluator.ExactP95Run2EvaluationRowV2(
        source_kind=source,
        seed=seed,
        update_index=update,
        actor_state_sha256="a" * 64 if actor_source else "",
        checkpoint_sha256="b" * 64 if actor_source else "",
        checkpoint_file_sha256="c" * 64 if actor_source else "",
        panel_index=panel_index,
        scene_rank=panel_index // 4,
        sample_id=f"scene-{panel_index // 4}",
        episode_id=f"episode-{panel_index // 4}",
        frame_id=1000 + panel_index // 4,
        network_profile=profile,
        radio_csv_row_number=panel_index,
        radio_row_sha256="d" * 64,
        executed_mode_id=11,
        requested_q=0.6,
        executed_q_e4=6000,
        q_perc=quality,
        p_edge_admission_given_sent=admission,
        total_transmitted_bytes=1000.0,
        datagram_count=1,
        latency_proxy_p50_ms=p95 - 20.0,
        latency_proxy_p95_ms=p95,
        latency_proxy_p99_ms=p95 + 20.0,
        p95_margin_ms=200.0 - p95,
        p95_miss=int(p95 > 200.0),
        d1_reward64=d1,
        d1_reward64_bits_hex=evaluator._float64_bits_hex(d1),
        v2_base_reward64=base,
        v2_base_reward64_bits_hex=evaluator._float64_bits_hex(base),
        v2_shaped_reward64=shaped,
        v2_shaped_reward64_bits_hex=evaluator._float64_bits_hex(shaped),
        v2_emitted_reward_float32=emitted,
        v2_emitted_reward_float32_bits_hex=evaluator._float32_bits_hex(emitted),
    )


def _complete_synthetic_rows():
    rows = []
    for panel in range(340):
        rows.append(
            _row(
                evaluator.SOURCE_ORACLE,
                evaluator.POOLED_SEED,
                -1,
                panel,
                quality=0.80,
                admission=1.0,
                p95=150.0,
            )
        )
        rows.append(
            _row(
                evaluator.SOURCE_FIXED,
                evaluator.POOLED_SEED,
                -1,
                panel,
                quality=0.60,
                admission=1.0,
                p95=180.0,
            )
        )
    for seed in (17, 29, 43):
        for panel in range(340):
            rows.append(
                _row(
                    evaluator.SOURCE_RANDOM,
                    seed,
                    -1,
                    panel,
                    quality=0.35,
                    admission=0.95,
                    p95=230.0,
                )
            )
            rows.append(
                _row(
                    evaluator.SOURCE_RUN1,
                    seed,
                    5000,
                    panel,
                    quality=0.62,
                    admission=1.0,
                    p95=220.0,
                )
            )
            rows.append(
                _row(
                    evaluator.SOURCE_V2,
                    seed,
                    0,
                    panel,
                    quality=0.55,
                    admission=1.0,
                    p95=220.0,
                )
            )
            rows.append(
                _row(
                    evaluator.SOURCE_V2,
                    seed,
                    5000,
                    panel,
                    quality=0.76,
                    admission=1.0,
                    p95=170.0,
                )
            )
    return tuple(rows)


def _full_schedule_synthetic_rows():
    rows = list(_complete_synthetic_rows())
    finals = [
        row
        for row in rows
        if row.source_kind == evaluator.SOURCE_V2 and row.update_index == 5000
    ]
    for update in evaluator.EVALUATION_UPDATES_V2[1:-1]:
        rows.extend(replace(row, update_index=update) for row in finals)
    return tuple(rows)


class ExactP95Run2EvaluatorV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.python_state = random.getstate()
        cls.torch_state = torch.get_rng_state().clone()
        cls.cuda_before = torch.cuda.is_initialized()

    @classmethod
    def tearDownClass(cls) -> None:
        if random.getstate() != cls.python_state:
            raise AssertionError("tests advanced global Python RNG")
        if not torch.equal(torch.get_rng_state(), cls.torch_state):
            raise AssertionError("tests advanced global Torch RNG")
        if not cls.cuda_before and torch.cuda.is_initialized():
            raise AssertionError("tests initialized CUDA")

    def test_protocol_is_frozen_before_outcomes(self) -> None:
        document = evaluator.evaluation_protocol_document_v2()
        self.assertEqual(
            evaluator.EVALUATION_PROTOCOL_V2_SHA256,
            "02e7f34702fc13d325047e3e32d2473a7129a2854e0cf4048b042626d7e22e50",
        )
        self.assertEqual(
            evaluator.canonical_sha256(document),
            evaluator.EVALUATION_PROTOCOL_V2_SHA256,
        )
        self.assertEqual(document["primary_update"], 5000)
        self.assertEqual(document["adaptation"]["bootstrap"]["resamples"], 10000)
        self.assertEqual(
            document["frozen_bindings"]["training_cli_commit"],
            "da58f919fcea41ef4b6c95881d6d773bef256a2e",
        )

    def test_binary64_order_float32_emission_and_exact_boundary(self) -> None:
        base, shaped, emitted = evaluator._score_v2(
            p_admit=0.75, quality=0.6, latency_p95_ms=200.0
        )
        expected = 0.75 * (0.6 - 0.25 * (200.0 / 200.0)) + 0.25 * -1.0
        self.assertEqual(base, expected)
        self.assertEqual(shaped, base)
        self.assertEqual(
            evaluator._float32_bits_hex(emitted),
            evaluator._float32_bits_hex(float(torch.tensor(base, dtype=torch.float32))),
        )
        _base2, shaped2, _emitted2 = evaluator._score_v2(
            p_admit=0.75,
            quality=0.6,
            latency_p95_ms=float.fromhex("0x1.9000000000001p+7"),
        )
        self.assertLess(shaped2, shaped)

    def test_row_rejects_noninteger_or_wrong_p95_miss(self) -> None:
        row = _row(
            evaluator.SOURCE_FIXED, -1, -1, 0, quality=0.6, admission=1.0, p95=201.0
        )
        self.assertEqual(row.p95_miss, 1)
        document = row.to_canonical_dict()
        document["p95_miss"] = 0
        with self.assertRaises(evaluator.ExactP95Run2EvaluationV2Error):
            evaluator.ExactP95Run2EvaluationRowV2(**document)

    def test_row_authenticates_d1_reward_and_exact_q_boundary(self) -> None:
        row = _row(
            evaluator.SOURCE_FIXED,
            -1,
            -1,
            0,
            quality=0.6,
            admission=0.9,
            p95=190.0,
        )
        reward_tamper = row.to_canonical_dict()
        reward_tamper["d1_reward64"] += 0.01
        reward_tamper["d1_reward64_bits_hex"] = evaluator._float64_bits_hex(
            reward_tamper["d1_reward64"]
        )
        with self.assertRaises(evaluator.ExactP95Run2EvaluationV2Error):
            evaluator.ExactP95Run2EvaluationRowV2(**reward_tamper)
        q_tamper = row.to_canonical_dict()
        q_tamper["requested_q"] = 0.7
        with self.assertRaises(evaluator.ExactP95Run2EvaluationV2Error):
            evaluator.ExactP95Run2EvaluationRowV2(**q_tamper)

    def test_aggregate_normalized_progress_is_unclamped(self) -> None:
        rows = _complete_synthetic_rows()
        aggregate = evaluator.aggregate_evaluation_rows_v2(rows)
        final = next(
            row
            for row in aggregate
            if row["source_kind"] == evaluator.SOURCE_V2
            and row["seed"] == evaluator.POOLED_SEED
            and row["update_index"] == 5000
            and row["network_profile"] == "ALL_PROFILES"
        )
        self.assertGreater(final["normalized_random_to_oracle_progress_unclamped"], 0.0)
        self.assertIsInstance(final["p95_miss_count"], int)
        profiled = next(
            row
            for row in aggregate
            if row["source_kind"] == evaluator.SOURCE_V2
            and row["seed"] == 17
            and row["update_index"] == 5000
            and row["network_profile"]
            == evaluator.NETWORK_PROFILE_ORDER[0]
        )
        self.assertIsNotNone(profiled["contextual_oracle_regret_mean"])
        self.assertIsNotNone(
            profiled["normalized_random_to_oracle_progress_unclamped"]
        )

    def test_success_tiers_apply_registered_thresholds(self) -> None:
        rows = _complete_synthetic_rows()
        result = evaluator.evaluate_success_tiers_v2(
            rows,
            mechanics=evaluator.evaluation_protocol_document_v2()[
                "success_tiers"
            ]["tier0"],
        )
        self.assertTrue(result["tier0"]["passed"])
        self.assertTrue(result["tier1"]["passed"])
        self.assertTrue(result["tier2"]["passed"])
        self.assertTrue(result["tier3"]["passed"])
        self.assertEqual(result["tier1"]["pooled_final_miss_count"], 0)
        incomplete = dict(
            evaluator.evaluation_protocol_document_v2()["success_tiers"]["tier0"]
        )
        incomplete.pop("finite_rows")
        self.assertFalse(
            evaluator.evaluate_success_tiers_v2(
                rows, mechanics=incomplete
            )["tier0"]["passed"]
        )
        self.assertIsNone(evaluator._relative_reduction(1, 0))

    def test_scene_cluster_bootstrap_is_deterministic_and_paired(self) -> None:
        rows = _complete_synthetic_rows()
        first = evaluator.paired_scene_cluster_bootstrap_v2(
            rows, resamples=200, seed=20260921
        )
        second = evaluator.paired_scene_cluster_bootstrap_v2(
            rows, resamples=200, seed=20260921
        )
        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap_cluster_count"], 85)
        self.assertEqual(first["positive_seed_count"], 3)
        self.assertTrue(first["passes_all"])

    def test_exact_checkpoint_loader_rejects_v1_or_foreign_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "foreign.pt"
            torch.save(EmpiricalSmokeConfigV1(), path)
            with self.assertRaises(evaluator.ExactP95Run2EvaluationV2Error):
                evaluator._load_exact_v2_checkpoint(path)

    def test_all_eight_registered_figures_render_png_and_pdf(self) -> None:
        rows = _full_schedule_synthetic_rows()
        evaluator._validate_collected_population(rows)
        with self.assertRaises(evaluator.ExactP95Run2EvaluationV2Error):
            evaluator._validate_collected_population(rows[:-1])
        aggregates = evaluator.aggregate_evaluation_rows_v2(rows)
        adaptation = evaluator.paired_scene_cluster_bootstrap_v2(
            rows, resamples=20, seed=evaluator.BOOTSTRAP_SEED_V2
        )
        with tempfile.TemporaryDirectory() as directory:
            names = evaluator._write_figures(
                Path(directory), rows, aggregates, adaptation
            )
            self.assertEqual(len(names), 16)
            self.assertEqual(len(set(names)), 16)
            self.assertTrue(all((Path(directory) / name).stat().st_size > 0 for name in names))

    def test_source_has_no_optimizer_replay_training_or_live_launch(self) -> None:
        source = inspect.getsource(evaluator)
        for forbidden in (
            "torch.optim",
            "ExactP95Run2ReplayV2(",
            "ExactP95Run2RunnerV2(",
            ".environment.reset(",
            ".environment.step(",
            "subprocess",
            "docker",
            "carla.Client",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
