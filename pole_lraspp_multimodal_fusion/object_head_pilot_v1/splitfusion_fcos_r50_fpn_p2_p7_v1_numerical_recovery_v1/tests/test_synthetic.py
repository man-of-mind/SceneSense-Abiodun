from __future__ import annotations

import math
import json
import inspect
import tempfile
import unittest
from pathlib import Path

import torch

from ..audit import audit_tree, require_finite_audit
from ..continue_scientific import (_expected_scheduler, _select_verified_resume_checkpoint,
                                   _verify_resume_provenance)
from ..contracts import (canonical_hash, current_commit, load_recovery_config, package_hashes, sha256,
                         verify_original_provenance)
from ..envelope import build_healthy_envelope
from ..guards import PreStepBreaker, proposed_sgd_metrics
from ..qualify_recovery import disposable_execution_accounting
from ..runner import aggregate_required_reachability, run_guarded_epoch
from ..safe_math import exp_dimensions_fp64, normalize_yaw_fp32
from ..state_guard import DiagnosticStateGuard, model_hash, optimizer_hash


class YawTests(unittest.TestCase):
    def test_zero_near_ordinary_large_and_autograd(self) -> None:
        raw = torch.tensor([[0.0, 0.0], [1e-8, -2e-8], [3.0, 4.0], [1e20, -1e20]],
                           dtype=torch.float64, requires_grad=True)
        result = normalize_yaw_fp32(raw, 1e-4)
        self.assertEqual(result.value.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result.value).all())
        self.assertEqual(result.diagnostics["below_tau_count"], 2)
        self.assertEqual(result.diagnostics["fallback_used"], False)
        result.value.square().sum().backward()
        self.assertIsNotNone(raw.grad); self.assertTrue(torch.isfinite(raw.grad).all())

    def test_ordinary_equation_is_exactly_unchanged(self) -> None:
        raw = torch.tensor([[3.0, 4.0], [-8.0, 6.0]], dtype=torch.float32)
        recovered = normalize_yaw_fp32(raw, 1e-2).value
        original_equation = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
        self.assertTrue(torch.equal(recovered, original_equation))
        self.assertTrue(torch.equal(torch.linalg.vector_norm(recovered, dim=-1), torch.ones(2)))

    def test_tau_is_fail_closed(self) -> None:
        raw = torch.ones(1, 2)
        for tau in (None, float("nan"), 2e-4):
            with self.assertRaises(RuntimeError):
                normalize_yaw_fp32(raw, tau)
        with self.assertRaises(FloatingPointError):
            normalize_yaw_fp32(torch.tensor([[float("inf"), 0.0]]), 1e-4)


class DimensionAndAuditTests(unittest.TestCase):
    def test_dimension_exp_and_empty_candidates(self) -> None:
        decoded = exp_dimensions_fp64(torch.tensor([[0.0, math.log(2.0), -2.0]]))
        self.assertEqual(decoded.dtype, torch.float64); self.assertTrue((decoded > 0).all())
        empty = exp_dimensions_fp64(torch.empty(0, 3))
        self.assertEqual(tuple(empty.shape), (0, 3))
        records = audit_tree({"candidates": empty, "scores": torch.empty(0),
                              "targets": torch.empty(0, dtype=torch.int64)}, "empty")
        require_finite_audit(records)
        self.assertTrue(all({"dtype", "shape", "finite", "min", "max", "absmax"} <= set(row)
                            for row in records.values()))

    def test_extreme_finite_representable_dimensions_and_edge_audits(self) -> None:
        dimensions = exp_dimensions_fp64(torch.tensor([[700.0, -740.0, 0.0]], dtype=torch.float64))
        yaw = normalize_yaw_fp32(torch.tensor([[1e-8, -1e-8]], requires_grad=True), 1e-4)
        records = audit_tree({"normal_candidates": torch.tensor([[0.2, 3.0, 4.0]]),
                              "zero_detections": torch.empty(0, 4),
                              "extreme_dimensions": dimensions, "near_zero_raw_yaw": torch.tensor([[1e-8, -1e-8]]),
                              "near_zero_raw_norm": yaw.raw_norm, "near_zero_normalized_yaw": yaw.value,
                              "empty_targets": torch.empty(0, 4)}, "edge_cases")
        require_finite_audit(records)
        self.assertEqual(records["edge_cases.zero_detections"]["shape"], [0, 4])

    def test_extreme_dimensions_fail_without_clamp(self) -> None:
        with self.assertRaises(OverflowError):
            exp_dimensions_fp64(torch.tensor([[1000.0, 0.0, 0.0]], dtype=torch.float64))
        with self.assertRaises(FloatingPointError):
            exp_dimensions_fp64(torch.tensor([[float("nan"), 0.0, 0.0]]))


class BreakerAndStateTests(unittest.TestCase):
    def _fixture(self) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.GroupNorm(1, 2))
        optimizer = torch.optim.SGD([{"params": list(model.parameters()), "name": "new", "lr": 0.01}],
                                    momentum=0.9, weight_decay=1e-4)
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        return model, optimizer

    def test_breaker_does_not_mutate_or_step_and_zero_is_not_abort(self) -> None:
        model, optimizer = self._fixture()
        ceilings = {"gradient_norm": {"new": 1.0, "global": 1.0}, "momentum_norm": {"new": 1.0},
                    "proposed_sgd_update_norm": {"new": 1.0}, "max_parameter_relative_update": 1.0}
        with tempfile.TemporaryDirectory() as temporary:
            breaker = PreStepBreaker(ceilings, Path(temporary))
            before = (model_hash(model), optimizer_hash(optimizer))
            record = breaker.check(model, optimizer, epoch=10, update_in_epoch=1, global_update_if_stepped=9469)
            self.assertEqual(record["action"], "allow_step")
            self.assertEqual(before, (model_hash(model), optimizer_hash(optimizer)))
            self.assertEqual(len(optimizer.state), 0)

    def test_breaker_abort_is_before_step(self) -> None:
        model, optimizer = self._fixture()
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 100.0)
        ceilings = {"gradient_norm": {"new": 0.1, "global": 0.1}, "momentum_norm": {"new": 0.1},
                    "proposed_sgd_update_norm": {"new": 0.1}, "max_parameter_relative_update": 0.1}
        before = (model_hash(model), optimizer_hash(optimizer))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FloatingPointError):
                PreStepBreaker(ceilings, Path(temporary)).check(
                    model, optimizer, epoch=10, update_in_epoch=447, global_update_if_stepped=9915,
                    context={"sample_ids": ["synthetic_a", "synthetic_b"]})
            record = json.loads((Path(temporary) / "BREAKER_E010_U0447.json").read_text())
            self.assertEqual(record["sample_ids"], ["synthetic_a", "synthetic_b"])
            self.assertEqual(record["action"], "abort_before_optimizer_step")
            self.assertEqual(record["nonmutation_proof"], "unit_tests_and_qualification_boundary_hashes")
        self.assertEqual(before, (model_hash(model), optimizer_hash(optimizer)))

    def test_streaming_sgd_metrics_match_reference_without_mutation(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[3.0, 4.0]]))
        model.weight.grad = torch.tensor([[1.0, 2.0]])
        optimizer = torch.optim.SGD([{"params": [model.weight], "name": "new", "lr": 0.1}],
                                    momentum=0.9, weight_decay=0.01)
        optimizer.state[model.weight]["momentum_buffer"] = torch.tensor([[0.5, -0.5]])
        before = (model_hash(model), optimizer_hash(optimizer))
        metrics = proposed_sgd_metrics(model, optimizer)
        direction = torch.tensor([[1.03, 2.04]], dtype=torch.float64)
        expected_buffer = 0.9 * torch.tensor([[0.5, -0.5]], dtype=torch.float64) + direction
        self.assertAlmostEqual(metrics["gradient_norm"]["new"], math.sqrt(5.0), places=12)
        self.assertAlmostEqual(metrics["momentum_norm"]["new"], math.sqrt(0.5), places=12)
        self.assertAlmostEqual(metrics["proposed_sgd_update_norm"]["new"],
                               0.1 * float(expected_buffer.norm()), places=7)
        self.assertAlmostEqual(metrics["max_parameter_relative_update"],
                               0.1 * float(expected_buffer.norm()) / 5.0, places=7)
        self.assertTrue(metrics["streaming_scalar_reductions"])
        self.assertEqual(metrics["retained_fp64_tensor_copies"], 0)
        self.assertEqual(before, (model_hash(model), optimizer_hash(optimizer)))
        self.assertNotIn("model_hash", inspect.getsource(PreStepBreaker.check))

    def test_full_diagnostic_state_restores(self) -> None:
        model, optimizer = self._fixture(); before = (model_hash(model), optimizer_hash(optimizer))
        with DiagnosticStateGuard(model, optimizer) as guard:
            with torch.no_grad():
                next(model.parameters()).add_(3.0)
            model.eval(); next(model.parameters()).requires_grad_(False); torch.rand(3)
        self.assertTrue(guard.report["restored_exactly"])
        self.assertEqual(before, (model_hash(model), optimizer_hash(optimizer)))


class EnvelopeAndProvenanceTests(unittest.TestCase):
    def test_ten_times_max_envelope(self) -> None:
        def row(update: int, scale: float) -> dict[str, object]:
            return {"source": "EPOCH10_EXPLICIT_REPLAY", "epoch": 10, "update_in_epoch": update, "finite": True,
                    "metrics": {"gradient_norm": {"pretrained_backbone": scale, "pretrained_fpn_heads": 2*scale,
                        "new": 3*scale, "global": 4*scale}, "momentum_norm": {"pretrained_backbone": scale,
                        "pretrained_fpn_heads": scale, "new": scale}, "proposed_sgd_update_norm": {
                        "pretrained_backbone": scale, "pretrained_fpn_heads": scale, "new": scale},
                        "max_parameter_relative_update": scale}}
        result = build_healthy_envelope([row(1, 1.0), row(2, 2.0)])
        self.assertEqual(result["ceilings"]["gradient_norm"]["global"], 80.0)
        self.assertEqual(result["threshold_formula"], "10 * maximum_healthy_value")

    def test_immutable_state_and_original_hashes(self) -> None:
        config = load_recovery_config()
        self.assertEqual(config["state"], "UNQUALIFIED_IMPLEMENTATION_ONLY")
        self.assertTrue(all(value == 0 for value in config["execution_counters"].values()))
        report = verify_original_provenance(checkpoint_metadata=False)
        self.assertEqual(report["checkpoint_sha256"], config["original"]["checkpoint_sha256"])


class ReviewDefectTests(unittest.TestCase):
    def test_aggregate_reachability_allows_isolated_zero(self) -> None:
        records = [
            {"yaw": {"required_this_stage": True, "finite": True, "nonzero": False},
             "p2": {"required_this_stage": True, "finite": True, "nonzero": True}},
            {"yaw": {"required_this_stage": True, "finite": True, "nonzero": True},
             "p2": {"required_this_stage": True, "finite": True, "nonzero": False}},
        ]
        report = aggregate_required_reachability(records)
        self.assertTrue(report["all_required_trainable_groups_finite_every_update"])
        self.assertTrue(report["all_required_trainable_groups_observed_nonzero"])
        self.assertEqual(report["groups"]["yaw"]["zero_update_count"], 1)
        self.assertNotIn("enforce_required_nonzero", inspect.signature(run_guarded_epoch).parameters)

    def test_aggregate_reachability_rejects_never_nonzero_group(self) -> None:
        report = aggregate_required_reachability([
            {"yaw": {"required_this_stage": True, "finite": True, "nonzero": False}}])
        self.assertTrue(report["all_required_trainable_groups_finite_every_update"])
        self.assertFalse(report["all_required_trainable_groups_observed_nonzero"])

    def test_disposable_execution_plan_has_no_candidate_replay_steps(self) -> None:
        report = disposable_execution_accounting(4)
        self.assertEqual(report["candidate_common_boundary_runs"], 4)
        self.assertEqual(report["candidate_boundary_optimizer_steps"], 0)
        self.assertEqual(report["selected_candidate_repeat_runs"], 1)
        self.assertEqual(report["total_disposable_optimizer_steps"], 1976)

    @staticmethod
    def _resume_fixture(root: Path, *, recovery: dict[str, object], partial: bool = False) -> tuple[dict[str, object], int]:
        for name in ("checkpoints", "training_metrics", "gradient_telemetry", "numerical_failures"):
            (root / name).mkdir(parents=True, exist_ok=True)
        config = {"scientific_seed": 20260829, "training": {
            "base_lrs": {"pretrained_backbone": 0.001, "pretrained_fpn_heads": 0.0025, "new": 0.01},
            "gamma": 0.1}}
        epoch = 10; global_update = 9468 + 1052
        if partial:
            (root / "training_metrics/epoch_010.json").write_text('{"epoch": 10}\n', encoding="utf-8")
            return config, global_update
        scheduler = _expected_scheduler(config, epoch)
        state = {"schema": "splitfusion_fcos_numerical_recovery_atomic_checkpoint_v1",
            "model": {"weight": torch.ones(1)},
            "optimizer": {"state": {}, "param_groups": [
                {"name": name, "lr": lr, "params": []} for name, lr in scheduler["lrs"].items()]},
            "scheduler": scheduler, "epoch": epoch, "global_optimizer_update": global_update,
            "amp": {"enabled": True, "dtype": "bfloat16", "grad_scaler": None},
            "rng": {"python": (), "numpy": (), "torch": torch.zeros(1, dtype=torch.uint8), "cuda": []},
            "sampler": {"length": 16827, "seed": 20260829, "epoch": epoch, "start_index": 0},
            "validation_accessed": False, "recovery": recovery}
        checkpoint = (root / "checkpoints/epoch_010.pt").resolve(); torch.save(state, checkpoint)
        (root / "checkpoints/epoch_010.json").write_text(json.dumps({"epoch": epoch, "path": str(checkpoint),
            "sha256": sha256(checkpoint), "global_optimizer_update": global_update}) + "\n", encoding="utf-8")
        for directory in ("training_metrics", "gradient_telemetry"):
            (root / directory / "epoch_010.json").write_text('{"epoch": 10}\n', encoding="utf-8")
        return config, global_update

    def test_verified_epoch_boundary_resume_selection(self) -> None:
        recovery = {"binding": "synthetic"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, expected_global = self._resume_fixture(root, recovery=recovery)
            epoch, global_update, state = _select_verified_resume_checkpoint(
                root, expected_recovery=recovery, config=config, train_frames=16827, effective_batch=16)
            self.assertEqual((epoch, global_update), (10, expected_global))
            self.assertEqual(state["recovery"], recovery)

    def test_epoch10_awaiting_review_is_an_exact_resume_boundary(self) -> None:
        recovery = {"binding": "prospective-epoch10"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, expected_global = self._resume_fixture(root, recovery=recovery)
            (root / "RECOVERED_EPOCH10_GATE_COMPLETE").write_text(
                "RECOVERED_EPOCH10_GATE_COMPLETE\n", encoding="utf-8")
            (root / "STATUS.json").write_text(json.dumps({
                "state": "awaiting_review", "epoch_complete": 10,
                "global_optimizer_update": expected_global, "validation_accessed": False,
                "epoch11_accessed": False}) + "\n", encoding="utf-8")
            epoch, global_update, _state = _select_verified_resume_checkpoint(
                root, expected_recovery=recovery, config=config, train_frames=16827,
                effective_batch=16)
            self.assertEqual((epoch, global_update), (10, expected_global))

    def test_partial_epoch_is_never_treated_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _global = self._resume_fixture(root, recovery={"binding": "synthetic"}, partial=True)
            with self.assertRaisesRegex(RuntimeError, "partial or noncontiguous"):
                _select_verified_resume_checkpoint(root, expected_recovery={"binding": "synthetic"},
                                                   config=config, train_frames=16827, effective_batch=16)

    def test_resume_checkpoint_rejects_binding_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _global = self._resume_fixture(root, recovery={"binding": "recorded"})
            with self.assertRaisesRegex(RuntimeError, "provenance verification failed"):
                _select_verified_resume_checkpoint(root, expected_recovery={"binding": "changed"},
                                                   config=config, train_frames=16827, effective_batch=16)

    def test_resume_experiment_provenance_is_exactly_bound(self) -> None:
        qualified = {"state": "synthetic"}; qualification = {"pass": True}
        original = {"checkpoint_sha256": "synthetic-checkpoint"}; hashes = {"artifact": "digest"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saved = {"schema": "splitfusion_fcos_scientific_recovery_v1",
                "source_commit": current_commit(), "source_files": package_hashes(),
                "source_files_sha256": canonical_hash(package_hashes()), "original_epoch9": original,
                "original_epochs_10_26": "CORRUPTED_FINITE_GRADIENT_TRAJECTORY_DO_NOT_USE",
                "qualified_config": qualified, "qualification_sha256": canonical_hash(qualification),
                "qualification_artifact_hashes": hashes, "start_epoch": 10, "validation_accessed": False}
            (root / "RECOVERY_PROVENANCE.json").write_text(json.dumps(saved) + "\n", encoding="utf-8")
            _verify_resume_provenance(root, qualified=qualified, qualification=qualification,
                                      provenance=original, qualification_hashes=hashes)
            with self.assertRaisesRegex(RuntimeError, "does not bind"):
                _verify_resume_provenance(root, qualified=qualified, qualification=qualification,
                                          provenance=original, qualification_hashes={"artifact": "changed"})


if __name__ == "__main__":
    unittest.main()
