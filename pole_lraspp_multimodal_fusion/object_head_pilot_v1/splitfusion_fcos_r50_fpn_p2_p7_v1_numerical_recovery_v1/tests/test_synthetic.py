from __future__ import annotations

import math
import json
import tempfile
import unittest
from pathlib import Path

import torch

from ..audit import audit_tree, require_finite_audit
from ..contracts import load_recovery_config, verify_original_provenance
from ..envelope import build_healthy_envelope
from ..guards import PreStepBreaker
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
            self.assertTrue(record["model_optimizer_unchanged"])
        self.assertEqual(before, (model_hash(model), optimizer_hash(optimizer)))

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


if __name__ == "__main__":
    unittest.main()
