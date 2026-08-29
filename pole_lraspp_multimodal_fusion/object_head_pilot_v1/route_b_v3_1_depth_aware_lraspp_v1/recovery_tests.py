from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from common import CONFIG_PATH, load_json, read_csv, sha256, utc_now, write_json_x
from data import load_objects
from decode import (decode_branch, inference_exp_float64, inference_expm1_float64)
from losses import log_dimension_loss
from model import COMMON_FIELDS, build_model
from train import latest_checkpoint


def dimension_loss_tests() -> dict[str, Any]:
    values = torch.tensor([-120.0, -72.0, -43.0, -13.8, 0.0, 31.0, 80.0, 95.5, 120.0],
                          dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([0.01, 0.02, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], dtype=torch.float32)
    loss = log_dimension_loss(values[:, None], targets[:, None])
    loss.backward()
    extreme = {
        "predictions": values.detach().tolist(), "loss": float(loss.item()),
        "forward_finite": bool(torch.isfinite(loss)),
        "backward_all_finite": bool(torch.isfinite(values.grad).all()),
        "backward_nonzero_count": int(torch.count_nonzero(values.grad)),
        "exact_optimum_zero_gradient": float(values.grad[4]) == 0.0,
    }
    safe_predictions = torch.tensor([-13.0, -10.0, -2.0, 0.0, 3.0, 31.0, 80.0], dtype=torch.float32)
    safe_targets = torch.tensor([0.01, 0.02, 0.2, 1.0, 3.0, 5.0, 20.0], dtype=torch.float32)
    old = F.smooth_l1_loss(
        torch.log(torch.exp(safe_predictions).clamp_min(1e-6)), torch.log(safe_targets), reduction="mean",
    )
    new = log_dimension_loss(safe_predictions, safe_targets)
    unsafe = torch.tensor([-120.0, -72.0, -43.0], dtype=torch.float32)
    old_unsafe = torch.log(torch.exp(unsafe).clamp_min(1e-6))
    safe = {
        "domain": "log(1e-6) <= prediction <= log(float32_max); test points keep clamp inactive",
        "old": float(old), "new": float(new), "absolute_delta": float(abs(old - new)),
        "tight_tolerance": 1e-6, "pass": bool(torch.allclose(old, new, rtol=1e-6, atol=1e-6)),
        "global_equivalence_claimed": False,
        "lower_clamp_active_example": {
            "predictions": unsafe.tolist(), "defective_transformed": old_unsafe.tolist(),
            "new_direct": unsafe.tolist(), "expressions_deliberately_differ": not torch.equal(old_unsafe, unsafe),
        },
    }
    positive_guard = False
    try:
        log_dimension_loss(torch.zeros(1, 1), torch.zeros(1, 1))
    except RuntimeError as error:
        positive_guard = str(error) == "dimension targets must be strictly positive"
    return {"extreme_direct_log": extreme, "safe_domain_equivalence": safe,
            "nonpositive_target_guard": positive_guard,
            "pass": (extreme["forward_finite"] and extreme["backward_all_finite"]
                     and extreme["backward_nonzero_count"] == len(values) - 1
                     and extreme["exact_optimum_zero_gradient"] and safe["pass"] and positive_guard)}


def eligible_dimension_targets(config: dict[str, Any]) -> dict[str, Any]:
    root = (Path.cwd() / config["dataset_root"]).resolve(strict=True)
    objects = load_objects(root)
    train_ids = {row["sample_id"] for row in read_csv(root / "dataset/manifest.csv") if row["split"] == "train"}
    values = []
    eligible = 0
    for sample_id in train_ids:
        for row in objects.get(sample_id, ()):
            if row.get("contract_state") != "POSITIVE":
                continue
            eligible += 1
            values.extend(max(0.01, float(row[name])) for name in ("gt_size_x_m", "gt_size_y_m", "gt_size_z_m"))
    return {"eligible_train_objects": eligible, "dimension_targets": len(values),
            "minimum_target_m": min(values), "strictly_positive": all(value > 0.0 for value in values)}


def prior_tests(model: torch.nn.Module) -> dict[str, Any]:
    shared = torch.randn(2, 128, 5, 7)
    result = {}
    trunk_nonzero = {}
    for class_name in ("vehicle", "person"):
        branch = getattr(model, class_name)
        outputs = branch(shared)
        fields = {}
        for name, head in branch.heads.items():
            expected = head.bias.detach().view(1, -1, 1, 1).expand_as(outputs[name])
            fields[name] = {
                "weight_exact_zero": int(torch.count_nonzero(head.weight)) == 0,
                "spatial_output_exact_bias": torch.equal(outputs[name], expected),
                "bias": head.bias.detach().tolist(),
            }
        result[class_name] = fields
        trunk_nonzero[class_name] = any(int(torch.count_nonzero(parameter)) > 0 for parameter in branch.trunk.parameters())
    neck_nonzero = any(int(torch.count_nonzero(parameter)) > 0 for parameter in model.depth_neck.parameters())
    expected_biases = {
        "vehicle": {"heatmap": [-4.6], "subcell": [0.0, 0.0],
                    "log_dimensions": list(torch.log(torch.tensor([4.0, 1.8, 1.6])).tolist()),
                    "yaw_sincos": [0.0, 1.0]},
        "person": {"heatmap": [-4.6], "subcell": [0.0, 0.0],
                   "log_dimensions": list(torch.log(torch.tensor([0.6, 0.6, 1.7])).tolist()),
                   "yaw_sincos": [0.0, 1.0]},
    }
    bias_pass = all(
        torch.allclose(torch.tensor(result[class_name][field]["bias"]), torch.tensor(expected), rtol=0, atol=1e-7)
        for class_name, values in expected_biases.items() for field, expected in values.items()
    )
    all_fields = all(value["weight_exact_zero"] and value["spatial_output_exact_bias"]
                     for branches in result.values() for value in branches.values())
    return {"branches": result, "expected_registered_biases_pass": bias_pass,
            "object_trunks_nonzero": trunk_nonzero, "shared_neck_nonzero": neck_nonzero,
            "pass": all_fields and bias_pass and all(trunk_nonzero.values()) and neck_nonzero}


def decoder_tests(model: torch.nn.Module) -> dict[str, Any]:
    safe = torch.tensor([-10.0, -2.0, 0.0, 3.0, 10.0, 20.0, 80.0], dtype=torch.float32)
    old_exp = torch.exp(safe)
    new_exp = inference_exp_float64(safe)
    old_expm1 = torch.expm1(safe)
    new_expm1 = inference_expm1_float64(safe)
    exp_parity = torch.allclose(new_exp.float(), old_exp, rtol=1e-6, atol=1e-6)
    expm1_parity = torch.allclose(new_expm1.float(), old_expm1, rtol=1e-6, atol=1e-6)
    high = inference_exp_float64(torch.tensor(95.5, dtype=torch.float32))
    fields = dict(COMMON_FIELDS)
    fields["parked"] = 1
    branch = {name: torch.zeros(1, channels, 1, 1) for name, channels in fields.items()}
    branch["heatmap"].fill_(10.0)
    branch["yaw_sincos"][:, 1].fill_(1.0)
    branch["log_dimensions"].fill_(800.0)
    explicit_failure = False
    try:
        decode_branch(
            branch, "vehicle", model.depth_anchors, model.depth_delta,
            np.eye(4), np.eye(3), 0.02, topk=1,
        )
    except FloatingPointError as error:
        explicit_failure = "non-finite scored detection" in str(error)
    return {
        "safe_values": safe.tolist(),
        "dimension_exp_fp32_tolerance_parity": bool(exp_parity),
        "depth_expm1_fp32_tolerance_parity": bool(expm1_parity),
        "log_dimension_95_5_float64": float(high),
        "log_dimension_95_5_finite": bool(torch.isfinite(high)),
        "scored_nonfinite_explicit_failure": explicit_failure,
        "clamp_or_bound_added": False,
        "pass": bool(exp_parity and expm1_parity and torch.isfinite(high) and explicit_failure),
    }


def checkpoint_discovery_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="depth_aware_epoch000_") as directory:
        root = Path(directory)
        payload = root / "epoch_000.pt"
        torch.save({"epoch": 0}, payload)
        (root / "epoch_000.json").write_text(json.dumps({
            "epoch": 0, "bytes": payload.stat().st_size, "sha256": sha256(payload), "complete": True,
        }), encoding="utf-8")
        valid = latest_checkpoint(root)
        (root / "epoch_001.pt").write_bytes(b"partial-without-sidecar")
        still_valid = latest_checkpoint(root)
        return {"epoch_000_recognized": valid == payload,
                "newer_incomplete_ignored": still_valid == payload,
                "pass": valid == payload and still_valid == payload}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve(strict=True)
    config = load_json(CONFIG_PATH)
    required = {
        "model_name": "route_b_v3_1_depth_aware_lraspp_v1",
        "scientific_seed": 20260829,
    }
    config_validation = {key: config.get(key) == value for key, value in required.items()}
    config_validation.update({
        "depth_bins_32": config["targets"]["depth_bins"] == 32,
        "depth_range_0_40": config["targets"]["depth_bin_range"] == [0.0, 40.0],
        "no_depth_input": config["inference"]["depth_input"] is False,
        "full_fp32": config["training"]["precision"] == "full_fp32",
    })
    model, loading = build_model(Path(config["pretrained"]["path"]), torch.device("cpu"))
    reports = {
        "schema": "route_b_v3_1_depth_aware_lraspp_recovery_unit_tests_v1",
        "created_utc": utc_now(), "config_validation": config_validation,
        "official_loading": loading, "dimension_loss": dimension_loss_tests(),
        "eligible_dimensions": eligible_dimension_targets(config),
        "initial_priors": prior_tests(model), "decoder": decoder_tests(model),
        "checkpoint_discovery": checkpoint_discovery_test(),
    }
    reports["pass"] = (
        all(config_validation.values()) and reports["dimension_loss"]["pass"]
        and reports["eligible_dimensions"]["strictly_positive"]
        and reports["initial_priors"]["pass"] and reports["decoder"]["pass"]
        and reports["checkpoint_discovery"]["pass"]
    )
    write_json_x(experiment / "UNIT_EQUIVALENCE_REPORT.json", reports)
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0 if reports["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
