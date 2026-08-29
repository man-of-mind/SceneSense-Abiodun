from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from common import (CONFIG_PATH, load_json, read_csv, rng_state, seed_everything, sha256,
                    tensor_state_hash, utc_now, write_json_x, write_text_x, write_torch_atomic_create)
from data import InferenceDataset
from model import (build_model, configure_two_stage, freeze_bn_running_state,
                   reset_private_object_branches)
from two_stage import (assert_allowlist, build_optimizer, is_representation, object_state,
                       representation_state, state_hash)


def tensor_hash(value: torch.Tensor) -> str:
    item = value.detach().cpu().contiguous(); digest = hashlib.sha256()
    digest.update(str(item.dtype).encode()); digest.update(str(tuple(item.shape)).encode()); digest.update(item.numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args(); experiment = args.experiment.resolve(strict=True)
    decision = load_json(experiment / "STAGE1_SELECTION.json")
    if not decision["stage2_authorized"]: raise RuntimeError("Stage 1 did not authorize Stage 2")
    config = load_json(CONFIG_PATH); registered = load_json(experiment / "REGISTERED_TWO_STAGE_DESIGN.json")
    checkpoint_path = Path(decision["selected_checkpoint"])
    if sha256(checkpoint_path) != decision["selected_checkpoint_sha256"]:
        raise RuntimeError("selected Stage-1 checkpoint SHA drift")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(config["stage2_seed"])); model, _ = build_model(Path(config["pretrained"]["path"]), device)
    model.load_state_dict(payload["model"], strict=True); model.eval(); freeze_bn_running_state(model)
    frozen_before = representation_state(model); frozen_hash = tensor_state_hash(frozen_before)
    root = (Path.cwd() / config["dataset_root"]).resolve(strict=True)
    rows = [row for row in read_csv(root / "dataset/manifest.csv") if row["split"] == "train"]
    probe = InferenceDataset(root, rows[:2]); inputs = torch.stack([probe[index][0] for index in range(2)]).to(device)
    with torch.inference_mode(): output_before = model.representation_outputs(inputs)
    reset_private_object_branches(model, int(config["stage2_initialization_seed"]))
    if tensor_state_hash(representation_state(model)) != frozen_hash:
        raise RuntimeError("representation changed during private object reset")
    with torch.inference_mode(): output_after = model.representation_outputs(inputs)
    prediction_equal = all(torch.equal(output_before[name], output_after[name]) for name in output_before)
    if not prediction_equal: raise RuntimeError("segmentation/dense output changed during Stage-2 reset")
    configure_two_stage(model, "stage2"); assert_allowlist(model, "stage2", registered["parameter_allowlists"]["stage2"])
    optimizer = build_optimizer(model, "stage2")
    if optimizer.state: raise RuntimeError("fresh Stage-2 optimizer unexpectedly has inherited state")
    reset_checks = {}
    for class_name in ("vehicle", "person"):
        branch = getattr(model, class_name)
        reset_checks[class_name] = {
            "trunk_kaiming_nonzero": any(torch.count_nonzero(value).item() > 0 for value in branch.trunk.parameters()),
            "all_final_weights_exact_zero": all(torch.count_nonzero(head.weight).item() == 0 for head in branch.heads.values()),
            "heatmap_bias": branch.heads["heatmap"].bias.detach().cpu().tolist(),
            "subcell_bias": branch.heads["subcell"].bias.detach().cpu().tolist(),
            "dimension_bias": branch.heads["log_dimensions"].bias.detach().cpu().tolist(),
            "yaw_bias": branch.heads["yaw_sincos"].bias.detach().cpu().tolist(),
        }
    if not all(item["trunk_kaiming_nonzero"] and item["all_final_weights_exact_zero"] for item in reset_checks.values()):
        raise RuntimeError("Stage-2 registered object initialization failed")
    seed_everything(int(config["stage2_seed"]))
    resolved_hash = sha256(experiment / "RESOLVED_CONFIG.json")
    checkpoint_dir = experiment / "stage2/checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=False)
    path = checkpoint_dir / "stage2_epoch_000.pt"
    stage2_payload = {"schema": "two_stage_lraspp_stage2_checkpoint_v1",
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "epoch": 0, "global_step": 0, "rng_state": rng_state(),
        "sampler_state": {"epoch": 1, "seed": int(config["stage2_seed"]) + 1,
                          "visited": 0, "unique": 0, "complete": False},
        "scheduler_state": {"next_epoch": 1, "schedule": "registered_stage2_warmup_cosine_v1"},
        "resolved_config_sha256": resolved_hash, "source_commit": config["source_commit"],
        "batch": 16, "accumulation": 1, "cumulative_wall_seconds": 0.0,
        "selected_stage1_checkpoint": str(checkpoint_path),
        "selected_stage1_checkpoint_sha256": decision["selected_checkpoint_sha256"],
        "frozen_representation_hash": frozen_hash}
    write_torch_atomic_create(path, stage2_payload)
    write_json_x(path.with_suffix(".json"), {"epoch": 0, "path": str(path), "bytes": path.stat().st_size,
                                            "sha256": sha256(path), "complete": True})
    audit = {"schema": "two_stage_lraspp_stage2_transition_v1", "created_utc": utc_now(),
        "selected_stage1_epoch": decision["selected_epoch"], "selected_stage1_checkpoint": str(checkpoint_path),
        "selected_stage1_checkpoint_sha256": decision["selected_checkpoint_sha256"],
        "verified_selected_sha": True, "frozen_representation_hash": frozen_hash,
        "frozen_tensor_hashes": {name: tensor_hash(value) for name, value in frozen_before.items()},
        "frozen_tensor_count": len(frozen_before), "reset_seed": int(config["stage2_initialization_seed"]),
        "reset_checks": reset_checks, "segmentation_dense_reset_bit_identical": prediction_equal,
        "fresh_optimizer": True, "fresh_optimizer_state_entries": len(optimizer.state),
        "optimizer_parameter_allowlist": registered["parameter_allowlists"]["stage2"],
        "stage2_epoch000": {"path": str(path), "bytes": path.stat().st_size,
                            "sha256": sha256(path), "verified": sha256(path) == load_json(path.with_suffix('.json'))['sha256']}}
    write_json_x(experiment / "STAGE2_TRANSITION.json", audit)
    write_text_x(experiment / "STAGE2_AUTHORIZED", "PASS\n")
    print(json.dumps({"stage2_authorized": True, "frozen_hash": frozen_hash,
                      "stage2_epoch000_sha256": sha256(path)}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
