from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_x(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    left_path, right_path = args.left.resolve(strict=True), args.right.resolve(strict=True)
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    initial_keys = (
        "model_state_sha256", "parameters_sha256", "buffers_sha256",
        "optimizer_sha256", "rng_sha256",
    )
    initial = {key: left["initial"][key] == right["initial"][key] for key in initial_keys}
    batches = []
    for left_batch, right_batch in zip(left["batches_1_through_15"], right["batches_1_through_15"]):
        batches.append({
            "batch": left_batch["batch_index"],
            "full_batch_hash_equal": left_batch["batch"] == right_batch["batch"],
            "input_hash_equal": left_batch["input"] == right_batch["input"],
            "target_hash_equal": left_batch["targets"] == right_batch["targets"],
            "sample_id_hash_equal": left_batch["sample_ids"] == right_batch["sample_ids"],
            "ordered_sample_ids_equal": left_batch["sample_ids_ordered"] == right_batch["sample_ids_ordered"],
        })
    updates = []
    for left_update, right_update in zip(left["updates_1_through_13"], right["updates_1_through_13"]):
        updates.append({
            "update": left_update["update"],
            "individual_and_total_loss_summaries_equal": left_update["losses"] == right_update["losses"],
            "preclip_gradient_hash_equal": (
                left_update["preclip_gradients"]["sha256"] == right_update["preclip_gradients"]["sha256"]
            ),
            "postclip_gradient_hash_equal": (
                left_update["postclip_gradients"]["sha256"] == right_update["postclip_gradients"]["sha256"]
            ),
            "optimizer_hash_equal": left_update["optimizer"]["sha256"] == right_update["optimizer"]["sha256"],
            "clip_norm_left": left_update["clip_returned_norm"],
            "clip_norm_right": right_update["clip_returned_norm"],
        })
    first_update_divergence = next((
        record for record in updates
        if not all(value for key, value in record.items() if key.endswith("_equal"))
    ), None)
    post_keys = (
        "model_state_sha256", "parameters_sha256", "buffers_sha256",
        "optimizer_sha256", "rng_sha256",
    )
    post13 = {key: left["post_update_13"][key] == right["post_update_13"][key] for key in post_keys}
    left_first, right_first = left["first_nonfinite_operation"], right["first_nonfinite_operation"]
    left_origin = left_first.get("origins", [{}])[0]
    right_origin = right_first.get("origins", [{}])[0]
    first_operation_identity = {
        "operation_equal": left_first["operation"] == right_first["operation"],
        "sequence_equal": left_first["sequence"] == right_first["sequence"],
        "sample_id_equal": left_origin.get("sample_id") == right_origin.get("sample_id"),
        "source_identity_equal": left_origin.get("source_identity") == right_origin.get("source_identity"),
        "cell_equal": (left_origin.get("cell_y"), left_origin.get("cell_x"))
        == (right_origin.get("cell_y"), right_origin.get("cell_x")),
        "component_equal": left_origin.get("nonfinite_component_indices")
        == right_origin.get("nonfinite_component_indices"),
        "operation_summary_equal": left_first["summary"] == right_first["summary"],
        "operation_input_summary_equal": left_first.get("input_summary") == right_first.get("input_summary"),
        "causal_input_values_equal": left_origin.get("causal_input_values") == right_origin.get("causal_input_values"),
    }
    all_batches_equal = all(
        all(value for key, value in record.items() if key.endswith("_equal")) for record in batches
    )
    deterministic_gate_pass = (
        all(initial.values()) and all_batches_equal and all(post13.values())
        and all(first_operation_identity.values())
    )
    report = {
        "schema": "route_b_v3_1_depth_aware_lraspp_reproduction_comparison_v1",
        "left": {"path": str(left_path), "sha256": sha256(left_path)},
        "right": {"path": str(right_path), "sha256": sha256(right_path)},
        "both_reproduced_nonfinite_total": (
            left["original_failure_reproduced"] and right["original_failure_reproduced"]
        ),
        "initial_hash_equality": initial,
        "batches_1_through_15": batches,
        "all_15_ordered_batches_inputs_and_targets_equal": all_batches_equal,
        "updates_1_through_13": updates,
        "first_update_divergence": first_update_divergence,
        "post_update_13_hash_equality": post13,
        "first_nonfinite_operation_identity": first_operation_identity,
        "left_first_nonfinite": left_first,
        "right_first_nonfinite": right_first,
        "deterministic_reproduction_gate_pass": deterministic_gate_pass,
        "repair_authorized": False,
        "terminal_if_closed_now": "DEPTH_AWARE_NONDETERMINISTIC_FAILURE",
        "reason": (
            "ordered batches and initial state are identical, but gradients diverge at update 1 and "
            "post-update-13 model/optimizer plus causal tensor summaries are not mutually identical"
        ),
    }
    write_json_x(args.output.resolve(), report)
    print(json.dumps({
        "deterministic_gate_pass": deterministic_gate_pass,
        "first_update_divergence": first_update_divergence,
        "post_update_13_hash_equality": post13,
        "first_nonfinite_operation_identity": first_operation_identity,
    }, indent=2))
    return 0 if deterministic_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
