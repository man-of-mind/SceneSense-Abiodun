"""Read-only import/wire replay of 49 sealed live numerical target references.

This does not run either new numerical implementation, reconstruct producer
history, measure performance, or prove the live scheduling integration. No raw
sensor archive is copied and only a new small result directory is written.
"""
import argparse
import csv
import io
import json
from pathlib import Path

import cv2
import numpy as np

from . import contract as C
from .evidence import admit_source_evidence, own_legacy_reference, reconcile_source
from .oracle_bridge import from_legacy, scorer_inputs

ROOT = Path(__file__).resolve().parents[2]
TRIAL = ROOT / "experiments/splitfusion_gt_main_thread_live_trial_v1/20261005_v11"
SEAL_SHA256 = "34be00c06c57091f5599408aafebcb3cee9977d26096b5a0d6aa88eacb57500d"
SCRATCH = TRIAL / "treatment/capture/cells/a71__favorable_stable/run4_phase6/gt_scratch_preserved"


def run(output):
    from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
    output = Path(output)
    C.require(not output.exists(), "create-only replay output required")
    C.require(np.__version__ == "1.26.4", "reference NumPy version differs")
    seal = (TRIAL / "OUTPUT_SHA256.json").read_bytes()
    C.require(C.sha256(seal) == SEAL_SHA256, "retained live seal changed")
    inventory = json.loads(seal)["files"]
    inputs = {str((TRIAL / "OUTPUT_SHA256.json").relative_to(ROOT)): SEAL_SHA256}
    source_pins_before = E.verify_sources()

    def read(path):
        path = Path(path)
        C.require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
                  "regular retained input required")
        relative = str(path.relative_to(TRIAL))
        raw = path.read_bytes()
        C.require(relative in inventory and C.sha256(raw) == inventory[relative], "retained input hash differs")
        inputs[str(path.relative_to(ROOT))] = C.sha256(raw)
        return raw

    decisions = list(csv.DictReader(io.StringIO(read(TRIAL / "comparison/treatment_decisions.csv").decode())))
    C.require(len(decisions) == 49 and len({int(d["frame_id"]) for d in decisions}) == 49,
              "treatment decision population differs")
    decision_by_frame = {int(d["frame_id"]): d for d in decisions}
    bundles = {}
    for path in sorted(SCRATCH.glob("target_*/READY.json")):
        ready = json.loads(read(path))
        frame = ready["frame_id"]
        C.require(frame not in bundles, "duplicate prepared frame")
        bundles[frame] = (path.parent, ready)
    C.require(set(bundles) == set(decision_by_frame), "target and policy frame populations differ")
    avos = {}
    for path in sorted(SCRATCH.glob("*.avo.json")):
        document = json.loads(read(path))
        frame = document["source_frame_id"]
        C.require(frame not in avos, "duplicate AVO frame")
        avos[frame] = document
    C.require(set(bundles).issubset(avos), "missing full AVO document")
    rows = []
    binding = C.sha256(C.canonical(dict(schema="OFFLINE_RECORDED_TARGET_CONTRACT_REPLAY_V1",
                                       retained_seal=SEAL_SHA256)))
    for frame in sorted(bundles):
        directory, ready = bundles[frame]
        saved = decision_by_frame[frame]
        target = json.loads(read(directory / "target.json"))
        native = json.loads(read(directory / "inputs.json"))
        for name in ("segmentation", "object_ignore"):
            raw = read(directory / (name + ".png"))
            C.require(C.sha256(raw) == ready["payloads"][name + ".png"]["sha256"], "READY PNG binding differs")
            target[name + "_png"] = raw
            target[name] = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        prepared = E.restore_prepared_target(frame_id=frame, sample_id=ready["sample_id"],
            target=target, content_sha256=ready["prepared_content_sha256"])
        source = admit_source_evidence(input_metadata=native, avo_document=avos[frame],
                                      expected_capture_identity=ready["identity"])
        full_source = json.loads(read(directory.with_suffix(".sources") / "SOURCE_EVIDENCE.json"))
        C.require(C.sha256(C.canonical(full_source)) == native["provenance"]["evidence_sha256"]["owned_source"],
                  "owned source binding differs")
        history = full_source["conflicts"]["causal_static_history"]
        policy = {k: saved[k] if k == "session_uuid" else int(saved[k]) for k in C.POLICY_FIELDS}
        observed_plan = dict(capture=ready["identity"], policy=policy,
            mode_id=int(saved["mode_id"]), q_e4=int(saved["q_e4"]),
            scope="OFFLINE_RECONSTRUCTED_ACTION_IDENTITY_NOT_LIVE_PLAN")
        plan_sha = C.sha256(C.canonical(observed_plan))
        admitted = from_legacy(prepared=prepared, source_evidence=source, policy_identity=policy,
            binding=binding, plan_sha256=plan_sha,
            history_before_sha256=history["previous_sha256"], history_after_sha256=history["chain_sha256"])
        reference = own_legacy_reference(admitted, source, target_tables=target["tables"],
                                         old_prepared_sha256=prepared.content_sha256)
        encoded = C.encode_target(admitted, ready_raw_ns=ready["ready_raw_ns"], ready_wall_ns=ready["ready_wall_ns"])
        decoded = C.decode_target(encoded, expected_binding=binding, expected_capture_identity=ready["identity"],
                                 expected_policy_identity=policy, expected_plan_sha256=plan_sha).target
        reconcile_source(decoded, source)
        numerical = scorer_inputs(decoded)
        C.require(decoded.semantic_sha256 == admitted.semantic_sha256, "semantic roundtrip changed")
        for name in ("segmentation", "object_ignore"):
            C.require(np.array_equal(numerical[name], target[name]), "numerical mask changed")
            try:
                numerical[name].setflags(write=True)
            except ValueError:
                pass
            else:
                raise C.ContractError("mask storage is mutable")
        C.require(C.thaw(numerical["object_boxes"]) == target["tables"]["object_boxes.csv"], "target order/value changed")
        C.require(C.thaw(numerical["avo"]) == avos[frame]["avo"], "AVO changed")
        C.require(C.thaw(reference.target_tables) == target["tables"], "legacy evidence tables changed")
        rows.append(dict(frame_id=frame, policy_identity=policy, semantic_sha256=decoded.semantic_sha256,
            original_prepared_sha256=prepared.content_sha256, source_evidence_sha256=source.content_sha256,
            packet_bytes=len(encoded), all_values_exact=True, mask_storage_immutable=True,
            recorded_quality_defined=saved["q_perc"] not in ("", "None", "null")))
    C.require(sum(not r["recorded_quality_defined"] for r in rows) == 8, "undefined quality population changed")
    for path, expected in inputs.items():
        C.require(C.sha256((ROOT / path).read_bytes()) == expected, "input changed during contract replay")
    C.require(E.verify_sources() == source_pins_before, "reference source pins changed")
    result = dict(status="RECORDED_NUMERICAL_CONTRACT_REPLAY_PASS_NOT_LIVE_QUALIFIED", frames=len(rows),
        undefined_quality_frames=8, rows=rows, input_sha256=inputs, reference_source_pins=source_pins_before,
        performance_measured=False, numerical_producer_or_scorer_replacement_tested=False,
        history_scope="Retained causal static-history digests only; no tracker clone or transactional-history proof",
        plan_scope="Replay binding of recorded identities/action; not a pre-existing live plan digest",
        retained_artifacts_rebuilt=False, artifact_tables_preserved=True,
        raw_sensor_archives_copied=False)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "RESULT.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    hashes = {"RESULT.json": C.sha256((output / "RESULT.json").read_bytes())}
    with (output / "OUTPUT_SHA256.json").open("x") as stream:
        json.dump(hashes, stream, indent=2, sort_keys=True); stream.write("\n")
    return dict(status=result["status"], frames=len(rows), undefined_quality_frames=8,
                inputs_verified=len(inputs), output=str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
