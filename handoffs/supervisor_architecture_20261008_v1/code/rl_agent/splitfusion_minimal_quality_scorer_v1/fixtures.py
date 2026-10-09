"""Read sealed oracle inputs only. No producer, broker, subprocess or writes."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .scorer import ROOT, canonical

V11 = ROOT / "experiments/splitfusion_gt_main_thread_live_trial_v1/20261005_v11"
V11_SEAL = "34be00c06c57091f5599408aafebcb3cee9977d26096b5a0d6aa88eacb57500d"
CELL_RELATIVE = "treatment/capture/cells/a71__favorable_stable"
V12_RESULT = ROOT / "experiments/splitfusion_gt_metadata_optimization_v1/20261005_v12/full49/RESULT.json"


class SealedReader:
    def __init__(self, root=V11, expected=V11_SEAL):
        self.root = root
        payload = (root / "OUTPUT_SHA256.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("reference output seal changed")
        self.seal = expected
        files = json.loads(payload)["files"]
        # The sealed v8 inventory uses repository-relative paths; v11 uses
        # evidence-root-relative paths. Normalize only after verifying its seal.
        prefix = str(root.relative_to(ROOT)) + "/"
        self.files = ({name[len(prefix):]: digest for name, digest in files.items() if name.startswith(prefix)}
                      if any(name.startswith(prefix) for name in files) else files)
        self.consumed = {}

    def read(self, path):
        relative = str(path.relative_to(self.root))
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("symbolic reference input")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != self.files.get(relative):
            raise ValueError("unbound reference input: " + relative)
        self.consumed[relative] = digest
        return payload

    def json(self, path):
        return json.loads(self.read(path))


def _load(root, expected_seal, cell_relative, population, selected_indices=None):
    from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
    reader = SealedReader(root, expected_seal)
    cell = root / cell_relative
    archive = cell / "profiler/edge/prediction_archive"
    manifest = reader.json(archive / "MANIFEST.json")
    if manifest["frames"] != population or manifest["evaluated_frames"] != population:
        raise ValueError("complete saved prediction population differs")
    for name, sha in manifest["sha256"].items():
        if reader.files.get(str((archive / name).relative_to(root))) != sha:
            raise ValueError("inner and outer prediction bindings disagree")
    feedback = reader.json(cell / "run4_phase6/PHASE6_UE_EVIDENCE.json")
    received = {row["frame_id"]: row for row in feedback["feedback_rows"]}
    if len(received) != len(feedback["feedback_rows"]) or len(received) != population:
        raise ValueError("feedback population differs")
    avo_documents = {}
    for path in sorted((cell / "run5_live_avo_inputs").glob("*.avo.json")):
        document = reader.json(path)
        frame = document["identity"]["frame_id"]
        if frame in avo_documents:
            raise ValueError("duplicate AVO frame")
        avo_documents[frame] = document
    cases = []
    paths = sorted(archive.glob("*.identity.json"))
    if len(paths) != population:
        raise ValueError("saved identity population differs")
    for index, path in enumerate(paths):
        if selected_indices is not None and index not in selected_indices:
            continue
        identity = reader.json(path)
        frame = identity["gt_identity"]["frame_id"]
        stem = path.name.removesuffix(".identity.json")
        records = reader.json(archive / (stem + ".records.json"))
        labels = np.load(io.BytesIO(reader.read(archive / (stem + ".labels.npy"))), allow_pickle=False)
        evidence_payload = reader.read(archive / (stem + ".evaluation.json"))
        target_directory = T.bundle_path(cell / "parity_complete_targets", identity["gt_identity"])
        ready = reader.json(target_directory / "READY.json")
        # Full native verification runs in the original boundary validator.
        # Its payload hashes must also agree with the sealed v11 inventory.
        for name, item in ready["payloads"].items():
            if reader.files.get(str((target_directory / name).relative_to(root))) != item["sha256"]:
                raise ValueError("inner and outer target bindings disagree")
        bundle = T.verify_full_inputs(cell / "parity_complete_targets",
            expected_identity=identity["gt_identity"],
            expected_manifest_sha256=reader.files[str((target_directory / "READY.json").relative_to(root))])
        for name, item in ready["payloads"].items():
            reader.consumed[str((target_directory / name).relative_to(root))] = item["sha256"]
        source = reader.json(target_directory.with_suffix(".sources") / "SOURCE_EVIDENCE.json")
        if hashlib.sha256(canonical(source)).hexdigest() != bundle.prepared.target["provenance"]["evidence_sha256"]["owned_source"]:
            raise ValueError("original source evidence binding differs")
        cases.append(dict(prepared=bundle.prepared, prediction_identity=identity, predicted_labels=labels,
            prediction_contract=manifest["prediction_contract"],
            causal_history=source["conflicts"]["causal_static_history"],
            predictions=records, avo_document=avo_documents[frame], input_metadata=T._input_metadata(bundle.inputs),
            run_binding=reader.seal, saved_evidence=json.loads(evidence_payload),
            saved_evidence_sha256=hashlib.sha256(evidence_payload).hexdigest(), feedback=received[frame]))
    cases.sort(key=lambda case: case["prepared"].frame_id)
    count = population if selected_indices is None else len(selected_indices)
    if len(cases) != count or len({case["prepared"].frame_id for case in cases}) != count:
        raise ValueError("unique scoring population differs")
    return cases, reader, feedback


def load_v11():
    return _load(V11, V11_SEAL, CELL_RELATIVE, 49)


def load_older_sample():
    """Three distinct older fixtures: first/middle/last sorted archive keys."""
    return _load(ROOT / "experiments/splitfusion_live_parity_runtime_v8/20261004_handoff_live_v8",
        "9e8e74b63d95b1487a5b2520c8fffda3eb42527bdf5a9f949df6115ad31ade13",
        "qualification_100/capture/cells/a71__favorable_stable", 48, (0, 24, 47))


def admission_arguments(case):
    return {k: case[k] for k in ("prepared", "prediction_identity", "predicted_labels", "predictions",
        "avo_document", "input_metadata", "run_binding", "prediction_contract")}


def oracle_arguments(case):
    return dict(identity=case["prediction_identity"]["action_identity"], prepared=case["prepared"],
                predicted_labels=case["predicted_labels"], predictions=case["predictions"],
                avo=case["avo_document"]["avo"])
