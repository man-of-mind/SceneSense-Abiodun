"""Read-only reconstruction of the 49 sealed v11 treatment frames (offline only).

Adapts the retained v12 harness loading inside this package: the sealed
``replay_source_cost.owned`` proxy rebuilds CARLA-free actors/images from the
retained target bundles and source archives; pedestrian yaw/pitch/roll come from
the archived AVO rows; bounding-box rotations from the registered population.
Nothing under the retained evidence tree is ever written.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rl_agent.splitfusion_live_gt_optimization_v1 import replay_source as R
from rl_agent.splitfusion_live_gt_optimization_v3 import benchmark_offload as B3

ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = ROOT / "experiments/splitfusion_gt_main_thread_live_trial_v1/20261005_v11"
INPUT_SEAL_SHA = "34be00c06c57091f5599408aafebcb3cee9977d26096b5a0d6aa88eacb57500d"
QUALIFICATION = INPUT_ROOT / "treatment"
CELL = QUALIFICATION / "capture/cells/a71__favorable_stable"
COHORT = 49
UNDEFINED_QUALITY = 8


class CaseError(ValueError):
    pass


def require(value, message):
    if not value:
        raise CaseError(message)


def sha(path: Path) -> str:
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "regular nonsymbolic file required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path, expected: str | None = None):
    raw = Path(path).read_bytes()
    if expected is not None:
        require(hashlib.sha256(raw).hexdigest() == expected, "retained input hash differs: " + str(path))
    return json.loads(raw)


def verify_inputs() -> dict[str, str]:
    """Recheck every sealed treatment file (same rule as the retained v12 replay)."""
    seal = INPUT_ROOT / "OUTPUT_SHA256.json"
    require(sha(seal) == INPUT_SEAL_SHA, "sealed latest live package changed")
    files = read(seal)["files"]
    selected = {name: digest for name, digest in files.items() if name.startswith("treatment/")}
    require(selected, "sealed treatment inventory empty")
    actual = {str(p.relative_to(INPUT_ROOT)) for p in QUALIFICATION.rglob("*") if p.is_file()}
    require(actual == set(selected), "saved treatment file inventory changed")
    for name, digest in selected.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe sealed input path")
        require(sha(INPUT_ROOT / relative) == digest, "saved input changed: " + name)
    return selected


@dataclass
class Case:
    record: dict
    bundle: Any                 # T.TargetBundle (verified native inputs + prepared target)
    evidence: dict              # retained SOURCE_EVIDENCE.json
    kwargs: dict                # producer inputs (identity, rgb, images, frozen_world, cameras, radar, evidence)
    radar: np.ndarray
    avo_document: dict          # retained scored AVO document
    avo_prescore: dict          # retained pre-score document
    avo_sha256: str
    prescore_sha256: str
    target_json: dict
    segmentation_png: bytes
    object_ignore_png: bytes
    ready: dict
    target_dir: Path
    source_dir: Path

    @property
    def frame_id(self) -> int:
        return int(self.record["frame_id"])


def load_cases(limit: int | None = None) -> list[Case]:
    reference = R.load_reference()         # sealed replay_source_cost helper (SHA checked)
    reference.CELL = CELL
    summary = read(CELL / "run5_live_avo_gt_summary.json")
    records = summary["records"]
    require(len(records) == COHORT and len({r["frame_id"] for r in records}) == COHORT,
            "all 49 distinct saved treatment frames required")
    scope = read(CELL / "PAIRED_INPUT_SCOPE.json")
    require(scope.get("complete") is True and scope.get("policy_opportunities") == COHORT,
            "saved GT population differs from exact-input prepared cohort")
    cases = []
    for record in records[: (COHORT if limit is None else limit)]:
        bundle, evidence, kwargs = reference.owned(record)
        require(bundle.native_inputs_verified, "full target input verification was omitted")
        avo_name = Path(record["path"]).name
        avo = read(CELL / "run5_live_avo_inputs" / avo_name, record["sha256"])
        prescore_name = avo["prescore_input"]["filename"]
        prescore = read(CELL / "run5_live_avo_inputs" / prescore_name, avo["prescore_input"]["sha256"])
        people = {int(row["gt_actor_id"]): row for row in avo["raw_pedestrian_rows"]}
        for actor in kwargs["frozen_world"].get_actors():
            if actor.id in people:
                row, rotation = people[actor.id], actor.get_transform().rotation
                rotation.yaw, rotation.pitch, rotation.roll = (row["object_" + k + "_deg"] for k in ("yaw", "pitch", "roll"))
        original = dict(kwargs)
        fixture_kwargs, radar = B3.fixture((bundle, evidence, original, record, avo))
        fixture_kwargs = dict(fixture_kwargs, radar_world_xyz=radar,
                              dynamic_snapshot_evidence=evidence["dynamic_snapshot_evidence"])
        target_dir = CELL / "parity_complete_targets" / Path(record["complete_target"]["directory"]).name
        source_dir = CELL / "parity_complete_targets" / Path(record["complete_sources"]["directory"]).name
        cases.append(Case(record=record, bundle=bundle, evidence=evidence, kwargs=fixture_kwargs, radar=radar,
            avo_document=avo, avo_prescore=prescore, avo_sha256=record["sha256"],
            prescore_sha256=avo["prescore_input"]["sha256"], target_json=read(target_dir / "target.json"),
            segmentation_png=(target_dir / "segmentation.png").read_bytes(),
            object_ignore_png=(target_dir / "object_ignore.png").read_bytes(), ready=read(target_dir / "READY.json"),
            target_dir=target_dir, source_dir=source_dir))
    frame_ids = [c.frame_id for c in cases]
    require(frame_ids == sorted(set(frame_ids)), "complete chronological history required")
    return cases


def load_predictions(cases: list[Case]) -> dict[int, dict]:
    """Saved native predictions, evaluation evidence and received Q_perc per frame."""
    archive = CELL / "profiler/edge/prediction_archive"
    manifest = read(archive / "MANIFEST.json")
    require(manifest["frames"] == manifest["evaluated_frames"] == COHORT, "complete 49 native predictions required")
    feedback = read(CELL / "run4_phase6/PHASE6_UE_EVIDENCE.json")["feedback_rows"]
    received = {row["frame_id"]: row for row in feedback}
    require(len(received) == len(feedback) == COHORT, "feedback frame population differs")
    by_frame = {}
    for path in sorted(archive.glob("*.identity.json")):
        document = read(path)
        frame = document["gt_identity"]["frame_id"]
        require(frame not in by_frame, "duplicate saved prediction frame")
        stem = path.name.removesuffix(".identity.json")
        by_frame[frame] = dict(identity=document, labels=np.load(archive / (stem + ".labels.npy"), allow_pickle=False),
            records=read(archive / (stem + ".records.json")), evidence=read(archive / (stem + ".evaluation.json")),
            evidence_sha256=sha(archive / (stem + ".evaluation.json")), q_perc=received[frame]["q_perc"])
    require(len(by_frame) == COHORT, "native prediction identities incomplete")
    require(sum(value["q_perc"] is None for value in by_frame.values()) == UNDEFINED_QUALITY,
            "latest49 undefined-quality population differs")
    for case in cases:
        require(by_frame[case.frame_id]["identity"]["gt_identity"] == case.kwargs["identity"], "prediction/GT identity mismatch")
    return by_frame


def producer_arguments(case: Case) -> dict:
    population = case.evidence["dynamic_snapshot_evidence"]
    return dict(static_catalogue_dir=CELL / "parity_static_catalogue",
        experiment_id=case.bundle.inputs.frame["experiment_id"], ego_id=population["population"]["ego_id"],
        dynamic_population_ids=population["registered_population_ids"],
        population_evidence_sha256=case.evidence["population_evidence_sha256"], max_frames=100)
