"""Presentation-only folder renames in the existing supervisor handoff.

Source contents/imports stay byte-identical to their originals. This is a code
reading snapshot, not a package refactor or a deployable release.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "handoffs/supervisor_architecture_20261008_v1"
OLD = "code/pole_lraspp_multimodal_fusion"
PACKAGE = OLD + "/object_head_pilot_v1"
NEW = "code/fcos_model"
RENAMES = {
    "splitfusion_fcos_r50_fpn_p2_p7_v1": "network",
    "splitfusion_fcos_r50_fpn_p2_p7_ae_v1": "feature_autoencoders",
    "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1": "spatial_compression",
    "splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1": "postprocessing",
    "splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1": "person_score_calibration",
    "splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1": "person_instance_consolidation",
    "splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1": "quality_metrics",
    "splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1": "training_recovery",
    "route_b_v3_1_clean_base_v1": "training_target_contract",
    "route_b_v3_1_targeted_refinement_v1": "gt_target_refinement",
}


def new_path(relative):
    for old, new in RENAMES.items():
        prefix = PACKAGE + "/" + old
        if relative == prefix or relative.startswith(prefix + "/"):
            return NEW + "/" + new + relative[len(prefix):]
    helper = OLD + "/pole_lraspp_multimodal_fusion"
    if relative == helper or relative.startswith(helper + "/"):
        return NEW + "/legacy_shared_helpers" + relative[len(helper):]
    if relative == PACKAGE or relative.startswith(PACKAGE + "/"):
        return NEW + relative[len(PACKAGE):]
    return relative


def main():
    manifest_path = SNAPSHOT / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    assert (SNAPSHOT / PACKAGE).is_dir() and not (SNAPSHOT / NEW).exists()
    before = {str(p.relative_to(SNAPSHOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (SNAPSHOT / OLD).rglob("*") if p.is_file()}
    assert len({new_path(p) for p in before}) == len(before)
    for record in manifest["files"]:
        assert hashlib.sha256((SNAPSHOT / record["path"]).read_bytes()).hexdigest() == record["sha256"]

    (SNAPSHOT / PACKAGE).rename(SNAPSHOT / NEW)
    for old, new in RENAMES.items():
        (SNAPSHOT / NEW / old).rename(SNAPSHOT / NEW / new)
    (SNAPSHOT / OLD / "pole_lraspp_multimodal_fusion").rename(
        SNAPSHOT / NEW / "legacy_shared_helpers")
    (SNAPSHOT / OLD).rmdir()  # Empty directory only; no source deletion.

    for record in manifest["files"]:
        previous = record["path"]
        record["path"] = new_path(previous)
        if previous != record["path"]:
            record["snapshot_path_before_rename"] = previous
    manifest["presentation_layout"] = {
        "renamed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_contents_unchanged": True,
        "imports_retain_original_repository_names": True,
        "not_a_runnable_package_refactor": True,
        "path_mapping": {p: new_path(p) for p in before},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    readme = SNAPSHOT / "README.md"
    text = readme.read_text()
    for old in sorted(before, key=len, reverse=True):
        text = text.replace(old, new_path(old))
    readme.write_text(text)
    for previous, digest in before.items():
        assert hashlib.sha256((SNAPSHOT / new_path(previous)).read_bytes()).hexdigest() == digest
    print(json.dumps({"renamed_source_files": len(before), "fcos_directory": str(SNAPSHOT/NEW)}))


if __name__ == "__main__":
    main()
