#!/usr/bin/env python3
"""Create-only symlink view over the expanded Route B v3.1 corpus, plus its verification.

The view copies nothing.  ``dataset`` and ``contracts`` are directory symlinks onto the
expanded v3.1 camera-plane contract; the split membership files are the only new payload.

Verification is fail-closed and is what makes the view usable:  exactly ten train and two
validation episodes, globally unique sample IDs, disjoint train/validation episode IDs and
seed pairs, validation artifacts byte-identical to the reference native-grid validation
view, no locked-test token or path anywhere, every symlink target resolvable, recorded
source hashes reproduced, no retained post-intervention-excluded sample, and collision
windows handled exactly as the existing contract handles them.

Test absence is established without opening test contents: the manifest is checked for
``split == "test"`` rows, every symlink target is checked against the admitted episode set,
and text artifacts are scanned for locked identifiers.  No locked directory is listed,
resolved or read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
CONTRACTS = ("v010", "v025")
CONTRACT_FILES = ("object_boxes.csv", "object_ignore_regions.csv", "target_manifest.csv")
LOCKED_TOKENS = ("canonical_v3_07", "canonical_v3_08", "_test_", "/test/")
FAILED = "ROUTE_B_V3_1_EXPANDED_VIEW_BUILD_FAILED"

TRAIN_EPISODES = (
    "canonical_v3_01_train_30_30_s501_tm1501",
    "canonical_v3_02_train_50_50_s502_tm1502",
    "canonical_v3_03_train_30_30_s503_tm1503",
    "canonical_v3_04_train_50_50_s504_tm1504",
    "extra_v3_09_train_30_30_s801_tm1801",
    "extra_v3_10_train_50_50_s802_tm1802",
    "extra_v3_11_train_30_30_s803_tm1803",
    "extra_v3_12_train_50_50_s804_tm1804",
    "extra_v3_13_train_30_30_s805_tm1805",
    "extra_v3_14_train_50_50_s806_tm1806",
)
VAL_EPISODES = (
    "canonical_v3_05_val_30_30_s601_tm1601",
    "canonical_v3_06_val_50_50_s602_tm1602",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def rows_digest(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> str:
    """Order-preserving digest of a row subset, independent of the surrounding file."""
    digest = hashlib.sha256()
    digest.update(("\x1f".join(fields) + "\x1e").encode("utf-8"))
    for row in rows:
        digest.update(("\x1f".join(str(row.get(field, "")) for field in fields) + "\x1e").encode("utf-8"))
    return digest.hexdigest()


def episode_seed_pair(episode: str, corpus: Path) -> tuple[int, int]:
    metadata = json.loads((corpus / episode / "metadata.json").read_text(encoding="utf-8"))
    return int(metadata["scenario_seed"]), int(metadata["traffic_manager_seed"])


def write_json_x(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")



def recorded_hash_pairs(root: Path) -> list[tuple[str, str]]:
    """(relative path, recorded SHA-256) pairs from whichever provenance a layer writes.

    The clean-base v3.1 layer records ``MANIFEST_HASHES.json``; the derived camera-plane
    layer records its hashes inside ``CAMERA_PLANE_CONTRACT_SUMMARY.json``.  Both are read
    when present, and a layer that publishes neither is a failure, not a silent skip.
    """
    pairs: list[tuple[str, str]] = []
    manifest_hashes = root / "MANIFEST_HASHES.json"
    if manifest_hashes.is_file():
        recorded = json.loads(manifest_hashes.read_text(encoding="utf-8"))
        pairs.extend((item["path"], item["sha256"]) for item in recorded["files"])
    camera_plane = root / "CAMERA_PLANE_CONTRACT_SUMMARY.json"
    if camera_plane.is_file():
        summary = json.loads(camera_plane.read_text(encoding="utf-8"))
        for contract in CONTRACTS:
            for split in ("train", "val"):
                entry = summary["summaries"][contract][split]
                for filename, key in (
                        ("object_boxes.csv", "derived_object_boxes_sha256"),
                        ("object_ignore_regions.csv", "derived_object_ignore_regions_sha256"),
                        ("target_manifest.csv", "derived_target_manifest_sha256")):
                    pairs.append((f"contracts/{contract}/{split}/{filename}", entry[key]))
        pairs.append(("dataset/manifest.csv", summary["dataset"]["manifest_sha256"]))
        pairs.append(("dataset/object_boxes.csv", summary["dataset"]["object_boxes_sha256"]))
    return pairs


def compare_validation(new_root: Path, reference: Path) -> dict[str, Any]:
    """Byte-level comparison of every validation artifact between two v3.1 contract roots."""
    entry: dict[str, Any] = {"reference": str(reference), "new": str(new_root), "files": {}}
    identical = True
    for contract in CONTRACTS:
        for filename in CONTRACT_FILES:
            new_hash = sha256(new_root / "contracts" / contract / "val" / filename)
            ref_hash = sha256(reference / "contracts" / contract / "val" / filename)
            entry["files"][f"{contract}/val/{filename}"] = {
                "new": new_hash, "reference": ref_hash, "equal": new_hash == ref_hash}
            identical = identical and new_hash == ref_hash
    new_manifest = read_csv(new_root / "dataset/manifest.csv")
    ref_manifest = read_csv(reference / "dataset/manifest.csv")
    new_val = [row for row in new_manifest if row["split"] == "val"]
    ref_val = [row for row in ref_manifest if row["split"] == "val"]
    fields = list(new_manifest[0]) if new_manifest else []
    ref_fields = list(ref_manifest[0]) if ref_manifest else []
    entry["validation_manifest_field_sets_equal"] = fields == ref_fields
    shared = [field for field in fields if field in ref_fields]
    entry["validation_manifest_rows"] = {
        "new": rows_digest(new_val, shared), "reference": rows_digest(ref_val, shared),
        "compared_fields": shared, "new_rows": len(new_val), "reference_rows": len(ref_val)}
    entry["validation_manifest_rows"]["equal"] = (
        entry["validation_manifest_rows"]["new"] == entry["validation_manifest_rows"]["reference"]
        and len(new_val) == len(ref_val))
    identical = identical and entry["validation_manifest_rows"]["equal"] and fields == ref_fields
    new_boxes = read_csv(new_root / "dataset/object_boxes.csv")
    ref_boxes = read_csv(reference / "dataset/object_boxes.csv")
    new_ids = {row["sample_id"] for row in new_val}
    ref_ids = {row["sample_id"] for row in ref_val}
    box_fields = list(new_boxes[0]) if new_boxes else []
    entry["validation_object_boxes"] = {
        "new": rows_digest([r for r in new_boxes if r["sample_id"] in new_ids], box_fields),
        "reference": rows_digest([r for r in ref_boxes if r["sample_id"] in ref_ids], box_fields)}
    entry["validation_object_boxes"]["equal"] = (
        entry["validation_object_boxes"]["new"] == entry["validation_object_boxes"]["reference"])
    identical = identical and entry["validation_object_boxes"]["equal"]
    entry["identical"] = identical
    return entry


# ------------------------------------------------------------------------------ checks

def verify(view: Path, contract_root: Path, views_root: Path, references: Mapping[str, Path],
           corpus: Path, base_comparisons: Mapping[str, tuple[Path, Path]]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    observed: dict[str, Any] = {}

    manifest = read_csv(contract_root / "dataset/manifest.csv")
    by_split: dict[str, list[dict[str, str]]] = {}
    for row in manifest:
        by_split.setdefault(row["split"], []).append(row)
    train_rows = by_split.get("train", [])
    val_rows = by_split.get("val", [])

    train_episodes = tuple(dict.fromkeys(row["experiment_id"] for row in train_rows))
    val_episodes = tuple(dict.fromkeys(row["experiment_id"] for row in val_rows))
    observed["train_episodes"] = list(train_episodes)
    observed["validation_episodes"] = list(val_episodes)
    observed["train_frames"] = len(train_rows)
    observed["validation_frames"] = len(val_rows)
    checks["exactly_10_train_episodes"] = (
        len(train_episodes) == 10 and set(train_episodes) == set(TRAIN_EPISODES))
    checks["exactly_2_validation_episodes"] = (
        len(val_episodes) == 2 and set(val_episodes) == set(VAL_EPISODES))

    sample_ids = [row["sample_id"] for row in manifest]
    checks["sample_ids_globally_unique"] = len(set(sample_ids)) == len(sample_ids)
    checks["sample_ids_episode_namespaced"] = all(
        row["sample_id"].startswith(row["experiment_id"] + "_") for row in manifest)

    train_seeds = {episode: episode_seed_pair(episode, corpus) for episode in train_episodes}
    val_seeds = {episode: episode_seed_pair(episode, corpus) for episode in val_episodes}
    observed["train_seed_pairs"] = train_seeds
    observed["validation_seed_pairs"] = val_seeds
    checks["train_validation_episode_ids_disjoint"] = not (set(train_episodes) & set(val_episodes))
    checks["train_validation_seed_pairs_disjoint"] = not (
        set(train_seeds.values()) & set(val_seeds.values()))
    checks["train_seed_pairs_all_distinct"] = len(set(train_seeds.values())) == len(train_seeds)

    # ---- validation byte-identity against the reference native-grid validation view(s)
    validation_identity: dict[str, Any] = {}
    identical = True
    for name, reference in references.items():
        entry = compare_validation(contract_root, reference)
        identical = identical and entry["identical"]
        validation_identity[name] = entry
    observed["validation_identity"] = validation_identity
    checks["validation_byte_identical_to_reference_views"] = identical

    base_identity: dict[str, Any] = {}
    base_ok = True
    for name, (new_root, reference) in base_comparisons.items():
        entry = compare_validation(new_root, reference)
        base_ok = base_ok and entry["identical"]
        base_identity[name] = entry
    observed["base_layer_validation_identity"] = base_identity
    checks["base_layer_validation_byte_identical"] = base_ok

    # every validation mask on disk still hashes to what its target manifest recorded
    mask_mismatches: list[str] = []
    for contract in CONTRACTS:
        contract_dir = contract_root / "contracts" / contract / "val"
        for row in read_csv(contract_dir / "target_manifest.csv"):
            for path_field, hash_field in (
                    ("segmentation_mask_path", "segmentation_mask_sha256"),
                    ("object_ignore_mask_path", "object_ignore_mask_sha256")):
                path = contract_dir / row[path_field]
                if not path.exists() or sha256(path) != row[hash_field]:
                    mask_mismatches.append(f"{contract}/val/{row[path_field]}")
    observed["validation_mask_hash_mismatches"] = mask_mismatches[:20]
    observed["validation_mask_hash_mismatch_count"] = len(mask_mismatches)
    checks["validation_masks_match_recorded_hashes"] = not mask_mismatches

    # ---- test absence, established without opening any locked directory
    checks["zero_test_rows"] = not by_split.get("test")
    observed["manifest_splits"] = sorted(by_split)
    symlinks: list[dict[str, Any]] = []
    broken: list[str] = []
    admitted = set(TRAIN_EPISODES) | set(VAL_EPISODES)
    foreign: list[str] = []
    for base in (view, contract_root, views_root):
        for path in [base, *base.rglob("*")]:
            if not path.is_symlink():
                continue
            target = os.readlink(path)
            resolved = Path(os.path.realpath(path))
            symlinks.append({"link": str(path), "target": target})
            if not resolved.exists():
                broken.append(str(path))
            if resolved.parent == corpus and resolved.name not in admitted:
                foreign.append(str(path))
    observed["symlink_count"] = len(symlinks)
    observed["broken_symlinks"] = broken
    observed["symlinks_outside_admitted_episodes"] = foreign
    checks["all_symlink_targets_exist"] = not broken
    checks["zero_symlinks_to_unadmitted_episodes"] = not foreign

    leaks: list[str] = []
    for base in (view, contract_root):
        for path in base.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() not in {".json", ".csv", ".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            lowered = text.lower()
            if any(token in lowered for token in LOCKED_TOKENS) or "split,test" in lowered:
                leaks.append(str(path))
    observed["locked_token_artifacts"] = leaks
    checks["no_locked_test_token_or_path"] = not leaks

    # ---- recorded source hashes reproduce, for every contract layer in the lineage
    hash_failures: list[str] = []
    hash_checked: dict[str, int] = {}
    layers = {"final": contract_root}
    layers.update({f"base:{name}": new_root for name, (new_root, _) in base_comparisons.items()})
    for label, root in layers.items():
        pairs = recorded_hash_pairs(root)
        hash_checked[label] = len(pairs)
        hash_failures.extend(f"{label}:{rel}" for rel, expected in pairs
                             if sha256(root / rel) != expected)
    observed["source_hash_failures"] = hash_failures
    observed["source_hashes_checked"] = hash_checked
    checks["recorded_source_hashes_reproduce"] = not hash_failures and all(
        count > 0 for count in hash_checked.values())

    # ---- post-intervention exclusion and collision-window handling
    manifest_ids = set(sample_ids)
    retained_excluded: list[str] = []
    excluded_total = 0
    for split in ("train", "val"):
        for row in read_csv(views_root / f"provenance/{split}_excluded_post_intervention_samples.csv"):
            excluded_total += 1
            if row["sample_id"] in manifest_ids:
                retained_excluded.append(row["sample_id"])
    observed["post_intervention_excluded_source_rows"] = excluded_total
    observed["retained_post_intervention_excluded"] = retained_excluded[:20]
    checks["no_post_intervention_excluded_sample_retained"] = not retained_excluded

    collision: dict[str, Any] = {}
    dropped: list[str] = []
    for split in ("train", "val"):
        rows = read_csv(views_root / f"provenance/{split}_collision_window_samples.csv")
        present = [row["sample_id"] for row in rows if row["sample_id"] in manifest_ids]
        collision[split] = {"source_rows": len(rows), "retained_in_view": len(present)}
        dropped.extend(row["sample_id"] for row in rows
                       if row["sample_id"] not in manifest_ids
                       and row["sample_id"] not in set(retained_excluded))
    excluded_ids = set()
    for split in ("train", "val"):
        excluded_ids |= {row["sample_id"] for row in read_csv(
            views_root / f"provenance/{split}_excluded_post_intervention_samples.csv")}
    dropped = [item for item in dropped if item not in excluded_ids]
    collision["dropped_outside_post_intervention_exclusion"] = dropped[:20]
    observed["collision_windows"] = collision
    checks["collision_windows_retained_with_provenance"] = not dropped

    return {"passed": all(checks.values()), "checks": checks, "observed": observed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", required=True, type=Path)
    parser.add_argument("--contract-root", required=True, type=Path,
                        help="expanded v3.1 camera-plane contract experiment directory")
    parser.add_argument("--views-root", required=True, type=Path,
                        help="expanded aggregate view experiment directory (provenance source)")
    parser.add_argument("--reference", action="append", default=[], metavar="NAME=PATH",
                        help="reference validation view(s) to prove byte-identity against")
    parser.add_argument("--base-comparison", action="append", default=[],
                        metavar="NAME=NEWPATH=REFPATH",
                        help="an upstream contract layer to prove validation-identical too")
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data_collection/experiments/route_b_perception_v3")
    args = parser.parse_args()

    view = args.view.resolve()
    contract_root = args.contract_root.resolve(strict=True)
    views_root = args.views_root.resolve(strict=True)
    corpus = args.corpus.resolve(strict=True)
    references = {}
    for item in args.reference:
        name, _, path = item.partition("=")
        references[name] = (ROOT / path).resolve(strict=True)
    base_comparisons: dict[str, tuple[Path, Path]] = {}
    for item in args.base_comparison:
        name, new_path, ref_path = item.split("=", 2)
        base_comparisons[name] = ((ROOT / new_path).resolve(strict=True),
                                  (ROOT / ref_path).resolve(strict=True))
    if view.exists():
        raise FileExistsError(f"refusing to overwrite {view}")

    started = time.monotonic()
    view.mkdir(parents=True)
    try:
        (view / "dataset").symlink_to(contract_root / "dataset", target_is_directory=True)
        (view / "contracts").symlink_to(contract_root / "contracts", target_is_directory=True)
        manifest = read_csv(contract_root / "dataset/manifest.csv")
        splits = view / "splits"
        splits.mkdir()
        for split, filename in (("train", "train.txt"), ("val", "val.txt")):
            ids = [row["sample_id"] for row in manifest if row["split"] == split]
            with (splits / filename).open("x", encoding="utf-8") as stream:
                stream.write("\n".join(ids) + "\n")
        # No test.txt is created: the locked split has no row, no symlink and no file here.

        result = verify(view, contract_root, views_root, references, corpus,
                        base_comparisons)
        summary = {
            "schema": "route_b_v3_1_native_grid_expanded_train_view_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "view": str(view), "contract_root": str(contract_root), "views_root": str(views_root),
            "references": {name: str(path) for name, path in references.items()},
            "base_layer_comparisons": {name: [str(a), str(b)]
                                       for name, (a, b) in base_comparisons.items()},
            "train_episodes": list(TRAIN_EPISODES), "validation_episodes": list(VAL_EPISODES),
            "test": {"present": False, "rows": 0, "symlinks": 0, "payload_references": 0,
                     "established_by": "manifest split column, symlink target set and text scan; "
                                       "no locked directory was listed, resolved or read"},
            "corpus_payload_copies": 0,
            "verification": result,
            "wall_seconds": time.monotonic() - started,
        }
        write_json_x(view / "EXPANDED_TRAIN_VIEW_SUMMARY.json", summary)
        if not result["passed"]:
            failed = sorted(name for name, value in result["checks"].items() if not value)
            (view / "TERMINAL_VERDICT.txt").write_text(f"{FAILED}\n", encoding="utf-8")
            print(json.dumps({"terminal": FAILED, "failed_checks": failed}, indent=2))
            return 1
        (view / "TERMINAL_VERDICT.txt").write_text(
            "ROUTE_B_V3_1_EXPANDED_VIEW_READY\n", encoding="utf-8")
        (view / "VIEW_COMPLETE").write_text("ROUTE_B_V3_1_EXPANDED_VIEW_READY\n", encoding="utf-8")
        print(json.dumps({"terminal": "ROUTE_B_V3_1_EXPANDED_VIEW_READY",
                          "checks": result["checks"],
                          "train_frames": result["observed"]["train_frames"],
                          "validation_frames": result["observed"]["validation_frames"]}, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        if not (view / "TERMINAL_VERDICT.txt").exists():
            (view / "TERMINAL_VERDICT.txt").write_text(f"{FAILED}\n", encoding="utf-8")
            write_json_x(view / "view_failure.json",
                         {"terminal": FAILED, "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
