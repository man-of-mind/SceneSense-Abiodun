#!/usr/bin/env python3
"""Registered Run-5 validation scenes: the unseen ``held_scene`` grid split.

All 256 ``held_scene`` candidates of the registered quality grid are
enumerated.  A scene is eligible iff its quality ground truth is prospectively
valid, using exactly the rule the training catalogue applied to FIT scenes
(``scene_source.FitSceneCatalog``):

* ``camera_si`` present;
* corrected radar P40 present in the registered read-only sidecar;
* the complete 12-mode x 11-q grid present; and
* every grid ``q_perc`` defined (no ``UNDEFINED_NO_LOCALIZATION_ELIGIBLE_GT``).

The rule reads only ground-truth availability, never a policy, reward,
latency or outcome.  Training never loaded this split (``FIT_SPLIT = "fit"``).
The eligible scenes are ordered by (episode_id, frame_id, sample_id), i.e. route
order, and the ordered key list is hashed.  ``HeldSceneCatalogV1`` exposes the
same ``keys / scene_descriptors / draw`` surface as the training catalogue so
the unchanged Run-4 kernel can consume it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from rl_agent.ue_production_queue_capture_v1 import contract as V1
from rl_agent.ue_production_transport_model_v2 import contract_v2 as C2
from rl_agent.ue_production_transport_model_v2 import scene_source as SS

HELD_SPLIT = "held_scene"
TRAINING_SPLIT = SS.FIT_SPLIT
EXPECTED_CANDIDATES = 256
PARTITION_SCHEMA = "splitfusion.run5.held_scene_partition.v1"
PACKAGE = Path(__file__).resolve().parent
SEALED_PARTITION = PACKAGE / "RUN5_HELD_SCENE_PARTITION.json"


class HeldScenePartitionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HeldScenePartitionError(message)


def _key(row: dict) -> str:
    return f"{row['episode_id']}|{row['sample_id']}|{row['frame_id']}"


def _order(key: str) -> tuple:
    episode, sample, frame = key.split("|")
    return (episode, int(frame), sample)


def load_split(repo_root: Path, split: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    database = repo_root / V1.PAYLOAD_AUTHORITY_RELPATH
    manifest = repo_root / V1.PAYLOAD_AUTHORITY_MANIFEST_RELPATH
    require(C2.sha256_file(manifest) == V1.PAYLOAD_AUTHORITY_MANIFEST_SHA256,
            "payload authority manifest drifted")
    sidecar_path = repo_root / SS.P40_SIDECAR_RELPATH
    require(C2.sha256_file(sidecar_path) == SS.P40_SIDECAR_SHA256, "P40 sidecar drifted")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    require(sidecar["status"] == "COMPLETE" and sidecar["no_zero_imputation"] is True,
            "P40 sidecar incomplete")
    corrected = {f"{r['episode_id']}|{r['sample_id']}|{r['frame_id']}": r["corrected_p40"]
                 for r in sidecar["records"] if r["grid_split"] == split}
    scenes: dict[str, dict[str, Any]] = {}
    connection = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
    try:
        for (blob,) in connection.execute(
                "SELECT row_json FROM quality_rows WHERE grid_split = ?", (split,)):
            row = json.loads(blob)
            key = _key(row)
            entry = scenes.setdefault(key, {"camera_si": row.get("camera_si"),
                                            "radar_p40": corrected.get(key), "grid": {}})
            entry["grid"][(int(row["mode_id"]), int(row["q_e4"]))] = (
                row.get("q_perc"), int(row["total_transmitted_bytes"]),
                int(row["udp_application_bytes"]), str(row["row_sha256"]))
    finally:
        connection.close()
    sources = {"payload_authority_manifest_sha256": V1.PAYLOAD_AUTHORITY_MANIFEST_SHA256,
               "database_sha256": C2.sha256_file(database),
               "p40_sidecar_sha256": SS.P40_SIDECAR_SHA256}
    return scenes, sources


def eligibility(entry: dict[str, Any]) -> list[str]:
    reasons = []
    if entry["camera_si"] is None:
        reasons.append("CAMERA_SI_MISSING")
    if entry["radar_p40"] is None:
        reasons.append("RADAR_P40_MISSING")
    if len(entry["grid"]) != SS.MODE_COUNT * len(SS.Q_E4_GRID):
        reasons.append("QUALITY_GRID_INCOMPLETE")
    undefined = sum(1 for item in entry["grid"].values() if item[0] is None)
    if undefined:
        reasons.append(f"UNDEFINED_Q_PERC_ROWS_{undefined}")
    return reasons


def build_partition(repo_root: Path) -> dict[str, Any]:
    scenes, sources = load_split(repo_root, HELD_SPLIT)
    require(len(scenes) == EXPECTED_CANDIDATES, f"expected 256 candidates, got {len(scenes)}")
    training = set(SS.FitSceneCatalog(repo_root).keys)
    candidates = sorted(scenes, key=_order)
    excluded = {k: eligibility(scenes[k]) for k in candidates if eligibility(scenes[k])}
    eligible = [k for k in candidates if k not in excluded]
    require(not set(eligible) & training, "a held scene appears in the training catalogue")
    return {
        "schema": PARTITION_SCHEMA, "split": HELD_SPLIT,
        "eligibility_rule": ("camera_si present AND corrected radar P40 present AND complete "
                             "12x11 grid AND every q_perc defined (training-catalogue rule; "
                             "ground-truth availability only)"),
        "order": "route order: (episode_id, frame_id, sample_id)",
        "candidate_count": len(candidates), "eligible_count": len(eligible),
        "excluded": excluded,
        "overlap_with_training_catalogue": 0,
        "training_catalogue_split": TRAINING_SPLIT,
        "eligible_keys": eligible,
        "eligible_keys_sha256": hashlib.sha256(json.dumps(eligible).encode()).hexdigest(),
        "sources": sources,
    }


def partition_sha256(document: dict[str, Any]) -> str:
    return C2.canonical_sha256(document)


def load_sealed_partition(repo_root: Path) -> dict[str, Any]:
    sealed = json.loads(SEALED_PARTITION.read_text())
    require(sealed == build_partition(repo_root), "sealed held-scene partition drifted")
    return sealed


class HeldSceneCatalogV1:
    """Held-scene twin of ``FitSceneCatalog`` over the sealed eligible keys."""

    def __init__(self, repo_root: Path, eligible_keys: list[str]) -> None:
        scenes, _ = load_split(repo_root, HELD_SPLIT)
        require(all(k in scenes and not eligibility(scenes[k]) for k in eligible_keys),
                "catalogue key is not eligible")
        self._scenes = {k: scenes[k] for k in eligible_keys}
        self._keys = tuple(eligible_keys)
        self.binding_sha256 = C2.canonical_sha256({"schema": PARTITION_SCHEMA,
                                                   "keys": list(self._keys)})

    keys = property(lambda self: self._keys)
    scene_count = property(lambda self: len(self._keys))

    def scene_descriptors(self, scene_key: str) -> tuple[float, float]:
        entry = self._scenes[scene_key]
        return float(entry["camera_si"]), float(entry["radar_p40"])

    # Identical interpolation semantics to the training catalogue.
    draw = SS.FitSceneCatalog.draw


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args(argv)
    document = build_partition(args.evidence_root.resolve())
    if args.seal:
        with SEALED_PARTITION.open("x") as handle:
            json.dump(document, handle, indent=1, sort_keys=True)
            handle.write("\n")
    print(json.dumps({"candidates": document["candidate_count"],
                      "eligible": document["eligible_count"], "excluded": document["excluded"],
                      "eligible_keys_sha256": document["eligible_keys_sha256"],
                      "partition_sha256": partition_sha256(document)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
