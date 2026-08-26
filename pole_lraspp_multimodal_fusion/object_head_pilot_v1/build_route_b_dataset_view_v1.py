#!/usr/bin/env python3
"""Build the Route B symlink dataset view for the noAE precision pilot.

The canonical Route B corpus (``data_collection/experiments/route_b_perception_v2``)
is ~71 GiB. Training must not copy it, so this builder creates a *view*:

* one symlink per source episode under ``<experiment_dir>/dataset/<episode>``;
* a combined ``dataset/manifest.csv`` whose payload paths are rewritten to
  ``<episode>/<relative path>`` so they resolve through those symlinks;
* a combined ``dataset/object_boxes.csv`` restricted to retained samples;
* episode provenance, the resolved split table and source hashes.

Only the two train and two validation episodes are admitted. The three test
episodes are locked: this builder refuses to run if a test episode is named, and
asserts after the fact that no ``split == "test"`` row and no test payload path
entered the view.

The single content filter is the post-intervention exclusion: a saved frame is
dropped when its timestamp lies in ``[t0, t0 + 0.2]`` seconds of a roadblock
``DESTROYED`` intervention event, because the 200 ms radar window can still carry
the destroyed vehicle's previous sweep while the ground truth for that frame no
longer contains it. The rule is applied identically to train and validation.
Nothing else is filtered: collision-window frames stay in the primary dataset and
are only *recorded* so a later sensitivity analysis can exclude them offline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Payload columns rewritten to point at the episode symlink.
PATH_COLUMNS = (
    "rgb_path",
    "mask_path",
    "instance_raw_path",
    "radar_tensor_path",
    "radar_points_path",
)

# Provenance/index files hashed per episode. The 71 GiB of payload is not hashed;
# these are the files whose content decides what the view contains.
HASHED_FILES = (
    "manifest.csv",
    "object_boxes.csv",
    "collision_incident_windows.csv",
    "metadata.json",
    "route_summary.json",
)

POST_INTERVENTION_WINDOW_S = 0.2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def intervention_events(supervised_path: Path) -> List[Dict[str, Any]]:
    """Roadblock DESTROYED events recorded by the supervised collector."""
    payload = json.loads(supervised_path.read_text(encoding="utf-8"))
    events: List[Dict[str, Any]] = []
    for attempt in payload.get("attempts", []):
        policy = attempt.get("intervention_policy", {}) or {}
        for event in policy.get("intervention_events", []) or []:
            if str(event.get("action", "")).upper() != "DESTROYED":
                continue
            events.append(dict(event))
    return events


def build(
    *,
    corpus_dir: Path,
    experiment_dir: Path,
    episodes: Sequence[str],
    expected_train: int,
    expected_val: int,
) -> int:
    dataset_dir = experiment_dir / "dataset"
    provenance_dir = experiment_dir / "provenance"
    if dataset_dir.exists():
        print(f"refusing to overwrite an existing dataset view: {dataset_dir}", file=sys.stderr)
        return 2
    for name in episodes:
        if "_test_" in name:
            print(f"refusing a locked test episode: {name}", file=sys.stderr)
            return 2

    dataset_dir.mkdir(parents=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []
    manifest_fieldnames: List[str] = []
    box_rows: List[Dict[str, str]] = []
    box_fieldnames: List[str] = []
    excluded_records: List[Dict[str, Any]] = []
    collision_records: List[Dict[str, Any]] = []
    episode_provenance: List[Dict[str, Any]] = []
    seen_sample_ids: Dict[str, str] = {}

    for episode in episodes:
        source = (corpus_dir / episode).resolve(strict=True)
        link = dataset_dir / episode
        link.symlink_to(source, target_is_directory=True)

        rows = read_csv_rows(source / "manifest.csv")
        if not manifest_fieldnames:
            manifest_fieldnames = list(rows[0].keys())
        elif list(rows[0].keys()) != manifest_fieldnames:
            print(f"manifest schema drift in {episode}", file=sys.stderr)
            return 2

        splits = {str(row.get("split", "")) for row in rows}
        if splits & {"test"}:
            print(f"test rows present in {episode}", file=sys.stderr)
            return 2
        if len(splits) != 1:
            print(f"episode {episode} mixes splits {sorted(splits)}", file=sys.stderr)
            return 2
        episode_split = splits.pop()

        events = intervention_events(corpus_dir / f"{episode}_supervised.json")
        # Sample ids namespaced by episode: the collector already prefixes every id
        # with the episode name, and this asserts it so a collision is impossible.
        for row in rows:
            sample_id = str(row["sample_id"])
            if not sample_id.startswith(episode):
                print(f"sample id {sample_id} is not namespaced by episode {episode}", file=sys.stderr)
                return 2
            if sample_id in seen_sample_ids:
                print(f"duplicate sample id {sample_id}", file=sys.stderr)
                return 2
            seen_sample_ids[sample_id] = episode

        excluded_ids = set()
        for row in rows:
            timestamp = float(row["timestamp"])
            for event in events:
                start = float(event["sim_s"])
                if start <= timestamp <= start + POST_INTERVENTION_WINDOW_S:
                    excluded_ids.add(str(row["sample_id"]))
                    excluded_records.append(
                        {
                            "episode": episode,
                            "split": episode_split,
                            "sample_id": str(row["sample_id"]),
                            "frame_id": str(row.get("frame_id", "")),
                            "frame_timestamp_s": timestamp,
                            "event_action": str(event.get("action", "")),
                            "event_sim_s": start,
                            "event_actor_id": event.get("actor_id"),
                            "event_actor_type": event.get("actor_type"),
                            "event_x": event.get("x"),
                            "event_y": event.get("y"),
                            "window_end_s": start + POST_INTERVENTION_WINDOW_S,
                        }
                    )
                    break

        retained = [row for row in rows if str(row["sample_id"]) not in excluded_ids]
        for row in retained:
            out = dict(row)
            for column in PATH_COLUMNS:
                value = str(out.get(column, ""))
                if value:
                    out[column] = f"{episode}/{value}"
            manifest_rows.append(out)

        boxes = read_csv_rows(source / "object_boxes.csv")
        if boxes:
            if not box_fieldnames:
                box_fieldnames = list(boxes[0].keys())
            elif list(boxes[0].keys()) != box_fieldnames:
                print(f"object_boxes schema drift in {episode}", file=sys.stderr)
                return 2
        retained_ids = {str(row["sample_id"]) for row in retained}
        episode_boxes = [row for row in boxes if str(row["sample_id"]) in retained_ids]
        box_rows.extend(episode_boxes)

        # Collision-window frames stay in the dataset; record them for the
        # offline sensitivity analysis only.
        for row in read_csv_rows(source / "collision_incident_windows.csv"):
            if str(row.get("incident_window", "0")) != "1":
                continue
            sample_id = str(row["sample_id"])
            collision_records.append(
                {
                    "episode": episode,
                    "split": episode_split,
                    "sample_id": sample_id,
                    "frame_id": str(row.get("frame_id", "")),
                    "timestamp_s": row.get("timestamp_s", ""),
                    "nearest_collision_dt_s": row.get("nearest_collision_dt_s", ""),
                    "retained_in_dataset": int(sample_id in retained_ids),
                }
            )

        episode_provenance.append(
            {
                "episode": episode,
                "split": episode_split,
                "source_dir": str(source),
                "symlink": str(link),
                "manifest_rows": len(rows),
                "retained_rows": len(retained),
                "excluded_rows": len(rows) - len(retained),
                "object_box_rows_source": len(boxes),
                "object_box_rows_retained": len(episode_boxes),
                "intervention_events": events,
                "scenario_id": rows[0].get("scenario_id", ""),
                "map_name": rows[0].get("map_name", ""),
                "traffic_density": rows[0].get("traffic_density", ""),
                "pedestrian_density": rows[0].get("pedestrian_density", ""),
                "source_hashes": {
                    name: sha256_file(source / name)
                    for name in HASHED_FILES
                    if (source / name).is_file()
                },
                "supervised_json_sha256": sha256_file(corpus_dir / f"{episode}_supervised.json"),
                "population_events_sha256": sha256_file(
                    corpus_dir / f"{episode}_population_events.jsonl"
                ),
            }
        )

    # Preserve the recorded split verbatim; never re-split by frame.
    split_counts: Dict[str, int] = {}
    for row in manifest_rows:
        split_counts[str(row["split"])] = split_counts.get(str(row["split"]), 0) + 1
    if "test" in split_counts:
        print("test rows leaked into the training view", file=sys.stderr)
        return 2
    for row in manifest_rows:
        for column in PATH_COLUMNS:
            if "_test_" in str(row.get(column, "")):
                print(f"test payload path leaked: {row[column]}", file=sys.stderr)
                return 2

    manifest_path = dataset_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=manifest_fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    boxes_path = dataset_dir / "object_boxes.csv"
    with boxes_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=box_fieldnames)
        writer.writeheader()
        writer.writerows(box_rows)

    def write_table(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)

    write_table(
        provenance_dir / "excluded_post_intervention_samples.csv",
        excluded_records,
        [
            "episode", "split", "sample_id", "frame_id", "frame_timestamp_s",
            "event_action", "event_sim_s", "window_end_s", "event_actor_id",
            "event_actor_type", "event_x", "event_y",
        ],
    )
    write_table(
        provenance_dir / "collision_window_samples.csv",
        collision_records,
        ["episode", "split", "sample_id", "frame_id", "timestamp_s",
         "nearest_collision_dt_s", "retained_in_dataset"],
    )
    write_table(
        provenance_dir / "resolved_split_table.csv",
        [
            {
                "episode": item["episode"],
                "split": item["split"],
                "source_rows": item["manifest_rows"],
                "excluded_post_intervention": item["excluded_rows"],
                "retained_rows": item["retained_rows"],
            }
            for item in episode_provenance
        ],
        ["episode", "split", "source_rows", "excluded_post_intervention", "retained_rows"],
    )

    summary = {
        "view": "route_b_noae_precision_pilot_v1",
        "corpus_dir": str(corpus_dir.resolve()),
        "experiment_dir": str(experiment_dir.resolve()),
        "episodes": episode_provenance,
        "post_intervention_window_s": POST_INTERVENTION_WINDOW_S,
        "post_intervention_rule": (
            "exclude saved frames with event.sim_s <= timestamp <= event.sim_s + 0.2 "
            "for every roadblock DESTROYED intervention event; applied identically to "
            "train and val"
        ),
        "collision_windows_retained_in_dataset": True,
        "collision_window_samples": len(collision_records),
        "source_rows": {
            "train": expected_train,
            "val": expected_val,
        },
        "retained_rows": split_counts,
        "excluded_post_intervention_rows": len(excluded_records),
        "object_box_rows": len(box_rows),
        "manifest_sha256": sha256_file(manifest_path),
        "object_boxes_sha256": sha256_file(boxes_path),
        "test_episodes_present": False,
    }
    (provenance_dir / "dataset_view_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    source_train = sum(e["manifest_rows"] for e in episode_provenance if e["split"] == "train")
    source_val = sum(e["manifest_rows"] for e in episode_provenance if e["split"] == "val")
    if source_train != expected_train or source_val != expected_val:
        print(
            f"pre-exclusion row counts differ from the collection result: "
            f"train={source_train} (expected {expected_train}) "
            f"val={source_val} (expected {expected_val})",
            file=sys.stderr,
        )
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--episode", action="append", required=True, dest="episodes")
    parser.add_argument("--expected-train-rows", type=int, default=3293)
    parser.add_argument("--expected-val-rows", type=int, default=3600)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return build(
        corpus_dir=args.corpus_dir,
        experiment_dir=args.experiment_dir,
        episodes=args.episodes,
        expected_train=args.expected_train_rows,
        expected_val=args.expected_val_rows,
    )


if __name__ == "__main__":
    raise SystemExit(main())
