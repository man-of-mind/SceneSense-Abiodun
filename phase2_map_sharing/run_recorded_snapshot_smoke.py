#!/usr/bin/env python3
"""Smoke the Phase-2 adapter on existing two-stream map recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from .adapters import snapshot_stream_to_contribution
from .engine import RecipientMapEngine
from .schemas import MapContribution
from .transport import ChunkReassembler, chunk_payload


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "recorded_snapshot_smoke_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(config: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    source = str(config["source_stream_id"])
    recipient = str(config["recipient_ue_id"])
    class_counts: dict[str, int] = {}
    sequence = 0
    for recording_name in config["recordings"]:
        recording = REPO_ROOT / str(recording_name)
        engine = RecipientMapEngine(
            recipient,
            association_gate_m=float(config["association_gate_m"]),
            track_ttl_s=float(config["track_ttl_s"]),
            max_transport_age_s=float(config["max_transport_age_s"]),
            warning_horizon_s=float(config["warning_horizon_s"]),
            confidence_floor=float(config["confidence_floor"]),
        )
        for line_number, line in enumerate(recording.read_text(encoding="utf-8").splitlines(), 1):
            record = json.loads(line)
            snapshot = record.get("snap", record)
            active = {
                str(item.get("stream_id")): item
                for item in snapshot.get("active_streams", [])
                if isinstance(item, dict)
            }
            if source not in active or recipient not in active:
                continue
            sequence += 1
            published_at_s = float(record.get("t", 0.0))
            captured_at_s = published_at_s - max(0.0, float(active[source].get("age_s", 0.0)))
            contribution = snapshot_stream_to_contribution(
                snapshot,
                source_stream_id=source,
                recipient_ue_id=recipient,
                sequence_number=sequence,
                captured_at_s=captured_at_s,
                published_at_s=published_at_s,
                profile_id="recorded_mprime_map_output",
            )
            encoded = contribution.to_json_bytes()
            chunks = chunk_payload(encoded, message_id=sequence, chunk_bytes=int(config["chunk_bytes"]))
            receiver = ChunkReassembler(timeout_s=float(config["max_transport_age_s"]))
            reassembled = None
            for chunk_index, chunk in enumerate(reversed(chunks)):
                reassembled = receiver.ingest(
                    f"recording:{source}",
                    chunk,
                    received_at_s=published_at_s + chunk_index * 1e-6,
                )
            round_trip = reassembled is not None and reassembled.payload == encoded
            decoded = MapContribution.from_json_bytes(reassembled.payload) if reassembled else contribution
            identity_clean = all(token not in encoded for token in (b"actor_id", b"ground_truth_id"))
            status = engine.install(decoded, published_at_s + (len(chunks) - 1) * 1e-6)
            for obj in decoded.objects:
                name = obj.class_name.lower()
                class_counts[name] = class_counts.get(name, 0) + 1
            rows.append(
                {
                    "recording": str(recording.relative_to(REPO_ROOT)),
                    "line_number": line_number,
                    "sequence_number": sequence,
                    "captured_at_s": captured_at_s,
                    "published_at_s": published_at_s,
                    "source_age_s": published_at_s - captured_at_s,
                    "object_count": len(decoded.objects),
                    "payload_bytes": decoded.payload_bytes,
                    "chunk_count": len(chunks),
                    "round_trip_exact": round_trip,
                    "runtime_identity_clean": identity_clean,
                    "install_status": status,
                    "recipient_track_count": len(engine.snapshot(published_at_s)["tracks"]),
                }
            )
    frame = pd.DataFrame(rows)
    accepted = int((frame["install_status"] == "accepted").sum()) if len(frame) else 0
    gates = {
        "enough_two_active_snapshots": len(frame)
        >= int(config["acceptance"]["minimum_two_active_snapshots"]),
        "enough_accepted_contributions": accepted
        >= int(config["acceptance"]["minimum_accepted_contributions"]),
        "pedestrian_and_vehicle_present": {"pedestrian", "vehicle"}.issubset(class_counts),
        "exact_chunk_round_trip": bool(len(frame)) and bool(frame["round_trip_exact"].all()),
        "no_runtime_actor_identity": bool(len(frame)) and bool(frame["runtime_identity_clean"].all()),
    }
    summary = {
        "implementation_status": "recorded_snapshot_adapter_validation",
        "research_evidence": False,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "eligible_two_active_snapshots": len(frame),
        "accepted_contributions": accepted,
        "rejected_contributions": len(frame) - accepted,
        "class_observation_counts": class_counts,
        "mean_application_payload_bytes": float(frame["payload_bytes"].mean()) if len(frame) else None,
        "gates": gates,
    }
    return frame, summary


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frame, summary = evaluate(config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "phase2_map_sharing" / "experiments" / "snapshot_adapter" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved = run_dir / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    events = run_dir / "adapter_events.csv"
    frame.to_csv(events, index=False)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# Recorded two-stream adapter smoke",
            "",
            f"**{summary['verdict']} — existing-recording integration check only; not C2 evidence.**",
            "",
            f"- Eligible two-active-stream snapshots: {summary['eligible_two_active_snapshots']}.",
            f"- Accepted contributions: {summary['accepted_contributions']}; rejected: {summary['rejected_contributions']}.",
            f"- Runtime class observations: {summary['class_observation_counts']}.",
            f"- Mean canonical JSON payload: {summary['mean_application_payload_bytes']:.1f} B.",
            "- Warning lead is not scored because these recordings have no separate synchronized hazard-truth stream.",
            "",
        ]
    )
    report_path = run_dir / "RESULTS.md"
    report_path.write_text(report, encoding="utf-8")
    sources = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "adapters.py",
        Path(__file__).resolve().parent / "schemas.py",
        Path(__file__).resolve().parent / "transport.py",
        config_path,
        *(REPO_ROOT / str(name) for name in config["recordings"]),
    ]
    manifest = {
        "experiment_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_status": summary["implementation_status"],
        "verdict": summary["verdict"],
        "source_sha256": {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in sources},
        "artifact_sha256": {
            path.name: _sha256(path) for path in (resolved, events, summary_path, report_path)
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "COMPLETED.json").write_text(json.dumps({"verdict": summary["verdict"]}, indent=2) + "\n", encoding="utf-8")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(run(args.config.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
