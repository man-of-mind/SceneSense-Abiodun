#!/usr/bin/env python3
"""Offline Phase-2 contract acceptance; this produces no C2 research claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from .engine import RecipientMapEngine
from .evaluation import TruthTrajectory, match_warning_to_truth
from .schemas import EgoState, MapContribution, MapObjectObservation, with_exact_payload_bytes
from .selection import select_recipient_hazards
from .transport import chunk_payload


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "local_acceptance_v1.yaml"


def _engine(config: Mapping[str, object]) -> RecipientMapEngine:
    return RecipientMapEngine(
        str(config["recipient_ue_id"]),
        association_gate_m=float(config["association_gate_m"]),
        track_ttl_s=float(config["track_ttl_s"]),
        max_transport_age_s=float(config["max_transport_age_s"]),
        warning_horizon_s=float(config["warning_horizon_s"]),
        confidence_floor=float(config["confidence_floor"]),
        safety_radius_m_by_class=config["safety_radius_m_by_class"],
    )


def _object(track_id: str, x_m: float, y_m: float, captured_at_s: float) -> MapObjectObservation:
    return MapObjectObservation(
        source_track_id=track_id,
        class_name="pedestrian",
        x_m=x_m,
        y_m=y_m,
        vx_mps=0.0,
        vy_mps=0.0,
        confidence=0.90,
        observed_at_s=captured_at_s,
    )


def _contribution(
    config: Mapping[str, object],
    *,
    source: str,
    recipient: str | None = None,
    sequence: int,
    captured_at_s: float,
    include_hazard: bool = True,
    include_benign: bool = False,
) -> MapContribution:
    fixture = config["synthetic_fixture"]
    objects = []
    if include_hazard:
        objects.append(
            _object(
                "source_local_hazard_track",
                float(fixture["hazard_x_m"]),
                float(fixture["hazard_y_m"]),
                captured_at_s,
            )
        )
    if include_benign:
        objects.append(
            _object(
                "source_local_benign_track",
                float(fixture["hazard_x_m"]),
                float(fixture["benign_y_m"]),
                captured_at_s,
            )
        )
    contribution = MapContribution(
        contribution_id=f"{source}:{sequence}",
        source_ue_id=source,
        recipient_ue_id=recipient or str(config["recipient_ue_id"]),
        sequence_number=sequence,
        captured_at_s=captured_at_s,
        published_at_s=captured_at_s,
        profile_id="synthetic_contract_fixture",
        payload_bytes=0,
        objects=tuple(objects),
    )
    return with_exact_payload_bytes(contribution)


def _ego(config: Mapping[str, object], timestamp_s: float) -> EgoState:
    speed = float(config["synthetic_fixture"]["ego_speed_mps"])
    return EgoState(
        recipient_ue_id=str(config["recipient_ue_id"]),
        timestamp_s=timestamp_s,
        x_m=speed * timestamp_s,
        y_m=0.0,
        vx_mps=speed,
        vy_mps=0.0,
    )


def evaluate_fixture(config: Mapping[str, object]) -> tuple[pd.DataFrame, dict]:
    fixture = config["synthetic_fixture"]
    recipient = str(config["recipient_ue_id"])
    helper = str(config["helper_ue_id"])
    helper_at = float(fixture["helper_publish_s"])
    ego_at = float(fixture["ego_first_observation_s"])
    send_everything = _contribution(
        config,
        source=helper,
        sequence=1,
        captured_at_s=helper_at,
        include_benign=True,
    )
    selected = select_recipient_hazards(
        send_everything.objects,
        _ego(config, helper_at),
        capture_at_s=helper_at,
        horizon_s=float(config["warning_horizon_s"]),
        confidence_floor=float(config["confidence_floor"]),
        safety_radius_m_by_class=config["safety_radius_m_by_class"],
    )
    hazard_only = replace(
        send_everything,
        contribution_id=f"{helper}:hazard-only:1",
        payload_bytes=0,
        objects=selected,
    )
    hazard_only = with_exact_payload_bytes(hazard_only)
    strategies = {
        "ego_only": _contribution(
            config,
            source=recipient,
            sequence=1,
            captured_at_s=ego_at,
        ),
        "send_everything": send_everything,
        "hazard_only": hazard_only,
    }
    rows = []
    first_warnings: dict[str, float] = {}
    warning_aoi: dict[str, float] = {}
    truth = (
        TruthTrajectory(
            truth_id="evaluation_truth_hazard_1",
            class_name="pedestrian",
            x_m=float(fixture["hazard_x_m"]),
            y_m=float(fixture["hazard_y_m"]),
            safety_hazard=True,
        ),
        TruthTrajectory(
            truth_id="evaluation_truth_benign_1",
            class_name="pedestrian",
            x_m=float(fixture["hazard_x_m"]),
            y_m=float(fixture["benign_y_m"]),
            safety_hazard=False,
        ),
    )
    for strategy, contribution in strategies.items():
        engine = _engine(config)
        status = engine.install(contribution, contribution.published_at_s)
        warnings = engine.warnings(_ego(config, contribution.published_at_s))
        matches = [match_warning_to_truth(warning, truth) for warning in warnings]
        hazard_warnings = [
            warning for warning, match in zip(warnings, matches) if match.safety_hazard
        ]
        if hazard_warnings:
            warning = hazard_warnings[0]
            first_warnings[strategy] = warning.warning_at_s
            warning_aoi[strategy] = warning.map_aoi_s
            warning_payload = asdict(warning)
        else:
            warning_payload = {}
        rows.append(
            {
                "event": "strategy_evaluation",
                "strategy": strategy,
                "install_status": status,
                "captured_at_s": contribution.captured_at_s,
                "received_at_s": contribution.published_at_s,
                "payload_bytes": contribution.payload_bytes,
                "onwire_bytes": sum(
                    len(chunk) + 28
                    for chunk in chunk_payload(
                        contribution.to_json_bytes(),
                        message_id=contribution.sequence_number,
                        chunk_bytes=int(config["transport"]["chunk_bytes"]),
                    )
                ),
                "warning_count": len(warnings),
                "hazard_warning_count": len(hazard_warnings),
                "first_warning_at_s": first_warnings.get(strategy),
                "map_aoi_at_warning_s": warning_aoi.get(strategy),
                "warning_json": json.dumps(warning_payload, sort_keys=True),
            }
        )

    # A benign-only fixture is scored separately so a true hazard warning cannot
    # mask an unrelated false warning in the same contribution.
    benign_engine = _engine(config)
    benign = _contribution(
        config,
        source=helper,
        sequence=1,
        captured_at_s=helper_at,
        include_hazard=False,
        include_benign=True,
    )
    benign_engine.install(benign, helper_at)
    false_warnings = len(benign_engine.warnings(_ego(config, helper_at)))

    integrity = _engine(config)
    wrong = _contribution(
        config,
        source=helper,
        recipient="different_ego",
        sequence=1,
        captured_at_s=0.0,
    )
    wrong_status = integrity.install(wrong, 0.0)
    accepted = _contribution(
        config,
        source=helper,
        sequence=2,
        captured_at_s=0.2,
    )
    accepted_status = integrity.install(accepted, 0.2)
    sequence_status = integrity.install(accepted, 0.2)
    stale = _contribution(
        config,
        source="stale_helper",
        sequence=1,
        captured_at_s=0.0,
    )
    stale_status = integrity.install(stale, float(config["max_transport_age_s"]) + 0.1)

    lead_full = first_warnings["ego_only"] - first_warnings["send_everything"]
    lead_hazard = first_warnings["ego_only"] - first_warnings["hazard_only"]
    threshold = float(config["acceptance"]["minimum_warning_lead_gain_s"])
    gates = {
        "positive_full_warning_lead": lead_full >= threshold,
        "positive_hazard_warning_lead": lead_hazard >= threshold,
        "no_false_warning": false_warnings <= int(config["acceptance"]["maximum_false_warnings"]),
        "recipient_isolation": wrong_status == "rejected_wrong_recipient",
        "sequence_rejection": accepted_status == "accepted" and sequence_status == "rejected_sequence",
        "stale_rejection": stale_status == "rejected_transport_stale",
    }
    summary = {
        "implementation_status": "synthetic_contract_validation",
        "research_evidence": False,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "first_warning_at_s": first_warnings,
        "warning_lead_gain_vs_ego_only_s": {
            "send_everything": lead_full,
            "hazard_only": lead_hazard,
        },
        "false_warning_count": false_warnings,
        "application_bytes_per_advanced_warning": {
            "send_everything": strategies["send_everything"].payload_bytes,
            "hazard_only": strategies["hazard_only"].payload_bytes,
        },
        "payload_reduction_pct_hazard_vs_full": 100.0
        * (1.0 - strategies["hazard_only"].payload_bytes / strategies["send_everything"].payload_bytes),
        "integrity_status": {
            "wrong_recipient": wrong_status,
            "first_sequence": accepted_status,
            "duplicate_sequence": sequence_status,
            "stale": stale_status,
        },
        "gates": gates,
    }
    return pd.DataFrame(rows), summary


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_figure(frame: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].bar(frame["strategy"], frame["first_warning_at_s"], color=["#A5A5A5", "#4472C4", "#70AD47"])
    axes[0].set_ylabel("First hazard warning (s; lower is earlier)")
    axes[0].tick_params(axis="x", rotation=18)
    axes[1].bar(frame["strategy"], frame["payload_bytes"] / 1024.0, color=["#A5A5A5", "#4472C4", "#70AD47"])
    axes[1].set_ylabel("Delivered payload (KiB)")
    axes[1].tick_params(axis="x", rotation=18)
    figure.suptitle("Phase-2 synthetic contract validation (not C2 evidence)")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run(config_path: Path, output_root: Path | None = None) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frame, summary = evaluate_fixture(config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = output_root or REPO_ROOT / "phase2_map_sharing" / "experiments"
    run_dir = root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    resolved = run_dir / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    events = run_dir / "event_log.csv"
    frame.to_csv(events, index=False)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _save_figure(frame, run_dir / "figures" / "warning_lead_and_payload")
    report = "\n".join(
        [
            "# Phase-2 local acceptance",
            "",
            f"**{summary['verdict']} — synthetic contract validation only; not C2 evidence.**",
            "",
            f"- Send-everything and hazard-only each advanced the warning by {summary['warning_lead_gain_vs_ego_only_s']['hazard_only']:.2f} s versus ego-only.",
            f"- Hazard-only reduced exact serialized application payload by {summary['payload_reduction_pct_hazard_vs_full']:.2f}% while preserving the synthetic warning time.",
            f"- False warnings: {summary['false_warning_count']}.",
            f"- Recipient, sequence, and stale-message guards: {'PASS' if all(summary['gates'][key] for key in ('recipient_isolation', 'sequence_rejection', 'stale_rejection')) else 'FAIL'}.",
            "",
            "The lead is constructed by the controlled fixture. A paired CARLA occlusion run with a separate truth stream is required before claiming cooperation gain.",
            "",
        ]
    )
    (run_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    artifacts = [resolved, events, summary_path, run_dir / "RESULTS.md", run_dir / "figures" / "warning_lead_and_payload.png", run_dir / "figures" / "warning_lead_and_payload.pdf"]
    manifest = {
        "experiment_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_status": summary["implementation_status"],
        "verdict": summary["verdict"],
        "source_config": str(config_path.resolve()),
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parent / "schemas.py",
                Path(__file__).resolve().parent / "adapters.py",
                Path(__file__).resolve().parent / "selection.py",
                Path(__file__).resolve().parent / "engine.py",
                Path(__file__).resolve().parent / "evaluation.py",
                Path(__file__).resolve().parent / "transport.py",
                config_path.resolve(),
            )
        },
        "artifacts": {str(path.relative_to(run_dir)): _sha256(path) for path in artifacts},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "COMPLETED.json").write_text(json.dumps({"verdict": summary["verdict"]}, indent=2) + "\n", encoding="utf-8")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    path = run(args.config.resolve(), args.output_root.resolve() if args.output_root else None)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
