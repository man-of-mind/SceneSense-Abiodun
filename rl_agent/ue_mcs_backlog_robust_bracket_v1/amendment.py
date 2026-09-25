#!/usr/bin/env python3
"""Create-only builder and fail-closed offline verifier for the amendment.

The preserved capacity attempt remains refused.  This package adopts only its
physically verified empirical/bootstrap uncertainty set for a prospective,
byte-only downstream queue design.  Nothing here launches a RAN, simulator,
container, model, CUDA runtime, or network client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rl_agent.ue_mcs_backlog_near_capacity_v1 import capacity_qualification as CQ
from rl_agent.ue_mcs_backlog_near_capacity_v1 import capacity_runner as CR
from rl_agent.ue_mcs_backlog_near_capacity_v1 import contract as NC
from rl_agent.ue_mcs_backlog_near_capacity_v1 import radio_binding as RB

from . import selector as S


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_RELPATH = "rl_agent/ue_mcs_backlog_robust_bracket_v1"
CONFIG_RELPATH = f"{PACKAGE_RELPATH}/config_v1.json"
DEFAULT_CONFIG = ROOT / CONFIG_RELPATH
DEFAULT_SEALED_DIR = ROOT / PACKAGE_RELPATH / "sealed"

AMENDMENT_FILENAME = "PROSPECTIVE_CAPACITY_AMENDMENT.json"
MANIFEST_FILENAME = "manifest.json"
TERMINAL_FILENAME = "AMENDMENT_ADOPTED.json"
AMENDMENT_SCHEMA = "scenesense.ue_mcs_backlog_robust_bracket_amendment.v1"
MANIFEST_SCHEMA = "scenesense.ue_mcs_backlog_robust_bracket_manifest.v1"
TERMINAL_SCHEMA = "scenesense.ue_mcs_backlog_robust_bracket_terminal.v1"
STATUS = (
    "PHYSICAL_CAPACITY_INTERVAL_ADOPTED_FOR_PROSPECTIVE_DOWNSTREAM_DESIGN_"
    "ONLY__ORIGINAL_CAPACITY_QUALIFICATION_REFUSED__ORIGINAL_TIER_SELECTION_"
    "NOT_QUALIFIED"
)
ORIGINAL_STATUS = "CAPACITY_QUALIFICATION_REFUSED"
INSTABILITY_PROBLEM = (
    "tier stability: bootstrap capacity interval changes the deterministic "
    "tier triplet"
)
INSTABILITY_DETAIL = (
    "bootstrap capacity interval changes the deterministic tier triplet"
)

EXPECTED_BINDINGS: Mapping[str, tuple[str, str]] = {
    "result": (
        "CAPACITY_QUALIFICATION_RESULT.json",
        "0f3b1a3e4d8b04da33876b679e966a0ca11080c20ebed039d5fd18bd9cb17a43",
    ),
    "manifest": (
        "manifest.json",
        "8a77f0ff913ffb493cc226313bd2fd221394ce7330041770a0b346bda29d0293",
    ),
    "refused_terminal": (
        "CAPACITY_QUALIFICATION_REFUSED.json",
        "6540cbdd003169f46165df564f533d0303999dd51cc845b2ba6b2ca4778defc7",
    ),
    "final_cold_state": (
        "final_cold_state.json",
        "d062eb00c0d0f5d1fea88ddcec7904ea3412ea554d371fabf6fae9ffab12bda5",
    ),
}
EXPECTED_ATTEMPT_RELPATH = (
    "rl_agent/experiments/ue_mcs_backlog_near_capacity_capacity_v1/"
    "20260925_001332"
)
EXPECTED_CATALOG_JSON_SHA256 = (
    "07e0690f8a55bdd6068b8b283d14b7e165ccbf44742dd0a9568cfdd5dcac54c3"
)
EXPECTED_CATALOG_CSV_SHA256 = (
    "0512cb39982178e8c7c96a65ed26e272b3aa3a5aec0020a8dd9cf1cdb6696fbb"
)

SOURCE_RELPATHS: tuple[str, ...] = (
    f"{PACKAGE_RELPATH}/__init__.py",
    f"{PACKAGE_RELPATH}/__main__.py",
    f"{PACKAGE_RELPATH}/selector.py",
    f"{PACKAGE_RELPATH}/amendment.py",
    f"{PACKAGE_RELPATH}/config_v1.json",
    f"{PACKAGE_RELPATH}/PREREGISTRATION.md",
)


class AmendmentError(RuntimeError):
    """A source, evidence, rule, candidate, or seal gate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AmendmentError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AmendmentError(f"cannot open {label} JSON {path}: {exc}") from exc
    require(type(value) is dict, f"{label} must be a JSON object")
    return value


def write_json_create(path: Path, value: Any) -> None:
    """Write canonical presentation JSON without ever replacing a file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _safe_relative_path(root: Path, value: Any) -> tuple[str, Path]:
    require(type(value) is str and bool(value),
            "manifest relative_path must be a non-empty string")
    relative = Path(value)
    require(not relative.is_absolute(), f"manifest path must be relative: {value!r}")
    require(value == relative.as_posix() and not any(
        part in ("", ".", "..") for part in relative.parts),
        f"manifest path is not canonical or escapes its root: {value!r}")
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    require(resolved_root == candidate.parent or resolved_root in candidate.parents,
            f"manifest path escapes its root: {value!r}")
    return value, candidate


def _file_inventory(root: Path, *, excluded: Iterable[str] = ()) -> list[dict[str, Any]]:
    excluded_set = set(excluded)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_file() and relative not in excluded_set:
            rows.append({
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return rows


def _verify_inventory_rows(
    root: Path, rows: Any, *, label: str,
) -> list[dict[str, Any]]:
    require(type(rows) is list, f"{label} must be a list")
    seen: set[str] = set()
    for row in rows:
        require(type(row) is dict, f"{label} entry must be an object")
        relative, path = _safe_relative_path(root, row.get("relative_path"))
        require(relative not in seen, f"duplicate {label} path: {relative}")
        seen.add(relative)
        require(path.is_file(), f"{label} file is missing: {relative}")
        require(type(row.get("size_bytes")) is int and row["size_bytes"] >= 0,
                f"{label} size is invalid: {relative}")
        require(path.stat().st_size == row["size_bytes"],
                f"{label} size changed: {relative}")
        digest = row.get("sha256")
        require(type(digest) is str and bool(re.fullmatch(r"[0-9a-f]{64}", digest)),
                f"{label} digest is invalid: {relative}")
        require(sha256_file(path) == digest, f"{label} digest changed: {relative}")
    return rows


def source_inventory(repo_root: Path = ROOT) -> dict[str, Any]:
    """Hash every source that defines this amendment and its preregistration."""

    files: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_RELPATHS:
        path = repo_root / relative
        require(path.is_file(), f"amendment source is missing: {relative}")
        files[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    body = {"repo_files": files}
    return {**body, "inventory_sha256": S.canonical_sha256(body)}


def load_config(repo_root: Path = ROOT) -> dict[str, Any]:
    config = _json_object(repo_root / CONFIG_RELPATH, label="amendment config")
    require(config.get("schema") ==
            "scenesense.ue_mcs_backlog_robust_bracket_config.v1",
            "amendment config schema mismatch")
    require(config.get("package_id") == "ue_mcs_backlog_robust_bracket_v1",
            "amendment config package identity mismatch")
    require(config.get("status") == STATUS, "amendment config status drifted")
    preserved = config.get("preserved_attempt")
    require(type(preserved) is dict, "config has no preserved-attempt binding")
    require(preserved.get("relative_path") == EXPECTED_ATTEMPT_RELPATH,
            "config preserved-attempt path drifted")
    for name, (filename, digest) in EXPECTED_BINDINGS.items():
        row = preserved.get(name)
        require(type(row) is dict
                and row.get("filename") == filename
                and row.get("sha256") == digest,
                f"config {name} binding drifted")
    catalogue = config.get("catalogue")
    require(type(catalogue) is dict
            and catalogue.get("json_relative_path") == NC.ACTION_CATALOG_RELPATH
            and catalogue.get("json_sha256") == EXPECTED_CATALOG_JSON_SHA256
            and catalogue.get("csv_relative_path") == NC.ACTION_CATALOG_CSV_RELPATH
            and catalogue.get("csv_sha256") == EXPECTED_CATALOG_CSV_SHA256,
            "config catalogue binding drifted")
    uncertainty = config.get("retained_empirical_bootstrap_uncertainty_set_mbps")
    require(uncertainty == {
        "lower": 28.512,
        "point": 30.576,
        "upper": 31.584,
        "interpretation": S.UNCERTAINTY_INTERPRETATION,
    }, "config uncertainty-set interpretation or endpoints drifted")
    selector = config.get("selector")
    registered_rule = S.registered_rule()
    require(type(selector) is dict
            and selector.get("mode") == S.SELECTION_MODE
            and selector.get("group_identity_fields") == list(S.GROUP_IDENTITY_FIELDS)
            and selector.get("cross_group_mixing_allowed") is False
            and selector.get("exterior_margin_fraction") == 0.1
            and selector.get("target_ratios_to_point") == {
                "low": 0.5, "medium": 1.0, "high": 1.4,
            }
            and selector.get("group_ranking") ==
                registered_rule["feasible_group_ranking"],
            "config robust selector rule drifted")
    expected = config.get("expected_selection")
    require(type(expected) is dict
            and expected.get("family") == "AE64"
            and expected.get("quantizer") == "UINT8"
            and expected.get("action_ids") == S.EXPECTED_ACTION_IDS
            and expected.get("median_payload_bytes") == S.EXPECTED_PAYLOAD_BYTES
            and expected.get("offered_mbps") == {
                tier: float(S.EXPECTED_OFFERED_MBPS[tier]) for tier in S.TIER_ORDER
            }, "config expected robust selection drifted")
    governance = config.get("governance")
    require(type(governance) is dict
            and governance.get("scope") ==
                "PROSPECTIVE_DOWNSTREAM_BYTE_ONLY_QUEUE_DESIGN"
            and governance.get("catalogue_contract_tier") == "EMERGENCY_ONLY"
            and governance.get("original_capacity_qualification_overturned") is False
            and governance.get("original_tier_selection_qualified") is False
            and governance.get("live_execution_authorized") is False
            and governance.get("perception_endorsement") is False
            and type(governance.get("caveat")) is str
            and "EMERGENCY_ONLY" in governance["caveat"]
            and "byte-only queue design" in governance["caveat"],
            "config governance/perception caveat drifted")
    return config


def load_catalogue(
    repo_root: Path, config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = config["catalogue"]
    json_path = repo_root / binding["json_relative_path"]
    csv_path = repo_root / binding["csv_relative_path"]
    require(json_path.is_file() and csv_path.is_file(),
            "pinned catalogue JSON and CSV must both exist")
    observed_json = sha256_file(json_path)
    observed_csv = sha256_file(csv_path)
    require(observed_json == binding["json_sha256"],
            "pinned catalogue JSON digest changed")
    require(observed_csv == binding["csv_sha256"],
            "pinned catalogue CSV digest changed")
    catalogue = _json_object(json_path, label="action catalogue")
    return catalogue, {
        "json_relative_path": binding["json_relative_path"],
        "json_sha256": observed_json,
        "csv_relative_path": binding["csv_relative_path"],
        "csv_sha256": observed_csv,
    }


def validate_original_refusal_document(
    result: Mapping[str, Any], *, manifest: Mapping[str, Any] | None = None,
    terminal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Enforce the original disposition independently of hash seals."""

    require(result.get("schema") == CR.RESULT_SCHEMA,
            "preserved result schema mismatch")
    require(result.get("stage_id") == CQ.STAGE_ID,
            "preserved result stage identity mismatch")
    require(result.get("status") == ORIGINAL_STATUS,
            "original capacity result is no longer REFUSED")
    require(result.get("qualified") is False,
            "original capacity result qualified flag is not exactly false")
    require(result.get("selected_tiers") == [],
            "original capacity result selected tiers are not exactly empty")
    require(result.get("failure") == INSTABILITY_PROBLEM,
            "original capacity result failure reason drifted")
    audit = result.get("audit")
    require(type(audit) is dict, "original capacity result has no audit")
    require(audit.get("qualified") is False,
            "original capacity audit qualified flag is not exactly false")
    require(audit.get("problems") == [INSTABILITY_PROBLEM],
            "original capacity audit does not have the sole exact instability problem")
    stability = audit.get("tier_stability")
    require(type(stability) is dict
            and stability.get("stable") is False
            and stability.get("problems") == [INSTABILITY_DETAIL],
            "original tier-stability refusal semantics drifted")
    interval = stability.get("interval")
    require(type(interval) is dict
            and interval.get("lower_mbps") == 28.512
            and interval.get("point_estimate_mbps") == 30.576
            and interval.get("upper_mbps") == 31.584
            and interval.get("seed") == CQ.BOOTSTRAP_SEED
            and interval.get("draws") == CQ.BOOTSTRAP_DRAWS
            and interval.get("confidence") == CQ.BOOTSTRAP_CONFIDENCE
            and interval.get("sample_count") == 100,
            "retained empirical/bootstrap uncertainty set drifted")
    if manifest is not None:
        require(manifest.get("schema") == CR.MANIFEST_SCHEMA
                and manifest.get("status") == ORIGINAL_STATUS,
                "original manifest no longer records REFUSED")
    if terminal is not None:
        require(terminal.get("schema") == CR.TERMINAL_SCHEMA
                and terminal.get("status") == ORIGINAL_STATUS,
                "original terminal no longer records REFUSED")
    return {
        "status": ORIGINAL_STATUS,
        "qualified": False,
        "selected_tiers": [],
        "sole_audit_problem": INSTABILITY_PROBLEM,
        "original_tier_selection_qualified": False,
        "original_capacity_qualification_overturned": False,
    }


def _verify_point_records(
    result: Mapping[str, Any], *, repo_root: Path,
) -> tuple[list[CQ.CapacityPoint], Sequence[float], dict[str, Any]]:
    config = _json_object(
        repo_root / "rl_agent/ue_mcs_backlog_near_capacity_v1/config_v1.json",
        label="original registered config")
    expected_packetization = CQ.verify_probe_packetization(
        config["capacity_qualification"]["probe"]["packetization"],
        production_chunk_bytes=NC.CHUNK_BYTES,
        ssburst_header_bytes=CR.U3.HEADER.size)
    require(result.get("probe_packetization") == expected_packetization,
            "preserved result probe packetization does not reproduce")
    probe = CQ.verify_probe_action(repo_root)
    require(result.get("probe_identity") == probe,
            "preserved result probe action does not reproduce")

    records = result.get("points")
    require(type(records) is list and len(records) == 3,
            "preserved result must retain exactly three point records")
    points: list[CQ.CapacityPoint] = []
    boundary_samples: Sequence[float] | None = None
    labels: list[str] = []
    for record in records:
        require(type(record) is dict, "capacity point record must be an object")
        try:
            point = CQ.CapacityPoint(**record["capacity_point"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AmendmentError(f"capacity point record is malformed: {exc}") from exc
        labels.append(point.label)
        require(record.get("label") == point.label
                and record.get("target_snr_db") == point.target_snr_db,
                f"{point.label}: point identity does not reproduce")
        prime = record.get("prime")
        require(type(prime) is dict
                and prime.get("commanded_noise_power_db")
                    == point.commanded_noise_power_db
                and prime.get("read_back_noise_power_db")
                    == point.commanded_noise_power_db,
                f"{point.label}: RF command/read-back does not reproduce")
        require(record.get("packetization") == expected_packetization,
                f"{point.label}: packetization does not reproduce")
        sender = record.get("sender")
        require(type(sender) is dict
                and sender.get("frames") == CR.POINT_FRAMES
                and sender.get("payload_bytes") == CQ.PROBE_PAYLOAD_BYTES
                and sender.get("chunk_bytes") == CQ.PROBE_CHUNK_PAYLOAD_BYTES
                and sender.get("chunks_per_frame") == CR.PROBE_CHUNKS
                and sender.get("packetization") == expected_packetization
                and sender.get("unexpected_socket_errors") == 0,
                f"{point.label}: sender accounting is invalid")
        require(type(sender.get("chunks_handed_to_socket")) is int
                and type(sender.get("chunks_dropped_by_socket")) is int
                and sender["chunks_handed_to_socket"] >= 0
                and sender["chunks_dropped_by_socket"] >= 0
                and sender["chunks_handed_to_socket"]
                    + sender["chunks_dropped_by_socket"]
                    == CR.POINT_FRAMES * CR.PROBE_CHUNKS,
                f"{point.label}: sender datagram totals do not reconcile")
        samples = record.get("service_mbps_samples")
        require(type(samples) is list and len(samples) == point.samples
                and all(type(value) in (int, float)
                        and math.isfinite(float(value)) and float(value) >= 0
                        for value in samples),
                f"{point.label}: retained service samples are invalid")
        sink = record.get("sink")
        require(type(sink) is dict
                and sink.get("clean_duration_complete") is True
                and sink.get("expected_frames") == CR.POINT_FRAMES
                and sink.get("expected_chunks_per_frame") == CR.PROBE_CHUNKS
                and sink.get("header_bytes_excluded") == CR.U3.HEADER.size
                and sink.get("packetization") == expected_packetization
                and sink.get("malformed_datagrams") == 0
                and sink.get("packetization_mismatch_datagrams") == 0
                and sink.get("outside_registered_probe") == 0,
                f"{point.label}: ext-DN sink gate is invalid")
        bins = sink.get("payload_bytes_per_100ms_bin")
        require(type(bins) is list and len(bins) >= CR.POINT_BINS
                and all(type(value) is int and value >= 0 for value in bins),
                f"{point.label}: ext-DN payload bins are invalid")
        derived = [
            float(value) * 8.0 / CQ.SAMPLE_PERIOD_S / 1e6
            for value in bins[CR.SETTLE_BINS:CR.SETTLE_BINS + CR.MEASURE_BINS]
        ]
        require(samples == derived,
                f"{point.label}: service samples do not derive from payload bins")
        percentiles = (
            CQ._percentile(derived, 0.10),
            CQ._percentile(derived, 0.50),
            CQ._percentile(derived, 0.90),
        )
        require((point.service_mbps_p10, point.service_mbps_p50,
                 point.service_mbps_p90) == percentiles,
                f"{point.label}: service percentiles do not reproduce")
        achieved = record.get("achieved_pusch_snr_db")
        require(type(achieved) is dict and type(achieved.get("values")) is list,
                f"{point.label}: achieved-PUSCH samples are absent")
        achieved_values = achieved["values"]
        require(len(achieved_values) == point.achieved_pusch_snr_samples
                and all(type(value) in (int, float)
                        and math.isfinite(float(value)) for value in achieved_values)
                and achieved.get("samples") == len(achieved_values)
                and achieved.get("p50") == statistics.median(achieved_values)
                and point.achieved_pusch_snr_db_p50
                    == statistics.median(achieved_values),
                f"{point.label}: achieved-PUSCH summary does not reproduce")
        corroboration = record.get("corroboration")
        require(type(corroboration) is dict,
                f"{point.label}: corroborating telemetry is absent")
        backlog = corroboration.get("backlog")
        require(type(backlog) is dict,
                f"{point.label}: backlog evidence is absent")
        flags = backlog.get("bin_backlogged")
        counts = backlog.get("tick_counts_per_bin")
        minima = backlog.get("minimum_backlog_per_bin")
        require(type(flags) is list and len(flags) == CR.MEASURE_BINS
                and all(type(value) is bool for value in flags)
                and type(counts) is list and len(counts) == CR.MEASURE_BINS
                and all(type(value) is int and value >= 0 for value in counts)
                and type(minima) is list and len(minima) == CR.MEASURE_BINS
                and all(value is None or type(value) is int and value >= 0
                        for value in minima),
                f"{point.label}: backlog-bin evidence is malformed")
        reproduced_flags = [
            count > 0 and minimum is not None and minimum > 0
            for count, minimum in zip(counts, minima)
        ]
        fraction = sum(reproduced_flags) / CR.MEASURE_BINS
        require(flags == reproduced_flags
                and backlog.get("backlogged_fraction") == fraction
                and point.backlogged_fraction == fraction,
                f"{point.label}: backlog saturation does not reproduce")
        require(corroboration.get("primary_extdn_unique_application_payload_bytes")
                == sum(bins[CR.SETTLE_BINS:CR.SETTLE_BINS + CR.MEASURE_BINS]),
                f"{point.label}: primary ext-DN byte total does not reproduce")
        sources = tuple(corroboration.get(name) for name in (
            "ue_rlc_tx_sdu", "ue_rlc_tx_dequeue", "gnb_pdcp_rx_deliver"))
        complete = all(
            type(source) is dict
            and type(source.get("events")) is int and source["events"] > 0
            and type(source.get("bytes")) is int and source["bytes"] >= 0
            for source in sources
        ) and any(count > 0 for count in counts)
        require(complete and corroboration.get("corroboration_complete") is True,
                f"{point.label}: corroborating telemetry is incomplete")
        drain = record.get("post_probe_drain")
        require(type(drain) is dict and drain.get("drained") is True
                and drain.get(
                    "no_new_pdcp_or_rlc_ingress_during_quiet_interval") is True
                and int(drain.get("quiet_elapsed_ns", -1))
                    >= int(drain.get("quiet_interval_ns", 0)) > 0,
                f"{point.label}: post-probe drain/quiet proof is invalid")
        points.append(point)
        if point.label == CQ.BOUNDARY_OPERATING_POINT:
            boundary_samples = [float(value) for value in samples]
    require(labels == ["p25", "p50", "p75"],
            f"capacity point order/identity is invalid: {labels}")
    require(boundary_samples is not None, "boundary samples are absent")
    return points, boundary_samples, expected_packetization


def verify_preserved_attempt(
    repo_root: Path = ROOT, config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deeply verify seals, raw physical gates, teardown, and cold state.

    This intentionally accepts only the one hash-bound REFUSED attempt.  The
    physical gates must reproduce, while the tier-instability decision must
    remain the sole exact refusal problem.
    """

    config = dict(config) if config is not None else load_config(repo_root)
    attempt_root = (repo_root / EXPECTED_ATTEMPT_RELPATH).resolve()
    require(attempt_root.is_dir(), f"preserved attempt is missing: {attempt_root}")
    paths = {
        name: attempt_root / filename
        for name, (filename, _digest) in EXPECTED_BINDINGS.items()
    }
    for name, (_filename, digest) in EXPECTED_BINDINGS.items():
        require(paths[name].is_file(), f"preserved {name} is missing")
        require(sha256_file(paths[name]) == digest,
                f"preserved {name} digest differs from the exact binding")

    result = _json_object(paths["result"], label="preserved result")
    manifest = _json_object(paths["manifest"], label="preserved manifest")
    terminal = _json_object(paths["refused_terminal"], label="refused terminal")
    cold_file = _json_object(paths["final_cold_state"], label="final cold state")
    disposition = validate_original_refusal_document(
        result, manifest=manifest, terminal=terminal)
    require(manifest.get("result_sha256") == EXPECTED_BINDINGS["result"][1],
            "original manifest does not bind the exact result")
    require(terminal.get("result_sha256") == EXPECTED_BINDINGS["result"][1]
            and terminal.get("manifest_sha256") == EXPECTED_BINDINGS["manifest"][1],
            "original refused terminal does not bind the exact result/manifest")

    manifest_rows = _verify_inventory_rows(
        attempt_root, manifest.get("files"), label="preserved manifest inventory")
    expected_manifest = _file_inventory(
        attempt_root, excluded=(MANIFEST_FILENAME,
                                EXPECTED_BINDINGS["refused_terminal"][0]))
    require(manifest_rows == expected_manifest,
            "preserved manifest is not the exact current attempt inventory")
    expected_evidence = _file_inventory(
        attempt_root, excluded=(EXPECTED_BINDINGS["result"][0], MANIFEST_FILENAME,
                                EXPECTED_BINDINGS["refused_terminal"][0]))
    require(result.get("evidence_files") == expected_evidence,
            "preserved result evidence inventory is incomplete or stale")

    require(result.get("radio_profile_id") == RB.RADIO_PROFILE_ID,
            "preserved radio identity mismatch")
    require(result.get("primary_service_measurement") ==
            "EXT_DN_UNIQUE_SSBURST_APPLICATION_PAYLOAD_BYTES_PER_FIXED_"
            "100MS_MONOTONIC_WINDOW",
            "preserved result uses an unregistered primary service metric")
    require(result.get("pusch_tb_is_primary") is False,
            "PUSCH transport-block bytes cannot be the primary service metric")
    inventory = result.get("source_inventory")
    require(type(inventory) is dict, "preserved result has no source inventory")
    inventory_body = {
        "repo_files": inventory.get("repo_files"),
        "runtime": inventory.get("runtime"),
    }
    require(inventory.get("inventory_sha256") == CR.canonical_sha256(inventory_body),
            "preserved source-inventory digest is invalid")
    require(inventory == CR.source_inventory(repo_root),
            "preserved source inventory differs from current execution sources")
    require(manifest.get("source_inventory_sha256") ==
            inventory.get("inventory_sha256")
            and terminal.get("source_inventory_sha256") ==
                inventory.get("inventory_sha256"),
            "preserved seals do not bind the source inventory")
    checks = result.get("source_verifications")
    require(type(checks) is list and bool(checks),
            "preserved result has no source/radio verification chain")
    required_stages = {
        "before_preflight", "before_point_p25", "before_point_p50",
        "before_point_p75", "final_sealing",
    }
    stages: set[str] = set()
    for check in checks:
        require(type(check) is dict, "source verification row must be an object")
        stages.add(str(check.get("stage")))
        require(check.get("source_inventory", {}).get("verified") is True
                and check.get("radio_binding", {}).get("verified") is True
                and check.get("protected_evidence", {}).get("all_unchanged") is True,
                "a preserved source/radio/protected-evidence gate did not pass")
    require(required_stages <= stages,
            "preserved result is missing a required source-verification stage")

    expected_containers = tuple(
        _json_object(
            repo_root / "rl_agent/ue_mcs_backlog_near_capacity_v1/config_v1.json",
            label="original registered config")["radio"]["core_containers"])
    CR._validate_container_image_bindings(
        result.get("container_images"), expected_names=expected_containers)
    points, boundary_samples, _packetization = _verify_point_records(
        result, repo_root=repo_root)
    recomputed = CQ.audit_points(
        points, boundary_service_samples=boundary_samples, repo_root=repo_root)
    require(recomputed == result.get("audit"),
            "preserved capacity audit does not reproduce from retained evidence")
    require(recomputed.get("qualified") is False
            and recomputed.get("problems") == [INSTABILITY_PROBLEM],
            "physical audit has a problem other than exact tier-action instability")
    require(result.get("adverse_capacity_mbps") == 30.576
            and result.get("adverse_capacity_mbps") ==
                recomputed.get("adverse_capacity_mbps"),
            "preserved point capacity does not reproduce")

    initial_drain = result.get("initial_drain")
    require(type(initial_drain) is dict and initial_drain.get("drained") is True
            and initial_drain.get(
                "no_new_pdcp_or_rlc_ingress_during_quiet_interval") is True,
            "preserved result lacks the initial drain/quiet proof")
    cold = result.get("final_cold_state")
    require(cold == cold_file,
            "standalone final-cold-state file differs from the embedded proof")
    require(type(cold) is dict and cold.get("cold") is True
            and cold.get("schema") == "scenesense.capacity_final_cold_state.v1"
            and cold.get("orphan_processes") == {}
            and cold.get("residual_ue_tunnels") == []
            and cold.get("carla_processes") == []
            and cold.get("probe_errors") == []
            and set(cold.get("core_containers", {})) == set(expected_containers)
            and not any(str(value).startswith("true")
                        for value in cold["core_containers"].values()),
            "preserved result lacks a complete cold final-state proof")
    teardown = result.get("teardown")
    core = teardown.get("core", {}) if type(teardown) is dict else {}
    require(type(teardown) is dict
            and teardown.get("extract_ttracer_ok") is True
            and teardown.get("ran_notes") == []
            and core.get("stopped") is True
            and core.get("returncode") == 0
            and set(core.get("core_after", {})) == set(expected_containers)
            and not any(str(value).startswith("true")
                        for value in core["core_after"].values()),
            "preserved teardown is incomplete or failed")
    require(type(result.get("radio_lineage")) is dict
            and bool(result["radio_lineage"]),
            "preserved result has no authorization/run lineage")

    return {
        "verified": True,
        "bindings": {
            name: {"filename": filename, "sha256": digest}
            for name, (filename, digest) in EXPECTED_BINDINGS.items()
        },
        "original_disposition": disposition,
        "physical_gates": {
            "all_manifested_evidence_hashes_verified": True,
            "source_radio_and_protected_evidence_stages_verified": sorted(
                required_stages),
            "radio_profile_id": RB.RADIO_PROFILE_ID,
            "primary_service_measurement": result["primary_service_measurement"],
            "pusch_tb_is_primary": False,
            "point_labels": [point.label for point in points],
            "point_gate_count": len(points),
            "achieved_snr_strictly_ordered": True,
            "monotonic_in_snr": True,
            "backlogged_fraction_by_point": {
                point.label: point.backlogged_fraction for point in points
            },
            "drain_and_quiet_proofs_verified": True,
            "corroborating_telemetry_verified": True,
            "physical_audit_problems": [],
            "sole_nonphysical_audit_problem": INSTABILITY_PROBLEM,
            "teardown_verified": True,
            "final_cold_state_verified": True,
        },
        "retained_empirical_bootstrap_uncertainty_set":
            S.registered_uncertainty_set(),
    }


def prove_old_verifier_rejects(repo_root: Path = ROOT) -> dict[str, Any]:
    """Prove the unmodified old verifier still refuses the preserved result."""

    path = repo_root / EXPECTED_ATTEMPT_RELPATH / EXPECTED_BINDINGS["result"][0]
    try:
        CR.verify_bound_capacity_result(path)
    except CR.CapacityRunError as exc:
        return {
            "rejected": True,
            "verifier": (
                "rl_agent.ue_mcs_backlog_near_capacity_v1.capacity_runner."
                "verify_bound_capacity_result"
            ),
            "exception_type": type(exc).__name__,
            "reason": str(exc),
            "original_verifier_modified": False,
        }
    raise AmendmentError(
        "old verify_bound_capacity_result unexpectedly accepted the refused attempt")


def build_amendment_document(repo_root: Path = ROOT) -> dict[str, Any]:
    """Construct the complete deterministic amendment in memory."""

    config = load_config(repo_root)
    catalogue, catalogue_binding = load_catalogue(repo_root, config)
    selection = S.select_robust_bracket(catalogue)
    physical = verify_preserved_attempt(repo_root, config)
    old_rejection = prove_old_verifier_rejects(repo_root)
    sources = source_inventory(repo_root)
    return {
        "schema": AMENDMENT_SCHEMA,
        "package_id": "ue_mcs_backlog_robust_bracket_v1",
        "status": STATUS,
        "scope": "PROSPECTIVE_DOWNSTREAM_BYTE_ONLY_QUEUE_DESIGN",
        "config": {
            "relative_path": CONFIG_RELPATH,
            "sha256": sha256_file(repo_root / CONFIG_RELPATH),
        },
        "source_inventory": sources,
        "catalogue_binding": catalogue_binding,
        "preserved_attempt": {
            "relative_path": EXPECTED_ATTEMPT_RELPATH,
            **physical,
            "old_verifier_rejection_proof": old_rejection,
        },
        "selector": selection,
        "adoption": {
            "retained_empirical_bootstrap_uncertainty_set":
                S.registered_uncertainty_set(),
            "selected_action_ids": dict(S.EXPECTED_ACTION_IDS),
            "selected_payload_bytes": dict(S.EXPECTED_PAYLOAD_BYTES),
            "selected_offered_mbps": {
                tier: float(S.EXPECTED_OFFERED_MBPS[tier]) for tier in S.TIER_ORDER
            },
            "selected_family": "AE64",
            "selected_quantizer": "UINT8",
            "catalogue_contract_tier": "EMERGENCY_ONLY",
            "byte_only_queue_design": True,
            "perception_endorsement": False,
        },
        "governance": dict(config["governance"]),
        "original_capacity_qualification_overturned": False,
        "original_tier_selection_qualified": False,
        "live_execution_performed": False,
        "cuda_used": False,
        "network_used": False,
    }


def build_amendment(
    output_dir: Path = DEFAULT_SEALED_DIR, *, repo_root: Path = ROOT,
) -> dict[str, Path]:
    """Create a new sealed artifact directory, refusing any overwrite."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    preserved_root = (repo_root / EXPECTED_ATTEMPT_RELPATH).resolve()
    require(output_dir != preserved_root and preserved_root not in output_dir.parents,
            "amendment output may not be inside the preserved attempt")
    require(not output_dir.exists(),
            f"create-only amendment output already exists: {output_dir}")
    document = build_amendment_document(repo_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    amendment_path = output_dir / AMENDMENT_FILENAME
    write_json_create(amendment_path, document)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": STATUS,
        "amendment_sha256": sha256_file(amendment_path),
        "source_inventory_sha256": document["source_inventory"]["inventory_sha256"],
        "candidate_universe_sha256":
            document["selector"]["candidate_universe_sha256"],
        "files": _file_inventory(
            output_dir, excluded=(MANIFEST_FILENAME, TERMINAL_FILENAME)),
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    write_json_create(manifest_path, manifest)
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": STATUS,
        "amendment_sha256": sha256_file(amendment_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_inventory_sha256": document["source_inventory"]["inventory_sha256"],
        "candidate_universe_sha256":
            document["selector"]["candidate_universe_sha256"],
        "preserved_result_sha256": EXPECTED_BINDINGS["result"][1],
        "preserved_manifest_sha256": EXPECTED_BINDINGS["manifest"][1],
        "preserved_refused_terminal_sha256":
            EXPECTED_BINDINGS["refused_terminal"][1],
        "preserved_final_cold_state_sha256":
            EXPECTED_BINDINGS["final_cold_state"][1],
    }
    terminal_path = output_dir / TERMINAL_FILENAME
    write_json_create(terminal_path, terminal)
    return {
        "amendment": amendment_path,
        "manifest": manifest_path,
        "terminal": terminal_path,
    }


def _verify_new_seals(amendment_path: Path) -> tuple[
        dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(amendment_path.name == AMENDMENT_FILENAME,
            f"amendment must be named {AMENDMENT_FILENAME}")
    root = amendment_path.parent
    manifest_path = root / MANIFEST_FILENAME
    terminal_path = root / TERMINAL_FILENAME
    require(amendment_path.is_file() and manifest_path.is_file()
            and terminal_path.is_file(),
            "amendment, manifest, and terminal seal must all exist")
    amendment = _json_object(amendment_path, label="prospective amendment")
    manifest = _json_object(manifest_path, label="amendment manifest")
    terminal = _json_object(terminal_path, label="amendment terminal")
    require(amendment.get("schema") == AMENDMENT_SCHEMA,
            "amendment schema mismatch")
    require(manifest.get("schema") == MANIFEST_SCHEMA,
            "amendment manifest schema mismatch")
    require(terminal.get("schema") == TERMINAL_SCHEMA,
            "amendment terminal schema mismatch")
    require(amendment.get("status") == STATUS
            and manifest.get("status") == STATUS
            and terminal.get("status") == STATUS,
            "amendment status mismatch")
    amendment_digest = sha256_file(amendment_path)
    require(manifest.get("amendment_sha256") == amendment_digest
            and terminal.get("amendment_sha256") == amendment_digest,
            "amendment digest is not bound by both seals")
    require(terminal.get("manifest_sha256") == sha256_file(manifest_path),
            "terminal does not bind the amendment manifest")
    rows = _verify_inventory_rows(
        root, manifest.get("files"), label="amendment manifest inventory")
    expected = _file_inventory(
        root, excluded=(MANIFEST_FILENAME, TERMINAL_FILENAME))
    require(rows == expected,
            "amendment manifest is not the exact current file inventory")
    require([row["relative_path"] for row in rows] == [AMENDMENT_FILENAME],
            "sealed amendment directory contains an unexpected file")
    return amendment, manifest, terminal


def verify_amendment(
    path: Path = DEFAULT_SEALED_DIR / AMENDMENT_FILENAME, *,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Offline, fail-closed verification of seals, sources, evidence, and rule."""

    repo_root = repo_root.resolve()
    amendment_path = path / AMENDMENT_FILENAME if path.is_dir() else path
    amendment_path = amendment_path.resolve()
    amendment, manifest, terminal = _verify_new_seals(amendment_path)

    # Cheap semantic guards run before the 569-MiB deep preserved inventory.
    require(amendment.get("package_id") == "ue_mcs_backlog_robust_bracket_v1"
            and amendment.get("scope") ==
                "PROSPECTIVE_DOWNSTREAM_BYTE_ONLY_QUEUE_DESIGN",
            "amendment identity/scope drifted")
    require(amendment.get("original_capacity_qualification_overturned") is False
            and amendment.get("original_tier_selection_qualified") is False,
            "amendment attempts to reopen the original refusal")
    require(amendment.get("live_execution_performed") is False
            and amendment.get("cuda_used") is False
            and amendment.get("network_used") is False,
            "amendment is not an offline-only derivation")
    preserved = amendment.get("preserved_attempt")
    require(type(preserved) is dict
            and preserved.get("relative_path") == EXPECTED_ATTEMPT_RELPATH,
            "amendment preserved-attempt binding drifted")
    expected_disposition = {
        "status": ORIGINAL_STATUS,
        "qualified": False,
        "selected_tiers": [],
        "sole_audit_problem": INSTABILITY_PROBLEM,
        "original_tier_selection_qualified": False,
        "original_capacity_qualification_overturned": False,
    }
    require(preserved.get("original_disposition") == expected_disposition,
            "amendment does not preserve the exact original refusal")
    require(preserved.get("bindings") == {
        name: {"filename": filename, "sha256": digest}
        for name, (filename, digest) in EXPECTED_BINDINGS.items()
    }, "amendment preserved evidence bindings drifted")

    config = load_config(repo_root)
    config_binding = amendment.get("config")
    require(config_binding == {
        "relative_path": CONFIG_RELPATH,
        "sha256": sha256_file(repo_root / CONFIG_RELPATH),
    }, "amendment config source binding drifted")
    current_sources = source_inventory(repo_root)
    require(amendment.get("source_inventory") == current_sources,
            "amendment source inventory drifted")
    require(manifest.get("source_inventory_sha256") ==
            current_sources["inventory_sha256"]
            and terminal.get("source_inventory_sha256") ==
                current_sources["inventory_sha256"],
            "amendment seals do not bind the source inventory")

    catalogue, catalogue_binding = load_catalogue(repo_root, config)
    require(amendment.get("catalogue_binding") == catalogue_binding,
            "amendment catalogue binding drifted")
    expected_selection = S.select_robust_bracket(catalogue)
    require(amendment.get("selector") == expected_selection,
            "amendment candidate universe or robust selection rule does not reproduce")
    candidate_digest = expected_selection["candidate_universe_sha256"]
    require(manifest.get("candidate_universe_sha256") == candidate_digest
            and terminal.get("candidate_universe_sha256") == candidate_digest,
            "amendment seals do not bind the complete candidate universe")
    adoption = amendment.get("adoption")
    require(type(adoption) is dict
            and adoption.get("selected_action_ids") == S.EXPECTED_ACTION_IDS
            and adoption.get("selected_payload_bytes") == S.EXPECTED_PAYLOAD_BYTES
            and adoption.get("selected_offered_mbps") == {
                tier: float(S.EXPECTED_OFFERED_MBPS[tier]) for tier in S.TIER_ORDER
            }
            and adoption.get("selected_family") == "AE64"
            and adoption.get("selected_quantizer") == "UINT8"
            and adoption.get("catalogue_contract_tier") == "EMERGENCY_ONLY"
            and adoption.get("byte_only_queue_design") is True
            and adoption.get("perception_endorsement") is False,
            "amendment adoption or EMERGENCY_ONLY caveat drifted")
    require(amendment.get("governance") == config["governance"],
            "amendment governance/perception caveat drifted")

    physical = verify_preserved_attempt(repo_root, config)
    require(preserved.get("physical_gates") == physical["physical_gates"]
            and preserved.get("retained_empirical_bootstrap_uncertainty_set") ==
                physical["retained_empirical_bootstrap_uncertainty_set"],
            "amendment physical/cold verification summary does not reproduce")
    old_rejection = prove_old_verifier_rejects(repo_root)
    require(preserved.get("old_verifier_rejection_proof") == old_rejection,
            "old verifier rejection proof does not reproduce")
    require(old_rejection["rejected"] is True,
            "old verifier rejection was not proved")

    expected_document = build_amendment_document(repo_root)
    require(amendment == expected_document,
            "amendment document is not the deterministic registered build")
    require(terminal.get("preserved_result_sha256") ==
            EXPECTED_BINDINGS["result"][1]
            and terminal.get("preserved_manifest_sha256") ==
                EXPECTED_BINDINGS["manifest"][1]
            and terminal.get("preserved_refused_terminal_sha256") ==
                EXPECTED_BINDINGS["refused_terminal"][1]
            and terminal.get("preserved_final_cold_state_sha256") ==
                EXPECTED_BINDINGS["final_cold_state"][1],
            "terminal preserved-attempt bindings drifted")
    return {
        **amendment,
        "binding_verified": True,
        "binding": {
            "amendment_sha256": sha256_file(amendment_path),
            "manifest_sha256": sha256_file(amendment_path.parent / MANIFEST_FILENAME),
            "terminal_sha256": sha256_file(amendment_path.parent / TERMINAL_FILENAME),
            "root": str(amendment_path.parent),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="create a new sealed amendment directory")
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--repo-root", type=Path, default=ROOT)
    verify = sub.add_parser("verify", help="verify a sealed amendment offline")
    verify.add_argument("--amendment", type=Path,
                        default=DEFAULT_SEALED_DIR / AMENDMENT_FILENAME)
    verify.add_argument("--repo-root", type=Path, default=ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        paths = build_amendment(args.output_dir, repo_root=args.repo_root)
        print(json.dumps({key: str(value) for key, value in paths.items()},
                         sort_keys=True))
    else:
        verified = verify_amendment(args.amendment, repo_root=args.repo_root)
        print(json.dumps({
            "binding_verified": verified["binding_verified"],
            "status": verified["status"],
            "selected_action_ids": verified["adoption"]["selected_action_ids"],
            "binding": verified["binding"],
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

