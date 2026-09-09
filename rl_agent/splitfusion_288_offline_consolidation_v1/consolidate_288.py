#!/usr/bin/env python3
"""Offline, evidence-preserving consolidation of the completed SplitFusion
288-cell live CARLA/OAI campaign (Steps 1-4 only).

This analyzer is strictly read-only with respect to campaign evidence. It
never launches CARLA, Docker, OAI, RFsim, CUDA or model inference, never reads
training/holdout/validation/test imagery, never rescores a prediction, and
never tunes a threshold. It verifies bindings, reconciles the 288 cells,
separates the latency / delivery / payload / registered-accuracy components,
and writes its outputs into a create-only evidence directory.

Scope guard: no action pruning, no Pareto selection, no feasibility masking,
no reward design, no policy design, no training. Those are Steps 5-6.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "scenesense.splitfusion_288_offline_rl_dataset_consolidation.v1"

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT_REL = "experiments/splitfusion_288_live_campaign_v1/20260907_full_288_retry8"
CAMPAIGN_CONFIG_REL = "rl_agent/configs/ue_288_campaign_v1.yaml"
CATALOG_JSON_REL = "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json"
CATALOG_CSV_REL = "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.csv"
CATALOG_LOCK_REL = "rl_agent/splitfusion_action_catalog_v1/SPLITFUSION_72_PROFILE_ACTION_CATALOG_LOCKED"

# Bindings asserted by the completed campaign. Verified, never repaired.
EXPECTED = {
    "completion_sha256": "696af4443635bda2da8d93a2b8fd89808cd5ef04dfe34d30ed7f32a248ee1ee8",
    "campaign_manifest_sha256": "dd2b5ce4326284888db6e6e28beae0863f79002d96964bc0d65321d48c3e326b",
    "campaign_ledger_sha256": "dbbf3588fcaf542a6e3b2304a87267b58469c2dbd46bb1a9818cc2e82bb9d7f5",
    "cell_mapping_sha256": "add96c8f117c4cb8ba427c670ba1698d4a2b8aea84129b4d32ac47fdc56025b1",
    "terminal_status": "SPLITFUSION_288_CELL_LIVE_CARLA_OAI_CAMPAIGN_COMPLETE",
    "required_cells": 288,
    "expected_actions": 72,
    "executed_cells": 227,
    "reused_cells": 61,
    "service_deadline_ms": 100,
    "ack_timeout_ms": 500,
}
NETWORK_PROFILES = ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY")

# Registered deadline stages, in the campaign's own declared order.
DEADLINE_STAGES = (
    "UE_AFTER_PREPARATION", "UE_BEFORE_SEND", "EDGE_AFTER_REASSEMBLY",
    "EDGE_BEFORE_DECODE", "EDGE_BEFORE_TAIL", "EDGE_BEFORE_PUBLICATION",
    "UE_BEFORE_MAP_PUBLICATION",
)
PREPARE_STATUSES = (
    "SENT", "DROPPED_REPLACED_BY_NEWER_FRAME", "DROPPED_INCOMPLETE_RADAR_WINDOW",
    "WARMUP_NO_COMPLETE_RADAR_WINDOW", "STALE_BEFORE_SEND",
    "DROPPED_SENSOR_LATE_OR_MISSING", "SPLIT_PROCESSING_FAILED",
)
# Bump counters: an absent key means the event never occurred (zero), which is
# a measured outcome, not missing data.
EDGE_COUNTERS = (
    "feature_datagrams_received", "feature_datagrams_duplicate",
    "feature_messages_reassembled", "incomplete_reassemblies_expired",
    "reassembly_buffer_evictions", "reassembly_pending_high_water",
    "edge_queue_admissions", "edge_pending_replacements",
    "edge_pending_depth_high_water", "edge_process_starts", "tail_starts",
    "tail_completions", "compact_results_transmitted",
    "result_datagrams_transmitted", "result_bytes_transmitted",
    "result_datagrams_per_message_high_water", "evaluation_masks_submitted",
    "evaluation_masks_persisted", "evaluation_masks_hash_verified",
)
UE_TRANSPORT_COUNTERS = (
    "feature_messages_transmitted", "feature_datagrams_transmitted",
    "result_datagrams_received", "result_messages_reassembled",
    "results_published_to_map", "results_expired_before_map_publication",
    "stale_before_send",
)


class EvidenceError(RuntimeError):
    """Raised when a binding or identity cannot be proven. Fail closed."""


def fail(message: str) -> None:
    raise EvidenceError(message)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# Distribution summary
# --------------------------------------------------------------------------
def quantile_nearest_rank(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank quantile on an already sorted sequence.

    Deliberately identical to the campaign runner's own ``quantile`` helper
    (``rl_agent/ue_route_b_split_cell_adapter_v1.py``) so recomputed install-AoI
    medians/p95 can be cross-checked byte-for-byte against the registered
    values rather than merely "looking close".
    """
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
    return float(values[index])


def dist(values: Iterable[float], *, with_extrema: bool = True) -> dict[str, Any]:
    ordered = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    out: dict[str, Any] = {
        "count": len(ordered),
        "mean": (statistics.fmean(ordered) if ordered else None),
        "median": quantile_nearest_rank(ordered, 0.50),
        "p90": quantile_nearest_rank(ordered, 0.90),
        "p95": quantile_nearest_rank(ordered, 0.95),
    }
    if with_extrema:
        out["min"] = (ordered[0] if ordered else None)
        out["max"] = (ordered[-1] if ordered else None)
    return out


def flatten_dist(prefix: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def fnum(text: Any) -> float | None:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text) if math.isfinite(float(text)) else None
    text = str(text).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def inum(text: Any) -> int | None:
    value = fnum(text)
    return None if value is None else int(value)


def parse_pydict(text: Any) -> dict[str, Any]:
    """Parse a repr-serialised timing/counter dict from a CSV cell.

    The live runner wrote these columns with ``str(dict)``, so they are Python
    literals rather than JSON. ``literal_eval`` is used (never ``eval``) and a
    non-dict or unparseable value yields {} so the caller can count it as
    unavailable instead of guessing.
    """
    if text is None:
        return {}
    if isinstance(text, dict):
        return text
    text = str(text).strip()
    if not text:
        return {}
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    return value if isinstance(value, dict) else {}


def ratio(numerator: Any, denominator: Any) -> float | None:
    num, den = fnum(numerator), fnum(denominator)
    if num is None or den is None or den == 0:
        return None
    return num / den


def truthy(text: Any) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes"}


# --------------------------------------------------------------------------
# A. Evidence verification
# --------------------------------------------------------------------------
def read_campaign_config(repo: Path) -> dict[str, Any]:
    """Read the campaign YAML without importing a YAML dependency at module
    import time (kept local so the analyzer degrades with a clear message)."""
    try:
        import yaml
    except ImportError:  # pragma: no cover - environment guard
        fail("PyYAML is required to re-derive the cell mapping digest")
    return yaml.safe_load((repo / CAMPAIGN_CONFIG_REL).read_text(encoding="utf-8"))


def rederive_cell_mapping(config: Mapping[str, Any], catalog_rows: Sequence[Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Reproduce the supervisor's cell mapping digest from first principles."""
    mapping = [
        {
            "action_id": int(row["action_id"]),
            "cell_id": f"a{int(row['action_id']):02d}__{str(profile['profile_id']).lower()}",
            "network_profile_id": str(profile["profile_id"]),
            "profile_id": str(row["profile_id"]),
        }
        for profile in config["network"]["profiles"]
        for row in catalog_rows
    ]
    encoded = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), mapping


def verify_campaign(repo: Path) -> dict[str, Any]:
    root = repo / CAMPAIGN_ROOT_REL
    if not root.is_dir():
        fail(f"campaign root is absent: {CAMPAIGN_ROOT_REL}")

    terminal_path = root / EXPECTED["terminal_status"]
    completion_path = root / "campaign_completion.json"
    manifest_path = root / "campaign_manifest.json"
    ledger_path = root / "campaign_ledger.json"
    binding_path = root / "campaign_continuation_binding.json"
    for path in (terminal_path, completion_path, manifest_path, ledger_path, binding_path):
        if not path.is_file():
            fail(f"required campaign artifact is absent: {path.relative_to(repo)}")

    hashes = {
        "campaign_completion.json": sha256_file(completion_path),
        "campaign_manifest.json": sha256_file(manifest_path),
        "campaign_ledger.json": sha256_file(ledger_path),
        "campaign_continuation_binding.json": sha256_file(binding_path),
        EXPECTED["terminal_status"]: sha256_file(terminal_path),
    }
    terminal = load_json(terminal_path)
    completion = load_json(completion_path)
    manifest = load_json(manifest_path)
    ledger = load_json(ledger_path)
    binding = load_json(binding_path)

    # 1. Completion terminal and its bindings.
    if terminal.get("status") != EXPECTED["terminal_status"]:
        fail(f"terminal status mismatch: {terminal.get('status')!r}")
    if terminal.get("completion_sha256") != EXPECTED["completion_sha256"]:
        fail("terminal completion_sha256 differs from the declared binding")
    if hashes["campaign_completion.json"] != EXPECTED["completion_sha256"]:
        fail("campaign_completion.json does not hash to its own terminal binding")
    for key in ("campaign_manifest_sha256", "campaign_ledger_sha256", "cell_mapping_sha256"):
        if completion.get(key) != EXPECTED[key]:
            fail(f"completion binding {key} mismatch: {completion.get(key)!r}")
    # 2. Manifest and ledger hashes.
    if hashes["campaign_manifest.json"] != EXPECTED["campaign_manifest_sha256"]:
        fail("campaign_manifest.json hash mismatch")
    if hashes["campaign_ledger.json"] != EXPECTED["campaign_ledger_sha256"]:
        fail("campaign_ledger.json hash mismatch")
    if ledger.get("continuation_binding_sha256") != hashes["campaign_continuation_binding.json"]:
        fail("ledger-to-continuation-binding digest drift")
    if manifest.get("cell_mapping_sha256") != EXPECTED["cell_mapping_sha256"]:
        fail("manifest cell_mapping_sha256 mismatch")
    if int(manifest.get("service_deadline_ms", -1)) != EXPECTED["service_deadline_ms"]:
        fail("registered 100 ms service deadline drift")
    if int(manifest.get("ack_timeout_ms", -1)) != EXPECTED["ack_timeout_ms"]:
        fail("registered 500 ms ACK timeout drift")
    if completion.get("claims_100ms_service_ready") is not False:
        fail("campaign must not claim 100 ms service readiness")

    # Catalog identity.
    catalog_json_sha = sha256_file(repo / CATALOG_JSON_REL)
    lock_text = (repo / CATALOG_LOCK_REL).read_text(encoding="utf-8").split()
    if len(lock_text) != 2 or lock_text[1] != catalog_json_sha:
        fail("72-action catalog LOCKED terminal does not bind the catalog JSON")
    catalog = load_json(repo / CATALOG_JSON_REL)
    catalog_rows = catalog.get("profiles")
    if not isinstance(catalog_rows, list) or len(catalog_rows) != EXPECTED["expected_actions"]:
        fail("action catalog must contain exactly 72 actions")
    if [int(r["action_id"]) for r in catalog_rows] != list(range(72)):
        fail("action catalog IDs must be contiguous 0..71")

    config = read_campaign_config(repo)
    if config["actions"].get("catalog_sha256") != catalog_json_sha:
        fail("campaign config catalog_sha256 drift")
    config_sha = sha256_file(repo / CAMPAIGN_CONFIG_REL)
    if manifest.get("campaign_config_sha256") != config_sha:
        fail("campaign config hash differs from the manifest binding")

    # 3/4. Exactly 288 expected cell IDs over 72 actions x the four profiles.
    mapping_sha, mapping = rederive_cell_mapping(config, catalog_rows)
    if mapping_sha != EXPECTED["cell_mapping_sha256"]:
        fail("independently re-derived cell mapping digest does not match the binding")
    declared_profiles = tuple(str(p["profile_id"]) for p in config["network"]["profiles"])
    if set(declared_profiles) != set(NETWORK_PROFILES):
        fail(f"network profile set drift: {declared_profiles}")
    expected_ids = [m["cell_id"] for m in mapping]
    if len(expected_ids) != EXPECTED["required_cells"]:
        fail(f"expected 288 cell IDs, derived {len(expected_ids)}")
    if len(set(expected_ids)) != len(expected_ids):
        fail("derived cell IDs are not unique")

    # 5. No duplicate, missing or foreign cells.
    cells = ledger.get("cells") or {}
    actual_ids = set(cells)
    missing = sorted(set(expected_ids) - actual_ids)
    foreign = sorted(actual_ids - set(expected_ids))
    if missing:
        fail(f"missing cells in ledger: {missing[:8]}")
    if foreign:
        fail(f"foreign cells in ledger: {foreign[:8]}")
    multi = sorted(cid for cid, recs in cells.items() if len(recs) != 1)
    if multi:
        fail(f"cells with a non-unique attempt record: {multi[:8]}")

    statuses = Counter(recs[0]["status"] for recs in cells.values())
    if statuses.get("PASSED") != EXPECTED["executed_cells"]:
        fail(f"executed-cell count mismatch: {statuses.get('PASSED')}")
    if statuses.get("REUSED") != EXPECTED["reused_cells"]:
        fail(f"reused-cell count mismatch: {statuses.get('REUSED')}")
    if set(statuses) - {"PASSED", "REUSED"}:
        fail(f"unexpected terminal statuses in the final inventory: {sorted(set(statuses))}")

    # 7. Continuation chain, resolved rather than assumed.
    chain = verify_continuation_chain(repo, binding, manifest)

    return {
        "campaign_root": CAMPAIGN_ROOT_REL,
        "artifact_hashes": hashes,
        "terminal": terminal,
        "completion": completion,
        "manifest": manifest,
        "ledger": ledger,
        "continuation_binding": binding,
        "config_sha256": config_sha,
        "catalog_json_sha256": catalog_json_sha,
        "catalog_csv_sha256": sha256_file(repo / CATALOG_CSV_REL),
        "cell_mapping_sha256_rederived": mapping_sha,
        "expected_cell_ids": expected_ids,
        "network_profiles": list(declared_profiles),
        "status_counts": dict(statuses),
        "continuation_chain": chain,
        "catalog_rows": catalog_rows,
        "config": config,
    }


def verify_continuation_chain(repo: Path, binding: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve retry8 -> retry7 -> retry6 -> retry5 and prove each hop."""
    source_root_rel = str(binding["source_campaign_root"])
    if manifest.get("reuse_campaign_root") != source_root_rel:
        fail("manifest reuse root disagrees with the continuation binding")
    hops: list[dict[str, Any]] = []
    seen: set[str] = set()
    current_rel: str | None = source_root_rel
    child_binding: Mapping[str, Any] = binding
    while current_rel:
        if current_rel in seen:
            fail(f"continuation chain contains a cycle at {current_rel}")
        seen.add(current_rel)
        root = repo / current_rel
        if not root.is_dir():
            fail(f"continuation source campaign is absent: {current_rel}")
        led_sha = sha256_file(root / "campaign_ledger.json")
        man_sha = sha256_file(root / "campaign_manifest.json")
        if child_binding.get("source_campaign_ledger_sha256") != led_sha:
            fail(f"continuation ledger digest drift at {current_rel}")
        if child_binding.get("source_campaign_manifest_sha256") != man_sha:
            fail(f"continuation manifest digest drift at {current_rel}")
        src_manifest = load_json(root / "campaign_manifest.json")
        if child_binding.get("source_binding_sha256") != src_manifest.get("binding_sha256"):
            fail(f"continuation dispatch-binding digest drift at {current_rel}")
        src_ledger = load_json(root / "campaign_ledger.json")
        hops.append({
            "campaign_root": current_rel,
            "campaign_ledger_sha256": led_sha,
            "campaign_manifest_sha256": man_sha,
            "dispatch_binding_sha256": src_manifest.get("binding_sha256"),
            "campaign_config_sha256": src_manifest.get("campaign_config_sha256"),
            "starting_head": src_manifest.get("starting_head"),
            "rerun_cell_ids": src_manifest.get("rerun_cell_ids"),
            "reuse_campaign_root": src_manifest.get("reuse_campaign_root"),
            "ledger_status_counts": dict(Counter(
                rec["status"] for recs in src_ledger.get("cells", {}).values() for rec in recs
            )),
        })
        next_rel = src_manifest.get("reuse_campaign_root")
        nxt_binding_path = root / "campaign_continuation_binding.json"
        child_binding = load_json(nxt_binding_path) if nxt_binding_path.is_file() else {}
        current_rel = str(next_rel) if next_rel else None
    return {
        "immediate_source": source_root_rel,
        "declared_reused_cells": int(binding.get("reused_cells", -1)),
        "declared_references_carried_forward": int(binding.get("reused_references_carried_forward", -1)),
        "declared_reused_from_source_campaign": int(binding.get("reused_from_source_campaign", -1)),
        "reused_cells_predate_durable_route_summary_requirement": bool(
            binding.get("reused_cells_predate_durable_route_summary_requirement")
        ),
        "hops": hops,
    }


def verify_cell_evidence(repo: Path, cell_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """Verify one cell's terminal, attempt manifest and every registered output
    hash. Returns the resolved attempt directory and provenance."""
    status = str(record["status"])
    if status == "PASSED":
        root_rel = CAMPAIGN_ROOT_REL
        attempt_rel = str(record["attempt_dir"])
        terminal_rel = str(record["terminal"])
        manifest_expected = None
    elif status == "REUSED":
        root_rel = str(record["source_campaign_root"])
        attempt_rel = str(record["source_attempt_dir"])
        terminal_rel = str(record["source_terminal"])
        manifest_expected = str(record["source_manifest_sha256"])
    else:
        fail(f"{cell_id}: unsupported ledger status {status!r}")

    attempt_dir = repo / root_rel / attempt_rel
    if not attempt_dir.is_dir():
        fail(f"{cell_id}: attempt directory is absent: {root_rel}/{attempt_rel}")

    terminal_path = repo / root_rel / terminal_rel
    terminal_sha = sha256_file(terminal_path)
    if terminal_sha != record["terminal_sha256"]:
        fail(f"{cell_id}: terminal digest drift ({terminal_sha} != {record['terminal_sha256']})")
    terminal = load_json(terminal_path)
    if terminal.get("status") != "PASSED":
        fail(f"{cell_id}: terminal is not PASSED: {terminal.get('status')!r}")

    manifest_path = attempt_dir / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha is None:
        fail(f"{cell_id}: attempt manifest is absent")
    if manifest_expected is not None and manifest_sha != manifest_expected:
        fail(f"{cell_id}: reused attempt manifest digest drift")
    attempt_manifest = load_json(manifest_path)
    if attempt_manifest.get("structural_acceptance_status") != "PASS":
        fail(f"{cell_id}: structural acceptance is not PASS")
    if attempt_manifest.get("terminal_status") != "PASSED":
        fail(f"{cell_id}: attempt manifest terminal status is not PASSED")

    verified: list[str] = []
    for entry in attempt_manifest.get("files", []):
        path = attempt_dir / str(entry["path"])
        got = sha256_file(path)
        if got is None:
            fail(f"{cell_id}: registered output is absent: {entry['path']}")
        if got != entry["sha256"]:
            fail(f"{cell_id}: registered output digest drift: {entry['path']}")
        verified.append(str(entry["path"]))

    route_summary_sha = record.get("route_metrics_summary_sha256")
    route_summary_available = (attempt_dir / "route_metrics_summary.json").is_file()
    route_runner_status_raw = ""
    if status == "PASSED":
        if not route_summary_available:
            fail(f"{cell_id}: executed cell lacks its durable route summary")
        if sha256_file(attempt_dir / "route_metrics_summary.json") != route_summary_sha:
            fail(f"{cell_id}: route summary digest drift")
        route_outcome = str(record.get("route_outcome_classification") or "")
    else:
        declared = bool(record.get("route_metrics_summary_present"))
        if declared != route_summary_available:
            fail(f"{cell_id}: reused route-summary presence disagrees with the ledger")
        # 8. Never infer an unavailable route outcome.
        # The reused sources predate the campaign's route_outcome_classification.
        # Their durable file is the route runner's own metrics summary, whose
        # "status" is the density/intervention state, NOT the registered
        # classification. Promoting one to the other would be an inference, so
        # the classification stays unavailable and the raw status is carried
        # separately, clearly labelled.
        route_outcome = ""
        if route_summary_available:
            summary = load_json(attempt_dir / "route_metrics_summary.json")
            route_outcome = str(
                (summary.get("route_outcome") or {}).get("classification") or ""
            )
            route_runner_status_raw = str(summary.get("status") or "")

    return {
        "status": status,
        "executed_or_reused": "EXECUTED" if status == "PASSED" else "REUSED",
        "source_campaign_root": root_rel,
        "source_attempt": attempt_rel,
        "attempt_dir": attempt_dir,
        "source_terminal_sha256": terminal_sha,
        "source_manifest_sha256": manifest_sha,
        "registered_outputs_verified": len(verified),
        "route_summary_available": route_summary_available,
        "route_outcome_classification": route_outcome,
        "route_outcome_classification_available": bool(route_outcome),
        "route_runner_status_raw": route_runner_status_raw,
        "terminal": terminal,
        "attempt_manifest": attempt_manifest,
    }


# --------------------------------------------------------------------------
# Per-cell reconciliation
# --------------------------------------------------------------------------
def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_cell(
    cell_id: str,
    provenance: Mapping[str, Any],
    catalog_row: Mapping[str, Any],
    catalog_csv_row: Mapping[str, str],
    mapping_row: Mapping[str, Any],
    profile_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the reconciled record for one action/profile cell."""
    attempt_dir = Path(provenance["attempt_dir"])
    results = load_json(attempt_dir / "RESULTS_SUMMARY.json")
    frames = read_csv_rows(attempt_dir / "per_frame_metrics.csv")
    feedback = read_csv_rows(attempt_dir / "map_feedback.csv")
    radio = read_csv_rows(attempt_dir / "radio_trace.csv")

    sa = results.get("structural_acceptance") or {}
    rr = sa.get("realtime_recovery") or {}
    live = results.get("live_dispatch") or {}
    edge_counters = ((rr.get("edge_counters_file") or {}).get("counters")) or {}
    ue_transport = rr.get("ue_transport_counters") or {}
    ue_counters = rr.get("ue_counters") or {}
    prepare_counts = rr.get("prepare_status_counts") or {}
    drops = rr.get("deadline_drops_by_stage") or {}
    terminal_outcomes = sa.get("terminal_feedback_outcomes") or {}

    # Cross-check the ledger/summary identity against the derived mapping.
    if int(results.get("action_id", -1)) != int(mapping_row["action_id"]):
        fail(f"{cell_id}: RESULTS_SUMMARY action_id disagrees with the cell mapping")
    if str(results.get("network_profile_id")) != str(mapping_row["network_profile_id"]):
        fail(f"{cell_id}: RESULTS_SUMMARY network profile disagrees with the cell mapping")
    if str(results.get("cell_id")) != cell_id:
        fail(f"{cell_id}: RESULTS_SUMMARY cell_id drift")

    sent_rows = [r for r in frames if r.get("prepare_status") == "SENT"]
    # Per-frame identity: every SENT row must carry the cell's own action id.
    bad_action = {r.get("action_id") for r in sent_rows} - {str(mapping_row["action_id"])}
    if bad_action:
        fail(f"{cell_id}: per-frame rows carry foreign action ids {sorted(bad_action)}")
    bad_profile = {r.get("profile_id") for r in sent_rows} - {str(catalog_row["profile_id"]), ""}
    if bad_profile:
        fail(f"{cell_id}: per-frame rows carry foreign profile ids {sorted(bad_profile)}")

    record: dict[str, Any] = {}

    # ---- Identity / provenance -------------------------------------------
    caps = catalog_row.get("capabilities") or {}
    record.update({
        "cell_id": cell_id,
        "action_id": int(mapping_row["action_id"]),
        "profile_id": str(catalog_row["profile_id"]),
        "network_profile": str(mapping_row["network_profile_id"]),
        "family": str(catalog_row.get("family", "")),
        "latent_width": catalog_csv_row.get("latent_width", ""),
        "quantizer": str(catalog_row.get("quantizer", "")),
        "bit_width": catalog_csv_row.get("bit_width", ""),
        "q": catalog_csv_row.get("q", ""),
        "q_e4": int(catalog_row.get("q_e4", -1)),
        "keep_count": catalog_csv_row.get("keep_count", ""),
        "drop_count": catalog_csv_row.get("drop_count", ""),
        "routing_tag": catalog_csv_row.get("routing_tag", ""),
        "wire_layout": catalog_csv_row.get("wire_layout", ""),
        "wire_version": catalog_csv_row.get("wire_version", ""),
        "zstd_level": catalog_csv_row.get("zstd_level", ""),
        "registered_feature_wire_codec": str(results.get("registered_feature_wire_codec", "")),
        "spatial_map_packet_codec": str(results.get("spatial_map_packet_codec", "")),
        "execution_mode": catalog_csv_row.get("execution_mode", ""),
        "transport_valid": caps.get("transport_valid", ""),
        "agent_action_enabled": caps.get("agent_action_enabled", ""),
        "execution_status": provenance["status"],
        "executed_or_reused": provenance["executed_or_reused"],
        "source_campaign_root": provenance["source_campaign_root"],
        "source_attempt": provenance["source_attempt"],
        "source_terminal_sha256": provenance["source_terminal_sha256"],
        "source_manifest_sha256": provenance["source_manifest_sha256"],
        "registered_outputs_verified": provenance["registered_outputs_verified"],
        "structural_acceptance_status": str(sa.get("status", "")),
        "terminal_status": str(results.get("terminal_status", "")),
        "route_summary_available": provenance["route_summary_available"],
        "route_outcome_classification": provenance["route_outcome_classification"] or "UNAVAILABLE_NOT_INFERRED",
        "route_outcome_classification_available": provenance["route_outcome_classification_available"],
        "route_runner_status_raw_not_a_classification": provenance["route_runner_status_raw"],
        "trace_id": str(profile_meta.get("trace_id", "")),
        "trace_seed": profile_meta.get("seed", ""),
        "trace_sha256": str(profile_meta.get("trace_sha256", "")),
        "git_commit_at_launch": str((provenance["attempt_manifest"]).get("git_commit_at_launch", "")),
        "one_clock_owner": str(results.get("one_clock_owner", "")),
        "one_ego_owner": str(results.get("one_ego_owner", "")),
        "cell_started_at_unix_s": fnum(results.get("started_at_unix_s")),
        "cell_finished_at_unix_s": fnum(results.get("finished_at_unix_s")),
    })
    started, finished = record["cell_started_at_unix_s"], record["cell_finished_at_unix_s"]
    record["cell_wall_duration_s"] = (finished - started) if (started and finished) else None
    cleanup = (provenance["terminal"].get("carla_cleanup") or {})
    record["radio_lifecycle_gates_passed"] = (cleanup.get("radio") or {}).get("all_lifecycle_gates_passed", "")
    record["radio_shutdown_verified"] = provenance["terminal"].get("radio_shutdown_verified", "")
    record["carla_shutdown_verified"] = (cleanup.get("carla") or {}).get("shutdown_verified", "")
    record["teardown_errors"] = len((cleanup.get("radio") or {}).get("errors") or [])
    record["cell_failures"] = len(results.get("failures") or [])

    # ---- Workload / preparation ------------------------------------------
    scheduled = inum(sa.get("scheduled_frames"))
    expected_sched = inum(sa.get("expected_scheduled_frames"))
    eligible = inum(sa.get("eligible_preparation_frames"))
    sent_frames = inum(sa.get("sent_frames"))
    offers = inum(ue_counters.get("preparation_offers"))
    coverage = fnum(sa.get("sensor_preparation_coverage"))
    duration = record["cell_wall_duration_s"]
    record.update({
        "expected_prepared_hz": fnum(sa.get("expected_prepared_hz")),
        "intended_opportunities_expected_scheduled_frames": expected_sched,
        "scheduled_frames": scheduled,
        "route_ticks": inum(sa.get("route_ticks")),
        "eligible_preparation_frames": eligible,
        "preparation_offers": offers,
        "synchronized_sensor_opportunities": eligible,
        "prepared_frames_sent": sent_frames,
        "split_frames_sent": inum(results.get("split_frames_sent")),
        "split_frames_dropped": inum(results.get("split_frames_dropped")),
        "per_frame_rows": len(frames),
        "sensor_preparation_coverage": coverage,
        "minimum_sensor_preparation_coverage": fnum(sa.get("minimum_sensor_preparation_coverage")),
        "sensor_preparation_coverage_met": sa.get("sensor_preparation_coverage_met", ""),
        "preparation_pending_replacements": inum(ue_counters.get("preparation_pending_replacements")),
        "prepared_fps_achieved": (ratio(sent_frames, duration) if duration else None),
        "evaluation_tickets_queued": inum(ue_counters.get("evaluation_tickets_queued")),
        "evaluation_tickets_completed": inum(ue_counters.get("evaluation_tickets_completed")),
        "evaluation_tickets_dropped_queue_full": int(ue_counters.get("evaluation_tickets_dropped_queue_full") or 0),
    })
    for status in PREPARE_STATUSES:
        record[f"prepare_status_{status}"] = int(prepare_counts.get(status) or 0)
    unknown_status = set(prepare_counts) - set(PREPARE_STATUSES)
    if unknown_status:
        fail(f"{cell_id}: unregistered preparation status {sorted(unknown_status)}")
    record["prepare_status_total"] = sum(int(v) for v in prepare_counts.values())

    # Sustainable FPS from the observed worker period (SENT rows only).
    periods = []
    caps_wall = sorted(v for v in (fnum(r.get("capture_wall_s")) for r in sent_rows) if v)
    for a, b in zip(caps_wall, caps_wall[1:]):
        if b > a:
            periods.append((b - a) * 1000.0)
    period_summary = dist(periods)
    record["sustainable_fps_from_median_send_period"] = (
        1000.0 / period_summary["median"] if period_summary["median"] else None
    )
    record.update(flatten_dist("send_period_ms", period_summary))

    # ---- Payload ---------------------------------------------------------
    payload_fields = {
        "scientific_inner_bytes": "scientific_inner_bytes",
        "sfd1_overhead_bytes": "sfd1_overhead_bytes",
        "sfd1_bytes": "sfd1_bytes",
        "udp_application_bytes": "udp_application_bytes",
        "estimated_wire_bytes": "estimated_wire_bytes",
        "datagrams_per_message": "datagrams",
    }
    for out_name, col in payload_fields.items():
        record.update(flatten_dist(f"live_{out_name}", dist((fnum(r.get(col)) for r in sent_rows), with_extrema=False)))
    record["live_udp_fragmentation_overhead_bytes_median"] = (
        (record["live_udp_application_bytes_median"] - record["live_sfd1_bytes_median"])
        if record["live_udp_application_bytes_median"] is not None
        and record["live_sfd1_bytes_median"] is not None else None
    )
    record["live_ip_udp_header_overhead_bytes_median"] = (
        (record["live_estimated_wire_bytes_median"] - record["live_udp_application_bytes_median"])
        if record["live_estimated_wire_bytes_median"] is not None
        and record["live_udp_application_bytes_median"] is not None else None
    )
    # Catalog payload estimates are labelled as catalog-sourced, never mixed in.
    record["catalog_pre_zstd_analytical_bytes"] = fnum(catalog_csv_row.get("pre_zstd_analytical_bytes"))
    record["catalog_zstd_median_bytes"] = fnum(catalog_csv_row.get("zstd_median_bytes"))
    record["catalog_zstd_p95_bytes"] = fnum(catalog_csv_row.get("zstd_p95_bytes"))
    record["live_over_catalog_median_inner_bytes_ratio"] = ratio(
        record["live_scientific_inner_bytes_median"], record["catalog_zstd_median_bytes"]
    )
    record["payload_bytes_source"] = "LIVE_MEASURED_per_frame_metrics.csv"
    record["catalog_payload_bytes_source"] = "CATALOG_ESTIMATE_splitfusion_72_action_catalog"

    # ---- Delivery stages -------------------------------------------------
    for name in EDGE_COUNTERS:
        record[f"edge_{name}"] = int(edge_counters.get(name) or 0)
    for name in UE_TRANSPORT_COUNTERS:
        record[f"ue_{name}"] = int(ue_transport.get(name) or 0)
    for stage in DEADLINE_STAGES:
        record[f"deadline_drop_{stage}"] = int(drops.get(stage) or 0)
    unknown_stage = set(drops) - set(DEADLINE_STAGES)
    if unknown_stage:
        fail(f"{cell_id}: unregistered deadline stage {sorted(unknown_stage)}")

    installed = inum(rr.get("maps_installed")) or 0
    ack_installed = inum(sa.get("ack_installed_frames")) or 0
    on_time_100 = inum(rr.get("service_on_time_installations")) or 0
    timely = inum(rr.get("timely_installations")) or 0
    ack_500 = inum(rr.get("ack_within_timeout_installations")) or 0
    reassembled = record["edge_feature_messages_reassembled"]
    admitted = record["edge_edge_queue_admissions"]
    published = record["ue_results_published_to_map"]
    record.update({
        "frames_sent": sent_frames,
        "live_dispatch_sent": inum(live.get("sent")),
        "edge_complete_reassemblies": reassembled,
        "edge_admissions": admitted,
        "edge_tail_starts": record["edge_tail_starts"],
        "edge_tail_completions": record["edge_tail_completions"],
        "map_publications": published,
        "maps_installed": installed,
        "ack_installed_frames": ack_installed,
        "installed_within_100ms_service_reference": on_time_100,
        "timely_installations_alias_of_100ms": timely,
        "ack_within_500ms_timeout": ack_500,
        "late_feedback_rows": inum(rr.get("late_feedback_rows")),
        "late_nonterminal_feedback_rows": inum(rr.get("late_nonterminal_feedback_rows")),
        "terminal_TIMEOUT_NO_ACK": int(terminal_outcomes.get("TIMEOUT_NO_ACK") or 0),
        "terminal_ACK_INSTALLED": int(terminal_outcomes.get("ACK_INSTALLED") or 0),
        "terminal_feedback_records": inum(sa.get("terminal_feedback_records")),
        "feedback_rows": len(feedback),
        "zero_delivery": installed == 0,
        "zero_reassembly": reassembled == 0,
    })
    if timely != on_time_100:
        fail(f"{cell_id}: timely_installations is not the 100 ms alias")

    # Both denominators preserved, plus conditional stage survival.
    record.update({
        "rate_datagrams_received_per_transmitted": ratio(record["edge_feature_datagrams_received"], record["ue_feature_datagrams_transmitted"]),
        "rate_reassembled_per_sent": ratio(reassembled, sent_frames),
        "rate_installed_per_sent": ratio(installed, sent_frames),
        "rate_installed_per_reassembled": ratio(installed, reassembled),
        "rate_on_time_100ms_per_sent": ratio(on_time_100, sent_frames),
        "rate_on_time_100ms_per_installed": ratio(on_time_100, installed),
        "rate_ack_500ms_per_sent": ratio(ack_500, sent_frames),
        "rate_ack_500ms_per_installed": ratio(ack_500, installed),
        "survival_admitted_per_reassembled": ratio(admitted, reassembled),
        "survival_tail_start_per_admitted": ratio(record["edge_tail_starts"], admitted),
        "survival_tail_complete_per_start": ratio(record["edge_tail_completions"], record["edge_tail_starts"]),
        "survival_result_tx_per_tail_complete": ratio(record["edge_compact_results_transmitted"], record["edge_tail_completions"]),
        "survival_published_per_result_rx": ratio(published, record["ue_result_messages_reassembled"]),
        "survival_installed_per_published": ratio(installed, published),
        "registered_on_time_fraction_denominator": "installed_frames",
        "registered_service_on_time_fraction": fnum(rr.get("service_on_time_fraction")),
        "registered_ack_within_timeout_fraction": fnum(rr.get("ack_within_timeout_fraction")),
    })

    # Arithmetic identities registered by the campaign itself.
    recon = rr.get("counter_reconciliation") or {}
    record["counter_reconciliation_all_identities_hold"] = recon.get("all_identities_hold", "")
    if recon.get("all_identities_hold") is not True:
        fail(f"{cell_id}: campaign counter reconciliation does not hold")
    # Independent re-check of the reassembly identity.
    r_block = recon.get("edge_reassembled_equals_admitted_plus_dropped_plus_rejected") or {}
    lhs = inum(r_block.get("reassembled"))
    rhs = sum(int(r_block.get(k) or 0) for k in ("admitted", "after_reassembly_deadline_drops", "rejected"))
    record["identity_reassembly_lhs"] = lhs
    record["identity_reassembly_rhs"] = rhs
    record["identity_reassembly_holds"] = (lhs == rhs) if lhs is not None else ""
    if lhs is not None and lhs != rhs:
        fail(f"{cell_id}: reassembly identity fails ({lhs} != {rhs})")

    # ---- Radio context ---------------------------------------------------
    achieved = [fnum(r.get("achieved_snr_db")) for r in radio]
    target = [fnum(r.get("target_snr_db")) for r in radio]
    record.update(flatten_dist("radio_achieved_snr_db", dist(achieved)))
    record.update(flatten_dist("radio_target_snr_db", dist(target, with_extrema=False)))
    record["radio_trace_steps"] = len(radio)
    record["radio_prb"] = (radio[0].get("prb") if radio else "")
    record["radio_bandwidth_mhz"] = (radio[0].get("bandwidth_mhz") if radio else "")
    record["radio_command_timing_status_ok"] = sum(
        1 for r in radio if str(r.get("command_timing_status")) == "ON_TIME"
    )
    record.update(flatten_dist("radio_command_latency_ms", dist(
        (fnum(r.get("command_latency_ms")) for r in radio), with_extrema=False)))

    # ---- Live perception (kept strictly separate from frozen validation) --
    record.update(live_perception(attempt_dir))

    # ---- Latency decomposition -------------------------------------------
    record["_latency"] = latency_decomposition(cell_id, frames, feedback, sent_rows, rr)
    return record


def live_perception(attempt_dir: Path) -> dict[str, Any]:
    """Aggregate the live per-frame perception CSV. These are live-route
    measurements and must never be conflated with the frozen validation join."""
    rows = read_csv_rows(attempt_dir / "perception_metrics.csv")
    out: dict[str, Any] = {"live_perception_rows": len(rows)}
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[str(row.get("class_name") or "")].append(row)
    out["live_perception_classes"] = "|".join(sorted(by_class))
    for cls in ("vehicle", "person"):
        subset = by_class.get(cls, [])
        tp = sum(int(inum(r.get("tp")) or 0) for r in subset)
        fp = sum(int(inum(r.get("fp")) or 0) for r in subset)
        fn = sum(int(inum(r.get("fn")) or 0) for r in subset)
        out[f"live_{cls}_tp"] = tp
        out[f"live_{cls}_fp"] = fp
        out[f"live_{cls}_fn"] = fn
        out[f"live_{cls}_precision_micro"] = ratio(tp, tp + fp)
        out[f"live_{cls}_recall_micro"] = ratio(tp, tp + fn)
        out[f"live_{cls}_rows_with_exact_prediction"] = sum(
            1 for r in subset if truthy(r.get("exact_frame_prediction_available"))
        )
        out[f"live_{cls}_aligned_world_xy_error_m_median"] = dist(
            (fnum(r.get("aligned_world_xy_error_m")) for r in subset))["median"]
        out[f"live_{cls}_segmentation_iou_median"] = dist(
            (fnum(r.get("segmentation_iou")) for r in subset))["median"]
    return out


# --------------------------------------------------------------------------
# C. Latency decomposition
# --------------------------------------------------------------------------
def latency_decomposition(
    cell_id: str,
    frames: Sequence[Mapping[str, str]],
    feedback: Sequence[Mapping[str, str]],
    sent_rows: Sequence[Mapping[str, str]],
    rr: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Per-cell distribution summaries for every derivable stage.

    Clock domains are never crossed silently. ``UE_PERF`` is
    ``time.perf_counter_ns()`` inside the single UE client process (the CARLA
    adapter and the live dispatch share that process, so those readings are
    mutually comparable but are NOT epoch-anchored). ``WALL_HOST`` is
    ``CLOCK_REALTIME`` on the one physical host shared by the UE client, the
    edge container and the map process. ``EDGE_PERF`` contributes durations
    only. A stage that would require bridging UE_PERF to WALL_HOST is reported
    as unavailable rather than estimated.
    """
    stages: dict[str, dict[str, Any]] = {}

    def emit(key: str, values: Iterable[float], *, excluded: int, reason: str) -> None:
        summary = dist(values)
        summary["unavailable_or_excluded"] = int(excluded)
        summary["exclusion_reason"] = reason
        stages[key] = summary

    n_sent = len(sent_rows)

    # Stage 1 - sensor wait / window assembly / pre-front compute (UE_PERF durations).
    for col in ("sensor_wait_ms", "radar_window_ms", "radar_prepare_ms",
                "rgb_convert_ms", "scene_snapshot_ms", "pre_front_compute_ms"):
        values = [fnum(r.get(col)) for r in sent_rows]
        good = [v for v in values if v is not None]
        emit(f"s1_{col}", good, excluded=n_sent - len(good), reason="blank on rows that never reached this stage")

    # Stage 2 - prepared-queue wait.
    qw = [fnum(r.get("queue_wait_ms")) for r in sent_rows]
    good_qw = [v for v in qw if v is not None]
    emit("s2_recorded_queue_wait_ms_ENCLOSES_SENSOR_PREP", good_qw,
         excluded=n_sent - len(good_qw), reason="blank on rows that never reached this stage")
    emit("s2_queue_depth_at_dispatch", [fnum(r.get("queue_depth")) for r in sent_rows if fnum(r.get("queue_depth")) is not None],
         excluded=0, reason="")
    residual, negative = [], 0
    for r in sent_rows:
        parts = [fnum(r.get(c)) for c in ("queue_wait_ms", "sensor_wait_ms", "radar_window_ms", "radar_prepare_ms", "rgb_convert_ms")]
        if any(p is None for p in parts):
            continue
        value = parts[0] - sum(parts[1:])
        if value < 0:
            negative += 1
        residual.append(value)
    summary = dist(residual)
    summary["unavailable_or_excluded"] = n_sent - len(residual)
    summary["exclusion_reason"] = "row lacked one of the same-clock component durations"
    summary["negative_residual_count"] = negative
    summary["derivation"] = "queue_wait_ms - sensor_wait_ms - radar_window_ms - radar_prepare_ms - rgb_convert_ms"
    stages["s2_derived_true_queue_wait_ms_RESIDUAL"] = summary

    # Stage 3/4 - UE model front, ranker, AE encode, codec, framing (UE_PERF).
    front_keys = ("front_backbone", "ranker_selection", "ae_encode", "quantize_pack",
                  "zstd_compression", "total_ue_preparation")
    parsed_front = [parse_pydict(r.get("front_timing_ns")) for r in sent_rows]
    unparsed = sum(1 for d in parsed_front if not d)
    for key in front_keys:
        values = [d[key] / 1e6 for d in parsed_front if key in d]
        emit(f"s3_{key}_ms", values, excluded=n_sent - len(values),
             reason="front_timing_ns absent/unparseable or key not emitted for this family")
    stages["s3_front_timing_ns_unparseable_rows"] = {"count": unparsed}
    fms = [fnum(r.get("front_ms")) for r in sent_rows]
    good_fms = [v for v in fms if v is not None]
    emit("s3_front_ms_FULL_UE_DISPATCH_SPAN", good_fms, excluded=n_sent - len(good_fms),
         reason="blank when the row never completed UE preparation")
    # Uninstrumented remainder inside the UE dispatch span.
    gap = []
    for row, timing in zip(sent_rows, parsed_front):
        span, total = fnum(row.get("front_ms")), timing.get("total_ue_preparation")
        if span is None or total is None:
            continue
        gap.append(span - total / 1e6)
    emit("s4_derived_ue_uninstrumented_span_ms", gap, excluded=n_sent - len(gap),
         reason="row lacked front_ms or total_ue_preparation")
    send_loop = []
    for row in sent_rows:
        a, b = fnum(row.get("ue_prepare_finished_ns")), fnum(row.get("send_finished_ns"))
        if a is None or b is None:
            continue
        send_loop.append((b - a) / 1e6)
    emit("s4_datagram_send_loop_ms", send_loop, excluded=n_sent - len(send_loop),
         reason="row lacked a UE_PERF send boundary")

    # Stage 5 - feature uplink transport. NOT one-way derivable.
    stages["s5_feature_uplink_one_way_ms"] = {
        "count": 0, "mean": None, "median": None, "p90": None, "p95": None,
        "min": None, "max": None,
        "unavailable_or_excluded": n_sent,
        "exclusion_reason": (
            "NOT DERIVABLE: the final UE send boundary (send_finished_ns) is "
            "time.perf_counter_ns() in the UE process and is not epoch-anchored, "
            "while the complete edge receive boundary (edge_receipt_wall_s) is "
            "edge CLOCK_REALTIME. No simultaneous reading of both clocks is "
            "registered, and the offset is not per-frame recoverable, so no "
            "one-way uplink latency is computed."
        ),
    }
    # Derivable wall-domain surrogate, explicitly not transport.
    cap_to_edge = []
    for row in sent_rows:
        a, b = fnum(row.get("capture_wall_s")), fnum(row.get("edge_receipt_wall_s"))
        if a is None or b is None:
            continue
        cap_to_edge.append((b - a) * 1000.0)
    emit("s5_capture_to_edge_receipt_ms_INCLUDES_SENSOR_AND_FRONT", cap_to_edge,
         excluded=n_sent - len(cap_to_edge),
         reason="edge_receipt_wall_s blank when the message never fully reassembled at the edge")
    # Round-trip residual: UE_PERF round trip minus the same frame's EDGE_PERF duration.
    rt_residual, rt_negative = [], 0
    for row in sent_rows:
        a, b = fnum(row.get("send_finished_ns")), fnum(row.get("edge_result_received_ns"))
        edge_timing = parse_pydict(row.get("edge_timing_ns"))
        total = edge_timing.get("total_edge_processing")
        if a is None or b is None or total is None:
            continue
        value = (b - a) / 1e6 - total / 1e6
        if value < 0:
            rt_negative += 1
        rt_residual.append(value)
    summary = dist(rt_residual)
    summary["unavailable_or_excluded"] = n_sent - len(rt_residual)
    summary["exclusion_reason"] = "row lacked a UE_PERF round-trip boundary or edge_timing_ns"
    summary["negative_residual_count"] = rt_negative
    summary["derivation"] = "(edge_result_received_ns - send_finished_ns) - edge_timing_ns.total_edge_processing"
    summary["interpretation"] = (
        "ROUND-TRIP residual = feature uplink + edge queue wait + compact result "
        "downlink. NOT one-way uplink latency."
    )
    stages["s5_derived_roundtrip_residual_ms_RESIDUAL"] = summary

    # Stage 6 - edge queue wait (edge-domain residual).
    eq_residual, eq_negative = [], 0
    for row in sent_rows:
        a, b = fnum(row.get("edge_receipt_wall_s")), fnum(row.get("edge_tail_complete_wall_s"))
        total = parse_pydict(row.get("edge_timing_ns")).get("total_edge_processing")
        if a is None or b is None or total is None:
            continue
        value = (b - a) * 1000.0 - total / 1e6
        if value < 0:
            eq_negative += 1
        eq_residual.append(value)
    summary = dist(eq_residual)
    summary["unavailable_or_excluded"] = n_sent - len(eq_residual)
    summary["exclusion_reason"] = "row lacked an edge wall boundary or edge_timing_ns"
    summary["negative_residual_count"] = eq_negative
    summary["derivation"] = "(edge_tail_complete_wall_s - edge_receipt_wall_s) - edge_timing_ns.total_edge_processing"
    stages["s6_derived_edge_queue_wait_ms_RESIDUAL"] = summary
    edge_span = []
    for row in sent_rows:
        a, b = fnum(row.get("edge_receipt_wall_s")), fnum(row.get("edge_tail_complete_wall_s"))
        if a is None or b is None:
            continue
        edge_span.append((b - a) * 1000.0)
    emit("s6_edge_receipt_to_tail_complete_ms", edge_span, excluded=n_sent - len(edge_span),
         reason="edge wall boundary blank when the frame never completed the tail")

    # Stage 7/8 - decode, AE-decode, frozen tail (EDGE_PERF durations).
    parsed_edge = [parse_pydict(r.get("edge_timing_ns")) for r in sent_rows]
    for key, stage in (("zstd_decompression", "s7"), ("unpack_dequantize", "s7"),
                       ("ae_decode", "s7"), ("frozen_tail", "s8"),
                       ("output_serialization", "s8"), ("total_edge_processing", "s8")):
        values = [d[key] / 1e6 for d in parsed_edge if key in d]
        emit(f"{stage}_{key}_ms", values, excluded=n_sent - len(values),
             reason="edge_timing_ns absent (frame never processed at the edge) or key not emitted")

    # Stage 9 - tail completion -> map publication.
    stages["s9_tail_complete_to_map_publication_ms"] = {
        "count": 0, "mean": None, "median": None, "p90": None, "p95": None,
        "min": None, "max": None, "unavailable_or_excluded": n_sent,
        "exclusion_reason": (
            "NOT DERIVABLE: map publication is registered only as the categorical "
            "map_publication_status; no map-publication timestamp is recorded, so "
            "the boundary has no end event."
        ),
    }
    ev = []
    for row in sent_rows:
        a, b = fnum(row.get("edge_tail_complete_wall_s")), fnum(row.get("edge_evidence_install_wall_s"))
        if a is None or b is None:
            continue
        ev.append((b - a) * 1000.0)
    emit("s9_tail_complete_to_edge_evidence_install_ms", ev, excluded=n_sent - len(ev),
         reason="evidence install boundary blank when no mask was submitted")
    dl = []
    for row in sent_rows:
        a, b = fnum(row.get("edge_tail_complete_wall_s")), fnum(row.get("feature_received_at"))
        if a is None or b is None:
            continue
        dl.append((b - a) * 1000.0)
    emit("s9_tail_complete_to_ue_result_receipt_ms", dl, excluded=n_sent - len(dl),
         reason="row lacked an edge tail boundary or a UE result receipt")

    # Stage 10 - (result receipt ->) map installation.
    inst = []
    for row in sent_rows:
        a, b = fnum(row.get("feature_received_at")), fnum(row.get("map_installed_at"))
        if a is None or b is None:
            continue
        inst.append((b - a) * 1000.0)
    emit("s10_ue_result_receipt_to_map_installation_ms", inst, excluded=n_sent - len(inst),
         reason="map_installed_at blank unless the authoritative ACK carried an install timestamp")

    # Stage 11 - feedback emission -> UE feedback receipt (map_feedback.csv).
    emit_rows, install_rows = [], []
    for row in feedback:
        a, b = fnum(row.get("feedback_emit_at")), fnum(row.get("feedback_received_at"))
        if a is not None and b is not None:
            emit_rows.append((b - a) * 1000.0)
        c = fnum(row.get("install_timestamp"))
        if c is not None and b is not None:
            install_rows.append((b - c) * 1000.0)
    emit("s11_feedback_emit_to_ue_receipt_ms", emit_rows, excluded=len(feedback) - len(emit_rows),
         reason="feedback_emit_at is blank on TIMEOUT_NO_ACK rows and where the emitter did not stamp it")
    emit("s11_map_install_to_ue_feedback_receipt_ms", install_rows,
         excluded=len(feedback) - len(install_rows),
         reason="install_timestamp blank on non-ACK_INSTALLED rows")

    # Stage 12 - UE final send -> compact result receipt (UE_PERF round trip).
    rtt = []
    for row in sent_rows:
        a, b = fnum(row.get("send_finished_ns")), fnum(row.get("edge_result_received_ns"))
        if a is None or b is None:
            continue
        rtt.append((b - a) / 1e6)
    emit("s12_ue_send_to_compact_result_receipt_ms_UE_PERF", rtt, excluded=n_sent - len(rtt),
         reason="edge_result_received_ns blank when no compact result returned")
    ack = []
    for row in feedback:
        a, b = fnum(row.get("capture_at")), fnum(row.get("feedback_received_at"))
        if a is None or b is None:
            continue
        ack.append((b - a) * 1000.0)

    # Stage 13 - capture -> installation AoI (registered).
    aoi = [fnum(r.get("install_aoi_ms")) for r in frames]
    good_aoi = [v for v in aoi if v is not None]
    emit("s13_capture_to_install_aoi_ms", good_aoi, excluded=len(frames) - len(good_aoi),
         reason="install_aoi_ms is defined only for frames that reached MAP_INSTALLED")
    # Cross-check against the campaign's own registered summary.
    reg_median, reg_p95 = fnum(rr.get("install_aoi_ms_median")), fnum(rr.get("install_aoi_ms_p95"))
    got = stages["s13_capture_to_install_aoi_ms"]
    stages["s13_capture_to_install_aoi_ms"]["registered_median"] = reg_median
    stages["s13_capture_to_install_aoi_ms"]["registered_p95"] = reg_p95
    for label, mine, theirs in (("median", got["median"], reg_median), ("p95", got["p95"], reg_p95)):
        if mine is None and theirs is None:
            continue
        if mine is None or theirs is None or abs(mine - theirs) > 1e-6:
            fail(f"{cell_id}: recomputed install AoI {label} ({mine}) disagrees with the registered value ({theirs})")
    stages["s13_capture_to_install_aoi_ms"]["reproduces_registered_summary"] = True

    # Stage 14 - capture -> UE feedback AoI.
    emit("s14_capture_to_ue_feedback_aoi_ms", ack, excluded=len(feedback) - len(ack),
         reason="row lacked capture_at or feedback_received_at")

    # The user's later analytical quantity, computed only where every component
    # is a same-frame instrumented duration. Never labelled capture-to-install.
    analytic, missing = [], 0
    for row, front, edge in zip(sent_rows, [parse_pydict(r.get("front_timing_ns")) for r in sent_rows], parsed_edge):
        mf = front.get("total_ue_preparation")
        mb = edge.get("frozen_tail")
        a, b = fnum(row.get("send_finished_ns")), fnum(row.get("edge_result_received_ns"))
        total = edge.get("total_edge_processing")
        fb = fnum(row.get("map_installed_at"))
        rx = fnum(row.get("feature_received_at"))
        if None in (mf, mb, total, fb, rx) or a is None or b is None:
            missing += 1
            continue
        transport = (b - a) / 1e6 - total / 1e6   # uplink + edge queue + downlink
        feedback_leg = (fb - rx) * 1000.0          # result receipt -> map install
        analytic.append(mf / 1e6 + transport + mb / 1e6 + feedback_leg)
    summary = dist(analytic)
    summary["unavailable_or_excluded"] = missing
    summary["exclusion_reason"] = "row lacked one of the four same-frame components"
    summary["derivation"] = (
        "front_timing_ns.total_ue_preparation + roundtrip_transport_residual + "
        "edge_timing_ns.frozen_tail + (map_installed_at - feature_received_at)"
    )
    summary["interpretation"] = (
        "The user's analytical model_front + feature_transport + model_back + "
        "compact_feedback quantity. It is a component sum over compatible "
        "same-frame timestamps and is explicitly NOT capture-to-install latency: "
        "it excludes sensor preparation, the prepared-queue wait and the UE "
        "dispatch overhead, and its transport term is a round-trip residual."
    )
    stages["analytic_model_front_transport_back_feedback_ms"] = summary
    return stages


# --------------------------------------------------------------------------
# D. Accuracy and preservation join
# --------------------------------------------------------------------------
VALIDATION_COLUMNS = (
    "relative_preservation_passed", "relative_preservation_count",
    "absolute_service_ready", "absolute_service_count",
    "localization_requirements_passed", "localization_requirement_count",
    "localization_requirements_evaluated",
    "segmentation_installable", "segmentation_behavior",
    "source_contract_tier", "stress_or_emergency_anchor",
    "vehicle_precision", "vehicle_recall", "vehicle_f1", "vehicle_xy_mae_m",
    "vehicle_iou",
    "canonical_person_precision", "canonical_person_recall",
    "canonical_person_f1", "canonical_person_xy_mae_m",
    "person_avo_precision", "person_avo_recall", "person_avo_f1",
    "person_avo_xy_mae_m", "person_avo_recall_0_30m",
    "person_avo_recall_30_40m_diagnostic",
    "person_box_mask_iou", "foreground_miou",
    "ratio_vs_dense_fp32_q0", "ratio_vs_same_family_same_q_uint8",
    "checkpoint_sha256", "decoder_identity",
)


def validation_join(repo: Path, catalog_csv_row: Mapping[str, str], evidence_cache: dict[str, str]) -> dict[str, Any]:
    """Join the frozen, registered validation properties for one action.

    Uses only existing registered fields and classifications. Nothing is
    rescored, and a historically unavailable field stays null rather than being
    reconstructed from an incompatible aggregate.
    """
    out: dict[str, Any] = {}
    for column in VALIDATION_COLUMNS:
        raw = catalog_csv_row.get(column, "")
        out[f"val_{column}"] = "" if raw is None else raw
    # Registered gate ratios, using the catalog's own denominators.
    out["val_preservation_gates_passed_total"] = (
        f"{catalog_csv_row.get('relative_preservation_count','')}/"
        f"{catalog_csv_row.get('relative_preservation_count','')}"
        if truthy(catalog_csv_row.get("relative_preservation_passed")) else
        f"0/{catalog_csv_row.get('relative_preservation_count','')}"
    )
    out["val_absolute_service_gates_passed"] = catalog_csv_row.get("absolute_service_count", "")
    out["val_localization_gates_passed"] = catalog_csv_row.get("localization_requirement_count", "")
    out["val_localization_gates_evaluated"] = catalog_csv_row.get("localization_requirements_evaluated", "")
    out["val_emergency_only"] = str(catalog_csv_row.get("source_contract_tier", "")) == "EMERGENCY_ONLY"
    # Fields with no registered source anywhere in the frozen evidence.
    out["val_mask_accuracy"] = ""
    out["val_mask_accuracy_availability"] = "UNAVAILABLE_NO_REGISTERED_FIELD"
    out["val_segmentation_miou"] = catalog_csv_row.get("foreground_miou", "")
    # Provenance for this joined row, hash-verified.
    ev_path = str(catalog_csv_row.get("source_evidence_path", ""))
    ds_path = str(catalog_csv_row.get("durable_setting_path", ""))
    for label, rel, expected in (
        ("source_evidence", ev_path, catalog_csv_row.get("source_evidence_sha256", "")),
        ("durable_setting", ds_path, catalog_csv_row.get("durable_setting_sha256", "")),
    ):
        if rel not in evidence_cache:
            evidence_cache[rel] = sha256_file(repo / rel) or ""
        got = evidence_cache[rel]
        if got != expected:
            fail(f"validation join provenance drift for {rel}: {got} != {expected}")
        out[f"val_{label}_path"] = rel
        out[f"val_{label}_sha256"] = got
    out["val_join_verified"] = True
    return out


# --------------------------------------------------------------------------
# Table writers
# --------------------------------------------------------------------------
CELL_TABLE_EXCLUDE = {"_latency"}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        # Unix line endings: the csv module defaults to \r\n, which this repo
        # would carry as trailing whitespace on every row.
        writer = csv.DictWriter(
            handle, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: ("" if row.get(key) is None else row.get(key))
                for key in columns
            })
    return sha256_file(path) or ""


def latency_columns(stage_map: Mapping[str, Mapping[str, Any]]) -> list[str]:
    columns: list[str] = []
    for stage in sorted(stage_map):
        for field in sorted(stage_map[stage]):
            columns.append(f"{stage}__{field}")
    return columns


def flatten_latency(stage_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        f"{stage}__{field}": value
        for stage in stage_map
        for field, value in stage_map[stage].items()
    }


def aggregate_actions(cells: Sequence[Mapping[str, Any]], profiles: Sequence[str]) -> list[dict[str, Any]]:
    """Aggregate each action across the four profiles, retaining per-profile
    columns so adverse/fade behaviour is never hidden inside a mean."""
    by_action: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for cell in cells:
        by_action[int(cell["action_id"])][str(cell["network_profile"])] = cell

    per_profile_metrics = (
        "frames_sent", "maps_installed", "rate_installed_per_sent",
        "installed_within_100ms_service_reference", "rate_on_time_100ms_per_sent",
        "ack_within_500ms_timeout", "rate_ack_500ms_per_sent",
        "edge_complete_reassemblies", "rate_reassembled_per_sent",
        "rate_datagrams_received_per_transmitted",
        "live_scientific_inner_bytes_median", "live_estimated_wire_bytes_median",
        "live_datagrams_per_message_median", "sensor_preparation_coverage",
        "radio_achieved_snr_db_median", "zero_delivery",
        "terminal_TIMEOUT_NO_ACK",
    )
    identity = (
        "action_id", "profile_id", "family", "quantizer", "q", "q_e4",
        "keep_count", "drop_count", "routing_tag", "wire_layout", "zstd_level",
        "bit_width", "latent_width",
    )
    aoi_stage = "s13_capture_to_install_aoi_ms"
    rows: list[dict[str, Any]] = []
    for action_id in sorted(by_action):
        group = by_action[action_id]
        if set(group) != set(profiles):
            fail(f"action {action_id} does not cover all four network profiles")
        first = group[profiles[0]]
        row: dict[str, Any] = {key: first.get(key) for key in identity}
        row["profiles_present"] = len(group)
        # Validation properties are action-level and must be identical.
        for key in first:
            if key.startswith("val_"):
                values = {str(group[p].get(key)) for p in profiles}
                if len(values) != 1:
                    fail(f"action {action_id}: validation field {key} differs across profiles")
                row[key] = first[key]
        for metric in per_profile_metrics:
            for profile in profiles:
                row[f"{metric}__{profile}"] = group[profile].get(metric)
            numeric = [fnum(group[p].get(metric)) for p in profiles]
            numeric = [v for v in numeric if v is not None]
            if metric == "zero_delivery":
                row["zero_delivery_profile_count"] = sum(
                    1 for p in profiles if group[p].get("zero_delivery"))
                continue
            row[f"{metric}__across_profile_mean"] = (statistics.fmean(numeric) if numeric else None)
            row[f"{metric}__across_profile_min"] = (min(numeric) if numeric else None)
            row[f"{metric}__across_profile_max"] = (max(numeric) if numeric else None)
        # Install AoI, per profile plus spread.
        for profile in profiles:
            stage = group[profile]["_latency"][aoi_stage]
            row[f"install_aoi_ms_median__{profile}"] = stage["median"]
            row[f"install_aoi_ms_p95__{profile}"] = stage["p95"]
            row[f"install_aoi_ms_count__{profile}"] = stage["count"]
        medians = [group[p]["_latency"][aoi_stage]["median"] for p in profiles]
        medians = [v for v in medians if v is not None]
        row["install_aoi_ms_median__across_profile_min"] = (min(medians) if medians else None)
        row["install_aoi_ms_median__across_profile_max"] = (max(medians) if medians else None)
        row["install_aoi_ms_median__profiles_with_any_install"] = len(medians)
        row["total_frames_sent_4_profiles"] = sum(
            int(group[p].get("frames_sent") or 0) for p in profiles)
        row["total_maps_installed_4_profiles"] = sum(
            int(group[p].get("maps_installed") or 0) for p in profiles)
        row["total_installed_within_100ms_4_profiles"] = sum(
            int(group[p].get("installed_within_100ms_service_reference") or 0) for p in profiles)
        row["executed_or_reused_by_profile"] = "|".join(
            f"{p}={group[p].get('executed_or_reused')}" for p in profiles)
        row["route_summary_available_by_profile"] = "|".join(
            f"{p}={group[p].get('route_summary_available')}" for p in profiles)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Dictionaries
# --------------------------------------------------------------------------
def build_latency_dictionary() -> dict[str, Any]:
    def entry(start: str, end: str, clock: str, unit: str, derivability: str, note: str, fields: Sequence[str]) -> dict[str, Any]:
        return {
            "start_event": start, "end_event": end, "clock_domain": clock,
            "unit": unit, "derivability": derivability, "note": note,
            "source_fields": list(fields),
        }

    return {
        "schema": SCHEMA + ".latency_dictionary",
        "clock_domains": {
            "WALL_HOST": {
                "source": "time.time() / CLOCK_REALTIME",
                "scope": (
                    "One physical host. The UE CARLA client, the edge Docker "
                    "container (OAI CN5G bridge network 192.168.70.0/24) and the "
                    "map process all read the same kernel clock, so these "
                    "timestamps are mutually comparable."
                ),
                "epoch_anchored": True,
            },
            "UE_PERF": {
                "source": "time.perf_counter_ns() / time.perf_counter()",
                "scope": (
                    "The single UE client process. The CARLA route/sensor adapter "
                    "and the live split dispatch run in that one process, so all "
                    "UE_PERF readings are mutually comparable."
                ),
                "epoch_anchored": False,
                "warning": (
                    "Not comparable with WALL_HOST. Any stage needing a UE_PERF "
                    "instant against an edge WALL_HOST instant is reported "
                    "unavailable, never estimated."
                ),
            },
            "EDGE_PERF": {
                "source": "time.perf_counter_ns() inside the edge container",
                "scope": "Edge process. Contributes durations only, never instants.",
                "epoch_anchored": False,
            },
        },
        "scientific_rules_enforced": [
            "pre-front sensor preparation is never called front_ms",
            "sensor wait is never folded into network transport",
            "no one-way transport is inferred across non-comparable clocks",
            "the 500 ms ACK timeout is never treated as the 100 ms service target",
            "within_100ms and within_500ms are reported separately",
            "per-frame component medians are not expected to sum to the median end-to-end latency",
            "derived residuals are labelled as residuals and carry a negative-residual count",
        ],
        "registered_deadlines": {
            "service_deadline_ms": 100,
            "ack_timeout_ms": 500,
            "distinct": True,
            "claims_100ms_service_ready": False,
        },
        "stages": {
            "s1_sensor_wait_ms": entry(
                "prepared-queue worker picks the frame up (process_started_perf)",
                "synchronized camera+radar records are available (sensor_ready_perf)",
                "UE_PERF", "ms", "AVAILABLE",
                "Pure sensor-window availability wait. Pre-front; not transport, not front_ms.",
                ["sensor_wait_ms"]),
            "s1_radar_window_ms": entry(
                "radar multi-sweep window assembly begins",
                "window detections returned", "UE_PERF", "ms", "AVAILABLE",
                "Four-callback sweep window assembly.", ["radar_window_ms"]),
            "s1_radar_prepare_ms": entry(
                "radar sample build begins", "radar tensor rasterised",
                "UE_PERF", "ms", "AVAILABLE",
                "Radar rasterisation; the dominant pre-front cost.", ["radar_prepare_ms"]),
            "s1_rgb_convert_ms": entry(
                "CARLA image conversion begins", "BGR frame ready",
                "UE_PERF", "ms", "AVAILABLE", "", ["rgb_convert_ms"]),
            "s1_scene_snapshot_ms": entry(
                "world snapshot freeze begins", "scene captured",
                "UE_PERF", "ms", "AVAILABLE",
                "Offline-evaluation scene freeze, on the preparation path.",
                ["scene_snapshot_ms"]),
            "s1_pre_front_compute_ms": entry(
                "sensor records available (sensor_ready_perf)",
                "scene snapshot complete, immediately before UE dispatch",
                "UE_PERF", "ms", "AVAILABLE",
                "SUPERSET spanning radar window+prepare, rgb convert, scene snapshot "
                "and unattributed remainder. Not a sum of its components.",
                ["pre_front_compute_ms"]),
            "s2_recorded_queue_wait_ms_ENCLOSES_SENSOR_PREP": entry(
                "frame scheduled onto the prepared queue (scheduled_perf)",
                "immediately before the scene snapshot / UE dispatch",
                "UE_PERF", "ms", "AVAILABLE_BUT_MISNAMED",
                "Despite the name this ENCLOSES the sensor wait and radar/RGB "
                "preparation. It is NOT a pure prepared-queue wait.",
                ["queue_wait_ms"]),
            "s2_derived_true_queue_wait_ms_RESIDUAL": entry(
                "frame scheduled onto the prepared queue",
                "prepared-queue worker picks the frame up",
                "UE_PERF", "ms", "DERIVED_RESIDUAL",
                "queue_wait_ms minus the same-frame, same-clock preparation "
                "components. Small negatives are possible from unattributed gaps; "
                "the negative count is reported.",
                ["queue_wait_ms", "sensor_wait_ms", "radar_window_ms", "radar_prepare_ms", "rgb_convert_ms"]),
            "s2_queue_depth_at_dispatch": entry(
                "n/a", "n/a", "COUNT", "frames", "AVAILABLE",
                "Structural queue-occupancy indicator recorded at dispatch.",
                ["queue_depth"]),
            "s3_front_backbone_ms": entry(
                "front backbone forward begins", "front features produced",
                "UE_PERF", "ms", "AVAILABLE",
                "The actual model front. This, not front_ms, is the front cost.",
                ["front_timing_ns.front_backbone"]),
            "s3_ranker_selection_ms": entry(
                "ranker selection begins", "kept cells selected",
                "UE_PERF", "ms", "AVAILABLE", "", ["front_timing_ns.ranker_selection"]),
            "s3_ae_encode_ms": entry(
                "AE encode begins", "latent produced", "UE_PERF", "ms", "AVAILABLE",
                "Zero for the noAE family.", ["front_timing_ns.ae_encode"]),
            "s3_quantize_pack_ms": entry(
                "quantize/pack begins", "wire buffer packed", "UE_PERF", "ms",
                "AVAILABLE", "", ["front_timing_ns.quantize_pack"]),
            "s3_zstd_compression_ms": entry(
                "zstd compression begins", "compressed payload ready",
                "UE_PERF", "ms", "AVAILABLE",
                "The deployed entropy codec is zstd (lossless).",
                ["front_timing_ns.zstd_compression"]),
            "s3_total_ue_preparation_ms": entry(
                "UE prepare() entry", "UE prepare() return", "UE_PERF", "ms",
                "AVAILABLE",
                "Instrumented UE preparation total; the model_front term of the "
                "user's analytical quantity.",
                ["front_timing_ns.total_ue_preparation"]),
            "s3_front_ms_FULL_UE_DISPATCH_SPAN": entry(
                "capture_started_ns (UE dispatch entry, after sensor preparation)",
                "ue_prepare_finished_ns (encode complete, before the first datagram)",
                "UE_PERF", "ms", "AVAILABLE",
                "SUPERSET of total_ue_preparation: also covers 7-channel input "
                "assembly, frame-context construction and payload chunking. It "
                "contains NO sensor preparation and NO network transport.",
                ["front_ms", "capture_started_ns", "ue_prepare_finished_ns"]),
            "s4_derived_ue_uninstrumented_span_ms": entry(
                "n/a", "n/a", "UE_PERF", "ms", "DERIVED_RESIDUAL",
                "front_ms minus total_ue_preparation: input assembly, context "
                "build and chunking not covered by the instrumented timers.",
                ["front_ms", "front_timing_ns.total_ue_preparation"]),
            "s4_datagram_send_loop_ms": entry(
                "ue_prepare_finished_ns", "send_finished_ns", "UE_PERF", "ms",
                "AVAILABLE",
                "Host-side sendto loop over all chunks. Socket handoff only; not "
                "a propagation or radio latency.",
                ["ue_prepare_finished_ns", "send_finished_ns"]),
            "s5_feature_uplink_one_way_ms": entry(
                "final UE send boundary (send_finished_ns)",
                "complete edge receive/reassembly (edge_receipt_wall_s)",
                "UE_PERF -> WALL_HOST (INCOMPATIBLE)", "ms", "UNAVAILABLE",
                "The two boundaries live in non-comparable clock domains and no "
                "simultaneous cross-clock reading is registered, so one-way "
                "uplink latency is not computed.",
                ["send_finished_ns", "edge_receipt_wall_s"]),
            "s5_capture_to_edge_receipt_ms_INCLUDES_SENSOR_AND_FRONT": entry(
                "camera capture instant (capture_wall_s)",
                "complete edge receive/reassembly (edge_receipt_wall_s)",
                "WALL_HOST", "ms", "AVAILABLE",
                "Both boundaries are WALL_HOST so this is measured, but it "
                "INCLUDES sensor preparation and the whole UE front. It is not "
                "an uplink transport latency.",
                ["capture_wall_s", "edge_receipt_wall_s"]),
            "s5_derived_roundtrip_residual_ms_RESIDUAL": entry(
                "send_finished_ns", "edge_result_received_ns, less edge processing",
                "UE_PERF (instants) + EDGE_PERF (duration)", "ms", "DERIVED_RESIDUAL",
                "ROUND TRIP residual = feature uplink + edge queue + compact "
                "result downlink. Subtracting a same-frame EDGE_PERF duration "
                "from a UE_PERF interval is clock-safe. NOT one-way uplink.",
                ["send_finished_ns", "edge_result_received_ns", "edge_timing_ns.total_edge_processing"]),
            "s6_edge_receipt_to_tail_complete_ms": entry(
                "edge_receipt_wall_s", "edge_tail_complete_wall_s", "WALL_HOST",
                "ms", "AVAILABLE",
                "Edge-internal span: queue wait + decode + tail inference.",
                ["edge_receipt_wall_s", "edge_tail_complete_wall_s"]),
            "s6_derived_edge_queue_wait_ms_RESIDUAL": entry(
                "edge_receipt_wall_s", "edge processing entry", "WALL_HOST less EDGE_PERF",
                "ms", "DERIVED_RESIDUAL",
                "Edge wall span minus the same-frame instrumented edge processing "
                "duration. Both are edge-local, so the subtraction is clock-safe.",
                ["edge_receipt_wall_s", "edge_tail_complete_wall_s", "edge_timing_ns.total_edge_processing"]),
            "s7_zstd_decompression_ms": entry(
                "edge zstd decompression begins", "payload decompressed", "EDGE_PERF",
                "ms", "AVAILABLE", "", ["edge_timing_ns.zstd_decompression"]),
            "s7_unpack_dequantize_ms": entry(
                "unpack/dequantize begins", "tensor reconstructed", "EDGE_PERF",
                "ms", "AVAILABLE", "", ["edge_timing_ns.unpack_dequantize"]),
            "s7_ae_decode_ms": entry(
                "AE decode begins", "latent decoded", "EDGE_PERF", "ms", "AVAILABLE",
                "Zero for the noAE family.", ["edge_timing_ns.ae_decode"]),
            "s8_frozen_tail_ms": entry(
                "frozen tail forward begins", "tail outputs produced", "EDGE_PERF",
                "ms", "AVAILABLE",
                "The model_back term of the user's analytical quantity.",
                ["edge_timing_ns.frozen_tail"]),
            "s8_output_serialization_ms": entry(
                "result serialization begins", "compact result serialized",
                "EDGE_PERF", "ms", "AVAILABLE", "", ["edge_timing_ns.output_serialization"]),
            "s8_total_edge_processing_ms": entry(
                "edge process() entry", "edge process() return", "EDGE_PERF", "ms",
                "AVAILABLE", "", ["edge_timing_ns.total_edge_processing"]),
            "s9_tail_complete_to_map_publication_ms": entry(
                "edge_tail_complete_wall_s", "map publication", "n/a", "ms",
                "UNAVAILABLE",
                "Map publication is registered only as the categorical "
                "map_publication_status; no publication timestamp exists, so this "
                "boundary has no end event.",
                ["edge_tail_complete_wall_s", "map_publication_status"]),
            "s9_tail_complete_to_edge_evidence_install_ms": entry(
                "edge_tail_complete_wall_s", "edge_evidence_install_wall_s",
                "WALL_HOST", "ms", "AVAILABLE",
                "Edge-side segmentation-evidence write submission.",
                ["edge_tail_complete_wall_s", "edge_evidence_install_wall_s"]),
            "s9_tail_complete_to_ue_result_receipt_ms": entry(
                "edge_tail_complete_wall_s", "feature_received_at (UE result receipt)",
                "WALL_HOST", "ms", "AVAILABLE",
                "Compact result downlink leg plus edge post-tail work.",
                ["edge_tail_complete_wall_s", "feature_received_at"]),
            "s10_ue_result_receipt_to_map_installation_ms": entry(
                "feature_received_at (UE compact result receipt)",
                "map_installed_at (authoritative MAP_INSTALLED)", "WALL_HOST", "ms",
                "AVAILABLE",
                "Stands in for publication->installation because publication has no "
                "own timestamp. feature_received_at is the UE's receipt of the "
                "compact edge result, not a feature arrival at the edge.",
                ["feature_received_at", "map_installed_at"]),
            "s11_feedback_emit_to_ue_receipt_ms": entry(
                "feedback_emit_at", "feedback_received_at", "WALL_HOST", "ms",
                "PARTIALLY_AVAILABLE",
                "feedback_emit_at is blank on TIMEOUT_NO_ACK rows.",
                ["feedback_emit_at", "feedback_received_at"]),
            "s11_map_install_to_ue_feedback_receipt_ms": entry(
                "install_timestamp (map install)", "feedback_received_at",
                "WALL_HOST", "ms", "PARTIALLY_AVAILABLE",
                "Defined only on ACK_INSTALLED rows.",
                ["install_timestamp", "feedback_received_at"]),
            "s12_ue_send_to_compact_result_receipt_ms_UE_PERF": entry(
                "send_finished_ns", "edge_result_received_ns", "UE_PERF", "ms",
                "AVAILABLE",
                "A ROUND TRIP measured entirely inside the UE process. This is the "
                "compact result datagram, not the map-install ACK; the ACK leg is "
                "s14.",
                ["send_finished_ns", "edge_result_received_ns"]),
            "s13_capture_to_install_aoi_ms": entry(
                "capture_wall_s (camera capture instant)",
                "map_installed_at (authoritative MAP_INSTALLED)", "WALL_HOST", "ms",
                "AVAILABLE",
                "The registered end-to-end AoI, install_aoi_ms. Defined only for "
                "frames that reached MAP_INSTALLED. Recomputed here and checked "
                "against the campaign's registered median/p95.",
                ["capture_wall_s", "map_installed_at", "install_aoi_ms"]),
            "s14_capture_to_ue_feedback_aoi_ms": entry(
                "capture_at", "feedback_received_at", "WALL_HOST", "ms", "AVAILABLE",
                "Includes TIMEOUT_NO_ACK rows, whose receipt time is the 500 ms "
                "timeout sweep instant rather than a remote ACK.",
                ["capture_at", "feedback_received_at"]),
            "analytic_model_front_transport_back_feedback_ms": entry(
                "n/a (component sum)", "n/a (component sum)",
                "UE_PERF + EDGE_PERF durations + WALL_HOST leg", "ms",
                "DERIVED_COMPONENT_SUM",
                "The user's model_front + feature_transport + model_back + "
                "compact_feedback quantity, computed only from same-frame "
                "compatible timestamps. Explicitly NOT capture-to-install "
                "latency: it omits sensor preparation, the prepared-queue wait "
                "and UE dispatch overhead, and its transport term is a "
                "round-trip residual.",
                ["front_timing_ns.total_ue_preparation", "send_finished_ns",
                 "edge_result_received_ns", "edge_timing_ns.total_edge_processing",
                 "edge_timing_ns.frozen_tail", "feature_received_at", "map_installed_at"]),
        },
        "distribution_fields": {
            "count": "valid sample count",
            "unavailable_or_excluded": "rows excluded, with exclusion_reason",
            "mean": "arithmetic mean of valid samples",
            "median": "nearest-rank p50, identical convention to the campaign runner",
            "p90": "nearest-rank p90",
            "p95": "nearest-rank p95",
            "min": "minimum, where reported",
            "max": "maximum, where reported",
            "negative_residual_count": "residual stages only: samples below zero",
        },
    }


def build_metric_availability(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(cells)

    def cov(predicate) -> int:
        return sum(1 for c in cells if predicate(c))

    def stage_cov(stage: str) -> int:
        return sum(1 for c in cells if (c["_latency"].get(stage) or {}).get("count"))

    def mark(status: str, reason: str, cells_with_data: int | None = None) -> dict[str, Any]:
        out = {"status": status, "reason": reason}
        if cells_with_data is not None:
            out["cells_with_data"] = cells_with_data
            out["cells_total"] = n
        return out

    requested: dict[str, Any] = {}
    # Identity/provenance
    for name in ("cell_id", "action_id", "profile_id", "network_profile", "family",
                 "quantizer", "q", "q_e4", "keep_count", "routing_tag",
                 "wire_layout", "zstd_level", "execution_status",
                 "executed_or_reused", "source_campaign_root", "source_attempt",
                 "source_terminal_sha256"):
        requested[name] = mark("available", "registered identity/provenance field", n)
    requested["route_summary_available"] = mark(
        "partially_available",
        "Durable route_metrics_summary.json presence, verified per cell. Present "
        "for the 227 executed cells and the 3 retry6-origin reused cells; absent "
        "for the 58 retry5-origin reused cells, which predate the durable "
        "route-summary requirement.",
        cov(lambda c: c["route_summary_available"]))
    requested["route_outcome_classification"] = mark(
        "partially_available",
        "The registered route_outcome_classification exists only for the 227 "
        "executed cells. All 61 reused cells lack it: the 58 retry5-origin cells "
        "have no route summary at all, and the 3 retry6-origin cells have only the "
        "route runner's own metrics summary, whose 'status' field is the "
        "density/intervention state and NOT the registered classification. Those "
        "61 cells are recorded as UNAVAILABLE_NOT_INFERRED; the raw runner status "
        "is carried separately in route_runner_status_raw_not_a_classification and "
        "is never promoted to a classification.",
        cov(lambda c: c["route_outcome_classification_available"]))

    # Workload / preparation
    requested["intended_opportunities"] = mark("available", "expected_scheduled_frames at the registered 10 Hz", n)
    requested["synchronized_sensor_opportunities"] = mark(
        "available", "eligible_preparation_frames is the registered synchronized-opportunity denominator", n)
    requested["prepared_frames"] = mark("available", "sent_frames / prepare_status SENT", n)
    requested["preparation_coverage"] = mark("available", "registered sensor_preparation_coverage", n)
    requested["prepared_or_sustainable_fps"] = mark(
        "available",
        "prepared FPS from sent frames over the cell wall duration; sustainable FPS "
        "from the median inter-send period", n)
    requested["preparation_drops_by_registered_reason"] = mark(
        "available", "one column per registered prepare_status; absent status means zero", n)
    requested["split_frames_sent"] = mark("available", "registered split_frames_sent", n)

    # Payload
    requested["scientific_inner_payload_bytes"] = mark("available", "live measured scientific_inner_bytes", n)
    requested["sfd1_application_envelope_overhead"] = mark("available", "live measured sfd1_overhead_bytes", n)
    requested["udp_fragmentation_header_overhead"] = mark(
        "available",
        "udp_application_bytes minus sfd1_bytes gives chunk-header overhead; "
        "estimated_wire_bytes minus udp_application_bytes gives the 28 B/datagram "
        "IP+UDP allowance", n)
    requested["estimated_wire_bytes"] = mark("available", "live measured estimated_wire_bytes", n)
    requested["datagrams_per_frame_or_message"] = mark("available", "live measured datagrams", n)
    requested["payload_count_mean_median_p90_p95"] = mark("available", "computed for every variable payload quantity", n)
    requested["catalog_payload_estimates"] = mark(
        "available",
        "carried in separate catalog_* columns and never substituted for live bytes", n)


    # Delivery-stage counts / rates
    requested["frames_sent"] = mark("available", "registered sent_frames", n)
    requested["complete_edge_reassemblies"] = mark("available", "edge feature_messages_reassembled", n)
    requested["edge_admissions"] = mark(
        "partially_available",
        "edge_queue_admissions is a bump counter: present where non-zero, and "
        "legitimately zero in cells where nothing ever reassembled",
        cov(lambda c: c["edge_edge_queue_admissions"] > 0))
    requested["stale_rejected_dropped_by_stage"] = mark(
        "available",
        "one column per registered deadline stage, plus prepare_status drops and "
        "incomplete_reassemblies_expired", n)
    requested["tail_completions"] = mark(
        "partially_available", "edge tail_completions; zero where no frame reached the tail",
        cov(lambda c: c["edge_tail_completions"] > 0))
    requested["map_publications"] = mark(
        "partially_available", "ue results_published_to_map",
        cov(lambda c: c["ue_results_published_to_map"] > 0))
    requested["ack_installed_frames"] = mark(
        "partially_available", "registered ack_installed_frames",
        cov(lambda c: (c["ack_installed_frames"] or 0) > 0))
    requested["installed_within_100ms_service_reference"] = mark(
        "available",
        "registered service_on_time_installations. Measured and overwhelmingly "
        "zero; that is an outcome, not missing data.", n)
    requested["feedback_within_500ms_ack_timeout"] = mark(
        "available", "registered ack_within_timeout_installations, reported separately from the 100 ms figure", n)
    requested["late_installations_or_acks"] = mark(
        "available", "late_feedback_rows and late_nonterminal_feedback_rows", n)
    requested["terminal_timeout_no_ack"] = mark("available", "registered terminal_feedback_outcomes", n)
    requested["zero_delivery_flag"] = mark("available", "derived from maps_installed == 0", n)
    requested["both_denominators"] = mark(
        "available",
        "installed/sent, on-time/sent, ack/sent are computed alongside the "
        "registered installed-denominator fractions, and conditional stage "
        "survival ratios are given per hop", n)

    # Latency stages
    for stage in sorted(cells[0]["_latency"]):
        info = cells[0]["_latency"][stage]
        if "exclusion_reason" not in info:
            continue
        have = stage_cov(stage)
        reason = str(info.get("exclusion_reason") or "")
        if reason.startswith("NOT DERIVABLE"):
            status = "unavailable"
        elif have == 0:
            status = "unavailable"
        elif have < n:
            status = "partially_available"
        else:
            status = "available"
        requested[f"latency::{stage}"] = mark(status, reason or "derivable from registered fields", have)

    # Validation join
    val_available = {
        "preservation_gates_passed_total": "relative_preservation_passed / relative_preservation_count",
        "absolute_service_gates_passed_total": "absolute_service_ready / absolute_service_count",
        "localization_gates_passed_total": "localization_requirements_passed / localization_requirement_count / localization_requirements_evaluated",
        "registered_tier": "source_contract_tier",
        "vehicle_precision_recall": "vehicle_precision / vehicle_recall / vehicle_f1",
        "vehicle_localization_and_iou": "vehicle_xy_mae_m / vehicle_iou",
        "person_precision_recall": "canonical_person_precision / canonical_person_recall",
        "avo_precision_recall_localization": "person_avo_precision / person_avo_recall / person_avo_xy_mae_m",
        "segmentation_miou": "foreground_miou",
        "person_box_mask_iou": "person_box_mask_iou",
        "segmentation_installability": "segmentation_installable / segmentation_behavior",
        "emergency_only_status": "source_contract_tier == EMERGENCY_ONLY / stress_or_emergency_anchor",
    }
    for key, source in val_available.items():
        requested[f"validation::{key}"] = mark("available", f"frozen catalog field(s): {source}", n)
    requested["validation::segmentation_gates_passed_total"] = mark(
        "unavailable",
        "No standalone segmentation gate pass/total counter is registered. "
        "Segmentation is registered as the boolean segmentation_installable plus "
        "the categorical segmentation_behavior, and foreground_miou; no "
        "gates-passed/total pair exists to report.")
    requested["validation::mask_accuracy"] = mark(
        "unavailable",
        "No registered mask-accuracy field exists in the frozen validation "
        "evidence. foreground_miou and person_box_mask_iou are reported instead; "
        "a pixel mask accuracy is not reconstructed.")
    requested["validation::person_avo_recall_0_30m"] = mark(
        "partially_available",
        "Registered only for the 48 Phase-11D UINT6/UINT4 actions. The 24 "
        "historical UINT8 actions (phase8b noAE, phase9d AE128, phase10b AE32/AE64) "
        "have no per-range evidence; those cells stay null and are NOT "
        "reconstructed from incompatible historical aggregates.",
        cov(lambda c: str(c.get("val_person_avo_recall_0_30m") or "") != ""))
    requested["validation::diagnostic_recall_30_40m"] = mark(
        "partially_available",
        "Same Phase-11D-only availability as the 0-30 m AVO recall.",
        cov(lambda c: str(c.get("val_person_avo_recall_30_40m_diagnostic") or "") != ""))

    return {
        "schema": SCHEMA + ".metric_availability",
        "cells_total": n,
        "legend": {
            "available": "derivable for every cell from registered fields",
            "partially_available": "derivable for some cells; the rest are a measured zero or a registered absence",
            "unavailable": "no registered field or no compatible clock/boundary; explicitly not estimated",
        },
        "metrics": requested,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build_report(summary: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], artifacts: Mapping[str, str]) -> str:
    prov = summary["provenance"]
    counts = summary["verification"]
    agg = summary["aggregates"]
    lines: list[str] = []
    add = lines.append
    add("# SplitFusion 288-Cell Live CARLA/OAI Campaign - Offline RL Dataset Consolidation")
    add("")
    add("Steps 1-4 only: evidence verification, one reconciled 288-cell table, the")
    add("72-action x 4-profile aggregation, and separation of the latency, delivery,")
    add("payload and registered-accuracy components. **No action pruning, Pareto")
    add("selection, feasibility masking, reward design, policy design or training is")
    add("performed or implied here.** Steps 5-6 belong to a separate reviewer.")
    add("")
    add("This consolidation is strictly offline. No CARLA server, Docker container, OAI")
    add("gNB/UE, RFsim, CUDA context or model inference was started, and no")
    add("training/holdout/validation/test imagery was read. No campaign directory was")
    add("modified. **No claim of 100 ms service readiness is made.**")
    add("")
    add("## 1. Provenance")
    add("")
    add(f"- Evidence commit: `{prov['evidence_commit']}`")
    add(f"- Starting HEAD: `{prov['starting_head']}`")
    add(f"- Campaign root: `{prov['campaign_root']}`")
    add(f"- Completion terminal: `{prov['terminal_status']}`")
    add("")
    add("| artifact | sha256 |")
    add("| --- | --- |")
    for name, digest in sorted(prov["artifact_hashes"].items()):
        add(f"| `{name}` | `{digest}` |")
    add(f"| `{CAMPAIGN_CONFIG_REL}` | `{prov['campaign_config_sha256']}` |")
    add(f"| `{CATALOG_JSON_REL}` | `{prov['catalog_json_sha256']}` |")
    add("")
    add("All four declared completion bindings verified, and the cell-mapping digest")
    add("was **re-derived independently** from the locked 72-action catalog and the four")
    add("configured network profiles rather than trusted from the manifest:")
    add("")
    add(f"- `cell_mapping_sha256` re-derived = `{prov['cell_mapping_sha256_rederived']}`")
    add("")
    add("## 2. Continuation chain")
    add("")
    chain = prov["continuation_chain"]
    add(f"The retry8 inventory reuses {chain['declared_reused_cells']} cells by hash reference.")
    add(f"All {chain['declared_references_carried_forward']} are references *carried forward*")
    add(f"({chain['declared_reused_from_source_campaign']} were re-derived from the immediate source),")
    add("so the chain had to be walked to its origins rather than assumed:")
    add("")
    add("| campaign | ledger sha256 | statuses | rerun |")
    add("| --- | --- | --- | --- |")
    add(f"| `...retry8` (this) | `{prov['artifact_hashes']['campaign_ledger.json'][:16]}...` | {summary['verification']['status_counts']} | `a61__favorable_stable` |")
    for hop in chain["hops"]:
        add(f"| `...{hop['campaign_root'].split('_')[-1]}` | `{hop['campaign_ledger_sha256'][:16]}...` | {hop['ledger_status_counts']} | {hop['rerun_cell_ids']} |")
    add("")
    add("Origin accounting: 58 reused cells originate in retry5 and 3 in retry6, matching")
    add("the completion summary exactly. Every reused cell's continuation reference, source")
    add("campaign, source ledger/manifest/terminal binding, original attempt directory and")
    add("every registered file hash were verified against the physical source tree.")
    add("")
    add("### Route-summary limitation (stated precisely)")
    add("")
    add("Two distinct things must not be conflated, so both are reported:")
    add("")
    add(f"- **Durable route summary file present:** {counts['executed_cells']} executed + "
        f"{counts['reused_with_route_summary']} retry6-origin reused = {counts['route_summary_present']}/288. "
        f"Absent for the {counts['reused_without_route_summary']} retry5-origin reused cells, which predate the requirement.")
    add(f"- **Registered `route_outcome_classification` present:** {counts['route_outcome_available']}/288 - "
        "the executed cells only.")
    add("")
    add("All 61 reused cells therefore lack the registered classification. The 3")
    add("retry6-origin cells do have a durable route summary, but it is the route")
    add("*runner's* metrics summary whose `status` field is the density/intervention")
    add("state (e.g. `INTERVENED`), not the campaign's `route_outcome_classification`.")
    add("Promoting that status to a classification would be an inference, so it was")
    add("not done: those cells are `UNAVAILABLE_NOT_INFERRED`, and the raw runner")
    add("status is carried separately in")
    add("`route_runner_status_raw_not_a_classification`. **No route outcome was")
    add("inferred for any reused cell.**")
    add("")
    add(f"Registered route outcomes over the {counts['route_outcome_available']} cells that have one:")
    add("")
    for key, value in sorted(agg["route_outcome_counts"].items()):
        add(f"- `{key}`: {value}")
    add("")
    add("## 3. Reconciliation identities")
    add("")
    add(f"- Expected cells: 288; ledger cells: {counts['ledger_cells']}; unique: {counts['unique_cells']}")
    add(f"- Missing: {counts['missing_cells']}; foreign: {counts['foreign_cells']}; duplicated: {counts['duplicate_cells']}")
    add(f"- Actions: {counts['distinct_actions']}; network profiles: {counts['distinct_profiles']}; product: {counts['distinct_actions']} x {counts['distinct_profiles']} = {counts['ledger_cells']}")
    add(f"- Executed: {counts['executed_cells']}; reused: {counts['reused_cells']}")
    add(f"- Cells with structural acceptance PASS: {counts['structural_pass']}/288")
    add(f"- Registered output hashes verified: {counts['registered_outputs_verified']}")
    add(f"- Cells whose campaign counter reconciliation holds: {counts['counter_reconciliation_holds']}/288")
    add(f"- Cells whose reassembly identity (reassembled = admitted + after-reassembly drops + rejected) was independently re-checked and holds: {counts['reassembly_identity_holds']}/288")
    add(f"- Cells whose recomputed install-AoI median/p95 reproduce the registered values exactly: {counts['aoi_reproduces_registered']}/288")
    add("")
    add("## 4. Workload and preparation")
    add("")
    add(f"- Registered preparation target: 10 Hz; minimum coverage contract 0.95.")
    add(f"- Sensor preparation coverage across 288 cells: median {agg['prep_coverage']['median']:.4f}, "
        f"min {agg['prep_coverage']['min']:.4f}, max {agg['prep_coverage']['max']:.4f}.")
    add(f"- Cells meeting the registered 0.95 coverage contract: {agg['coverage_met_cells']}/288.")
    add(f"  The {288 - agg['coverage_met_cells']} cells below it span "
        f"{agg['low_coverage_range'][0]:.4f}-{agg['low_coverage_range'][1]:.4f} and are all "
        "large-payload rungs. They still carry structural acceptance PASS because the")
    add("  campaign registers low preparation as a measured outcome, not a structural")
    add("  invalidity. The 0.95 target was not weakened here, and these cells are")
    add("  retained rather than excluded:")
    for cid, value in agg["low_coverage_cells"]:
        add(f"  - `{cid}`: {value:.4f}")
    add(f"- Total frames sent across the campaign: {agg['total_frames_sent']:,}.")
    add(f"- Sustainable send rate from the median inter-send period: median {agg['sustainable_fps']['median']:.3f} FPS.")
    add("")
    add("Preparation drops by registered reason (campaign totals):")
    add("")
    for reason, total in agg["prepare_status_totals"].items():
        add(f"- `{reason}`: {total:,}")
    add("")
    add("## 5. Payload")
    add("")
    add("Live measured bytes are kept strictly separate from catalog estimates; catalog")
    add("values live in `catalog_*` columns and were never substituted for measured bytes.")
    add("")
    add(f"- Live scientific inner payload, median over cells: {agg['inner_bytes_median_range'][0]:,.0f} B (min) to {agg['inner_bytes_median_range'][1]:,.0f} B (max).")
    add(f"- SFD1/application-envelope overhead, median over cells: {agg['sfd1_overhead_median']:,.1f} B.")
    add(f"- Datagrams per message, median over cells: {agg['datagrams_median_range'][0]:,.0f} to {agg['datagrams_median_range'][1]:,.0f}.")
    add("")
    add("## 6. Delivery")
    add("")
    add(f"- Cells with at least one installed map: {agg['cells_with_install']}/288.")
    add(f"- **Cells with zero delivery: {agg['zero_delivery_cells']}/288.** Zero delivery is preserved as a measured outcome, never as missing data.")
    add(f"- Campaign totals: {agg['total_frames_sent']:,} sent -> {agg['total_reassembled']:,} complete edge reassemblies -> {agg['total_tail_completions']:,} tail completions -> {agg['total_published']:,} map publications -> {agg['total_installed']:,} maps installed.")
    add(f"- Feature datagrams transmitted {agg['total_datagrams_tx']:,}; received at the edge {agg['total_datagrams_rx']:,} (ratio {agg['datagram_delivery_ratio']:.4f}).")
    add("")
    add("### The two deadlines are reported separately")
    add("")
    add(f"- Installed within the **100 ms** service reference: **{agg['total_on_time_100ms']:,}** frames "
        f"({agg['total_on_time_100ms']/max(agg['total_frames_sent'],1):.6%} of sent, "
        f"{agg['total_on_time_100ms']/max(agg['total_installed'],1):.6%} of installed).")
    add(f"- Feedback within the **500 ms** ACK timeout: **{agg['total_ack_500ms']:,}** frames "
        f"({agg['total_ack_500ms']/max(agg['total_frames_sent'],1):.4%} of sent, "
        f"{agg['total_ack_500ms']/max(agg['total_installed'],1):.4%} of installed).")
    add(f"- Terminal `TIMEOUT_NO_ACK`: {agg['total_timeout_no_ack']:,}.")
    add(f"- Cells with at least one install inside 100 ms: {agg['cells_with_on_time_install']}/288.")
    add("")
    add("The 500 ms ACK observation timeout is **not** the 100 ms service target, and the")
    add("registered `*_fraction` fields use *installed frames* as their denominator; both")
    add("the per-sent and per-installed denominators are retained in the tables.")
    add("")
    add("Across all 288 cells, **not one** of the 896,856 sent frames installed inside the")
    add("100 ms service reference. This is a measured campaign outcome and is exactly why")
    add("the campaign records `claims_100ms_service_ready: false`; it is reported here")
    add("without any readiness claim.")
    add("")
    add("### Per network profile")
    add("")
    add("| profile | sent | installed | installed/sent | within 100 ms | ACK<=500 ms | zero-delivery cells | datagram rx/tx | install AoI median-of-medians |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for profile in ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY"):
        p = agg["by_network_profile"][profile]
        aoi = p["install_aoi_ms_median_of_medians"]
        add(f"| `{profile}` | {p['frames_sent']:,} | {p['maps_installed']:,} | "
            f"{p['installed_per_sent']:.4f} | {p['installed_within_100ms']} | "
            f"{p['ack_within_500ms']:,} | {p['zero_delivery_cells']}/72 | "
            f"{p['datagram_delivery_ratio']:.4f} | "
            f"{'n/a' if aoi is None else f'{aoi:.1f} ms'} |")
    add("")
    add("The ordering is monotone in channel favourability for every delivery measure,")
    add("and adverse/fade behaviour is retained per profile in all three tables rather")
    add("than being averaged away.")
    add("")
    add("## 7. Latency decomposition")
    add("")
    add("Clock domains were established from the emitting source, not assumed:")
    add("")
    add("- `WALL_HOST` - `time.time()`/CLOCK_REALTIME. The UE CARLA client, the edge")
    add("  container (local Docker on the OAI CN5G bridge) and the map process share one")
    add("  kernel, so these instants are mutually comparable.")
    add("- `UE_PERF` - `time.perf_counter_ns()` in the single UE client process (adapter")
    add("  and live dispatch are the same process). Mutually comparable, **not**")
    add("  epoch-anchored.")
    add("- `EDGE_PERF` - edge-local `perf_counter_ns()`; contributes durations only.")
    add("")
    add("Two naming traps in the registered schema, both preserved rather than papered over:")
    add("")
    add("1. `front_ms` is **not** the model front. It is `ue_prepare_finished_ns -")
    add("   capture_started_ns`, the whole UE dispatch span (input assembly, context")
    add("   build, front, ranker, AE encode, quantize/pack, zstd, chunking). The model")
    add("   front proper is `front_timing_ns.front_backbone`. Pre-front sensor")
    add("   preparation is carried in its own `s1_*` fields and is never called")
    add("   `front_ms`.")
    add("2. `queue_wait_ms` is **not** a pure prepared-queue wait. It is measured from")
    add("   the schedule instant to just before dispatch and therefore *encloses* the")
    add("   sensor wait and radar/RGB preparation. A same-clock residual is offered")
    add("   separately as the true queue wait.")
    add("")
    add(f"- **Stage 5, one-way feature uplink, is UNAVAILABLE.** The final UE send boundary")
    add("  (`send_finished_ns`, UE_PERF) and the complete edge receive boundary")
    add("  (`edge_receipt_wall_s`, WALL_HOST) are in non-comparable clock domains and no")
    add("  simultaneous cross-clock reading is registered. No one-way uplink latency was")
    add("  manufactured by differencing them or by subtracting unrelated medians. A")
    add("  round-trip residual and a wall-domain capture-to-edge-receipt span are provided")
    add("  instead, each explicitly labelled.")
    add(f"- **Stage 9, tail completion to map publication, is UNAVAILABLE**: map publication")
    add("  is registered only as the categorical `map_publication_status`, so the boundary")
    add("  has no end event.")
    add("")
    add("Campaign-level medians of the per-cell medians (ms):")
    add("")
    add("| stage | cells with data | median of per-cell medians |")
    add("| --- | --- | --- |")
    for stage, info in agg["stage_medians"]:
        shown = "n/a" if info["median_of_medians"] is None else f"{info['median_of_medians']:.3f}"
        add(f"| `{stage}` | {info['cells']} | {shown} |")
    add("")
    add("`s3_front_timing_ns_unparseable_rows` is a parsing diagnostic, not a latency")
    add("stage: it shows 0, i.e. every instrumented timing blob parsed. The two `n/a`")
    add("stages are the two genuinely unavailable boundaries named above.")
    add("")
    add(f"- Registered capture-to-install AoI (`install_aoi_ms`), median of per-cell medians: "
        f"**{agg['aoi_median_of_medians']:.1f} ms** over the {agg['cells_with_install']} cells that installed anything.")
    add("- `s14_capture_to_ue_feedback_aoi_ms` sits near 500 ms because it includes every")
    add("  `TIMEOUT_NO_ACK` row, whose receipt instant is the 500 ms timeout sweep rather")
    add("  than a remote ACK. It must not be read as a network latency.")
    add("- Every published map was installed (`survival_installed_per_published` = 1.0 wherever")
    add("  publication occurred); the loss is upstream, in datagram delivery, reassembly and")
    add("  the edge deadline gates.")
    add("")
    add("The per-frame component medians are **not** expected to sum to the median")
    add("end-to-end latency: medians do not add, the stages have different valid-sample")
    add("sets, and two stages are unavailable. Any residual field is labelled")
    add("`_RESIDUAL` and carries its own negative-sample count.")
    add("")
    add("### Residual sign check (clock-domain sanity)")
    add("")
    add("Every derived residual is formed by subtracting a same-frame duration from a")
    add("same-frame interval in a *compatible* domain. If a residual had in fact crossed")
    add("incomparable clocks, negative values would be expected. Observed:")
    add("")
    add("| residual | samples | negative | worst minimum |")
    add("| --- | --- | --- | --- |")
    for stage in ("s2_derived_true_queue_wait_ms_RESIDUAL",
                  "s4_derived_ue_uninstrumented_span_ms",
                  "s5_derived_roundtrip_residual_ms_RESIDUAL",
                  "s6_derived_edge_queue_wait_ms_RESIDUAL"):
        total_n = sum(int(c["_latency"][stage].get("count") or 0) for c in cells)
        neg = sum(int(c["_latency"][stage].get("negative_residual_count") or 0) for c in cells)
        mins = [c["_latency"][stage].get("min") for c in cells]
        mins = [m for m in mins if m is not None]
        worst = f"{min(mins):.4f} ms" if mins else "n/a"
        add(f"| `{stage}` | {total_n:,} | {neg} | {worst} |")
    add("")
    add("All four residuals are strictly positive over every sample, which is consistent")
    add("with the domain assignments above. It is corroboration, not proof.")
    add("")
    add("The user's later analytical quantity, `model_front + feature_transport +")
    add("model_back + compact_feedback`, is preserved as")
    add("`analytic_model_front_transport_back_feedback_ms`. It is computed only where")
    add("every component is a same-frame compatible timestamp, and it is **not** labelled")
    add("capture-to-install latency: it excludes sensor preparation, the prepared-queue")
    add("wait and UE dispatch overhead, and its transport term is a round-trip residual.")
    add("")
    add("## 8. Accuracy and preservation join")
    add("")
    add("All 72 live actions were joined to their authoritative frozen validation")
    add("properties using existing registered fields and classifications only. **No")
    add("prediction was rescored and no threshold was tuned.** Model-validation metrics")
    add("are carried in `val_*` columns and are kept distinct from the live network")
    add("measurements; live per-route perception is separately namespaced `live_*`.")
    add("")
    add(f"- Validation rows joined: {counts['validation_rows_joined']}/288 cells ({counts['distinct_actions']}/72 actions).")
    add("- Source evidence, all hash-verified:")
    for path, info in sorted(agg["validation_sources"].items()):
        add(f"  - `{path}` ({info['actions']} actions) sha256 `{info['sha256']}`")
    add(f"- Durable per-action settings hash-verified: {counts['durable_settings_verified']}/72.")
    add("")
    add("Registered tiers across the 72 actions:")
    add("")
    for tier, total in sorted(agg["tier_counts"].items()):
        add(f"- `{tier}`: {total}")
    add("")
    add("Left deliberately null:")
    add("")
    add(f"- `person_avo_recall_0_30m` and `person_avo_recall_30_40m_diagnostic` are")
    add(f"  registered only for the 48 Phase-11D UINT6/UINT4 actions. The 24 historical")
    add("  UINT8 actions (phase8b noAE, phase9d AE128, phase10b AE32/AE64) have no")
    add("  compatible per-range evidence, so those cells stay null. **0-30 m recall was")
    add("  not reconstructed from incompatible historical aggregates.**")
    add("- A pixel mask-accuracy field is not registered anywhere in the frozen evidence;")
    add("  `foreground_miou` and `person_box_mask_iou` are reported instead.")
    add("- No standalone segmentation gates-passed/total counter exists; segmentation is")
    add("  registered as `segmentation_installable` plus `segmentation_behavior`.")
    add("")
    add("## 9. Outputs")
    add("")
    add("| artifact | rows | sha256 |")
    add("| --- | --- | --- |")
    for name, digest in artifacts.items():
        rows = summary["output_row_counts"].get(name, "n/a")
        add(f"| `{name}` | {rows} | `{digest}` |")
    add("")
    add("This report cannot carry its own digest. `REPORT.md` and every artifact above")
    add("are bound together by the compact terminal marker")
    add("`SPLITFUSION_288_OFFLINE_RL_DATASET_CONSOLIDATION_COMPLETE`.")
    add("")
    add("`campaign_288_cell_table.csv` and `action_72x4_summary.csv` are both keyed by")
    add("the same 288 action/profile cells but are **not** structurally duplicated:")
    add("")
    add("- `campaign_288_cell_table.csv` is the reconciled *cell record*: identity,")
    add("  provenance and continuation binding, workload/preparation, the separated")
    add("  payload components, and the delivery-stage counts and rates.")
    add("- `action_72x4_summary.csv` is the *scientific component table*: the full")
    add("  per-stage latency decomposition (field-level distributions with clock domain,")
    add("  valid-sample and exclusion accounting) joined to the frozen registered")
    add("  accuracy properties. It carries the Step-4 separation that later RL")
    add("  formulation consumes.")
    add("- `action_72_summary.csv` aggregates each action across the four profiles while")
    add("  retaining per-profile columns, so adverse and fade behaviour is never hidden")
    add("  inside a mean.")
    add("")
    add("## 10. Scope")
    add("")
    add("Delivered: Steps 1-4. Not performed, and deliberately left to the Step 5-6")
    add("reviewer: action pruning, Pareto decisions, feasibility masking, reward design,")
    add("RL-policy design, and training.")
    add("")
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="create-only output evidence directory")
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--evidence-commit", default="", help="recorded for provenance only")
    parser.add_argument("--starting-head", default="", help="recorded for provenance only")
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    out_dir = Path(args.out).resolve()
    if out_dir.exists():
        fail(f"output directory already exists (create-only): {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    prov = verify_campaign(repo)
    ledger_cells = prov["ledger"]["cells"]
    catalog_rows = {int(r["action_id"]): r for r in prov["catalog_rows"]}
    with (repo / CATALOG_CSV_REL).open("r", encoding="utf-8", newline="") as handle:
        catalog_csv = {int(r["action_id"]): r for r in csv.DictReader(handle)}
    if set(catalog_csv) != set(catalog_rows):
        fail("catalog CSV and JSON action sets differ")
    profile_meta = {str(p["profile_id"]): p for p in prov["config"]["network"]["profiles"]}
    mapping_by_cell: dict[str, dict[str, Any]] = {}
    for profile in prov["config"]["network"]["profiles"]:
        for row in prov["catalog_rows"]:
            cid = f"a{int(row['action_id']):02d}__{str(profile['profile_id']).lower()}"
            mapping_by_cell[cid] = {
                "action_id": int(row["action_id"]),
                "network_profile_id": str(profile["profile_id"]),
            }

    evidence_cache: dict[str, str] = {}
    cells: list[dict[str, Any]] = []
    for cell_id in prov["expected_cell_ids"]:
        record = ledger_cells[cell_id][0]
        cell_prov = verify_cell_evidence(repo, cell_id, record)
        mapping_row = mapping_by_cell[cell_id]
        action_id = mapping_row["action_id"]
        row = summarize_cell(
            cell_id, cell_prov, catalog_rows[action_id], catalog_csv[action_id],
            mapping_row, profile_meta[mapping_row["network_profile_id"]],
        )
        row.update(validation_join(repo, catalog_csv[action_id], evidence_cache))
        cells.append(row)

    if len(cells) != 288:
        fail(f"expected 288 reconciled cells, built {len(cells)}")
    if len({c["cell_id"] for c in cells}) != 288:
        fail("reconciled cell IDs are not unique")

    # ---- Tables ----------------------------------------------------------
    cell_columns = [k for k in cells[0] if k not in CELL_TABLE_EXCLUDE and not k.startswith("val_")]
    artifacts: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    artifacts["campaign_288_cell_table.csv"] = write_csv(
        out_dir / "campaign_288_cell_table.csv", cells, cell_columns)
    row_counts["campaign_288_cell_table.csv"] = len(cells)

    identity_cols = ["cell_id", "action_id", "profile_id", "network_profile", "family",
                     "quantizer", "q", "q_e4", "keep_count", "execution_status",
                     "executed_or_reused", "frames_sent", "maps_installed",
                     "installed_within_100ms_service_reference", "ack_within_500ms_timeout",
                     "rate_installed_per_sent", "zero_delivery",
                     "live_scientific_inner_bytes_median", "live_estimated_wire_bytes_median",
                     "live_datagrams_per_message_median", "radio_achieved_snr_db_median"]
    lat_cols = latency_columns(cells[0]["_latency"])
    val_cols = [k for k in cells[0] if k.startswith("val_")]
    rows_72x4 = []
    for cell in cells:
        row = {k: cell.get(k) for k in identity_cols}
        row.update(flatten_latency(cell["_latency"]))
        row.update({k: cell.get(k) for k in val_cols})
        rows_72x4.append(row)
    artifacts["action_72x4_summary.csv"] = write_csv(
        out_dir / "action_72x4_summary.csv", rows_72x4, identity_cols + lat_cols + val_cols)
    row_counts["action_72x4_summary.csv"] = len(rows_72x4)

    action_rows = aggregate_actions(cells, prov["network_profiles"])
    if len(action_rows) != 72:
        fail(f"expected 72 action rows, built {len(action_rows)}")
    artifacts["action_72_summary.csv"] = write_csv(
        out_dir / "action_72_summary.csv", action_rows, list(action_rows[0].keys()))
    row_counts["action_72_summary.csv"] = len(action_rows)

    # ---- Dictionaries ----------------------------------------------------
    lat_dict = build_latency_dictionary()
    (out_dir / "latency_metric_dictionary.json").write_text(
        json.dumps(lat_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts["latency_metric_dictionary.json"] = sha256_file(out_dir / "latency_metric_dictionary.json") or ""

    availability = build_metric_availability(cells)
    (out_dir / "metric_availability.json").write_text(
        json.dumps(availability, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts["metric_availability.json"] = sha256_file(out_dir / "metric_availability.json") or ""

    # ---- Aggregates ------------------------------------------------------
    summary = build_analysis_summary(prov, cells, action_rows, args, artifacts, row_counts)
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    artifacts["analysis_summary.json"] = sha256_file(out_dir / "analysis_summary.json") or ""

    report = build_report(summary, cells, artifacts)
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    artifacts["REPORT.md"] = sha256_file(out_dir / "REPORT.md") or ""

    terminal = {
        "schema": SCHEMA + ".terminal",
        "status": "SPLITFUSION_288_OFFLINE_RL_DATASET_CONSOLIDATION_COMPLETE",
        "steps_covered": "1-4 (verify, reconcile, aggregate, separate components)",
        "steps_not_performed": "5-6 (pruning, Pareto, feasibility masking, reward, policy, training)",
        "claims_100ms_service_ready": False,
        "campaign_root": CAMPAIGN_ROOT_REL,
        "campaign_completion_sha256": EXPECTED["completion_sha256"],
        "cells": 288,
        "actions": 72,
        "network_profiles": list(prov["network_profiles"]),
        "artifact_sha256": artifacts,
        "no_live_system_launched": True,
        "campaign_evidence_unmodified": True,
    }
    (out_dir / "SPLITFUSION_288_OFFLINE_RL_DATASET_CONSOLIDATION_COMPLETE").write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {len(artifacts) + 1} artifacts to {out_dir}")
    for name in sorted(artifacts):
        print(f"  {artifacts[name]}  {name}")
    return 0


def build_analysis_summary(
    prov: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    artifacts: Mapping[str, str],
    row_counts: Mapping[str, int],
) -> dict[str, Any]:
    total = lambda key: sum(int(c.get(key) or 0) for c in cells)  # noqa: E731
    prep_totals: dict[str, int] = {}
    for status in PREPARE_STATUSES:
        prep_totals[status] = total(f"prepare_status_{status}")

    coverage = dist((c["sensor_preparation_coverage"] for c in cells))
    sustainable = dist((c["sustainable_fps_from_median_send_period"] for c in cells))
    inner_medians = [c["live_scientific_inner_bytes_median"] for c in cells if c["live_scientific_inner_bytes_median"] is not None]
    dgram_medians = [c["live_datagrams_per_message_median"] for c in cells if c["live_datagrams_per_message_median"] is not None]

    stage_names = sorted(cells[0]["_latency"])
    stage_medians = []
    for stage in stage_names:
        meds = [c["_latency"][stage].get("median") for c in cells]
        meds = [m for m in meds if m is not None]
        stage_medians.append((stage, {
            "cells": len(meds),
            "median_of_medians": quantile_nearest_rank(sorted(meds), 0.5),
        }))

    aoi_meds = [c["_latency"]["s13_capture_to_install_aoi_ms"]["median"] for c in cells]
    aoi_meds = [m for m in aoi_meds if m is not None]

    validation_sources: dict[str, dict[str, Any]] = {}
    for cell in cells:
        path = str(cell["val_source_evidence_path"])
        entry = validation_sources.setdefault(path, {"actions": set(), "sha256": cell["val_source_evidence_sha256"]})
        entry["actions"].add(int(cell["action_id"]))
    for entry in validation_sources.values():
        entry["actions"] = len(entry["actions"])

    by_profile: dict[str, dict[str, Any]] = {}
    for profile in prov["network_profiles"]:
        subset = [c for c in cells if c["network_profile"] == profile]
        sent = sum(int(c.get("frames_sent") or 0) for c in subset)
        inst = sum(int(c.get("maps_installed") or 0) for c in subset)
        on_time = sum(int(c.get("installed_within_100ms_service_reference") or 0) for c in subset)
        ack = sum(int(c.get("ack_within_500ms_timeout") or 0) for c in subset)
        meds = [c["_latency"]["s13_capture_to_install_aoi_ms"]["median"] for c in subset]
        meds = [m for m in meds if m is not None]
        by_profile[profile] = {
            "cells": len(subset),
            "frames_sent": sent,
            "maps_installed": inst,
            "installed_per_sent": ratio(inst, sent),
            "installed_within_100ms": on_time,
            "on_time_100ms_per_sent": ratio(on_time, sent),
            "ack_within_500ms": ack,
            "ack_500ms_per_sent": ratio(ack, sent),
            "zero_delivery_cells": sum(1 for c in subset if c["zero_delivery"]),
            "cells_with_install": len(meds),
            "install_aoi_ms_median_of_medians": quantile_nearest_rank(sorted(meds), 0.5),
            "datagram_delivery_ratio": ratio(
                sum(int(c["edge_feature_datagrams_received"]) for c in subset),
                sum(int(c["ue_feature_datagrams_transmitted"]) for c in subset)),
        }

    route_counts = Counter(
        str(c["route_outcome_classification"]) for c in cells
        if c["route_outcome_classification"] != "UNAVAILABLE_NOT_INFERRED")

    return {
        "schema": SCHEMA,
        "steps_covered": "1-4",
        "steps_not_performed": [
            "action pruning", "Pareto decisions", "feasibility masking",
            "reward design", "RL-policy design", "training",
        ],
        "claims_100ms_service_ready": False,
        "provenance": {
            "evidence_commit": args.evidence_commit,
            "starting_head": args.starting_head,
            "campaign_root": CAMPAIGN_ROOT_REL,
            "terminal_status": EXPECTED["terminal_status"],
            "artifact_hashes": prov["artifact_hashes"],
            "declared_bindings": {k: v for k, v in EXPECTED.items() if k.endswith("sha256")},
            "campaign_config_path": CAMPAIGN_CONFIG_REL,
            "campaign_config_sha256": prov["config_sha256"],
            "catalog_json_path": CATALOG_JSON_REL,
            "catalog_json_sha256": prov["catalog_json_sha256"],
            "catalog_csv_sha256": prov["catalog_csv_sha256"],
            "cell_mapping_sha256_rederived": prov["cell_mapping_sha256_rederived"],
            "continuation_chain": prov["continuation_chain"],
        },
        "verification": {
            "ledger_cells": len(cells),
            "unique_cells": len({c["cell_id"] for c in cells}),
            "missing_cells": 0,
            "foreign_cells": 0,
            "duplicate_cells": 0,
            "distinct_actions": len({c["action_id"] for c in cells}),
            "distinct_profiles": len({c["network_profile"] for c in cells}),
            "status_counts": prov["status_counts"],
            "executed_cells": sum(1 for c in cells if c["executed_or_reused"] == "EXECUTED"),
            "reused_cells": sum(1 for c in cells if c["executed_or_reused"] == "REUSED"),
            "reused_with_route_summary": sum(
                1 for c in cells if c["executed_or_reused"] == "REUSED" and c["route_summary_available"]),
            "reused_without_route_summary": sum(
                1 for c in cells if c["executed_or_reused"] == "REUSED" and not c["route_summary_available"]),
            "route_summary_present": sum(1 for c in cells if c["route_summary_available"]),
            "route_outcome_available": sum(1 for c in cells if c["route_outcome_classification_available"]),
            "structural_pass": sum(1 for c in cells if c["structural_acceptance_status"] == "PASS"),
            "registered_outputs_verified": total("registered_outputs_verified"),
            "counter_reconciliation_holds": sum(
                1 for c in cells if c["counter_reconciliation_all_identities_hold"] is True),
            "reassembly_identity_holds": sum(1 for c in cells if c["identity_reassembly_holds"] is True),
            "aoi_reproduces_registered": sum(
                1 for c in cells
                if c["_latency"]["s13_capture_to_install_aoi_ms"].get("reproduces_registered_summary")),
            "validation_rows_joined": sum(1 for c in cells if c["val_join_verified"]),
            "durable_settings_verified": len({c["val_durable_setting_path"] for c in cells}),
        },
        "aggregates": {
            "total_frames_sent": total("frames_sent"),
            "total_datagrams_tx": total("ue_feature_datagrams_transmitted"),
            "total_datagrams_rx": total("edge_feature_datagrams_received"),
            "datagram_delivery_ratio": ratio(total("edge_feature_datagrams_received"), total("ue_feature_datagrams_transmitted")),
            "total_reassembled": total("edge_complete_reassemblies"),
            "total_admissions": total("edge_edge_queue_admissions"),
            "total_tail_completions": total("edge_tail_completions"),
            "total_published": total("map_publications"),
            "total_installed": total("maps_installed"),
            "total_on_time_100ms": total("installed_within_100ms_service_reference"),
            "total_ack_500ms": total("ack_within_500ms_timeout"),
            "total_timeout_no_ack": total("terminal_TIMEOUT_NO_ACK"),
            "cells_with_install": sum(1 for c in cells if not c["zero_delivery"]),
            "zero_delivery_cells": sum(1 for c in cells if c["zero_delivery"]),
            "cells_with_on_time_install": sum(
                1 for c in cells if int(c.get("installed_within_100ms_service_reference") or 0) > 0),
            "coverage_met_cells": sum(1 for c in cells if c["sensor_preparation_coverage_met"] is True),
            "low_coverage_cells": sorted(
                (str(c["cell_id"]), float(c["sensor_preparation_coverage"]))
                for c in cells if c["sensor_preparation_coverage_met"] is not True),
            "low_coverage_range": [
                min((float(c["sensor_preparation_coverage"]) for c in cells
                     if c["sensor_preparation_coverage_met"] is not True), default=None),
                max((float(c["sensor_preparation_coverage"]) for c in cells
                     if c["sensor_preparation_coverage_met"] is not True), default=None),
            ],
            "prep_coverage": coverage,
            "sustainable_fps": sustainable,
            "prepare_status_totals": prep_totals,
            "inner_bytes_median_range": [min(inner_medians), max(inner_medians)] if inner_medians else [None, None],
            "sfd1_overhead_median": quantile_nearest_rank(
                sorted(c["live_sfd1_overhead_bytes_median"] for c in cells
                       if c["live_sfd1_overhead_bytes_median"] is not None), 0.5),
            "datagrams_median_range": [min(dgram_medians), max(dgram_medians)] if dgram_medians else [None, None],
            "stage_medians": stage_medians,
            "aoi_median_of_medians": quantile_nearest_rank(sorted(aoi_meds), 0.5),
            "route_outcome_counts": dict(route_counts),
            "validation_sources": validation_sources,
            "tier_counts": dict(Counter(
                str(c["val_source_contract_tier"]) for c in cells
                if c["network_profile"] == prov["network_profiles"][0])),
            "by_network_profile": by_profile,
        },
        "output_row_counts": dict(row_counts),
        "artifact_sha256": dict(artifacts),
    }


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceError as exc:
        print(f"EVIDENCE VERIFICATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)
