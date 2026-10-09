"""Read-only native-prediction verification, never live qualification.

Mechanically retained v1 verifier with explicit v2 contract, dimension and
memory-accounting checks. Original target/source/evaluation checks remain.

This is deliberately not a launcher or an actor/training selector. It verifies
complete GT bundles and the predictions actually owned by the evaluator. GT
unavailability remains visible, including when no complete target was produced.
No old raw-semantic GT or historical sample can substitute for a missing bundle.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from rl_agent.splitfusion_live_parity_repair_v1 import evaluation, target_transport
from .prediction_archive import (PredictionArchive, SCHEMA as ARCHIVE_SCHEMA, canonical,
    PredictionContract, contract_or_native, validate_labels, budget_document, MAX_MEMORY_BYTES)

SCHEMA = "splitfusion.parity_repair.offline_evidence_check.v2"
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_TICKETS = 1000
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY_KEYS = {"gt_identity", "action_identity", "capture_timestamp_ns", "enqueued_wall_ns"}
_ARCHIVE_KEYS = _IDENTITY_KEYS | {"schema", "prediction_digest", "snapshot_raw_ns"}


class ReadinessError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise ReadinessError(message)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _digest(value):
    _require(isinstance(value, str) and _SHA.fullmatch(value), "invalid SHA-256 binding")
    return value


def _read(path: Path, digest: str | None = None) -> bytes:
    path = Path(path)
    _require(not path.is_symlink() and path.is_file(), f"missing/nonregular/symlink evidence: {path.name}")
    _require(path.stat().st_size <= MAX_PAYLOAD_BYTES, "evidence payload bound exceeded")
    data = path.read_bytes()
    if digest is not None:
        _require(_sha(data) == _digest(digest), f"evidence hash mismatch: {path.name}")
    return data


def _json(payload):
    try:
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ReadinessError("duplicate JSON key")
                result[key] = value
            return result
        value = json.loads(payload, object_pairs_hook=unique_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ReadinessError("nonfinite JSON")))
        canonical(value)
        return value
    except (TypeError, ValueError) as exc:
        raise ReadinessError(f"invalid evidence JSON: {exc}") from exc


def _identity(value):
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
    _require(isinstance(value, Mapping) and set(value) == _IDENTITY_KEYS, "full exact ticket identity required")
    owned = _json(canonical(value))
    gt = target_transport.validate_identity(owned["gt_identity"])
    action = owned["action_identity"]
    MP.validate_run4_identity(action)
    _require(isinstance(action["session_uuid"], str) and bool(action["session_uuid"]), "missing session UUID")
    _require(action["reward_requested"] and action["frame_kind"] == "POLICY_DECISION", "nonreward archive identity")
    _require(gt["frame_id"] == action["frame_id"] and gt["action_id"] == action["anchor_action_id"],
             "GT/action frame or anchor identity mismatch")
    for key in ("capture_timestamp_ns", "enqueued_wall_ns"):
        _require(type(owned[key]) is int and owned[key] > 0, f"invalid ticket {key}")
    _require(owned["capture_timestamp_ns"] == gt["capture_timestamp_ns"], "capture timestamp mismatch")
    return owned


def _gt_key(identity):
    return _sha(canonical(identity))


def _verify_source_archive(descriptor, target):
    from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry
    from rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync import validate_source_alignment
    directory = Path(descriptor["directory"])
    _require(directory.is_dir() and not directory.is_symlink(), "source archive directory missing or symbolic")
    manifest_raw = _read(directory / "MANIFEST.json", descriptor["manifest_sha256"])
    manifest = _json(manifest_raw)
    _require(isinstance(manifest, dict) and set(manifest) == {"schema", "identity", "sha256"}
             and manifest["schema"] == "splitfusion.live_parity_source_archive.v1", "source archive schema drift")
    _require(manifest["identity"] == descriptor["identity"], "source archive full identity mismatch")
    files = manifest["sha256"]
    names = {"SOURCE_EVIDENCE.json", "rgb_bgra.npy", "radar_world_xyz.npy"}
    _require(isinstance(files, dict) and set(files) == names, "source archive file inventory drift")
    _require({p.name for p in directory.iterdir()} == names | {"MANIFEST.json"}, "unbound/missing source archive file")
    evidence = _json(_read(directory / "SOURCE_EVIDENCE.json", files["SOURCE_EVIDENCE.json"]))
    binding = target["provenance"]["evidence_sha256"]
    _require(_sha(canonical(evidence)) == binding.get("owned_source"), "native source evidence digest differs from target")
    _require(evidence.get("identity") == descriptor["identity"], "source evidence full identity mismatch")
    _require(evidence.get("sources") == source_geometry.verify_sources(), "source geometry pins differ")
    synchronization = evidence.get("source_synchronization", {})
    _require(synchronization.get("identity") == descriptor["identity"], "source synchronization full identity mismatch")
    validate_source_alignment(synchronization)
    population = evidence.get("dynamic_snapshot_evidence", {})
    _require(population.get("frame_id") == target["frame_id"], "population evidence frame mismatch")
    _require(evidence.get("population_evidence_sha256") == binding.get("population")
             == population.get("population_sha256") == _sha(canonical(population.get("population"))),
             "owned population digest mismatch")
    _require(_sha(canonical(evidence.get("conflicts"))) == binding.get("conflicts"),
             "conflict evidence digest mismatch")
    arrays = {}
    for name in ("rgb_bgra.npy", "radar_world_xyz.npy"):
        payload = _read(directory / name, files[name])
        try:
            arrays[name] = np.load(io.BytesIO(payload), allow_pickle=False)
        except Exception as exc:
            raise ReadinessError("invalid native source array") from exc
    rgb, radar = arrays["rgb_bgra.npy"], arrays["radar_world_xyz.npy"]
    _require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8
             and list(rgb.shape) == target["native_shape"] + [4], "native RGB dimensions/dtype mismatch")
    _require([synchronization["sources"]["rgb"]["height"], synchronization["sources"]["rgb"]["width"]]
             == target["native_shape"], "source metadata/native dimensions mismatch")
    _require(isinstance(radar, np.ndarray) and radar.dtype.kind == "f" and radar.ndim == 2
             and radar.shape[1] == 3 and radar.shape[0] <= 1_000_000 and np.isfinite(radar).all(),
             "invalid native radar world points")
    return {"identity": descriptor["identity"], "manifest_sha256": _sha(manifest_raw),
            "sha256": files, "rgb_shape": list(rgb.shape), "radar_shape": list(radar.shape)}


def verify_offline_evidence(*, target_directory: Path,
        target_descriptors: Sequence[Mapping], archive_directory: Path,
        expected_archive_manifest_sha256: str, evaluator_result: Mapping,
        expected_ticket_identities: Sequence[Mapping], source_descriptors: Sequence[Mapping] = (),
        expected_prediction_contract: PredictionContract | None = None) -> dict:
    """Verify bounded saved evidence without services, CUDA, writes or inference.

    ``target_descriptors`` has exact ``identity`` and ``manifest_sha256`` fields.
    ``expected_ticket_identities`` comes from the independently retained ticket
    population, not a population inferred from whatever files survived. Each is
    the full value returned by ``PredictionArchive.identity(ticket)``. The actual
    evaluator ``close()`` result must be supplied, not caller-created gate flags.

    Optional ``source_descriptors`` have exact ``identity``, ``directory`` and
    ``manifest_sha256`` fields for the GT bridge's RGB/radar/source archives.
    Omitting any leaves ``source_archives_complete`` false; no source integrity
    claim is made for missing auxiliary archives.

    Missing/unscored targets are counted, never replaced. Even a clean result
    cannot establish live timing, source completeness, or numerical parity by
    itself, and can never authorize training with the inherited latency model.
    """
    _require(0 < len(expected_ticket_identities) <= MAX_TICKETS, "explicit nonempty bounded ticket population required")
    _require(len(target_descriptors) <= MAX_TICKETS, "target population bound exceeded")
    _require(len(source_descriptors) <= MAX_TICKETS, "source archive population bound exceeded")
    _require(isinstance(evaluator_result, Mapping), "actual evaluator close result required")
    _require(evaluator_result.get("worker_alive") is False, "evaluator worker did not drain")
    _require(evaluator_result.get("parity_archive_complete") is True, "prediction archive incomplete")
    _require(evaluator_result.get("parity_archive_errors") == [], "prediction archive reports errors")
    prediction_contract = contract_or_native(expected_prediction_contract)
    sources = evaluation.verify_sources()
    expected, expected_gt = {}, set()
    for value in expected_ticket_identities:
        identity = _identity(value)
        key, gt_key = PredictionArchive.key(identity), _gt_key(identity["gt_identity"])
        _require(key not in expected and gt_key not in expected_gt, "duplicate expected ticket/frame identity")
        expected[key] = identity
        expected_gt.add(gt_key)

    bundles, verified_targets = {}, []
    for item in target_descriptors:
        _require(isinstance(item, Mapping) and set(item) == {"identity", "manifest_sha256"},
                 "exact complete-target descriptor required")
        identity = target_transport.validate_identity(item["identity"])
        key = _gt_key(identity)
        _require(key not in bundles, "duplicate complete-target identity")
        bundle = target_transport.verify_full_inputs(target_directory, expected_identity=identity,
                    expected_manifest_sha256=_digest(item["manifest_sha256"]))
        _require(bundle.native_inputs_verified and bundle.inputs is not None, "native target inputs unverified")
        _require(bundle.manifest.get("live_qualified") is False, "offline target must not claim live qualification")
        # Do not retain native arrays for 1,000 frames at once. One fully
        # verified frame is sufficient; joins need only its owned metadata.
        bundles[key] = {"frame_id": bundle.prepared.frame_id,
            "content_sha256": bundle.prepared.content_sha256,
            "provenance": bundle.prepared.target["provenance"],
            "native_shape": list(bundle.inputs.raw_semantic.shape)}
        verified_targets.append({"identity": identity, "manifest_sha256": bundle.manifest_sha256,
            "input_content_sha256": bundle.manifest["input_content_sha256"],
            "prepared_content_sha256": bundle.prepared.content_sha256})
        del bundle

    verified_source_archives, source_keys = [], set()
    for descriptor in source_descriptors:
        _require(isinstance(descriptor, Mapping)
                 and set(descriptor) == {"identity", "directory", "manifest_sha256"},
                 "exact source archive descriptor required")
        identity = target_transport.validate_identity(descriptor["identity"])
        key = _gt_key(identity)
        _require(key in bundles and key not in source_keys, "duplicate/foreign source archive identity")
        verified_source_archives.append(_verify_source_archive(descriptor, bundles[key]))
        source_keys.add(key)

    directory = Path(archive_directory)
    _require(directory.is_dir() and not directory.is_symlink(), "archive directory missing or symbolic")
    manifest_raw = _read(directory / "MANIFEST.json", expected_archive_manifest_sha256)
    manifest = _json(manifest_raw)
    _require(isinstance(manifest, dict), "prediction archive manifest must be an object")
    _require(manifest.get("schema") == ARCHIVE_SCHEMA, "wrong prediction archive schema")
    _require(type(manifest.get("frames")) is int and manifest["frames"] == len(expected),
             "archive/expected ticket population mismatch")
    _require(type(manifest.get("evaluated_frames")) is int and 0 <= manifest["evaluated_frames"] <= len(expected),
             "invalid evaluated population")
    _require(manifest.get("prediction_contract") == prediction_contract.document(),
             "archive prediction contract differs from expected calibrated native output")
    budget = manifest.get("memory_budget", {})
    try:
        expected_budget = budget_document(budget["max_frames"], budget["max_bytes"], prediction_contract)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReadinessError("invalid archive memory budget") from exc
    _require(budget == expected_budget and len(expected) <= budget["max_frames"],
             "archive frame/memory budget differs")
    _require(type(manifest.get("owned_bytes")) is int
             and 0 < manifest["owned_bytes"] <= budget["max_bytes"] <= MAX_MEMORY_BYTES,
             "archive owned-memory bound invalid")
    for flag in ("includes_gt_unavailable_tickets", "flush_after_feedback", "quality_and_terminals_unchanged"):
        _require(manifest.get(flag) is True, f"archive contract field missing: {flag}")
    _require(canonical(evaluator_result.get("parity_archive")) == canonical(manifest),
             "durable archive differs from evaluator close result")
    files = manifest.get("sha256")
    _require(isinstance(files, dict) and 3 * len(expected) <= len(files) <= 4 * len(expected),
             "archive file population invalid")
    _require(set(p.name for p in directory.iterdir()) == set(files) | {"MANIFEST.json"},
             "unbound or missing archive files")
    for name, digest in files.items():
        _require(isinstance(name, str) and re.fullmatch(r"[0-9a-f]{64}\.(labels\.npy|records\.json|identity\.json|evaluation\.json)", name),
                 "invalid/path-traversing archive filename")
        _digest(digest)

    used_files, evaluated, unavailable = set(), [], []
    verified_owned_bytes = 0
    for key, identity in expected.items():
        names = {kind: f"{key}.{kind}" for kind in ("identity.json", "records.json", "labels.npy")}
        for name in names.values():
            _require(name in files, "missing expected prediction payload")
        saved_identity = _json(_read(directory / names["identity.json"], files[names["identity.json"]]))
        _require(set(saved_identity) == _ARCHIVE_KEYS and saved_identity["schema"] == ARCHIVE_SCHEMA,
                 "archive identity schema drift")
        _require({k: saved_identity[k] for k in _IDENTITY_KEYS} == identity,
                 "foreign action/session/frame or ticket timestamp")
        _require(type(saved_identity["snapshot_raw_ns"]) is int and saved_identity["snapshot_raw_ns"] > 0,
                 "invalid prediction snapshot raw timestamp")
        records_raw = _read(directory / names["records.json"], files[names["records.json"]])
        records = _json(records_raw)
        _require(isinstance(records, list) and len(records) <= 4096, "invalid prediction record population")
        label_raw = _read(directory / names["labels.npy"], files[names["labels.npy"]])
        try:
            labels = np.load(io.BytesIO(label_raw), allow_pickle=False)
        except Exception as exc:
            raise ReadinessError("invalid owned prediction array") from exc
        try:
            validate_labels(labels, prediction_contract)
        except ValueError as exc:
            raise ReadinessError("invalid owned prediction labels: " + str(exc)) from exc
        verified_owned_bytes += labels.nbytes + len(records_raw) + len(canonical(identity))
        _require(_sha(labels.tobytes(order="C") + records_raw) == saved_identity["prediction_digest"],
                 "owned prediction digest mismatch")
        used_files.update(names.values())
        evaluation_name = f"{key}.evaluation.json"
        gt_key = _gt_key(identity["gt_identity"])
        if evaluation_name not in files:
            unavailable.append({"identity": identity, "complete_target_present": gt_key in bundles,
                                "status": "NOT_SCORED_REASON_NOT_IN_ARCHIVE"})
            continue
        _require(gt_key in bundles, "scored prediction lacks verified complete target; raw-GT fallback forbidden")
        target = bundles[gt_key]
        evaluation_raw = _read(directory / evaluation_name, files[evaluation_name])
        evidence = _json(evaluation_raw)
        verified_owned_bytes += len(evaluation_raw)
        _require(evidence.get("schema") == evaluation.SCHEMA and evidence.get("live_qualified") is False,
                 "evaluation evidence schema/qualification mismatch")
        _require(evidence.get("identity") == identity["action_identity"], "evaluation full action identity mismatch")
        _require(evidence.get("measurement", {}).get("frame_id") == identity["gt_identity"]["frame_id"],
                 "evaluation measurement frame mismatch")
        _require(evidence.get("target_content_sha256") == target["content_sha256"]
                 and evidence.get("target_provenance") == target["provenance"],
                 "evaluation target provenance/content mismatch")
        _require(evidence.get("source_pins") == sources, "evaluation sources differ from current verified sources")
        _require(evidence.get("prediction_records") == records
                 and evidence.get("prediction_records_sha256") == _sha(canonical(records)),
                 "evaluated prediction records differ from owned archive")
        _require(evidence.get("predicted_labels") == {"shape": list(labels.shape), "dtype": str(labels.dtype),
                 "array_sha256": _sha(labels.tobytes(order="C"))}, "evaluated labels differ from owned archive")
        avo = evidence.get("person_avo", {})
        _require(avo.get("frame_id") == target["frame_id"]
                 and all(k in avo for k in ("qualified", "structural_ignored")), "missing/foreign person AVO evidence")
        used_files.add(evaluation_name)
        evaluated.append(identity)
    _require(used_files == set(files), "extra foreign prediction/evaluation payloads")
    _require(len(evaluated) == manifest["evaluated_frames"], "evaluated count differs from durable payloads")
    _require(verified_owned_bytes == manifest["owned_bytes"], "archive owned-byte accounting differs")

    return {"schema": SCHEMA, "status": "OFFLINE_EVIDENCE_VERIFIED__LIVE_NOT_QUALIFIED",
        "prediction_contract": prediction_contract.document(), "memory_budget": budget,
        "verified_owned_bytes": verified_owned_bytes,
        "live_qualified": False, "cuda_qualified": False, "training_authorized": False,
        "inherited_latency_model_approved": False, "raw_gt_fallback_allowed": False,
        "expected_tickets": len(expected), "archived_predictions": len(expected),
        "scored_predictions": len(evaluated), "unscored_predictions": len(unavailable),
        "missing_complete_targets_for_archived_predictions": sum(_gt_key(x["gt_identity"]) not in bundles for x in expected.values()),
        "verified_complete_targets": len(bundles), "unscored_tickets": unavailable,
        "verified_targets": verified_targets, "archive_manifest_sha256": _sha(manifest_raw),
        "verified_source_archives": verified_source_archives,
        "missing_source_archives": len(bundles) - len(source_keys),
        "source_archives_complete": bool(bundles) and source_keys == set(bundles),
        "evaluator_result_sha256": _sha(canonical(evaluator_result)),
        "expected_ticket_population_sha256": _sha(canonical(list(expected_ticket_identities))),
        "verified_source_pins": sources,
        "limitations": ["Hashes establish saved-content integrity, not truthful sensor/population completeness.",
            "RGB/radar sources are verified only when explicitly supplied; catalogue files need their separate archive verification.",
            "This verifier does not replay numerical scoring or establish live accuracy, timing or CUDA parity.",
            "Unscored predictions are retained; their terminal reason must be joined from authoritative runtime evidence.",
            "No actor/state/reward/deadline or training-latency contract is changed or authorized."]}
