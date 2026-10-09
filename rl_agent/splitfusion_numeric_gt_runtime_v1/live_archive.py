"""Prospective numerical-GT archive values for an observed live frame.

The scorer has already admitted the complete native prediction and numerical
target before this code runs. This archive preserves those values, their source
bindings, the saved matching result and a separate observed clock ledger. It
does not reconstruct the historical AVO_PRE_SCORE_INPUTS document or claim its
pre-score-write clocks. A missing or never-scored ticket needs a separate
terminal evidence path. File durability remains the ArchiveClient's job.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, fields
import hashlib
import io
from types import MappingProxyType

import numpy as np

from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality as QUALITY
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
from rl_agent.splitfusion_minimal_quality_scorer_v1 import scorer as S, shared_admission as A
from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R, scoring as Q
from rl_agent.splitfusion_run5_live_avo_v1 import gt_source_sync as SY

from . import live_clock as CLOCK
from .scorer import _decode_legacy_png


SCHEMA = "splitfusion.numeric_gt.live_archive.v1"
SCORE_SCHEMA = "splitfusion.numeric_gt.live_score.v1"
MAX_FILES = 16
MAX_FILE_BYTES = 16 * 1024 * 1024
LEGACY_FILES = frozenset(("segmentation.png", "object_ignore.png", "object_boxes.csv",
    "object_ignore_regions.csv", "target_manifest.csv"))


class LiveArchiveError(ValueError):
    pass


def _require(ok, message):
    if not ok:
        raise LiveArchiveError(message)


def _canonical(value):
    return C.canonical(value)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _bounded_files(files):
    _require(type(files) is dict and 1 <= len(files) <= MAX_FILES, "bounded archive file inventory required")
    for name, raw in files.items():
        _require(type(name) is str and name and "/" not in name and "\\" not in name and ".." not in name,
                 "flat safe archive file names required")
        _require(type(raw) is bytes and len(raw) <= MAX_FILE_BYTES,
                 "bounded immutable archive file bytes required")
    return files


def _source_values(target, input_metadata, source_synchronization, numerical_avo):
    metadata = target.metadata
    capture = C.thaw(metadata["capture_identity"])
    # The live source export is canonical finite JSON, parsed into primitives
    # by the scorer before arrival here.  The existing edge-safe transport
    # canonicalizer also preserves the original NumPy-scalar JSON conversion
    # used by offline witnesses, without importing producer source geometry.
    inputs = C._parse(T._canonical(input_metadata))
    sync = C._json_owned(source_synchronization)
    avo = C._json_owned(numerical_avo)
    T._validate_metadata(inputs, capture)
    _require(_canonical(inputs["frame"]) == _canonical(metadata["frame"]),
             "complete source frame differs from admitted numerical target")
    _require(_canonical(inputs["provenance"]) == _canonical(metadata["provenance"])
             and inputs["provenance"]["input_content_sha256"] == metadata["input_content_sha256"],
             "complete input provenance/content differs from target")
    C._avo(avo, capture["frame_id"])
    _require(_canonical(avo) == _canonical(metadata["avo"]), "source numerical AVO differs from target")
    _require(sync.get("schema") == SY.SYNC_SCHEMA and sync.get("identity") == capture,
             "source synchronization schema/identity differs")
    SY.validate_source_alignment(sync)
    _require(type(sync.get("observed_raw_ns")) is int and 0 < sync["observed_raw_ns"] < 2**63,
             "source synchronization requires an observed RAW instant")
    rgb = sync["sources"]["rgb"]
    _require(rgb["timestamp"] == metadata["frame"]["timestamp"]
             and (rgb["height"], rgb["width"]) == (720, 1280),
             "source synchronization timestamp/native image geometry differs")
    camera = np.asarray(rgb["camera_matrix"], dtype=np.float64)
    target_camera = np.asarray(metadata["camera_matrix"], dtype=np.float64)
    _require(camera.shape == target_camera.shape == (4, 4)
             and np.isfinite(camera).all()
             and np.allclose(camera, target_camera, rtol=0, atol=SY.POSE_ATOL),
             "synchronized RGB camera differs from admitted frame context")
    return inputs, sync, avo


def _quality_value(result, q_perc, quality_spec):
    _require(type(quality_spec) is QUALITY.RewardSpecV1, "exact registered quality spec required")
    expected = QUALITY.evaluate_exact_quality(quality_spec, result.measurement)
    value = None if expected is None else float(expected.q_perc)
    _require((q_perc is None and value is None) or
             (type(q_perc) is float and value is not None and q_perc == value),
             "archived Q_perc differs from numerical score and registered formula")
    return value


def _scored_values(target, result, q_perc, quality_spec):
    R.require_ready(target)
    _require(type(result) is S.ScoreResult and type(result._frame) is S._AdmittedFrame
             and result._frame.issuer is S._ISSUER,
             "actual admitted numerical scorer result required")
    shared = result._frame.shared_context
    _require(type(shared) is Q._JoinedContext and shared.issuer is Q._JOIN_ISSUER
             and shared.target is target,
             "scored target differs from admitted ready target")
    prediction = shared.prediction
    _require(type(prediction) is A._AdmittedPrediction and prediction.issuer is A._PREDICTION_ISSUER
             and prediction.labels is result._frame.labels,
             "actual admitted native prediction required")
    action = C.thaw(target.metadata["action_identity"])
    capture = C.thaw(target.metadata["capture_identity"])
    _require(C._json_owned(prediction.identity["gt_identity"]) == capture
             and C._json_owned(prediction.identity["action_identity"]) == action
             and result.measurement["frame_id"] == capture["frame_id"],
             "scored prediction/frame/action identity differs")
    _require(prediction.identity["prediction_digest"] ==
             _digest(prediction.labels + prediction.raw_records),
             "native predicted labels/records digest differs")
    value = _quality_value(result, q_perc, quality_spec)
    associations = {item.name: getattr(result.associations, item.name) for item in fields(result.associations)}
    score = dict(schema=SCORE_SCHEMA, ready_identity=C._parse(target.identity_bytes),
        ready_semantic_sha256=target.semantic_sha256, quality_spec_sha256=quality_spec.canonical_sha256(),
        measurement=result.measurement, q_perc=value, quality_defined=value is not None,
        prediction_identity=C.thaw(prediction.identity), prediction_records=C._parse(prediction.raw_records),
        predicted_labels=dict(shape=list(prediction.shape), dtype="uint8", sha256=_digest(prediction.labels),
                              bytes=len(prediction.labels)),
        numerical_matches=associations, confusion_matrix=result.confusion,
        ignored_pixels=result.ignored_pixels, vehicle_detection=dict(result.vehicle_detection))
    encoded = _canonical(score)
    _require(len(encoded) <= C.MAX_METADATA_BYTES, "bounded numerical score evidence required")
    return prediction, encoded


def _legacy_files(target, supplied):
    if supplied is None:
        return {}
    # An explicit historical attachment is optional and outside the live
    # numerical archive path.  Preserve its pinned original verification only
    # when it is actually requested.
    from rl_agent.splitfusion_numeric_gt_producer_v1 import originals as O, payload as P
    _require(type(supplied) is dict and set(supplied) == LEGACY_FILES,
             "optional legacy attachment requires both PNGs and all three tables")
    files = _bounded_files(dict(supplied))
    _require(np.array_equal(_decode_legacy_png(files["segmentation.png"]), target.segmentation)
             and np.array_equal(_decode_legacy_png(files["object_ignore.png"]), target.object_ignore),
             "legacy PNG pixels differ from admitted numerical masks")
    builder = O.bound().builder
    for name, rows, columns in (("object_boxes.csv", target.metadata["object_boxes"], builder.OBJECT_FIELDS),
                                ("object_ignore_regions.csv", target.metadata["object_ignore_regions"],
                                 builder.IGNORE_FIELDS)):
        _require(files[name] == P.legacy_csv_bytes(rows, columns),
                 "legacy target table differs from numerical source: " + name)
    try:
        reader = csv.DictReader(io.StringIO(files["target_manifest.csv"].decode("utf-8"), newline=""))
        _require(reader.fieldnames == ["sample_id", "segmentation_mask_path", "segmentation_mask_sha256",
                    "object_ignore_mask_path", "object_ignore_mask_sha256"],
                 "legacy target manifest header differs")
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise LiveArchiveError("malformed legacy target manifest") from exc
    _require(len(rows) == 1 and rows[0]["sample_id"] == target.metadata["sample_id"]
             and rows[0]["segmentation_mask_sha256"] == _digest(files["segmentation.png"])
             and rows[0]["object_ignore_mask_sha256"] == _digest(files["object_ignore.png"]),
             "legacy manifest does not bind the numerical frame/PNG bytes")
    return files


@dataclass(frozen=True, slots=True)
class LiveArchive:
    files: object
    manifest_bytes: bytes
    manifest_sha256: str
    ready_identity_bytes: bytes
    semantic_sha256: str
    clock_descriptor_bytes: bytes


def build_live_archive(*, ready_target, score_result, q_perc, quality_spec,
                       input_metadata, source_synchronization, numerical_avo,
                       clock_ledger, legacy_artifacts=None) -> LiveArchive:
    """Build bounded evidence for one scored ticket with observed live clocks.

    The returned bytes can be submitted to the existing durable archive child.
    An actual completion receipt must be checked separately before declaring
    archival complete; this builder performs no filesystem work or feedback.
    """
    R.require_ready(ready_target)
    inputs, sync, avo = _source_values(ready_target, input_metadata,
                                      source_synchronization, numerical_avo)
    prediction, score_bytes = _scored_values(ready_target, score_result, q_perc, quality_spec)
    _require(type(clock_ledger) is CLOCK.LiveClockLedger
             and clock_ledger._identity_bytes == ready_target.identity_bytes
             and clock_ledger.semantic_sha256 == ready_target.semantic_sha256,
             "live clock ledger belongs to a different ready identity")
    sync_event = clock_ledger._events.get(("producer", "source_alignment_observed"))
    _require(sync_event is not None and sync_event["raw_ns"] == sync["observed_raw_ns"],
             "source synchronization RAW instant differs from its producer-owned clock event")
    legacy = _legacy_files(ready_target, legacy_artifacts)
    input_bytes, sync_bytes, avo_bytes = map(_canonical, (inputs, sync, avo))
    clock_bytes = clock_ledger.archive_descriptor(avo_sha256=_digest(avo_bytes),
        source_synchronization_sha256=_digest(sync_bytes))
    clock_doc = C._parse(clock_bytes)
    _require(clock_doc.get("schema") == (CLOCK.ASYNC_SCHEMA if
             clock_ledger.handoff_id == CLOCK.ASYNC_HANDOFF_ID else CLOCK.SCHEMA)
             and clock_doc.get("handoff_id") == clock_ledger.handoff_id
             and clock_doc.get("ready_identity_sha256") == _digest(ready_target.identity_bytes)
             and clock_doc.get("semantic_sha256") == ready_target.semantic_sha256,
             "versioned observed live clock descriptor differs")
    files = _bounded_files(dict(
        target_metadata_json=ready_target.metadata_bytes,
        input_metadata_json=input_bytes,
        source_synchronization_json=sync_bytes,
        numerical_avo_json=avo_bytes,
        numerical_score_json=score_bytes,
        live_clock_json=clock_bytes,
        segmentation_u8=ready_target.segmentation_bytes,
        object_ignore_u8=ready_target.object_ignore_bytes,
        predicted_labels_u8=prediction.labels,
    ))
    # Flat conventional filenames in the actual archive, without an old READY
    # claim. The manifest itself is excluded from the inventory to avoid a
    # circular digest.
    names = {key: key.replace("_json", ".json").replace("_u8", ".u8") for key in files}
    renamed = {names[key]: raw for key, raw in files.items()}
    _require(not (set(renamed) & set(legacy)), "legacy attachment collides with numerical archive")
    renamed.update(legacy)
    _bounded_files(renamed)
    inventory = {name: dict(bytes=len(raw), sha256=_digest(raw)) for name, raw in sorted(renamed.items())}
    manifest = dict(schema=SCHEMA, ready_identity=C._parse(ready_target.identity_bytes),
        ready_identity_sha256=_digest(ready_target.identity_bytes),
        ready_semantic_sha256=ready_target.semantic_sha256,
        producer_semantic_sha256=ready_target.metadata["producer_semantic_sha256"],
        input_content_sha256=ready_target.metadata["input_content_sha256"],
        source_provenance_sha256=_digest(_canonical(ready_target.metadata["provenance"])),
        quality_spec_sha256=quality_spec.canonical_sha256(),
        measurement_sha256=_digest(_canonical(score_result.measurement)),
        numerical_avo_sha256=_digest(avo_bytes),
        source_synchronization_sha256=_digest(sync_bytes),
        live_clock_sha256=_digest(clock_bytes),
        clock_semantics="OBSERVED_CLOCK_MONOTONIC_RAW_EVENTS_ONLY_NO_LEGACY_PRESCORE_CLAIM",
        q_perc=q_perc, quality_defined=q_perc is not None,
        deadline=clock_doc["deadline"],
        terminal_coverage="SCORED_TICKETS_ONLY", controller_terminal_claim=False,
        files=inventory, historical_avo_document=False, old_prepared_digest_claim=False,
        ready_manifest_claim=False)
    raw = _canonical(manifest)
    _require(len(raw) <= C.MAX_METADATA_BYTES and "live_archive.json" not in renamed,
             "bounded create-only live archive manifest required")
    renamed["live_archive.json"] = raw
    return LiveArchive(MappingProxyType(renamed), raw, _digest(raw), ready_target.identity_bytes,
                       ready_target.semantic_sha256, clock_bytes)


def verify_live_archive(archive: LiveArchive) -> dict:
    """Rehash a candidate byte inventory; durable receipt remains external."""
    _require(type(archive) is LiveArchive and type(archive.files) is MappingProxyType,
             "owned immutable live archive required")
    files = dict(archive.files)
    _bounded_files(files)
    _require(files.get("live_archive.json") == archive.manifest_bytes
             and _digest(archive.manifest_bytes) == archive.manifest_sha256,
             "live archive manifest bytes differ")
    document = C._parse(archive.manifest_bytes)
    _require(document.get("schema") == SCHEMA
             and _canonical(document["ready_identity"]) == archive.ready_identity_bytes
             and document.get("ready_semantic_sha256") == archive.semantic_sha256
             and document.get("live_clock_sha256") == _digest(archive.clock_descriptor_bytes)
             and files.get("live_clock.json") == archive.clock_descriptor_bytes,
             "live archive identity, semantic or clock differs")
    expected = {name: dict(bytes=len(raw), sha256=_digest(raw)) for name, raw in files.items()
                if name != "live_archive.json"}
    _require(document.get("files") == expected, "live archive file inventory/hash differs")
    return dict(schema=SCHEMA, files=len(files), archive_sha256=archive.manifest_sha256,
                ready_identity_sha256=_digest(archive.ready_identity_bytes),
                clock_sha256=_digest(archive.clock_descriptor_bytes),
                quality_defined=document["quality_defined"])
