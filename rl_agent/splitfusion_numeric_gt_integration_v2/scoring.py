"""Archive-free v2 admission to the unchanged, source-pinned numerical scorer.

``admit_prediction`` is the original complete prediction boundary, not a
reduced record validator. ``join_admitted_inputs`` needs only the private
ReadyTarget and that admitted prediction. Its result is scored by the original
NumericalScorer without PNG, source-file, JSON or archive-document work.

After scoring, ``attach_archive`` binds producer-created legacy evidence to
the numerical target. ``reconstruct_evidence`` and ``archive_payloads`` adapt a
COPY of the score result to the original deferred functions: saved matching
associations are reused, never recomputed. An attachment failure must be an
explicit archive fault, not a replacement score or a retroactive timeout.

The deferred attachment requires the actual prepared legacy target, not merely
its advertised digest. The frozen validator checks its PNG/array/table content
and old digest after numerical scoring; byte parity with retained artifacts is
also covered by the composed proof.
"""
from dataclasses import dataclass, fields, replace
import json
from pathlib import Path
from types import FunctionType

from rl_agent.splitfusion_minimal_quality_scorer_v1 import scorer as S
from rl_agent.splitfusion_minimal_quality_scorer_v1 import shared_admission as P
from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C, evidence as V

admit_prediction = P.admit_prediction
NumericalScorer = S.NumericalScorer
_JOIN_ISSUER = object()
_ARCHIVE_ISSUER = object()
MAX_ARCHIVE_BINDING_BYTES = 2 * C.MAX_METADATA_BYTES


@dataclass(frozen=True, slots=True)
class _JoinedContext:
    issuer: object
    target: object
    prediction: object


def join_admitted_inputs(scorer, *, target, prediction):
    """Join full capture/action/calibration identities; no archive prerequisite."""
    from .ready import require_ready
    C.require(type(scorer) is S.NumericalScorer, "exact scorer context required")
    target = require_ready(target)
    C.require(type(prediction) is P._AdmittedPrediction
              and prediction.issuer is P._PREDICTION_ISSUER
              and prediction.owner is scorer._owner,
              "private immutable prediction from this scorer required")
    meta = target.metadata
    capture, action = prediction.identity["gt_identity"], prediction.identity["action_identity"]
    C.require(capture == meta["capture_identity"] and action == meta["action_identity"],
              "complete capture/action identity differs")
    C.require(C._freeze(prediction.contract.document()) == meta["prediction_contract"],
              "prediction calibration/grid differs from admitted frame context")
    namespace = dict(scorer._detection, read_csv=lambda _: meta["object_boxes"])
    grouped, _ = FunctionType(namespace["load_gt"].__code__, namespace)(Path("/offline"), "v010")
    gt = grouped.get(meta["sample_id"], [])
    avo = meta["avo"]
    eligible_indices = tuple(i for i, row in enumerate(avo["qualified"]) if float(row["avo"]) >= .65)
    ignored_indices = tuple(i for i, row in enumerate(avo["qualified"]) if float(row["avo"]) < .65)

    def actors(rows):
        return S._frozen_rows({k: row[k] for k in ("world_x", "world_y", "gt_actor_id")} for row in rows)

    return S._issue_frame(S._ISSUER, scorer._owner, capture["frame_id"], target.segmentation_bytes,
        target.object_ignore_bytes, prediction.labels, prediction.shape, prediction.records, S._frozen_rows(gt),
        prediction.detection_indices, prediction.person_indices,
        actors(avo["qualified"][i] for i in eligible_indices), actors(avo["qualified"][i] for i in ignored_indices),
        actors(avo["structural_ignored"]), eligible_indices, ignored_indices,
        b"", scorer._source_pins, None, _JoinedContext(_JOIN_ISSUER, target, prediction))


@dataclass(frozen=True, slots=True, init=False)
class ArchiveReference:
    """Factory-issued, deeply immutable attachment; not an admitted target."""
    source: object
    target_tables: object
    old_prepared_sha256: str
    semantic_sha256: str
    identity_bytes: bytes
    binding_sha256: str
    owned_bytes: int
    _issuer: object

    def __init__(self, *args, **kwargs):
        raise C.ContractError("use attach_archive")


def require_archive_reference(value):
    C.require(type(value) is ArchiveReference and value._issuer is _ARCHIVE_ISSUER,
              "private v2 archive attachment required")
    return value


def _same(left, right, message):
    C.require(C.canonical(left) == C.canonical(right), message)


def attach_archive(target, *, source_evidence, target_tables, old_prepared_sha256, prepared_target):
    """Validate deferred full source/AVO document and ordered original tables.

    SourceEvidence must already have passed the original document validators.
    It contains the actual source frame, provenance, camera and complete AVO
    document. We additionally reconcile all those numerical fields against the
    admitted v2 target; an earlier numerical score need not retain this object.
    """
    from .ready import require_ready
    from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
    target = require_ready(target)
    source = V.require_source(source_evidence)
    meta, inputs, document = target.metadata, source.input_metadata, source.avo_document
    for left, right, description in (
        (meta["capture_identity"], document["identity"], "capture"),
        (meta["capture_identity"], inputs["provenance"]["capture_identity"], "source capture"),
        (meta["frame"], inputs["frame"], "complete source frame"),
        (meta["provenance"], inputs["provenance"], "complete provenance"),
        (meta["camera_inverse"], source.camera_inverse, "source camera pose"),
        (meta["avo"], document["avo"], "ordered numeric AVO"),
        (meta["camera_matrix"], document.get("camera_matrix"), "AVO camera pose"),
        (meta["camera_inverse"], document.get("camera_inverse"), "AVO inverse pose"),
        (meta["intrinsics"], document.get("intrinsics"), "AVO intrinsics")):
        _same(left, right, "archive/ready " + description + " differs")
    C.require(meta["input_content_sha256"] == inputs["provenance"]["input_content_sha256"]
              and meta["sample_id"] == inputs["frame"]["sample_id"],
              "archive source input digest/sample differs")
    tables = C.thaw(target_tables)
    C.require(type(tables) is dict and set(tables) == {
        "object_boxes.csv", "object_ignore_regions.csv", "target_manifest.csv"},
        "full original target tables required")
    _same(meta["object_boxes"], tables["object_boxes.csv"], "archive ordered boxes differ")
    _same(meta["object_ignore_regions"], tables["object_ignore_regions.csv"], "archive ordered ignore rows differ")
    manifest = tables["target_manifest.csv"]
    C.require(type(manifest) is list and len(manifest) == 1 and type(manifest[0]) is dict
              and manifest[0].get("sample_id") == meta["sample_id"], "foreign legacy target manifest")
    C.require(set(manifest[0]) == {"sample_id", "segmentation_mask_path", "segmentation_mask_sha256",
                                  "object_ignore_mask_path", "object_ignore_mask_sha256"},
              "legacy target manifest fields differ")
    for name in ("segmentation_mask", "object_ignore_mask"):
        path, digest = manifest[0][name + "_path"], manifest[0][name + "_sha256"]
        C.require(type(path) is str and bool(path) and not Path(path).is_absolute()
                  and ".." not in Path(path).parts and C.is_digest(digest),
                  "invalid legacy target manifest path/digest")
    C.require(C.is_digest(old_prepared_sha256), "original prepared digest required")
    tables_bytes = C.canonical(tables)
    C.require(len(tables_bytes) <= C.MAX_METADATA_BYTES, "archive target table bound exceeded")
    C.require(type(prepared_target) is E.PreparedEvaluationTarget
              and prepared_target.frame_id == meta["capture_identity"]["frame_id"]
              and prepared_target.sample_id == meta["sample_id"]
              and prepared_target.content_sha256 == old_prepared_sha256,
              "actual prepared legacy target/digest differs")
    for name in ("segmentation", "object_ignore"):
        encoded = prepared_target.target.get(name + "_png")
        C.require(type(encoded) is bytes and 0 < len(encoded) <= C.MAX_METADATA_BYTES,
                  "bounded actual legacy PNG required")
        C.require(C.sha256(encoded) == manifest[0][name + "_mask_sha256"],
                  "legacy manifest PNG hash differs from actual bytes")
    checked = E.restore_prepared_target(frame_id=prepared_target.frame_id,
        sample_id=prepared_target.sample_id, target=prepared_target.target,
        content_sha256=old_prepared_sha256)
    _same(checked.target["tables"], tables, "actual prepared legacy tables differ")
    _same(checked.target["provenance"], meta["provenance"], "actual prepared provenance differs")
    C.require(checked.target["segmentation"].tobytes(order="C") == target.segmentation_bytes
              and checked.target["object_ignore"].tobytes(order="C") == target.object_ignore_bytes,
              "actual legacy mask bytes differ from ready target")
    owned_bytes = len(source.canonical_bytes) + len(tables_bytes)
    C.require(owned_bytes <= MAX_ARCHIVE_BINDING_BYTES, "archive attachment bound exceeded")
    binding = C.sha256(C.canonical(dict(schema="splitfusion.numeric_gt.archive_attachment.v2",
        semantic_sha256=target.semantic_sha256, source_sha256=source.content_sha256,
        avo_document_sha256=source.avo_document_sha256, target_tables_sha256=C.sha256(tables_bytes),
        original_prepared_sha256=old_prepared_sha256)))
    value = object.__new__(ArchiveReference)
    members = dict(source=source, target_tables=C._freeze(tables), old_prepared_sha256=old_prepared_sha256,
        semantic_sha256=target.semantic_sha256, identity_bytes=target.identity_bytes, binding_sha256=binding,
        owned_bytes=owned_bytes, _issuer=_ARCHIVE_ISSUER)
    for field in fields(ArchiveReference):
        object.__setattr__(value, field.name, members[field.name])
    return value


def bind_archive_result(result, *, archive_reference):
    """Copy a v2 result for the unchanged deferred scorer, without rematching."""
    C.require(type(result) is S.ScoreResult, "original numerical score result required")
    frame = result._frame
    C.require(type(frame) is S._AdmittedFrame and frame.issuer is S._ISSUER
              and type(frame.shared_context) is _JoinedContext
              and frame.shared_context.issuer is _JOIN_ISSUER,
              "v2 joined score result required")
    context = frame.shared_context
    reference = require_archive_reference(archive_reference)
    C.require(reference.semantic_sha256 == context.target.semantic_sha256
              and reference.identity_bytes == context.target.identity_bytes,
              "archive attachment belongs to a different ready target")
    C.require(frame.owner is context.prediction.owner, "score prediction owner differs")
    evidence_inputs = S.canonical(dict(
        action_identity=C.thaw(context.prediction.identity["action_identity"]),
        records=json.loads(context.prediction.raw_records),
        target=dict(provenance=C.thaw(context.target.metadata["provenance"])),
        avo_document=C.thaw(reference.source.avo_document)))
    C.require(len(evidence_inputs) <= C.MAX_METADATA_BYTES, "deferred scorer evidence bound exceeded")
    changes = dict(evidence_inputs=evidence_inputs, original_content_sha256=reference.old_prepared_sha256,
                   shared_context=None)
    copied = S._issue_frame(*(changes.get(field.name, getattr(frame, field.name)) for field in fields(S._AdmittedFrame)))
    return replace(result, _frame=copied)


def reconstruct_evidence(result, *, archive_reference):
    """Original evidence from saved associations, after validated attachment."""
    return S.reconstruct_evidence(bind_archive_result(result, archive_reference=archive_reference))


def archive_payloads(result, *, archive_reference):
    """Original PNG/evaluation bytes; errors remain explicit archive failures."""
    return S.archive_payloads(bind_archive_result(result, archive_reference=archive_reference))
