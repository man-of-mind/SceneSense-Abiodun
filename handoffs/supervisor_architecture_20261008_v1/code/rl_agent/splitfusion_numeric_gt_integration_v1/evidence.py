"""Immutable source/evidence ownership separate from numerical scoring.

This validates full document identity and binds the numerical AVO/pose back to
its original metadata. It does not independently prove collector completeness
or re-read raw sensor arrays. Those are qualified producer responsibilities.
No file I/O, PNG encoding or legacy target building occurs here.
"""
from dataclasses import dataclass
import json

from . import contract as C

SOURCE_SCHEMA = "splitfusion.numeric_gt.source_evidence.v1"
_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class SourceEvidence:
    canonical_bytes: bytes
    content_sha256: str
    input_metadata: object
    avo_document: object
    avo_document_sha256: str
    camera_inverse: tuple
    _issuer: object

    def __init__(self, *args, **kwargs):
        raise C.ContractError("use admit_source_evidence")


def require_source(value):
    C.require(type(value) is SourceEvidence and value._issuer is _ISSUER, "admitted source evidence required")
    return value


def admit_source_evidence(*, input_metadata, avo_document, expected_capture_identity):
    # Reuse the pure original document validators at this one actual boundary.
    from rl_agent.splitfusion_live_gt_handoff_v1 import scoring_packet as P
    inputs = C.thaw(input_metadata)
    document = C.thaw(avo_document)
    identity = C.thaw(expected_capture_identity)
    C._capture(identity)
    P.T._validate_metadata(inputs, identity)
    C._capture(inputs["provenance"]["capture_identity"])
    C._capture(document.get("identity"))
    P._avo(document, identity)
    C._avo(document["avo"], identity["frame_id"])
    try:
        raw_camera = inputs["frame"]["camera_inverse_matrix_json"]
        C.require(type(raw_camera) is str, "original camera inverse text required")
        camera = C._parse(raw_camera.encode())
    except (KeyError, UnicodeError) as exc:
        raise C.ContractError("owned source camera inverse missing") from exc
    C.require(type(camera) is list and len(camera) == 4 and all(type(row) is list and len(row) == 4
        and all(type(x) in (int, float) and C._finite_numeric(x) for x in row) for row in camera),
        "source camera inverse malformed")
    encoded = C.canonical(dict(schema=SOURCE_SCHEMA, input_metadata=inputs, avo_document=document))
    C.require(len(encoded) <= C.MAX_METADATA_BYTES, "source evidence metadata bound exceeded")
    result = object.__new__(SourceEvidence)
    for name, value in dict(canonical_bytes=encoded, content_sha256=C.sha256(encoded),
        input_metadata=C._freeze(inputs), avo_document=C._freeze(document),
        avo_document_sha256=C.sha256(C.canonical(document)), camera_inverse=C._freeze(camera), _issuer=_ISSUER).items():
        object.__setattr__(result, name, value)
    return result


def reconcile_source(target, source):
    target = C.require_admitted(target)
    source = require_source(source)
    meta = target.metadata
    checks = ((meta["capture_identity"], source.avo_document["identity"]),
              (meta["provenance"], source.input_metadata["provenance"]),
              (meta["avo"], source.avo_document["avo"]),
              (meta["camera_inverse"], source.camera_inverse))
    C.require(all(C.canonical(a) == C.canonical(b) for a, b in checks), "numerical/source evidence differs")
    C.require(meta["artifact_source_sha256"] == source.content_sha256
        and meta["avo_document_sha256"] == source.avo_document_sha256
        and meta["sample_id"] == source.input_metadata["frame"]["sample_id"]
        and meta["input_content_sha256"] == source.input_metadata["provenance"]["input_content_sha256"],
        "numerical/source evidence digest differs")
    return source


@dataclass(frozen=True, slots=True)
class LegacyArtifactReference:
    """Offline/reference ownership, not a live target-readiness prerequisite."""
    source: SourceEvidence
    target_tables: object
    old_prepared_sha256: str
    semantic_sha256: str


def own_legacy_reference(target, source, *, target_tables, old_prepared_sha256):
    reconcile_source(target, source)
    tables = C.thaw(target_tables)
    C.require(type(tables) is dict and set(tables) == {
        "object_boxes.csv", "object_ignore_regions.csv", "target_manifest.csv"}, "full original target tables required")
    C.require(C.canonical(tables["object_boxes.csv"]) == C.canonical(target.metadata["object_boxes"])
        and C.canonical(tables["object_ignore_regions.csv"]) == C.canonical(target.metadata["object_ignore_regions"]),
        "original/numerical target table differs")
    sample = target.metadata["sample_id"]
    C.require(type(tables["target_manifest.csv"]) is list and len(tables["target_manifest.csv"]) == 1
        and tables["target_manifest.csv"][0].get("sample_id") == sample, "foreign legacy manifest")
    C.require(C.is_digest(old_prepared_sha256), "original prepared digest required")
    return LegacyArtifactReference(source, C._freeze(tables), old_prepared_sha256, target.semantic_sha256)
