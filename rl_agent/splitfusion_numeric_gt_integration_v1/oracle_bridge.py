"""Offline reference-fixture bridge; intentionally not the fast producer path.

The legacy validator runs once when importing a reference. New production code
must build the numerical metadata directly, otherwise this bridge would keep
PNG restoration on readiness. These helpers do not alter the legacy object.
"""
from . import contract as C
from .evidence import require_source, reconcile_source


def from_legacy(*, prepared, source_evidence, policy_identity,
                binding, plan_sha256, history_before_sha256, history_after_sha256):
    from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
    checked = E.restore_prepared_target(frame_id=prepared.frame_id, sample_id=prepared.sample_id,
        target=prepared.target, content_sha256=prepared.content_sha256)
    source = require_source(source_evidence)
    capture_identity = source.avo_document["identity"]
    metadata = dict(schema=C.SCHEMA, target_contract=C.TARGET_CONTRACT, binding=binding,
        capture_identity=capture_identity, policy_identity=policy_identity, plan_sha256=plan_sha256,
        sample_id=checked.sample_id, input_content_sha256=checked.target["provenance"]["input_content_sha256"],
        history_before_sha256=history_before_sha256, history_after_sha256=history_after_sha256,
        provenance=checked.target["provenance"], camera_inverse=source.camera_inverse,
        object_boxes=checked.target["tables"]["object_boxes.csv"],
        object_ignore_regions=checked.target["tables"]["object_ignore_regions.csv"], avo=source.avo_document["avo"],
        artifact_source_sha256=source.content_sha256, avo_document_sha256=source.avo_document_sha256)
    result = C.admit_target(metadata=metadata,
        segmentation=checked.target["segmentation"].tobytes(order="C"),
        object_ignore=checked.target["object_ignore"].tobytes(order="C"),
        expected_binding=binding, expected_capture_identity=capture_identity,
        expected_policy_identity=policy_identity, expected_plan_sha256=plan_sha256)
    reconcile_source(result, source)
    return result


def scorer_inputs(target):
    """Return immutable numerical views; no PNGs, disk reads or mutable exports.

    The external scorer retains original full prediction validation and exact
    metrics. This function does not implement, replace or qualify that scorer.
    """
    target = C.require_admitted(target)
    return dict(identity=target.metadata["policy_identity"], sample_id=target.metadata["sample_id"],
        segmentation=target.segmentation, object_ignore=target.object_ignore,
        object_boxes=target.metadata["object_boxes"], avo=target.metadata["avo"])
