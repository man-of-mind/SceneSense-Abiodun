"""Exact archive bytes and explicit pending receipts, never a fake disk READY.

Numerical validation still precedes scoring. Serialization happens after the
scoring packet is admitted; a separate process owns all file writes/fsyncs.
Raw-source serialization CPU is measured, not presented as eliminated work.
"""
import hashlib
import json
from pathlib import Path

from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T


class FileBatch:
    def __init__(self, roots, max_bytes=64*1024*1024):
        self.roots = tuple(Path(p).resolve(strict=True) for p in roots)
        self.max_bytes = max_bytes
        self.files = {}
        self.bytes = 0

    def add(self, path, payload):
        path = Path(path)
        if not path.is_absolute() or path.is_symlink():
            raise ValueError("absolute nonsymbolic evidence path required")
        owned = path.resolve(strict=False)
        if path != owned or not any(owned != r and owned.is_relative_to(r) for r in self.roots):
            raise ValueError("evidence path escapes owned roots")
        if type(payload) is not bytes or len(payload) > T.MAX_PAYLOAD_BYTES:
            raise ValueError("bounded immutable evidence bytes required")
        name = str(owned)
        if name in self.files:
            if self.files[name] != payload:
                raise ValueError("conflicting evidence write")
            return
        if self.bytes + len(payload) > self.max_bytes:
            raise ValueError("per-frame deferred evidence limit exceeded")
        self.files[name] = payload
        self.bytes += len(payload)


def target_files(batch, directory, *, identity, inputs, prepared):
    """Same target payloads as frozen write_bundle; durable clocks set by writer.

Caller has already executed the original complete-input and prepared-target
validators before handoff. The zero clocks are typed unpublished sentinels,
never sent to an evaluator or accepted by the live packet decoder.
"""
    destination = T.bundle_path(directory, identity)
    owned = prepared
    metadata = {k: v for k, v in owned.target.items() if k not in
        ("segmentation", "object_ignore", "segmentation_png", "object_ignore_png")}
    payloads = {"inputs.json": T._canonical(T._input_metadata(inputs)),
        "depth_bgra.npy": T._array_bytes(inputs.depth_bgra),
        "raw_semantic.npy": T._array_bytes(inputs.raw_semantic),
        "original_v3_mask.npy": T._array_bytes(inputs.original_v3_mask),
        "target.json": T._canonical(metadata),
        "segmentation.png": owned.target["segmentation_png"],
        "object_ignore.png": owned.target["object_ignore_png"]}
    for name, payload in payloads.items():
        batch.add(destination/name, payload)
    manifest = dict(schema=T.SCHEMA, identity=dict(identity), frame_id=owned.frame_id,
        sample_id=owned.sample_id, input_content_sha256=inputs.provenance["input_content_sha256"],
        prepared_content_sha256=owned.content_sha256, scope=T.LIVE_SCOPE, live_qualified=False,
        clock_domain="CLOCK_MONOTONIC_RAW", ready_raw_ns=0, ready_wall_ns=0,
        payloads={name: dict(filename=name, sha256=hashlib.sha256(p).hexdigest(), bytes=len(p))
                  for name, p in payloads.items()})
    ready = destination/T.READY_NAME
    batch.add(ready, T._canonical(manifest))
    return dict(directory=str(destination), manifest_sha256=None,
        frame_id=owned.frame_id, prepared_content_sha256=owned.content_sha256,
        input_content_sha256=inputs.provenance["input_content_sha256"],
        archive_pending=True), str(ready)


def resolve_records(records, completions):
    """Only after drain: bind authentic disk manifests, never fabricate success."""
    by_job = {r["job_id"]: r for r in completions}
    if len(by_job) != len(completions):
        raise ValueError("duplicate archive completion")
    for record in records:
        pending = record.get("archive_receipt")
        if not pending:
            raise ValueError("missing pending archive receipt")
        result = by_job.get(pending["job_id"])
        if result is None or result.get("status") != "COMPLETE":
            raise ValueError("archive request did not commit")
        if result.get("identity", {}).get("frame_id") != record["frame_id"]:
            raise ValueError("archive completion belongs to a different frame")
        descriptor = record.get("complete_target")
        if descriptor is not None:
            ready = Path(descriptor["directory"])/T.READY_NAME
            payload = ready.read_bytes()
            manifest = json.loads(payload)
            handoff = record["scoring_handoff"]
            if (manifest["identity"] != record["complete_identity"]
                    or handoff["identity"] != record["complete_identity"]
                    or manifest["prepared_content_sha256"] != descriptor["prepared_content_sha256"]
                    or manifest["prepared_content_sha256"] != handoff["prepared_content_sha256"]
                    or manifest["input_content_sha256"] != descriptor["input_content_sha256"]
                    or manifest["input_content_sha256"] != handoff["input_content_sha256"]):
                raise ValueError("archived target identity/content differs from scoring packet")
            digest = hashlib.sha256(payload).hexdigest()
            finalized = [r for r in result["finalized_ready"] if r["path"] == str(ready)]
            if (len(finalized) != 1 or finalized[0]["sha256"] != digest
                    or result["identity"] != record["complete_identity"]):
                raise ValueError("durable receipt does not bind this READY manifest")
            T.verify_full_inputs(ready.parent.parent, expected_identity=record["complete_identity"],
                                 expected_manifest_sha256=digest)
            descriptor["manifest_sha256"] = digest
            descriptor.pop("archive_pending", None)
        record["archive_receipt"] = {**pending, "durable": True, "completion": result}
    return records
