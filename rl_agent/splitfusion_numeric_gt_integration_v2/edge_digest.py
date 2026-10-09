"""Producer-payload semantic wire identity without producer-only imports.

This is the exact byte formula of numeric_gt_producer_v1.payload.semantic_digest.
The producer module itself also loads source geometry (and SciPy), which is
unavailable in the frozen edge image and unnecessary for wire admission.
"""
import hashlib
import struct


PAYLOAD_SCHEMA = "splitfusion.numeric_gt_producer.payload.v1"


def semantic_digest(metadata_bytes: bytes, segmentation: bytes, object_ignore: bytes) -> str:
    digest = hashlib.sha256(PAYLOAD_SCHEMA.encode() + b"\0" + struct.pack("!Q", len(metadata_bytes)))
    digest.update(metadata_bytes)
    digest.update(segmentation)
    digest.update(object_ignore)
    return digest.hexdigest()
