"""Small adapter around the sealed V4 exact source-snapshot codec.

No pickle, filesystem writes, simulator RPC, CUDA or geometry substitutions.
The unchanged snapshot codec transports native actor matrices AND inverses,
and restores local CARLA value constructors with its strict no-rounding check.
Population extras are primitive JSON and bound to the exact snapshot envelope.

This codec is not publisher authentication or a scheduler. A readonly view is
not proof of an immutable owner: for mapped input the caller MUST first verify
kernel seals using optimization_v4.memfd_transport.mapped and retain that mapped
owner until decoded views have been relinquished or copied into worker-owned
staging. Bytes inputs own their immutable storage directly.
"""
from __future__ import annotations

import json
import math
import mmap
import struct

from rl_agent.splitfusion_live_gt_optimization_v4 import snapshot as S
from rl_agent.splitfusion_live_gt_optimization_v4 import memfd_transport as M
from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C

SCHEMA = "splitfusion.numeric_gt.runtime.source_extras.v1"
MAX_EXTRAS_BYTES = 2 * 1024 * 1024
SOURCE_KEYS = frozenset(("identity", "rgb", "semantic_image", "depth_image", "frozen_world",
    "camera_matrix", "camera_inverse", "radar_world_xyz", "dynamic_snapshot_evidence"))
SNAPSHOT_KEYS = frozenset(("identity", "rgb", "semantic_image", "depth_image", "frozen_world",
    "camera_matrix", "camera_inverse", "policy_timestamp_s"))
EXTRA_KEYS = frozenset(("schema", "snapshot_body_sha256", "snapshot_bytes", "capture_identity",
                        "dynamic_snapshot_evidence"))
SourceCodecError = C.ContractError


def _timestamp(value):
    C.require(type(value) in (int, float) and math.isfinite(value), "finite policy timestamp required")
    return float(value)


def _own_extras(value):
    owned = C._json_owned(value)
    C.require(type(owned) is dict and set(owned) == EXTRA_KEYS and owned["schema"] == SCHEMA,
              "source extras schema/fields differ")
    C.require(len(C.canonical(owned)) <= MAX_EXTRAS_BYTES, "source extras exceed bound")
    C.require(C.is_digest(owned["snapshot_body_sha256"]), "snapshot digest missing")
    C.require(type(owned["snapshot_bytes"]) is int and S.HEADER_SIZE < owned["snapshot_bytes"] <= M.MAX_BYTES,
              "snapshot descriptor byte bound differs")
    C._capture(owned["capture_identity"])
    population = owned["dynamic_snapshot_evidence"]
    C.require(type(population) is dict and type(population.get("frame_id")) is int
              and population["frame_id"] == owned["capture_identity"]["frame_id"],
              "dynamic population/capture frame differs")
    for key in ("registry_complete", "owned_snapshot_complete", "no_unregistered_spawn"):
        C.require(population.get(key) is True, "incomplete dynamic population: " + key)
    for key in ("active_population_ids", "registered_population_ids"):
        ids = population.get(key)
        C.require(type(ids) is list and len(ids) <= S.DEFAULT_LIMITS.max_actors
                  and all(type(i) is int and 0 <= i < 2**63 for i in ids)
                  and len(set(ids)) == len(ids), "invalid dynamic population IDs")
    C.require(set(population["active_population_ids"]) <= set(population["registered_population_ids"]),
              "active actor absent from registered population")
    C.require(C.is_digest(population.get("population_sha256")), "dynamic population binding missing")
    return owned


def encode_source(kwargs, policy_timestamp_s):
    """Return immutable snapshot chunks and owned primitive population extras.

    The numeric producer has no policy_timestamp argument; it validates the RGB
    timestamp. A common caller may retain that redundant field only when it
    agrees exactly with the explicit policy timestamp passed here.
    """
    C.require(type(kwargs) is dict and SOURCE_KEYS <= set(kwargs)
              and set(kwargs) <= SOURCE_KEYS | {"policy_timestamp_s"}, "numeric source keyword inventory differs")
    stamp = _timestamp(policy_timestamp_s)
    if "policy_timestamp_s" in kwargs:
        C.require(_timestamp(kwargs["policy_timestamp_s"]).hex() == stamp.hex(), "conflicting policy timestamp")
    C._capture(C._json_owned(kwargs["identity"]))
    population = C._json_owned(kwargs["dynamic_snapshot_evidence"])
    C.require(C.canonical(population) == C.canonical(kwargs["frozen_world"].population_evidence),
              "source/world dynamic population differs")
    snap = {key: kwargs[key] for key in SNAPSHOT_KEYS - {"policy_timestamp_s"}}
    snap["policy_timestamp_s"] = stamp
    chunks = S.pack_chunks(snap, kwargs["radar_world_xyz"])
    size = sum(len(chunk) for chunk in chunks)
    C.require(0 < size <= M.MAX_BYTES and 0 < len(chunks) <= 64, "source exceeds sealed memfd bound")
    C.require(type(chunks[0]) is bytes and len(chunks[0]) == S.HEADER_SIZE
              and chunks[0][:8] == S.MAGIC, "snapshot header differs")
    metadata = C._parse(chunks[1])
    C.require(C.canonical(population) == C.canonical(metadata["world"]["population_evidence"]),
              "population changed while snapshot was owned")
    extras = _own_extras(dict(schema=SCHEMA, snapshot_body_sha256=chunks[0][16:48].hex(),
        snapshot_bytes=size, capture_identity=metadata["identity"], dynamic_snapshot_evidence=population))
    return chunks, extras


def decode_source(payload, extras):
    """Decode validated snapshot + population extras into exact producer kwargs.

    Caller owns verified sealed mmap lifetime. This routine deliberately does
    not invent unavailable actor pose components or recompute native inverses.
    """
    C.require(type(payload) in (bytes, memoryview), "immutable bytes or verified mapped view required")
    view = memoryview(payload)
    C.require(view.readonly and view.c_contiguous, "source view must be readonly and contiguous")
    owner = view.obj
    while type(owner) is memoryview:
        owner = owner.obj
    C.require(type(owner) is bytes or isinstance(owner, mmap.mmap),
              "readonly view over mutable/unverified buffer is forbidden")
    view = view.cast("B")
    owned = _own_extras(extras)
    C.require(len(view) == owned["snapshot_bytes"] and view[:8] == S.MAGIC,
              "source/extras packet size or magic differs")
    C.require(bytes(view[16:48]).hex() == owned["snapshot_body_sha256"],
              "source/extras snapshot binding differs")
    decoded, radar = S.unpack_snapshot(view)
    C.require(C.canonical(decoded["identity"]) == C.canonical(owned["capture_identity"]),
              "source/extras capture differs")
    C.require(C.canonical(decoded["frozen_world"].population_evidence)
              == C.canonical(owned["dynamic_snapshot_evidence"]), "source/extras population differs")
    stamp = decoded.pop("policy_timestamp_s")
    C.require(_timestamp(stamp).hex() == _timestamp(decoded["rgb"]["timestamp"]).hex(),
              "decoded source policy/RGB timestamp differs")
    decoded.update(radar_world_xyz=radar, dynamic_snapshot_evidence=owned["dynamic_snapshot_evidence"])
    C.require(set(decoded) == SOURCE_KEYS, "decoded producer keyword inventory differs")
    return decoded
