"""Bounded primitive messages over inherited private Unix sockets + sealed memfd.

No pickle, named listener, ambient file lookup or object-capability restoration.
Each endpoint owns its inherited socket; source binding, parent/child PID and
per-launch nonce are checked by process.py. SO_PEERCRED on socketpair records
the creator, not post-exec child PID; we do NOT claim otherwise.
"""
import hashlib
import json
import math
import os
import struct
import time
from collections.abc import Mapping

from rl_agent.splitfusion_live_gt_optimization_v4 import memfd_transport as M, worker as W

MAX_BYTES = M.MAX_BYTES
MAX_META = 8 * 1024 * 1024
MAX_BUFFERS = 62  # memfd transport accepts 64 chunks, including header + metadata.
HEADER = struct.Struct("!8sQ32s")
MAGIC = b"SFNUMIPC"


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def encode(value):
    buffers, total, nodes = [], 0, 0
    def node(item, depth=0):
        nonlocal total, nodes
        nodes += 1
        require(depth <= 40 and nodes <= 200000, "IPC nesting/node bound")
        if type(item) is bytes:
            require(len(buffers) < MAX_BUFFERS, "IPC buffer-count bound")
            index = len(buffers)
            total += len(item)
            require(total <= MAX_BYTES, "IPC binary byte bound")
            buffers.append(item)
            return ["bytes", index, len(item)]
        if item is None or type(item) in (bool, int, float, str):
            require(type(item) is not float or math.isfinite(item), "IPC nonfinite value")
            require(type(item) is not str or len(item) <= MAX_META, "IPC string bound")
            require(type(item) is not int or -(2 ** 63) <= item < 2 ** 64, "IPC integer bound")
            return ["value", item]
        if isinstance(item, Mapping):
            require(len(item) <= 100000 and all(type(k) is str for k in item), "IPC dictionary key/bound")
            return ["dict", [[k, node(v, depth + 1)] for k, v in item.items()]]
        if type(item) in (tuple, list):
            require(len(item) <= 100000, "IPC sequence bound")
            return ["list", [node(v, depth + 1) for v in item]]
        raise ValueError("IPC nonprimitive type: " + type(item).__name__)
    metadata = canonical(node(value))
    require(0 < len(metadata) <= MAX_META and HEADER.size + len(metadata) + total <= MAX_BYTES,
            "IPC complete packet bound")
    digest = hashlib.sha256(metadata)
    for buffer in buffers:
        digest.update(buffer)
    return [HEADER.pack(MAGIC, len(metadata), digest.digest()), metadata, *buffers]


def decode(payload):
    require(HEADER.size < len(payload) <= MAX_BYTES, "IPC packet bound")
    magic, size, digest = HEADER.unpack_from(payload)
    require(magic == MAGIC and 0 < size <= MAX_META and HEADER.size + size <= len(payload), "IPC header")
    require(hashlib.sha256(payload[HEADER.size:]).digest() == digest, "IPC checksum")
    tree = json.loads(bytes(payload[HEADER.size:HEADER.size + size]),
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("IPC nonfinite JSON")))
    offset, next_buffer, nodes = HEADER.size + size, 0, 0
    def restore(item, depth=0):
        nonlocal offset, next_buffer, nodes
        nodes += 1
        require(depth <= 40 and nodes <= 200000 and type(item) is list and item, "IPC node structure/bound")
        tag = item[0]
        if tag == "bytes":
            require(len(item) == 3 and type(item[1]) is int and item[1] == next_buffer
                and type(item[2]) is int and 0 <= item[2] <= MAX_BYTES, "IPC buffer descriptor")
            end = offset + item[2]
            require(end <= len(payload), "IPC truncated buffer")
            result = bytes(payload[offset:end])
            offset, next_buffer = end, next_buffer + 1
            require(next_buffer <= MAX_BUFFERS, "IPC buffer-count bound")
            return result
        require(len(item) == 2, "IPC primitive arity")
        if tag == "value":
            value = item[1]
            require(value is None or type(value) in (bool, int, float, str), "IPC primitive type")
            require(type(value) is not float or math.isfinite(value), "IPC nonfinite")
            require(type(value) is not int or -(2 ** 63) <= value < 2 ** 64, "IPC integer bound")
            return value
        require(tag in ("dict", "list") and type(item[1]) is list and len(item[1]) <= 100000, "IPC collection")
        if tag == "list":
            return [restore(v, depth + 1) for v in item[1]]
        result = {}
        for pair in item[1]:
            require(type(pair) is list and len(pair) == 2 and type(pair[0]) is str and pair[0] not in result,
                    "IPC duplicate/invalid key")
            result[pair[0]] = restore(pair[1], depth + 1)
        return result
    result = restore(tree)
    require(offset == len(payload), "IPC trailing bytes")
    return result


def send(sock, value, *, deadline):
    chunks = encode(value)
    descriptor = None
    try:
        descriptor, size = M.create(chunks)
        W.send(sock, W.frame_bytes(dict(schema="numeric_ipc_memfd_v1", size=size)), deadline)
        M.send_descriptor(sock, descriptor, deadline)
        return size
    finally:
        if descriptor is not None:
            os.close(descriptor)


def receive(sock, *, deadline):
    metadata, body = W.receive(sock, deadline)
    require(not body and set(metadata) == {"schema", "size"}
        and metadata["schema"] == "numeric_ipc_memfd_v1", "IPC transport envelope")
    descriptor = M.receive_descriptor(sock, deadline)
    try:
        with M.mapped(descriptor, metadata["size"]) as view:
            # Decoded bytes own their storage; no mapping survives response.
            return decode(view)
    finally:
        os.close(descriptor)


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
