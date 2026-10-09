"""Deterministic per-frame work accounting: calls and bytes, not wall time.

The counters wrap module attributes inside this process for the duration of a
``measure`` block and are always restored. They classify work as numerical
(geometry, projection, masks, components), repeated validation (re-hashing the
same arrays, re-verifying source files, re-decoding the same depth), conversion
(JSON canonicalisation, CSV write/parse, PNG encode/decode) and evidence
construction (np.save payload copies, evidence hashing). Timing is out of scope.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np

from rl_agent.splitfusion_live_gt_optimization_v1 import source_geometry as V1S
from rl_agent.splitfusion_live_parity_repair_v1 import conflicts as C
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG

from . import originals as O

CATEGORIES = {
    "sha256_calls": "evidence/validation hashing", "sha256_bytes": "evidence/validation hashing",
    "json_dumps_calls": "conversion (canonical JSON)", "json_dumps_bytes": "conversion (canonical JSON)",
    "json_loads_calls": "conversion (JSON parse)",
    "csv_writer_calls": "conversion (CSV write)", "csv_reader_calls": "conversion (CSV parse)",
    "png_encode_calls": "artifact construction (PNG encode)", "png_decode_calls": "artifact construction (PNG decode)",
    "image_file_reads": "artifact construction (image file read)",
    "np_save_calls": "evidence construction (np.save payload copy)",
    "verify_sources_calls": "repeated validation (source file re-hash)",
    "original_modules_calls": "repeated validation (module re-exec)",
    "input_content_sha256_calls": "repeated validation (complete-input digest)",
    "validate_calls": "repeated validation (complete-input validator)",
    "restore_prepared_calls": "repeated validation (prepared-target restore)",
    "target_digest_calls": "repeated validation (prepared-target digest)",
    "catalogue_digest_calls": "repeated validation (catalogue digest)",
    "depth_decode_calls": "repeated numerics (depth decode)",
    "depth_plausible_calls": "repeated validation (depth plausibility)",
    "isin_calls": "repeated numerics (full-image class mask)",
    "unique_calls": "repeated validation (label-set scan)",
    "visibility_calls": "indispensable numerics (depth-interval visibility)",
    "nearest_count_calls": "indispensable numerics (model-grid count)",
    "object_rows_calls": "indispensable numerics (actor projection)",
    "radar_transform_calls": "indispensable numerics (radar support transform)",
    "connected_components_calls": "indispensable numerics (semantic components)",
    "score_actor_calls": "indispensable numerics (AVO per qualified actor)",
    "concatenate_calls": "numerics (array concatenation)",
}


class Counters(SimpleNamespace):
    def __init__(self):
        super().__init__(**{name: 0 for name in CATEGORIES})

    def table(self) -> dict:
        return {name: {"count": getattr(self, name), "class": CATEGORIES[name]} for name in CATEGORIES}


def _wrap(target, name, counter, key, byte_key=None, byte_of=None):
    original = getattr(target, name)
    def call(*args, **kwargs):
        setattr(counter, key, getattr(counter, key) + 1)
        result = original(*args, **kwargs)
        if byte_key is not None:
            setattr(counter, byte_key, getattr(counter, byte_key) + byte_of(args, kwargs, result))
        return result
    call.__wrapped__ = original
    setattr(target, name, call)
    return (target, name, original)


class _HashProxy:
    """Forward to the real hash object while counting bytes fed through update()."""
    __slots__ = ("_counter", "_hash")

    def __init__(self, counter, value):
        self._counter, self._hash = counter, value

    def update(self, chunk):
        self._counter.sha256_bytes += len(chunk)
        return self._hash.update(chunk)

    def __getattr__(self, name):
        return getattr(self._hash, name)


class _Sha:
    def __init__(self, counter, original):
        self.counter, self.original = counter, original

    def __call__(self, data=b"", *args, **kwargs):
        self.counter.sha256_calls += 1
        self.counter.sha256_bytes += len(data) if hasattr(data, "__len__") else 0
        return _HashProxy(self.counter, self.original(data, *args, **kwargs))


@contextmanager
def measure(numeric_namespaces=None):
    """Count work inside the block; restores every wrapped attribute afterwards."""
    counter = Counters()
    restores = []
    bound = O.bound()
    original_sha = hashlib.sha256
    hashlib.sha256 = _Sha(counter, original_sha)
    try:
        restores.append(_wrap(json, "dumps", counter, "json_dumps_calls", "json_dumps_bytes", lambda a, k, r: len(r)))
        restores.append(_wrap(json, "loads", counter, "json_loads_calls"))
        restores.append(_wrap(csv, "DictWriter", counter, "csv_writer_calls"))
        restores.append(_wrap(csv, "DictReader", counter, "csv_reader_calls"))
        restores.append(_wrap(cv2, "imencode", counter, "png_encode_calls"))
        restores.append(_wrap(cv2, "imwrite", counter, "png_encode_calls"))
        restores.append(_wrap(cv2, "imdecode", counter, "png_decode_calls"))
        restores.append(_wrap(cv2, "imread", counter, "image_file_reads"))
        restores.append(_wrap(cv2, "connectedComponentsWithStats", counter, "connected_components_calls"))
        restores.append(_wrap(np, "save", counter, "np_save_calls"))
        restores.append(_wrap(np, "isin", counter, "isin_calls"))
        restores.append(_wrap(np, "unique", counter, "unique_calls"))
        restores.append(_wrap(np, "concatenate", counter, "concatenate_calls"))
        restores.append(_wrap(A, "verify_sources", counter, "verify_sources_calls"))
        restores.append(_wrap(A, "_original_modules", counter, "original_modules_calls"))
        restores.append(_wrap(A, "input_content_sha256", counter, "input_content_sha256_calls"))
        restores.append(_wrap(A, "_validate", counter, "validate_calls"))
        restores.append(_wrap(E, "restore_prepared_target", counter, "restore_prepared_calls"))
        restores.append(_wrap(E, "_target_digest", counter, "target_digest_calls"))
        restores.append(_wrap(C, "catalogue_content_sha256", counter, "catalogue_digest_calls"))
        # Original (pinned) namespaces reached by the retained producer.
        objects, semantic, static, visibility, _paint = S._original()
        restores.append(_wrap(visibility, "decode_depth_bgra", counter, "depth_decode_calls"))
        restores.append(_wrap(visibility, "depth_is_plausible", counter, "depth_plausible_calls"))
        restores.append(_wrap(visibility, "visibility_from_corners", counter, "visibility_calls"))
        restores.append(_wrap(visibility, "_nearest_model_count", counter, "nearest_count_calls"))
        for name, key in (("decode_depth_bgra", "depth_decode_calls"), ("depth_is_plausible", "depth_plausible_calls"),
                          ("visibility_from_corners", "visibility_calls")):
            original = static[name]
            def make(original, key):
                def call(*args, **kwargs):
                    setattr(counter, key, getattr(counter, key) + 1)
                    return original(*args, **kwargs)
                return call
            static[name] = make(original, key)
            restores.append((static, name, original))
        for name, key in (("project_actor_to_object_row", "object_rows_calls"),
                          ("transform_world_to_actor_local", "radar_transform_calls")):
            original = objects[name]
            def make2(original, key):
                def call(*args, **kwargs):
                    setattr(counter, key, getattr(counter, key) + 1)
                    return original(*args, **kwargs)
                return call
            objects[name] = make2(original, key)
            restores.append((objects, name, original))
        restores.append(_wrap(LG, "decode_depth", counter, "depth_decode_calls"))
        from data_collection.route_b_publication_actor_volume_visibility_v1 import scoring as AVOS
        restores.append(_wrap(AVOS, "score_actor_frame", counter, "score_actor_calls"))
        # Numeric producer's private namespaces (bound once at startup).
        for ns, name, key in ((bound.objects, "project_actor_to_object_row", "object_rows_calls"),
                              (bound.objects, "transform_world_to_actor_local", "radar_transform_calls"),
                              (bound.static, "visibility_from_corners", "visibility_calls"),
                              (bound.static, "depth_is_plausible", "depth_plausible_calls")):
            original = ns[name]
            def make3(original, key):
                def call(*args, **kwargs):
                    setattr(counter, key, getattr(counter, key) + 1)
                    return original(*args, **kwargs)
                return call
            ns[name] = make3(original, key)
            restores.append((ns, name, original))
        for name, key in (("decode_depth_bgra", "depth_decode_calls"), ("depth_is_plausible", "depth_plausible_calls"),
                          ("visibility_from_corners", "visibility_calls"), ("_nearest_model_count", "nearest_count_calls")):
            restores.append(_wrap(bound.visibility, name, counter, key))
        yield counter
    finally:
        hashlib.sha256 = original_sha
        for target, name, original in reversed(restores):
            if isinstance(target, dict):
                target[name] = original
            else:
                setattr(target, name, original)
