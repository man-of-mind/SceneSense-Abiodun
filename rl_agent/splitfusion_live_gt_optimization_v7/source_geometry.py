"""Exact, request-local reuse of invariant array work in the GT source builder.

The original floating-point radar transform, shapes, matmul ordering, mask
algorithm, validators and source evidence are unchanged. Only the radar
homogeneous input and static semantic membership result are reused within
their existing owning call. No cross-frame cache, GPU operation, thread, IO,
global patch, geometry filter or changed support/eligibility rule is added.

This module is an OFFLINE candidate, not a live-source binding or entry point.
"""
from types import FunctionType, SimpleNamespace

import numpy as np


def immutable_bytes_owner(array):
    value = array
    while isinstance(value, np.ndarray):
        if value.flags.writeable:
            return False
        value = value.base
    return type(value) is bytes


def clone(function, namespace):
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class RadarRequest:
    """One synchronous owned build; no actor transform or output is cached."""
    def __init__(self, points):
        self.points = points
        self.homogeneous = None
        self.calls = self.builds = self.reuses = self.fallbacks = 0

    def transform(self, original, points, actor):
        # Match the original's empty-input path and non-production inputs.
        # Fixed shape/strides/dtype preserve the same BLAS operation/layout.
        if (points is not self.points or type(points) is not np.ndarray
                or points.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or points.ndim != 2
                or points.shape[1] != 3 or not points.flags.c_contiguous
                or not immutable_bytes_owner(points)
                or not 0 < points.shape[0] <= 100000):
            self.fallbacks += 1
            return original(points, actor)
        self.calls += 1
        inverse = np.asarray(actor.get_transform().get_inverse_matrix(), dtype=np.float64)
        if self.homogeneous is None:
            self.homogeneous = np.concatenate(
                [points.astype(np.float64), np.ones((points.shape[0], 1))], axis=1)
            self.builds += 1
        else:
            self.reuses += 1
        return (inverse @ self.homogeneous.T).T[:, :3]

    def close(self):
        self.points = self.homogeneous = None


def static_rows(original, stats, static, frame, boxes, geometry, depth, semantic, radar):
    """Private NumPy namespace for the original one-frame static audit."""
    cache = {}

    def isin(elements, test_elements, *args, **kwargs):
        # The registered static audit passes exactly this one uint8 semantic
        # array and a list of integer tags. All other NumPy API calls fall back.
        if (elements is semantic and type(elements) is np.ndarray
                and elements.dtype == np.uint8 and elements.shape == (720, 1280)
                and type(test_elements) is list
                and all(type(tag) is int and 0 <= tag <= 255 for tag in test_elements)
                and not args and not kwargs):
            key = tuple(test_elements)
            stats["semantic_calls"] += 1
            if key not in cache or not np.array_equal(elements, cache[key][0]):
                computed = np.isin(elements, test_elements)
                value = np.frombuffer(computed.tobytes(), dtype=np.bool_).reshape(computed.shape)
                snapshot = elements.copy()
                snapshot.setflags(write=False)
                cache[key] = (snapshot, value)
                stats["semantic_builds"] += 1
            else:
                stats["semantic_reuses"] += 1
            return cache[key][1]
        return np.isin(elements, test_elements, *args, **kwargs)

    proxy = SimpleNamespace(**{**vars(np), "isin": isin})
    specialized = clone(original, dict(original.__globals__, np=proxy))
    try:
        return specialized(static, frame, boxes, geometry, depth, semantic, radar)
    finally:
        cache.clear()


def source_producer(base):
    """Wrap a private V2/V6 producer; preserve all original numerical methods."""
    original_build = base.build
    original_static = original_build.__globals__["_static_rows"]

    class RequestArrayReuseProducer(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._array_request = None
            self.last_array_reuse = None
            original_objects = self.objects
            objects = dict(original_objects)
            original_transform = original_objects["transform_world_to_actor_local"]

            def transform(points, actor):
                if self._array_request is None:
                    return original_transform(points, actor)
                return self._array_request.transform(original_transform, points, actor)

            for name, value in original_objects.items():
                if isinstance(value, FunctionType) and value.__globals__ is original_objects:
                    objects[name] = clone(value, objects)
            objects["transform_world_to_actor_local"] = transform
            self.objects = objects

        def build(self, **kwargs):
            if self._array_request is not None:
                raise RuntimeError("nested or concurrent source build is not allowed")
            request = RadarRequest(kwargs["radar_world_xyz"])
            stats = dict(semantic_calls=0, semantic_builds=0, semantic_reuses=0,
                         completed=False, retained_after_request=True)
            self._array_request = request

            def per_frame_static(*args):
                return static_rows(original_static, stats, *args)

            build = clone(original_build, dict(original_build.__globals__, _static_rows=per_frame_static))
            try:
                result = build(self, **kwargs)
                stats["completed"] = True
                return result
            finally:
                stats.update(radar_calls=request.calls, radar_builds=request.builds,
                             radar_reuses=request.reuses, radar_fallbacks=request.fallbacks)
                request.close()
                self._array_request = None
                stats["retained_after_request"] = False
                self.last_array_reuse = stats

    return RequestArrayReuseProducer
