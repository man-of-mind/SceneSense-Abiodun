"""Reuse identical depth only within one owned synchronous GT request.

Original decode arithmetic and downstream plausibility checks are unchanged.
The cache has immutable byte backing and compares all native bytes, including
alpha. Unsupported array layouts retain the original decoder, never a cache
hit. No frame value survives scope exit; no source file or NumPy is patched.
"""
from contextlib import contextmanager
import threading

import numpy as np


class DepthReuseError(ValueError):
    pass


class DepthReuse:
    def __init__(self, original_image_decoder, original_bgra_decoder):
        self.original_image_decoder = original_image_decoder
        self.original_bgra_decoder = original_bgra_decoder
        self._owner = self._identity = self._image = self._metadata = None
        self._native = self._decoded = self._shape = self._stats = None
        self.last_stats = None

    @property
    def active(self):
        return self._owner is not None

    @property
    def retained(self):
        return any(value is not None for value in (self._identity, self._image,
            self._metadata, self._native, self._decoded, self._shape))

    def _require_active(self):
        if self._owner != threading.get_ident():
            raise DepthReuseError("depth reuse outside owned request/thread")

    @staticmethod
    def _image_metadata(image):
        return (image.frame, image.timestamp, image.height, image.width)

    @contextmanager
    def request_scope(self, *, identity, depth_image):
        if self.active:
            raise DepthReuseError("nested/concurrent depth request forbidden")
        self.last_stats = None
        # Validate into locals before taking ownership. A malformed request
        # must not strand an active cache before the finally block exists.
        request_identity = dict(identity)
        metadata = self._image_metadata(depth_image)
        self._owner = threading.get_ident()
        self._identity = request_identity
        self._image = depth_image
        self._metadata = metadata
        self._stats = dict(frame_id=request_identity.get("frame_id"), calls=0, decodes=0,
                           reuses=0, fallbacks=0, completed=False)
        try:
            yield self
            self._stats["completed"] = True
        finally:
            self.last_stats = dict(self._stats)
            self._owner = self._identity = self._image = self._metadata = None
            self._native = self._decoded = self._shape = self._stats = None

    def _check_image(self, image):
        if (self._image_metadata(image) != self._metadata
                or self._metadata[0] != self._identity.get("frame_id")):
            raise DepthReuseError("depth image differs from request identity/metadata")
        # The preceding original AVO/source validators retain their own pose,
        # timestamp and plausibility checks. This gate additionally verifies
        # the complete native buffer before reuse, even for the same object.
        h, w = image.height, image.width
        if (type(h) is not int or type(w) is not int or not 0 < h <= 720 or not 0 < w <= 1280):
            raise DepthReuseError("bounded native depth dimensions required")
        raw = bytes(image.raw_data)
        if len(raw) != h*w*4:
            raise DepthReuseError("native depth byte count differs")
        return raw, (h, w, 4)

    def decode_image(self, image):
        self._require_active()
        self._stats["calls"] += 1
        if image is not self._image:
            raise DepthReuseError("depth decode image is not owned by this request")
        raw, shape = self._check_image(image)
        if self._native is not None:
            if raw != self._native:
                raise DepthReuseError("native depth source bytes changed")
            self._stats["reuses"] += 1
            return self._decoded
        # Invoke the original AVO decoder at its original location AFTER the
        # original synchronization/identity validators, not on scope entry.
        decoded = self.original_image_decoder(image)
        if self._check_image(image)[0] != raw:
            raise DepthReuseError("native depth mutated during decode")
        if (type(decoded) is not np.ndarray or decoded.dtype != np.float32
                or decoded.shape != shape[:2]):
            raise DepthReuseError("original depth decoder result differs")
        self._native, self._shape = raw, shape
        self._decoded = np.frombuffer(decoded.tobytes(order="C"), dtype=decoded.dtype).reshape(decoded.shape)
        self._stats["decodes"] += 1
        return self._decoded

    def decode_bgra(self, array):
        self._require_active()
        self._stats["calls"] += 1
        if self._decoded is None:
            raise DepthReuseError("AVO image decode must initialize this request")
        if self._check_image(self._image)[0] != self._native:
            raise DepthReuseError("bound native depth source changed before reuse")
        # Outside the exact production ndarray surface, execute the original
        # function, including its original shape/dtype validation. Never trust
        # subclass overrides, array identity or a reversible WRITEABLE flag.
        if (type(array) is not np.ndarray or array.ndim != 3 or array.shape[2] != 4
                or array.dtype != np.uint8 or not array.flags.c_contiguous):
            self._stats["fallbacks"] += 1
            return self.original_bgra_decoder(array)
        if array.shape != self._shape:
            raise DepthReuseError("native depth shape differs from request")
        if array.tobytes(order="C") != self._native:
            raise DepthReuseError("native depth source bytes differ")
        self._stats["reuses"] += 1
        return self._decoded
