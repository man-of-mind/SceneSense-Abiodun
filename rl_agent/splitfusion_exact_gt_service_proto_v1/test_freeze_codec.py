"""Adversarial admission tests for the binary freeze handoff (CPU only).

    cd abiodun && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        /usr/bin/python3 -m unittest rl_agent.splitfusion_exact_gt_service_proto_v1.test_freeze_codec -v
"""
import hashlib
import json
import os
import secrets
import unittest

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_live_gt_optimization_v4 import memfd_transport as M
from rl_agent.splitfusion_numeric_gt_runtime_v1 import staging as ST

from . import freeze_codec as FZ, source_compare as SCMP
from .test_deferred import corpus


def blob_of(chunks):
    return bytearray(b"".join(bytes(memoryview(c).cast("B")) for c in chunks))


def unsealed(raw):
    fd = os.memfd_create("unsealed_test", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    os.write(fd, bytes(raw))
    return fd


class FreezeGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = corpus()
        cls.cases = cls.c.cases

    def setUp(self):
        self.session = secrets.token_hex(32)
        self.gate = FZ.FreezeGate(self.session)
        self.fds = getattr(self, "fds", [])

    def tearDown(self):
        for fd in self.fds:
            os.close(fd)

    def frozen(self, index=0, sequence=1, session=None):
        case = self.cases[index]
        return FZ.freeze(self.c.sourcekwargs(case), case.kwargs["policy_timestamp_s"],
                         session_id=session or self.session, sequence=sequence)

    def sealed(self, raw_or_chunks):
        chunks = [bytes(raw_or_chunks)] if isinstance(raw_or_chunks, (bytes, bytearray)) else raw_or_chunks
        fd, _ = FZ.seal(chunks)
        self.fds.append(fd)
        return fd

    def admit(self, descriptor, fd):
        with self.gate.admit(descriptor, fd) as admitted:
            return admitted.descriptor["frame_id"]

    def edited(self, descriptor, change):
        value = C._parse(descriptor)
        change(value)
        return C.canonical(value)

    def with_buffer(self, descriptor, chunks, name, mutate):
        blob = blob_of(chunks)
        value = C._parse(descriptor)
        spec = next(b for b in value["buffers"] if b["name"] == name)
        view = np.frombuffer(blob, dtype=np.dtype(spec["dtype"]), count=spec["nbytes"] // np.dtype(spec["dtype"]).itemsize,
                             offset=spec["offset"]).reshape(spec["shape"])
        mutate(view)
        spec["sha256"] = hashlib.sha256(blob[spec["offset"]:spec["offset"] + spec["nbytes"]]).hexdigest()
        return C.canonical(value), blob

    def rejected(self, descriptor, fd, pattern=None):
        with self.assertRaises(Exception) as context:
            self.admit(descriptor, fd)
        if pattern:
            self.assertIn(pattern, str(context.exception))
        self.assertIsNotNone(self.gate.failure)
        # Fail closed: even a valid next descriptor is refused afterwards.
        d, chunks = self.frozen(1, self.gate.last_sequence + 1)
        with self.assertRaises(FZ.FreezeError):
            self.admit(d, self.sealed(chunks))

    # ---------------- accepted paths ----------------
    def test_valid_sequence_and_equality_with_v4(self):
        for seq, index in ((1, 0), (2, 1)):
            d, chunks = self.frozen(index, seq)
            fd = self.sealed(chunks)
            packet, extras = self.c.staged_packet(self.cases[index])
            v4 = ST.decode_stage(packet, extras)
            with self.gate.admit(d, fd) as admitted:
                fields = SCMP.compare(v4, admitted.kwargs)
            self.assertTrue(all(fields.values()), fields)
        self.assertEqual(self.gate.admitted, 2)

    def test_leaked_view_fails_closed(self):
        d, chunks = self.frozen()
        fd = self.sealed(chunks)
        with self.assertRaises(BufferError):
            with self.gate.admit(d, fd) as admitted:
                leaked = admitted.kwargs["semantic_image"].raw_data  # noqa: F841 retained past the context
        del leaked

    # ---------------- refused-freeze void ----------------
    def test_void_advances_sequence_then_next_admits(self):
        d1, _ = self.frozen(0, 1)
        self.assertEqual(self.gate.void(d1), dict(sequence=1, frame_id=self.cases[0].frame_id))
        d2, chunks = self.frozen(1, 2)
        self.assertEqual(self.admit(d2, self.sealed(chunks)), self.cases[1].frame_id)
        self.assertIsNone(self.gate.failure)

    def test_void_rejects_replay_and_foreign_and_fails_closed(self):
        d1, _ = self.frozen(0, 1)
        self.gate.void(d1)
        with self.assertRaises(FZ.FreezeError):
            self.gate.void(d1)                                   # replayed sequence
        self.assertIsNotNone(self.gate.failure)
        self.setUp()
        foreign, _ = self.frozen(0, 1, session=secrets.token_hex(32))
        with self.assertRaises(FZ.FreezeError):
            self.gate.void(foreign)
        d2, chunks = self.frozen(1, 1)
        with self.assertRaises(FZ.FreezeError):                  # gate stays closed
            self.admit(d2, self.sealed(chunks))

    # ---------------- session / ordering ----------------
    def test_foreign_session(self):
        d, chunks = self.frozen(session=secrets.token_hex(32))
        self.rejected(d, self.sealed(chunks), "foreign session")

    def test_replayed_descriptor(self):
        d, chunks = self.frozen()
        fd = self.sealed(chunks)
        self.admit(d, fd)
        self.rejected(d, fd, "sequence")

    def test_sequence_gap(self):
        d, chunks = self.frozen(0, 2)
        self.rejected(d, self.sealed(chunks), "sequence")

    def test_stale_frame(self):
        d, chunks = self.frozen(1, 1)
        self.admit(d, self.sealed(chunks))
        d, chunks = self.frozen(0, 2)
        self.rejected(d, self.sealed(chunks), "stale")

    # ---------------- malformed descriptors ----------------
    def test_malformed_variants(self):
        d, chunks = self.frozen()
        variants = {
            "not json": b"{not json",
            "non-canonical": json.dumps(C._parse(d), indent=1).encode(),
            "unknown field": self.edited(d, lambda v: v.update(extra=1)),
            "missing field": self.edited(d, lambda v: v.pop("world")),
            "schema": self.edited(d, lambda v: v.update(schema="other")),
            "oversize": b" " * (FZ.MAX_DESCRIPTOR_BYTES + 1),
            "nan": d.replace(b'"sequence":1', b'"sequence":NaN'),
        }
        for label, raw in variants.items():
            with self.subTest(label=label):
                self.setUp()
                self.rejected(raw, self.sealed(chunks))

    def test_action_bound_template(self):
        d, chunks = self.frozen()
        bad = self.edited(d, lambda v: v["capture_template"].update(action_id=71, profile_id="mode11_q9800_a71"))
        self.rejected(bad, self.sealed(chunks), "action-bound")

    # ---------------- memfd ownership / content ----------------
    def test_unsealed_memfd(self):
        d, chunks = self.frozen()
        fd = unsealed(blob_of(chunks))
        self.fds.append(fd)
        self.rejected(d, fd, "immutable")

    def test_size_mismatch(self):
        d, chunks = self.frozen()
        fd = self.sealed(bytes(blob_of(chunks)) + bytes(64))
        self.rejected(d, fd)

    def test_tampered_content(self):
        d, chunks = self.frozen()
        blob = blob_of(chunks)
        spec = next(b for b in C._parse(d)["buffers"] if b["name"] == "image_2")
        blob[spec["offset"] + 17] ^= 0xFF
        self.rejected(d, self.sealed(blob), "content differs")

    def test_other_frame_content_under_this_descriptor(self):
        d0, _ = self.frozen(0)
        _, chunks1 = self.frozen(1)
        self.rejected(d0, self.sealed(chunks1))

    # ---------------- semantic mismatches (digest kept consistent) ----------------
    def test_frame_identity_mismatch(self):
        d, chunks = self.frozen()
        def change(v):
            v["frame_id"] += 1
            v["capture_template"]["frame_id"] += 1
        self.rejected(self.edited(d, change), self.sealed(chunks), "mismatch")

    def test_pose_mismatch(self):
        d, chunks = self.frozen()
        bad, blob = self.with_buffer(d, chunks, "camera_matrix", lambda a: a.__setitem__((0, 3), a[0, 3] + 1.0))
        self.rejected(bad, self.sealed(blob), "camera")

    def test_camera_inverse_mismatch(self):
        d, chunks = self.frozen()
        bad, blob = self.with_buffer(d, chunks, "camera_inverse", lambda a: a.__setitem__((1, 1), a[1, 1] * 2.0))
        self.rejected(bad, self.sealed(blob), "inverse")

    def test_population_mismatch(self):
        d, chunks = self.frozen()
        bad = self.edited(d, lambda v: v["dynamic_snapshot_evidence"]["active_population_ids"].reverse())
        self.rejected(bad, self.sealed(chunks), "inventory")

    def test_duplicate_actor_id(self):
        d, chunks = self.frozen()
        bad, blob = self.with_buffer(d, chunks, "actor_ids", lambda a: a.__setitem__(1, a[0]))
        self.rejected(bad, self.sealed(blob))

    def test_nonfinite_radar(self):
        d, chunks = self.frozen()
        bad, blob = self.with_buffer(d, chunks, "radar", lambda a: a.__setitem__((0, 0), np.nan))
        self.rejected(bad, self.sealed(blob), "nonfinite")

    def test_layout_variants(self):
        d, chunks = self.frozen()
        def misalign(v):
            v["buffers"][3]["offset"] += 8
        def wrong_dtype(v):
            v["buffers"][6]["dtype"] = "<f4"
        def wrong_shape(v):
            v["buffers"][0]["shape"] = [360, 1280, 4]
        def reorder(v):
            v["buffers"][0], v["buffers"][1] = v["buffers"][1], v["buffers"][0]
        def actor_count(v):
            v["actor_type_ids"] = v["actor_type_ids"][:-1]
        def total(v):
            v["total_bytes"] += 64
        for label, change in dict(misalign=misalign, wrong_dtype=wrong_dtype, wrong_shape=wrong_shape,
                                  reorder=reorder, actor_count=actor_count, total=total).items():
            with self.subTest(label=label):
                self.setUp()
                self.rejected(self.edited(d, change), self.sealed(chunks))


if __name__ == "__main__":
    unittest.main()
