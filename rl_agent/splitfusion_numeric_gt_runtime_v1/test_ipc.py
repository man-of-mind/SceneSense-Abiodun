import hashlib
import os
import socket
import threading
import time
import unittest

from . import ipc as I


class IPCTests(unittest.TestCase):
    def test_nested_roundtrip(self):
        value = dict(a=[None, True, 1, 2.5, "test", b"abc"], b={"__bytes__": b"", "x": b"z" * 800000})
        self.assertEqual(I.decode(b"".join(I.encode(value))), value)

    def test_bad_digest_trailing_and_bounds(self):
        raw = b"".join(I.encode(dict(a=b"ab")))
        for packet in (raw[:-1], raw + b"x", raw[:-1] + b"z", b"X" + raw[1:]):
            with self.assertRaises(ValueError):
                I.decode(packet)
        for value in (float("nan"), object(), {3: "x"}, b"x" * I.MAX_BYTES):
            with self.assertRaises(ValueError):
                I.encode(value)

    def test_buffer_count_matches_memfd_chunk_bound(self):
        value = [b"x"] * I.MAX_BUFFERS
        chunks = I.encode(value)
        self.assertEqual(len(chunks), 64)
        self.assertEqual(I.decode(b"".join(chunks)), value)
        with self.assertRaisesRegex(ValueError, "buffer-count"):
            I.encode(value + [b"x"])

    def test_duplicate_wire_keys(self):
        meta = I.canonical(["dict", [["x", ["value", 1]], ["x", ["value", 2]]]])
        packet = I.HEADER.pack(I.MAGIC, len(meta), hashlib.sha256(meta).digest()) + meta
        with self.assertRaises(ValueError):
                I.decode(packet)

    def test_sealed_descriptor_handoff_and_no_fd_leak(self):
        before = len(os.listdir("/proc/self/fd"))
        left, right = socket.socketpair()
        out = []
        value = dict(x=b"abc" * 100000, identity={"frame": 1})
        thread = threading.Thread(target=lambda: out.append(I.receive(right, deadline=time.monotonic() + 5)))
        thread.start()
        try:
            I.send(left, value, deadline=time.monotonic() + 5)
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(out, [value])
        finally:
            left.close(); right.close()
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)

    def test_deadline_and_disconnect(self):
        left, right = socket.socketpair()
        try:
            with self.assertRaises(Exception):
                I.receive(left, deadline=time.monotonic() - 1)
            right.close()
            with self.assertRaises(Exception):
                I.receive(left, deadline=time.monotonic() + 1)
        finally:
            left.close(); right.close()


if __name__ == "__main__":
    unittest.main()
