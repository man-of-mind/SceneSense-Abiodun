"""Independent immutable-descriptor, framing, boundedness and leak tests."""
import array
import fcntl
import os
import socket
import tempfile
import time
import unittest
from unittest import mock

from . import memfd_transport as M


def fd_count():
    return len(os.listdir('/proc/self/fd'))


class MemfdTests(unittest.TestCase):
    def test_chunks_are_exact_immutable_and_close_on_scope(self):
        descriptor, size = M.create([b'abc', memoryview(b'\x00\xff'), b'', bytearray(b'def')])
        try:
            self.assertEqual(size, 8)
            self.assertEqual(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS), M.SEALS)
            with M.mapped(descriptor, size) as view:
                self.assertTrue(view.readonly)
                self.assertEqual(bytes(view), b'abc\x00\xffdef')
                with self.assertRaises(TypeError):
                    view[0] = 5
            for operation in (lambda: os.write(descriptor, b'x'), lambda: os.ftruncate(descriptor, size + 1),
                              lambda: os.ftruncate(descriptor, size - 1)):
                with self.assertRaises(OSError):
                    operation()
        finally:
            os.close(descriptor)

    def test_bad_sizes_descriptors_missing_seals_and_type_are_refused(self):
        descriptor, size = M.create([b'123'])
        try:
            for declared in (0, -1, True, 2, 4, M.MAX_BYTES + 1):
                with self.subTest(declared=declared), self.assertRaises(M.TransferError):
                    M.verify(descriptor, declared)
            with self.assertRaises(OSError):
                M.verify(-1, size)
            with tempfile.TemporaryFile() as regular:
                regular.write(b'123'); regular.flush()
                with self.assertRaises((M.TransferError, OSError)):
                    M.verify(regular.fileno(), size)
            unsealed = os.memfd_create('unsealed-test', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
            try:
                os.write(unsealed, b'123')
                with self.assertRaisesRegex(M.TransferError, 'immutable'):
                    M.verify(unsealed, size)
            finally:
                os.close(unsealed)
            reader, writer = os.pipe()
            try:
                with self.assertRaises(M.TransferError):
                    M.verify(reader, size)
            finally:
                os.close(reader); os.close(writer)
        finally:
            os.close(descriptor)

    def test_create_bounds_precede_memfd_and_failed_write_closes_fd(self):
        with mock.patch.object(M, 'MAX_BYTES', 4), mock.patch.object(M.os, 'memfd_create') as creation:
            for chunks in ([], [b''], [b'12345'], [b''] * 64 + [b'1']):
                with self.subTest(chunks=chunks), self.assertRaises(M.TransferError):
                    M.create(chunks)
            creation.assert_not_called()
        before = fd_count()
        with mock.patch.object(M.os, 'writev', side_effect=OSError('injected write failure')):
            with self.assertRaisesRegex(OSError, 'injected'):
                M.create([b'ab'])
        self.assertEqual(fd_count(), before)
        with mock.patch.object(M.os, 'writev', return_value=0):
            with self.assertRaisesRegex(M.TransferError, 'progress'):
                M.create([b'ab'])
        self.assertEqual(fd_count(), before)

    def test_short_writev_handled_without_concatenation(self):
        original = os.writev
        def partial(descriptor, chunks):
            for chunk in chunks:
                if len(chunk):
                    return original(descriptor, [chunk[:1]])
            return original(descriptor, chunks)
        with mock.patch.object(M.os, 'writev', side_effect=partial):
            descriptor, size = M.create([b'', b'abc', b'def', b''])
        try:
            with M.mapped(descriptor, size) as view:
                self.assertEqual(bytes(view), b'abcdef')
        finally:
            os.close(descriptor)

    def test_descriptor_roundtrip_is_cloexec_readonly_and_leak_free(self):
        before = fd_count()
        left, right = socket.socketpair()
        try:
            for index in range(128):
                payload = bytes([index]) * 4096
                original, size = M.create([payload[:2048], payload[2048:]])
                try:
                    M.send_descriptor(left, original, time.monotonic() + 1)
                    received = M.receive_descriptor(right, time.monotonic() + 1)
                    try:
                        self.assertFalse(os.get_inheritable(received))
                        with M.mapped(received, size) as view:
                            self.assertEqual(bytes(view), payload)
                            with self.assertRaises(TypeError):
                                view[0] = 0
                    finally:
                        os.close(received)
                finally:
                    os.close(original)
            self.assertEqual(fd_count(), before + 2)
        finally:
            left.close(); right.close()
        self.assertEqual(fd_count(), before)

    def test_multiple_truncated_and_wrong_token_rights_are_closed(self):
        for count, token in ((2, b'F'), (32, b'F'), (1, b'X'), (0, b'F')):
            with self.subTest(count=count, token=token):
                before = fd_count()
                left, right = socket.socketpair()
                descriptor, _ = M.create([b'proof'])
                try:
                    ancillary = [] if count == 0 else [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                                       array.array('i', [descriptor] * count))]
                    left.sendmsg([token], ancillary)
                    with self.assertRaisesRegex(M.TransferError, 'envelope'):
                        M.receive_descriptor(right, time.monotonic() + 1)
                    self.assertEqual(fd_count(), before + 3)
                finally:
                    os.close(descriptor); left.close(); right.close()
                self.assertEqual(fd_count(), before)

    def test_timeout_eof_expired_and_invalid_send_are_bounded(self):
        left, right = socket.socketpair()
        try:
            began = time.monotonic()
            with self.assertRaises((TimeoutError, M.TransferError)):
                M.receive_descriptor(right, began + .025)
            self.assertLess(time.monotonic() - began, .5)
            with self.assertRaisesRegex(M.TransferError, 'deadline'):
                M.receive_descriptor(right, time.monotonic() - 1)
            with self.assertRaises(OSError):
                M.send_descriptor(left, -1, time.monotonic() + 1)
            left.close()
            with self.assertRaisesRegex(M.TransferError, 'envelope'):
                M.receive_descriptor(right, time.monotonic() + 1)
        finally:
            left.close(); right.close()

    def test_malformed_mapped_snapshot_preserves_original_error_and_releases_views(self):
        from .snapshot import unpack_snapshot, SnapshotError
        before = fd_count()
        descriptor, size = M.create([b'X' * 100])
        try:
            with self.assertRaisesRegex(SnapshotError, 'magic'):
                with M.mapped(descriptor, size) as payload:
                    unpack_snapshot(payload)
        finally:
            os.close(descriptor)
        self.assertEqual(fd_count(), before)


if __name__ == '__main__':
    unittest.main()
