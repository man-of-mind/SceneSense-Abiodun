"""Actual bounded Unix socket lifecycle tests, CPU-only and temporary paths."""
import os
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
import unittest
from unittest import mock

from . import broker as B
from . import scoring_packet as P
from .test_scoring_packet import BINDING, packet_fixture


class BrokerTests(unittest.TestCase):
    def test_ancestor_namespace_pid_zero_preserves_authenticated_ids(self):
        from unittest.mock import Mock
        connection = Mock()
        connection.getsockopt.return_value = struct.pack("=iII", 0, 200171, 100117)
        self.assertEqual(B._peer_credentials(connection), dict(pid=0, uid=200171, gid=100117))
        connection.getsockopt.return_value = struct.pack("=iII", -1, 200171, 100117)
        with self.assertRaises(B.BrokerError):
            B._peer_credentials(connection)

    @classmethod
    def setUpClass(cls):
        cls.first = packet_fixture(10)
        cls.second = packet_fixture(11)
        cls.third = packet_fixture(12)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="scoring-handoff-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "scoring.sock"
        self.servers = []
        self.addCleanup(self.cleanup_servers)

    def cleanup_servers(self):
        for server in self.servers:
            result = server.close()
            self.assertFalse(result["worker_alive"])

    def server(self, **kwargs):
        server = B.Server(self.path, binding=BINDING, **kwargs)
        self.servers.append(server)
        return server.start()

    def publish(self, fixture=None, **kwargs):
        return B.publish(self.path, (self.first if fixture is None else fixture)[3],
            binding=kwargs.pop("binding", BINDING), timeout_s=1., **kwargs)

    def wait_failed(self, server):
        deadline = time.monotonic() + 2.
        while not server.stats()["failure"] and time.monotonic() < deadline:
            with server.condition:
                server.condition.wait(.01)
        self.assertIsNotNone(server.stats()["failure"])

    def test_exact_delivery_observation_and_clean_shutdown(self):
        server = self.server()
        self.assertIsNone(server.observe(self.first[0]))
        answer = self.publish()
        self.assertEqual(answer["packet_sha256"], P.digest(self.first[3]))
        observation = server.observe(self.first[0])
        self.assertFalse(observation["consumed"])
        self.assertGreater(observation["admitted_raw_ns"], 0)
        observation["identity"]["frame_id"] = 999
        self.assertEqual(server.observe(self.first[0])["identity"], self.first[0])
        actual = server.take(self.first[0], timeout_s=.1)
        self.assertEqual(actual, self.first[3])
        P.decode_packet(actual, expected_identity=self.first[0], binding=BINDING)
        self.assertTrue(server.observe(self.first[0])["consumed"])
        result = server.close()
        self.assertEqual((result["admitted"], result["consumed"], result["pending_jobs"]), (1, 1, 0))
        self.assertFalse(self.path.exists())
        self.assertIsNone(result["failure"])

    def test_explicit_same_user_credentials_and_socket_owner_are_recorded(self):
        server = self.server(publisher_uid=os.geteuid(), publisher_gid=os.getegid())
        info = self.path.lstat()
        self.assertEqual((info.st_uid, info.st_gid, info.st_mode & 0o7777),
                         (os.geteuid(), os.getegid(), 0o600))
        self.publish()
        expected = dict(pid=os.getpid(), uid=os.geteuid(), gid=os.getegid())
        row = server.observe(self.first[0])
        self.assertEqual(row["publisher_credentials"], expected)
        row["publisher_credentials"]["uid"] = 999999
        self.assertEqual(server.stats()["ledger"][0]["publisher_credentials"], expected)
        self.assertTrue(server.stats()["credential_policy"]["explicitly_supplied"])

    def test_incomplete_invalid_and_nonroot_foreign_credentials_refused(self):
        for kwargs in (dict(publisher_uid=1), dict(publisher_gid=1),
                       dict(publisher_uid=True, publisher_gid=1),
                       dict(publisher_uid=-1, publisher_gid=1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(B.BrokerError):
                B.Server(self.path, binding=BINDING, **kwargs)
        with mock.patch.object(B.os, "geteuid", return_value=1001), \
             mock.patch.object(B.os, "getegid", return_value=1001):
            with self.assertRaisesRegex(B.BrokerError, "only root"):
                B.Server(self.path, binding=BINDING, publisher_uid=1002, publisher_gid=1002)
        self.assertFalse(self.path.exists())

    def test_peer_rejected_before_any_application_bytes_are_read(self):
        server = self.server()
        foreign = dict(pid=os.getpid(), uid=os.geteuid() + 1, gid=os.getegid())
        with mock.patch.object(B, "_peer_credentials", return_value=foreign), \
             mock.patch.object(B, "_receive", side_effect=AssertionError("must not read")) as read:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(self.path))
                self.wait_failed(server)
            read.assert_not_called()
        self.assertEqual(server.stats()["rejected_credentials"], foreign)
        self.assertEqual(server.stats()["admitted"], 0)

    def test_foreign_owner_assignment_targets_only_pinned_new_socket_inode(self):
        # Exercise the root-only syscall branch without changing filesystem
        # ownership or claiming actual container/user crossing qualification.
        with mock.patch.object(B.os, "geteuid", return_value=0):
            server = B.Server(self.path, binding=BINDING,
                              publisher_uid=os.getuid() + 1, publisher_gid=os.getgid())
        self.servers.append(server)
        with mock.patch.object(B.os, "chown") as chown, \
             mock.patch.object(server, "_verify_endpoint", return_value={"mock_owner_check": True}):
            server.start()
        chown.assert_called_once()
        path, uid, gid = chown.call_args.args
        self.assertTrue(path.startswith("/proc/self/fd/"))
        self.assertEqual((uid, gid), (os.getuid() + 1, os.getgid()))
        self.assertEqual(self.path.lstat().st_mode & 0o7777, 0o600)

    def test_socket_permissions_drift_refused_before_packet_read(self):
        server = self.server()
        self.path.chmod(0o660)
        with mock.patch.object(B, "_receive", side_effect=AssertionError("must not read")) as read:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(self.path))
                self.wait_failed(server)
            read.assert_not_called()
        self.assertIn("identity/owner/mode", server.stats()["failure"])

    def test_timeout_preserves_later_packet_and_wrong_identity_does_not_consume(self):
        server = self.server()
        with self.assertRaises(B.NotReadyError):
            server.take(self.first[0], timeout_s=.001)
        self.publish()
        with self.assertRaises(B.NotReadyError):
            server.take(self.second[0], timeout_s=0)
        self.assertEqual(server.stats()["pending_jobs"], 1)
        self.assertEqual(server.take(self.first[0]), self.first[3])

    def test_wait_wakes_when_exact_packet_arrives(self):
        server = self.server()
        received = []
        worker = threading.Thread(target=lambda: received.append(server.take(self.first[0], timeout_s=1.)))
        worker.start()
        try:
            self.publish()
        finally:
            worker.join(timeout=2.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(received, [self.first[3]])

    def test_duplicates_are_sticky_even_after_consumption(self):
        for consumed in (False, True):
            with self.subTest(consumed=consumed):
                server = self.server()
                self.publish()
                if consumed:
                    server.take(self.first[0])
                with self.assertRaises(B.BrokerError):
                    self.publish()
                self.wait_failed(server)
                with self.assertRaises(B.BrokerError):
                    server.take(self.first[0])
                self.assertEqual(server.stats()["admitted"], 1)
                server.close()

    def test_pending_job_and_byte_bounds_fail_without_dropping_first(self):
        for limits in (dict(max_jobs=1), dict(max_bytes=len(self.first[3]))):
            with self.subTest(limits=limits):
                server = self.server(**limits)
                self.publish()
                with self.assertRaises(B.BrokerError):
                    self.publish(self.second)
                result = server.close()
                self.assertEqual(result["admitted"], 1)
                self.assertEqual(result["pending_jobs"], 1)
                self.assertEqual(result["pending_bytes"], len(self.first[3]))
                self.assertFalse(result["all_consumed"])

    def test_total_frame_bound_survives_consumption(self):
        server = self.server(max_jobs=1, max_frames=1)
        self.publish()
        server.take(self.first[0])
        with self.assertRaises(B.BrokerError):
            self.publish(self.second)
        self.assertEqual(server.stats()["admitted"], 1)
        self.assertEqual(server.stats()["consumed"], 1)

    def test_foreign_stream_and_regressing_identity_variation_rejected(self):
        variants = [packet_fixture(11, stream_id="foreign"), packet_fixture(10, capture_timestamp_ns=99999)]
        for variant in variants:
            server = self.server()
            self.publish()
            server.take(self.first[0])
            with self.assertRaises(B.BrokerError):
                self.publish(variant)
            self.assertEqual(server.stats()["admitted"], 1)
            server.close()

    def test_foreign_binding_malformed_packet_and_huge_header_fail_sticky(self):
        for message in (struct.pack("!Q", 8) + b"garbage!",
                        struct.pack("!Q", P.MAX_PACKET_BYTES + 1)):
            server = self.server(read_timeout_s=.1)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(self.path))
                connection.sendall(message)
                self.wait_failed(server)
            self.assertEqual(server.stats()["admitted"], 0)
            server.close()
        server = self.server()
        # The public producer rejects wrong binding locally. A raw publisher
        # must also be refused by the receiver's independent binding gate.
        with self.assertRaises(ValueError):
            self.publish(binding="b" * 64)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(self.path))
            from .test_scoring_packet import rewritten
            foreign = rewritten(self.first[3], lambda m, blobs: m.update(binding="b" * 64))
            connection.sendall(struct.pack("!Q", len(foreign)) + foreign)
            self.wait_failed(server)

    def test_slow_partial_connection_is_bounded_and_close_interrupts_it(self):
        server = self.server(read_timeout_s=.05)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(self.path))
            connection.sendall(b"\0")
            self.wait_failed(server)
        result = server.close()
        self.assertFalse(result["worker_alive"])
        server = self.server(read_timeout_s=3.)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(self.path))
            connection.sendall(b"\0")
            start = time.monotonic()
            result = server.close()
            self.assertLess(time.monotonic() - start, 1.)
        self.assertFalse(result["worker_alive"])

    def test_existing_socket_or_file_never_replaced(self):
        server = self.server()
        with self.assertRaises(B.BrokerError):
            B.Server(self.path, binding=BINDING).start()
        self.publish()
        self.assertEqual(server.take(self.first[0]), self.first[3])
        server.close()
        with self.path.open("xb") as stream:
            stream.write(b"owned existing file")
        with self.assertRaises(B.BrokerError):
            B.Server(self.path, binding=BINDING).start()
        self.assertEqual(self.path.read_bytes(), b"owned existing file")

    def test_close_reports_and_retains_unconsumed_packet(self):
        server = self.server()
        self.publish()
        result = server.close()
        self.assertEqual(result["payloads_retained_for_unconsumed"], 1)
        self.assertFalse(result["all_consumed"])
        self.assertEqual(server.take(self.first[0]), self.first[3])
        self.assertTrue(server.stats()["all_consumed"])

    def test_long_shared_directory_endpoint_and_symlink_refusal(self):
        directory = Path(self.temporary.name) / ("segment" * 15) / ("another" * 15)
        directory.mkdir(parents=True)
        self.path = directory / "scoring.sock"
        self.assertGreater(len(str(self.path)), 108)
        server = self.server()
        self.publish()
        self.assertEqual(server.take(self.first[0]), self.first[3])
        server.close()
        link = Path(self.temporary.name) / "link"
        link.symlink_to(directory, target_is_directory=True)
        with self.assertRaises(B.BrokerError):
            B.Server(link / "scoring.sock", binding=BINDING).start()
        with self.assertRaises(OSError):
            B.publish(link / "scoring.sock", self.first[3], binding=BINDING)


if __name__ == "__main__":
    unittest.main()
