"""Offline adversarial tests for the first-real split-host tensor proof."""

from __future__ import annotations

import hashlib
from pathlib import Path
import socket
import struct
from tempfile import TemporaryDirectory
import unittest
import uuid

from phase2_map_sharing.transport import CHUNK_HEADER, chunk_payload
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
    continuous_execution_v2 as X,
)
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1

from . import first_real_tensor_proof_v1 as P


SOURCE = "10.0.0.2"
DESTINATION = "192.168.70.140"
SOURCE_PORT = 41000
DESTINATION_PORT = 51002
FRAME_ID = 77


def _fixture(*, q_e4: int = 3000, inner_suffix: bytes = b""):
    inner = b"real-tensor-payload:" + bytes(range(256)) * 3 + inner_suffix
    envelope = X.ExecutionEnvelopeV3(
        mode_id=11,
        q_e4=q_e4,
        keep_count=15053,
        anchor_action_id=None,
        reward_requested=True,
        session_uuid=str(uuid.UUID(int=1)),
        controller_lineage_sha256="a" * 64,
        decision_seq=4,
        ticket_seq=4,
        frame_id=FRAME_ID,
        tensor_seq=9,
        capture_timestamp_ns=1_790_000_000_000_000_000,
        execution_bundle_sha256="b" * 64,
        inner_payload_sha256=hashlib.sha256(inner).hexdigest(),
        inner_payload=inner,
    )
    context = build_frame_context_v1(
        stream_id="ue288_split_host_proof",
        frame_id=envelope.frame_id,
        sequence_id=envelope.tensor_seq,
        capture_timestamp_ns=envelope.capture_timestamp_ns,
        ego_world_x=1.0,
        ego_world_y=2.0,
        ego_world_z=0.1,
        ego_world_pitch=0.0,
        ego_world_yaw=90.0,
        ego_world_roll=0.0,
    )
    wire = W.pack_sfd4(envelope, context)
    expected = {
        "session_uuid": envelope.session_uuid,
        "controller_lineage_sha256": envelope.controller_lineage_sha256,
        "decision_seq": envelope.decision_seq,
        "ticket_seq": envelope.ticket_seq,
        "frame_id": envelope.frame_id,
        "tensor_seq": envelope.tensor_seq,
        "capture_timestamp_ns": envelope.capture_timestamp_ns,
        "mode_id": envelope.mode_id,
        "q_e4": envelope.q_e4,
        "keep_count": envelope.keep_count,
        "anchor_action_id": envelope.anchor_action_id,
        "execution_bundle_sha256": envelope.execution_bundle_sha256,
    }
    frame = dict(expected)
    frame["action"] = {"keep_count": envelope.keep_count}
    identity = dict(expected)
    identity.update({
        "reward_requested": True,
        "frame_kind": "POLICY_DECISION",
    })
    ue_evidence = {
        "frames": [{**frame, "reward_requested": True}],
        "transmitted_identities": [{"run4_identity": identity}],
    }
    return wire, expected, ue_evidence


def _ipv4(payload: bytes, *, identification: int, offset: int = 0,
          more: bool = False) -> bytes:
    if more and (not payload or len(payload) % 8):
        raise AssertionError("non-final fixture fragment must be 8-byte aligned")
    if offset % 8:
        raise AssertionError("fixture fragment offset must be 8-byte aligned")
    fragment_word = (offset // 8) | (0x2000 if more else 0)
    header = struct.pack(
        "!BBHHHBBH4s4s",
        0x45, 0, 20 + len(payload), identification, fragment_word,
        64, 17, 0, socket.inet_aton(SOURCE), socket.inet_aton(DESTINATION),
    )
    return header + payload


def _udp(chunk: bytes) -> bytes:
    return struct.pack(
        "!HHHH", SOURCE_PORT, DESTINATION_PORT, 8 + len(chunk), 0,
    ) + chunk


def _packets_for_wire(wire: bytes, *, missing_fragment: bool = False,
                      conflicting_fragment: bool = False) -> list[bytes]:
    # Capacity is ceil(len/2), hence exactly two production !IHH chunks.
    chunks = chunk_payload(
        wire, message_id=FRAME_ID,
        chunk_bytes=((len(wire) + 1) // 2) + CHUNK_HEADER.size,
    )
    if len(chunks) != 2:
        raise AssertionError("fixture must have exactly two chunks")

    # Application chunk 1 arrives before chunk 0, followed by an identical
    # duplicate.  This is accepted but counted, and cannot change the wire.
    records = [
        _ipv4(_udp(chunks[1]), identification=201),
        _ipv4(_udp(chunks[1]), identification=202),
    ]

    # Chunk 0 is an IPv4-fragmented UDP datagram.  Its final fragment arrives
    # first, then an identical duplicate, then the first fragment.
    udp0 = _udp(chunks[0])
    split = 32
    first, final = udp0[:split], udp0[split:]
    records.append(_ipv4(final, identification=200, offset=split))
    duplicate = bytearray(final)
    if conflicting_fragment:
        duplicate[-1] ^= 1
    records.append(_ipv4(bytes(duplicate), identification=200, offset=split))
    if not missing_fragment:
        records.append(_ipv4(first, identification=200, more=True))
    return records


def _pcap(packets: list[bytes], *, linktype: int) -> bytes:
    body = bytearray(struct.pack(
        "<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, linktype))
    for index, packet in enumerate(packets):
        stored = packet if linktype == 101 else b"\0" * 12 + b"\x08\x00" + packet
        body.extend(struct.pack("<IIII", 100 + index, index, len(stored), len(stored)))
        body.extend(stored)
    return bytes(body)


def _write_capture(path: Path, wire: bytes, *, linktype: int,
                   missing_fragment: bool = False,
                   conflicting_fragment: bool = False) -> None:
    path.write_bytes(_pcap(_packets_for_wire(
        wire,
        missing_fragment=missing_fragment,
        conflicting_fragment=conflicting_fragment,
    ), linktype=linktype))


class FirstRealTensorProofTests(unittest.TestCase):
    def _paths(self, root: Path, *, local_wire: bytes, remote_wire: bytes | None = None,
               missing_fragment: bool = False,
               conflicting_fragment: bool = False):
        local = root / "local_oaitun_raw.pcap"
        remote = root / "remote_edge_ethernet.pcap"
        _write_capture(
            local, local_wire, linktype=101,
            missing_fragment=missing_fragment,
            conflicting_fragment=conflicting_fragment,
        )
        _write_capture(remote, remote_wire or local_wire, linktype=1)
        return local, remote

    def test_raw_local_ethernet_remote_fragmented_out_of_order_duplicates_pass(self):
        wire, expected, _ue = _fixture()
        with TemporaryDirectory() as temporary:
            local, remote = self._paths(
                Path(temporary), local_wire=wire)
            report, capture, receipt = P.reconcile_first_real_tensor(
                local_pcap=local, remote_pcap=remote,
                expected_identity=expected)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["proof_subject"],
                         "FIRST_REAL_REWARD_REQUESTED_SFD4")
        self.assertTrue(report["identity_hash_count_bytes_exact"])
        self.assertEqual(report["local_oaitun"]["duplicate_chunks"], 1)
        self.assertEqual(report["local_oaitun"]["fragmented_datagrams"], 1)
        self.assertEqual(report["local_oaitun"]["duplicate_ip_fragments"], 1)
        self.assertEqual(capture.shared_identity(), receipt.shared_identity())
        self.assertFalse(report["cross_host_latency_computed"])
        self.assertFalse(report["radio_tensor_path"]["cross_host_latency_computed"])

    def test_missing_fragment_fails_closed(self):
        wire, expected, _ue = _fixture()
        with TemporaryDirectory() as temporary:
            local, remote = self._paths(
                Path(temporary), local_wire=wire, missing_fragment=True)
            with self.assertRaisesRegex(P.FirstRealTensorProofError,
                                        "incomplete IPv4 fragment"):
                P.reconcile_first_real_tensor(
                    local_pcap=local, remote_pcap=remote,
                    expected_identity=expected)

    def test_conflicting_duplicate_fragment_fails_closed(self):
        wire, expected, _ue = _fixture()
        with TemporaryDirectory() as temporary:
            local, remote = self._paths(
                Path(temporary), local_wire=wire,
                conflicting_fragment=True)
            with self.assertRaisesRegex(P.FirstRealTensorProofError,
                                        "conflicting duplicate IPv4 fragment"):
                P.reconcile_first_real_tensor(
                    local_pcap=local, remote_pcap=remote,
                    expected_identity=expected)

    def test_remote_wire_mismatch_fails_closed(self):
        wire, expected, _ue = _fixture()
        changed_wire, _changed, _changed_ue = _fixture(q_e4=3001)
        with TemporaryDirectory() as temporary:
            local, remote = self._paths(
                Path(temporary), local_wire=wire, remote_wire=changed_wire)
            with self.assertRaisesRegex(P.FirstRealTensorProofError,
                                        "do not match exactly"):
                P.reconcile_first_real_tensor(
                    local_pcap=local, remote_pcap=remote,
                    expected_identity=expected)

    def test_ue_q_mismatch_fails_closed(self):
        wire, expected, _ue = _fixture()
        expected = dict(expected, q_e4=3001)
        with TemporaryDirectory() as temporary:
            local, remote = self._paths(Path(temporary), local_wire=wire)
            with self.assertRaisesRegex(P.FirstRealTensorProofError, "q_e4"):
                P.reconcile_first_real_tensor(
                    local_pcap=local, remote_pcap=remote,
                    expected_identity=expected)

    def test_create_only_proof_and_no_cross_host_packet_timestamps(self):
        wire, _expected, ue = _fixture()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            local, remote = self._paths(root, local_wire=wire)
            output = root / "FIRST_REAL_TENSOR_PROOF.json"
            report, _capture, _receipt = P.write_reconciled_proof(
                output, local_pcap=local, remote_pcap=remote,
                ue_evidence=ue)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                P.write_reconciled_proof(
                    output, local_pcap=local, remote_pcap=remote,
                    ue_evidence=ue)
            self.assertEqual(output.read_bytes(), original)

        def all_keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield key
                    yield from all_keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from all_keys(child)

        keys = set(all_keys(report))
        self.assertNotIn("first_record", keys)
        self.assertNotIn("pcap_timestamp_ns", keys)
        self.assertNotIn("packet_timestamp_ns", keys)
        self.assertNotIn("remote_monotonic_ns", keys)
        self.assertNotIn("local_monotonic_ns", keys)
        self.assertFalse(report["cross_host_latency_computed"])


if __name__ == "__main__":
    unittest.main()
