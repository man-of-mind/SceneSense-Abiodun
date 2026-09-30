"""Offline dual-capture proof for the first real reward-requested SFD4.

The local observer captures oaitun_ue1 and the remote observer captures the
edge container eth0.  This module reconstructs IPv4 fragments and then the
production !IHH application chunks.  The reconstructed SFD4 must match the
durable UE identity and must be byte-identical at both observers.  Packet
timestamps are deliberately neither exposed nor compared across hosts.

Importing this module reads nothing and starts no process.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import socket
import struct
from typing import Any, Mapping

from phase2_map_sharing.transport import CHUNK_HEADER
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_quality_feedback_probe_v1 import packet_evidence as PE

from . import contract as C
from . import split_host_phase6_coordinator_v1 as CO


SCHEMA = "scenesense.run4.split_host_first_real_tensor_proof.v1"
MAX_CHUNKS = 4096
_PCAP_FORMATS = {
    b"\xd4\xc3\xb2\xa1": "<",
    b"\xa1\xb2\xc3\xd4": ">",
    b"\x4d\x3c\xb2\xa1": "<",
    b"\xa1\xb2\x3c\x4d": ">",
}
_EXPECTED_FIELDS = (
    "session_uuid", "controller_lineage_sha256", "decision_seq", "ticket_seq",
    "frame_id", "tensor_seq", "capture_timestamp_ns", "mode_id", "q_e4",
    "keep_count", "anchor_action_id", "execution_bundle_sha256",
)


class FirstRealTensorProofError(RuntimeError):
    """A capture was incomplete, ambiguous, corrupt, or did not reconcile."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FirstRealTensorProofError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class _IpPayload:
    first_record: int
    source_ip: str
    destination_ip: str
    payload: bytes
    fragmented: bool
    duplicate_fragments: int


@dataclass(frozen=True)
class _UdpDatagram:
    first_record: int
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    payload: bytes
    fragmented: bool
    duplicate_fragments: int


@dataclass(frozen=True)
class CapturedRewardTensorV1:
    observer: str
    interface: str
    source_ip: str
    destination_ip: str
    destination_port: int
    message_id: int
    session_uuid: str
    controller_lineage_sha256: str
    decision_seq: int
    ticket_seq: int
    frame_id: int
    tensor_seq: int
    capture_timestamp_ns: int
    mode_id: int
    q_e4: int
    keep_count: int
    anchor_action_id: int | None
    execution_bundle_sha256: str
    inner_payload_sha256: str
    wire_sha256: str
    wire_bytes: int
    datagram_count: int
    duplicate_chunks: int
    fragmented_datagrams: int
    duplicate_ip_fragments: int
    pcap_sha256: str

    def wire_identity(self) -> tuple[Any, ...]:
        return (
            self.session_uuid, self.controller_lineage_sha256,
            self.decision_seq, self.ticket_seq, self.frame_id, self.tensor_seq,
            self.capture_timestamp_ns, self.mode_id, self.q_e4, self.keep_count,
            self.anchor_action_id, self.execution_bundle_sha256,
            self.inner_payload_sha256, self.message_id, self.wire_sha256,
            self.wire_bytes, self.datagram_count,
        )

    def as_observation(self) -> CO.RadioTensorObservationV1:
        return CO.RadioTensorObservationV1(
            observer=self.observer, session_uuid=self.session_uuid,
            frame_id=self.frame_id, tensor_seq=self.tensor_seq,
            payload_sha256=self.wire_sha256,
            datagram_count=self.datagram_count, payload_bytes=self.wire_bytes,
            destination=self.destination_ip,
            destination_port=self.destination_port, interface=self.interface,
        ).validate()

    def as_evidence(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _pcap_ipv4(path: Path) -> list[tuple[int, bytes]]:
    data = Path(path).read_bytes()
    _require(len(data) >= 24 and data[:4] in _PCAP_FORMATS,
             "capture is absent, truncated, or not classic pcap")
    endian = _PCAP_FORMATS[data[:4]]
    try:
        _magic, major, minor, _zone, _sigfigs, _snaplen, linktype = struct.unpack(
            endian + "IHHIIII", data[:24])
    except struct.error as exc:
        raise FirstRealTensorProofError("classic pcap header is corrupt") from exc
    _require((major, minor) == (2, 4), "classic pcap version drift")
    cursor = 24
    record = 0
    rows: list[tuple[int, bytes]] = []
    while cursor < len(data):
        _require(cursor + 16 <= len(data), "truncated pcap record header")
        _sec, _fraction, captured, original = struct.unpack(
            endian + "IIII", data[cursor:cursor + 16])
        cursor += 16
        _require(captured == original, "snaplen-truncated packet in tensor capture")
        _require(cursor + captured <= len(data), "truncated pcap packet")
        packet = data[cursor:cursor + captured]
        cursor += captured
        try:
            ip = PE._ipv4_payload(packet, int(linktype))
        except PE.PacketEvidenceError as exc:
            raise FirstRealTensorProofError(str(exc)) from exc
        if ip is not None:
            rows.append((record, ip))
        record += 1
    _require(cursor == len(data), "pcap trailing bytes")
    return rows


def _ipv4_datagrams(path: Path, *, destination: str) -> list[_IpPayload]:
    pending: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    complete: list[_IpPayload] = []
    for record, raw in _pcap_ipv4(path):
        if len(raw) < 20 or raw[0] >> 4 != 4:
            continue
        ihl = (raw[0] & 0x0F) * 4
        _require(ihl >= 20 and len(raw) >= ihl, "invalid IPv4 header length")
        total_length = struct.unpack("!H", raw[2:4])[0]
        _require(total_length >= ihl and total_length <= len(raw),
                 "truncated IPv4 packet")
        protocol = int(raw[9])
        source = socket.inet_ntoa(raw[12:16])
        target = socket.inet_ntoa(raw[16:20])
        if protocol != 17 or target != destination:
            continue
        identification = struct.unpack("!H", raw[4:6])[0]
        fragment_word = struct.unpack("!H", raw[6:8])[0]
        offset = (fragment_word & 0x1FFF) * 8
        more = bool(fragment_word & 0x2000)
        piece = bytes(raw[ihl:total_length])
        if more:
            _require(piece and len(piece) % 8 == 0,
                     "non-final IPv4 fragment is not 8-byte aligned")
        if offset == 0 and not more:
            complete.append(_IpPayload(
                record, source, target, piece, False, 0))
            continue

        key = (source, target, protocol, identification)
        item = pending.setdefault(key, {
            "first": record, "pieces": {}, "final": None, "duplicates": 0,
        })
        item["first"] = min(int(item["first"]), record)
        pieces: dict[int, bytes] = item["pieces"]
        if offset in pieces:
            _require(pieces[offset] == piece,
                     "conflicting duplicate IPv4 fragment")
            item["duplicates"] += 1
        else:
            end = offset + len(piece)
            for other_offset, other_piece in pieces.items():
                other_end = other_offset + len(other_piece)
                _require(end <= other_offset or offset >= other_end,
                         "overlapping IPv4 fragments")
            pieces[offset] = piece
        if not more:
            final = offset + len(piece)
            _require(item["final"] in (None, final),
                     "IPv4 final length changed")
            item["final"] = final
        if item["final"] is None:
            continue
        cursor = 0
        ordered: list[bytes] = []
        contiguous = True
        for start in sorted(pieces):
            if start != cursor:
                contiguous = False
                break
            ordered.append(pieces[start])
            cursor += len(pieces[start])
        if not contiguous or cursor < int(item["final"]):
            continue
        _require(cursor == int(item["final"]),
                 "IPv4 fragments exceed final length")
        complete.append(_IpPayload(
            int(item["first"]), source, target, b"".join(ordered), True,
            int(item["duplicates"])))
        del pending[key]
    _require(not pending, "incomplete IPv4 fragment set in tensor capture")
    return sorted(complete, key=lambda row: row.first_record)


def _udp_datagrams(path: Path, *, destination: str,
                   destination_port: int) -> list[_UdpDatagram]:
    rows: list[_UdpDatagram] = []
    for item in _ipv4_datagrams(path, destination=destination):
        _require(len(item.payload) >= 8, "reassembled UDP datagram is truncated")
        source_port, target_port, length, _checksum = struct.unpack(
            "!HHHH", item.payload[:8])
        _require(8 <= length == len(item.payload), "UDP length mismatch")
        if target_port != destination_port:
            continue
        rows.append(_UdpDatagram(
            item.first_record, item.source_ip, item.destination_ip,
            source_port, target_port, item.payload[8:length], item.fragmented,
            item.duplicate_fragments))
    return rows


def _single_reward_tensor(path: Path, *, observer: str,
                          interface: str) -> CapturedRewardTensorV1:
    topology = C.default_topology()
    datagrams = _udp_datagrams(
        path, destination=topology.edge_ip,
        destination_port=CO.EDGE_RECEIVE_PORT)
    _require(datagrams, "no tensor datagram reached the registered edge")
    pending: dict[tuple[str, int, int], dict[str, Any]] = {}
    completed: list[tuple[int, str, int, bytes, dict[str, Any]]] = []
    for datagram in datagrams:
        _require(len(datagram.payload) >= CHUNK_HEADER.size,
                 "tensor datagram is shorter than !IHH")
        message_id, index, total = CHUNK_HEADER.unpack_from(datagram.payload)
        _require(0 < total <= MAX_CHUNKS and index < total,
                 "invalid !IHH chunk index/count")
        key = (datagram.source_ip, datagram.source_port, message_id)
        item = pending.setdefault(key, {
            "total": total, "first": datagram.first_record, "chunks": {},
            "duplicates": 0, "fragmented": 0, "duplicate_fragments": 0,
            "destination_ip": datagram.destination_ip,
            "destination_port": datagram.destination_port,
        })
        _require(item["total"] == total, "chunk count changed within a message")
        item["first"] = min(int(item["first"]), datagram.first_record)
        chunks: dict[int, bytes] = item["chunks"]
        piece = datagram.payload[CHUNK_HEADER.size:]
        if index in chunks:
            _require(chunks[index] == piece, "conflicting duplicate !IHH chunk")
            item["duplicates"] += 1
        else:
            chunks[index] = piece
            item["fragmented"] += int(datagram.fragmented)
            item["duplicate_fragments"] += datagram.duplicate_fragments
        if len(chunks) == total:
            wire = b"".join(chunks[position] for position in range(total))
            completed.append((
                int(item["first"]), datagram.source_ip, message_id, wire, item))
            del pending[key]
    _require(not pending, "incomplete !IHH tensor message in capture")

    rewards: list[CapturedRewardTensorV1] = []
    for _first, source_ip, message_id, wire, item in sorted(completed):
        try:
            envelope, _context = W.unpack_sfd4(wire)
        except W.WireError as exc:
            raise FirstRealTensorProofError(
                f"captured tensor is not valid SFD4: {exc}") from exc
        _require(message_id == envelope.frame_id,
                 "!IHH message_id differs from SFD4 frame_id")
        if not envelope.reward_requested:
            continue
        rewards.append(CapturedRewardTensorV1(
            observer=observer, interface=interface, source_ip=source_ip,
            destination_ip=str(item["destination_ip"]),
            destination_port=int(item["destination_port"]),
            message_id=message_id, session_uuid=envelope.session_uuid,
            controller_lineage_sha256=envelope.controller_lineage_sha256,
            decision_seq=envelope.decision_seq, ticket_seq=envelope.ticket_seq,
            frame_id=envelope.frame_id, tensor_seq=envelope.tensor_seq,
            capture_timestamp_ns=envelope.capture_timestamp_ns,
            mode_id=envelope.mode_id, q_e4=envelope.q_e4,
            keep_count=envelope.keep_count,
            anchor_action_id=envelope.anchor_action_id,
            execution_bundle_sha256=envelope.execution_bundle_sha256,
            inner_payload_sha256=envelope.inner_payload_sha256,
            wire_sha256=hashlib.sha256(wire).hexdigest(),
            wire_bytes=len(wire), datagram_count=int(item["total"]),
            duplicate_chunks=int(item["duplicates"]),
            fragmented_datagrams=int(item["fragmented"]),
            duplicate_ip_fragments=int(item["duplicate_fragments"]),
            pcap_sha256=_sha256_file(path),
        ))
    _require(len(rewards) == 1,
             f"expected exactly one reward-requested SFD4, got {len(rewards)}")
    return rewards[0]


def expected_first_reward_from_ue_evidence(
    document: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Derive the one real policy-decision identity from durable UE evidence."""
    _require(type(document) is dict, "UE evidence must be an exact dictionary")
    frames = [row for row in document.get("frames") or ()
              if row.get("reward_requested") is True]
    transmitted = [
        row.get("run4_identity")
        for row in document.get("transmitted_identities") or ()
        if type(row.get("run4_identity")) is dict
        and row["run4_identity"].get("reward_requested") is True
    ]
    _require(len(frames) == 1 and len(transmitted) == 1,
             "bounded handshake must contain exactly one real reward tensor")
    frame, identity = frames[0], transmitted[0]
    _require(identity.get("frame_kind") == "POLICY_DECISION",
             "reward tensor is not a policy decision")
    action = frame.get("action")
    _require(type(action) is dict, "reward frame action is absent")
    expected = {
        "session_uuid": identity.get("session_uuid"),
        "controller_lineage_sha256": identity.get("controller_lineage_sha256"),
        "decision_seq": identity.get("decision_seq"),
        "ticket_seq": identity.get("ticket_seq"),
        "frame_id": identity.get("frame_id"),
        "tensor_seq": identity.get("tensor_seq"),
        "capture_timestamp_ns": frame.get("capture_timestamp_ns"),
        "mode_id": identity.get("mode_id"), "q_e4": identity.get("q_e4"),
        "keep_count": action.get("keep_count"),
        "anchor_action_id": identity.get("anchor_action_id"),
        "execution_bundle_sha256": identity.get("execution_bundle_sha256"),
    }
    for name in ("session_uuid", "decision_seq", "ticket_seq", "frame_id",
                 "tensor_seq", "mode_id", "q_e4", "anchor_action_id",
                 "execution_bundle_sha256"):
        _require(frame.get(name) == expected[name],
                 f"UE frame/transmitted identity drift: {name}")
    return expected


def reconcile_first_real_tensor(
    *, local_pcap: Path, remote_pcap: Path,
    expected_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], CO.RadioTensorObservationV1,
           CO.RadioTensorObservationV1]:
    """Reconcile one reward-requested SFD4 across radio and remote edge."""
    expected = dict(expected_identity)
    _require(set(expected) == set(_EXPECTED_FIELDS),
             "expected reward identity fields drifted")
    local = _single_reward_tensor(
        Path(local_pcap), observer="W10275_OAITUN_CAPTURE",
        interface=CO.POLICY_INTERFACE)
    remote = _single_reward_tensor(
        Path(remote_pcap), observer="L10319_EDGE_RECEIPT",
        interface="REMOTE_EDGE_RECEIVER")
    _require(local.wire_identity() == remote.wire_identity(),
             "local oaitun and remote eth0 SFD4 do not match exactly")
    for name in _EXPECTED_FIELDS:
        _require(getattr(local, name) == expected[name],
                 f"captured SFD4 differs from UE evidence: {name}")
    capture, receipt = local.as_observation(), remote.as_observation()
    radio = dict(CO.validate_radio_tensor_path(capture, receipt))
    report = {
        "schema": SCHEMA, "status": "PASS",
        "proof_subject": "FIRST_REAL_REWARD_REQUESTED_SFD4",
        "expected_identity": expected,
        "local_oaitun": local.as_evidence(),
        "remote_edge_eth0": remote.as_evidence(),
        "radio_tensor_path": radio,
        "identity_hash_count_bytes_exact": True,
        "cross_host_latency_computed": False,
    }
    return report, capture, receipt


def write_reconciled_proof(
    output: Path, *, local_pcap: Path, remote_pcap: Path,
    ue_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], CO.RadioTensorObservationV1,
           CO.RadioTensorObservationV1]:
    """Create one immutable proof JSON; an existing path is never replaced."""
    result = reconcile_first_real_tensor(
        local_pcap=local_pcap, remote_pcap=remote_pcap,
        expected_identity=expected_first_reward_from_ue_evidence(ue_evidence))
    report, capture, receipt = result
    with Path(output).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report, capture, receipt


__all__ = [
    "SCHEMA", "FirstRealTensorProofError", "CapturedRewardTensorV1",
    "expected_first_reward_from_ue_evidence", "reconcile_first_real_tensor",
    "write_reconciled_proof",
]
