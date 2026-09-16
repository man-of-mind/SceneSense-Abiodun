#!/usr/bin/env python3
"""Bounded packet evidence for compact edge-to-UE quality ACKs.

The quality sender lives in the edge container.  Its datagrams to the UE
control endpoint must therefore emerge from the qualified OAI downlink on
the host-owned ``oaitun_ue1`` created by the UE softmodem.  This module records
that interface, decodes only the two
versioned quality schemas, and joins their canonical digests to both edge and
UE durable ledgers.  Absence, capture loss, or a join mismatch fails closed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import socket
import struct
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import protocol


class PacketEvidenceError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketEvidenceError(message)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(dict(payload), sort_keys=True, indent=1) + "\n",
        encoding="utf-8",
    )
    os.link(temporary, path)
    temporary.unlink()


def _stop_root_group(process: subprocess.Popen[Any]) -> None:
    """Boundedly stop the sudo/tcpdump process group without a broad pkill."""

    pgid = int(process.pid)
    for sig, wait_s in ((signal.SIGINT, 5.0), (signal.SIGTERM, 2.0)):
        if process.poll() is not None:
            return
        subprocess.run(
            ("sudo", "-n", "kill", f"-{int(sig)}", "--", f"-{pgid}"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            process.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            pass
    if process.poll() is None:
        subprocess.run(
            ("sudo", "-n", "kill", "-9", "--", f"-{pgid}"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired as exc:
            raise PacketEvidenceError("tcpdump process group did not stop") from exc


class QualityAckCapture:
    """One create-only tcpdump bound to the UE tunnel and control endpoint."""

    def __init__(
        self, attempt_dir: Path, *, interface: str, ue_host: str, ue_port: int
    ) -> None:
        self.attempt_dir = Path(attempt_dir)
        self.interface = str(interface)
        self.ue_host = str(ue_host)
        self.ue_port = int(ue_port)
        self.pcap = self.attempt_dir / "quality_ack_oaitun_ue1.pcap"
        self.stderr_path = self.attempt_dir / "quality_ack_tcpdump.log"
        self._stderr: Any = None
        self.process: subprocess.Popen[Any] | None = None

    def link_check_argv(self) -> tuple[str, ...]:
        return (
            "sudo", "-n", "ip", "link", "show", self.interface,
        )

    def capture_argv(self) -> tuple[str, ...]:
        return (
            "sudo", "-n", "tcpdump", "-i", self.interface, "-nn", "-U", "-s", "0",
            "-w", str(self.pcap), "udp", "and", "dst", "host",
            self.ue_host, "and", "dst", "port", str(self.ue_port),
        )

    def start(self) -> None:
        _require(not self.pcap.exists(), f"capture already exists: {self.pcap}")
        link = subprocess.run(
            self.link_check_argv(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        _require(link.returncode == 0, f"UE tunnel unavailable: {self.interface}")
        self._stderr = self.stderr_path.open("xb")
        self.process = subprocess.Popen(
            self.capture_argv(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
            start_new_session=True,
        )
        time.sleep(0.25)
        _require(
            self.process.poll() is None,
            "tcpdump exited before the live cell started: "
            + self.stderr_path.read_text(encoding="utf-8", errors="replace"),
        )

    def stop(self) -> None:
        if self.process is not None:
            _stop_root_group(self.process)
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        # tcpdump may retain root ownership.  Make only this explicit artifact
        # readable by the unprivileged analyzer.
        if self.pcap.exists():
            subprocess.run(
                (
                    "sudo", "-n", "chown",
                    f"{os.getuid()}:{os.getgid()}", str(self.pcap),
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )


def _ipv4_payload(packet: bytes, linktype: int) -> bytes | None:
    if linktype == 1:  # Ethernet
        if len(packet) < 14:
            return None
        ethertype = struct.unpack("!H", packet[12:14])[0]
        offset = 14
        while ethertype in (0x8100, 0x88A8):
            if len(packet) < offset + 4:
                return None
            ethertype = struct.unpack("!H", packet[offset + 2:offset + 4])[0]
            offset += 4
        return packet[offset:] if ethertype == 0x0800 else None
    if linktype == 101:  # raw IPv4
        return packet
    if linktype == 113:  # Linux cooked v1
        return packet[16:] if len(packet) >= 16 and packet[14:16] == b"\x08\x00" else None
    if linktype == 276:  # Linux cooked v2
        return packet[20:] if len(packet) >= 20 and packet[0:2] == b"\x08\x00" else None
    raise PacketEvidenceError(f"unsupported pcap link type {linktype}")


def parse_quality_pcap(path: Path) -> tuple[int, list[dict[str, Any]]]:
    """Return packet count and decoded quality packets from classic pcap."""

    data = Path(path).read_bytes()
    _require(len(data) >= 24, "pcap is absent or truncated")
    magic = data[:4]
    formats = {
        b"\xd4\xc3\xb2\xa1": ("<", 1_000),
        b"\xa1\xb2\xc3\xd4": (">", 1_000),
        b"\x4d\x3c\xb2\xa1": ("<", 1),
        b"\xa1\xb2\x3c\x4d": (">", 1),
    }
    _require(magic in formats, "unsupported pcap magic")
    endian, fractional_to_ns = formats[magic]
    _, _, _, _, _, _, linktype = struct.unpack(endian + "IHHIIII", data[:24])
    cursor = 24
    packets = 0
    rows: list[dict[str, Any]] = []
    while cursor < len(data):
        _require(cursor + 16 <= len(data), "truncated pcap record header")
        seconds, fraction, captured, original = struct.unpack(
            endian + "IIII", data[cursor:cursor + 16]
        )
        cursor += 16
        _require(cursor + captured <= len(data), "truncated pcap packet")
        packet = data[cursor:cursor + captured]
        cursor += captured
        packets += 1
        ip = _ipv4_payload(packet, int(linktype))
        if ip is None or len(ip) < 20 or ip[0] >> 4 != 4 or ip[9] != 17:
            continue
        ihl = (ip[0] & 0x0F) * 4
        flags_fragment = struct.unpack("!H", ip[6:8])[0]
        _require((flags_fragment & 0x3FFF) == 0, "fragmented quality ACK observed")
        if len(ip) < ihl + 8:
            continue
        source_port, destination_port, udp_length, _ = struct.unpack(
            "!HHHH", ip[ihl:ihl + 8]
        )
        payload = ip[ihl + 8:ihl + int(udp_length)]
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict) or str(document.get("s") or "") not in {
            protocol.QUALITY_EVALUATED_ACK_SCHEMA,
            protocol.QUALITY_EVALUATION_FAILED_ACK_SCHEMA,
        }:
            continue
        protocol.validate(document)
        rows.append(
            {
                "packet_wall_ns": int(seconds) * 1_000_000_000
                + int(fraction) * int(fractional_to_ns),
                "source_ip": socket.inet_ntoa(ip[12:16]),
                "destination_ip": socket.inet_ntoa(ip[16:20]),
                "source_port": int(source_port),
                "destination_port": int(destination_port),
                "payload_bytes": len(payload),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "message_sha256": protocol.digest(document),
                "identity": list(protocol.identity(document)),
                "event": str(document["e"]),
                "detail_sha256": str(document["dh"]),
            }
        )
    return packets, rows


def _ledger_digests(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            str(row["message_sha256"])
            for row in csv.DictReader(handle)
            if row.get("message_sha256") and row.get("duplicate_identical") != "True"
        ]


def render_packet_evidence(
    attempt_dir: Path, *, ue_host: str, ue_port: int
) -> dict[str, Any]:
    attempt = Path(attempt_dir)
    pcap = attempt / "quality_ack_oaitun_ue1.pcap"
    edge_path = attempt / "direct_edge_map/quality_edge_report.json"
    ledger_path = attempt / "quality_feedback.csv"
    _require(pcap.is_file(), "quality ACK pcap was not preserved")
    _require(edge_path.is_file(), "quality edge report is absent")
    _require(ledger_path.is_file(), "quality UE ledger is absent")
    captured_packets, rows = parse_quality_pcap(pcap)
    edge = json.loads(edge_path.read_text(encoding="utf-8"))
    edge_digests = [str(item["sha256"]) for item in edge.get("messages", [])]
    edge_by_digest = {
        str(item["sha256"]): item for item in edge.get("messages", [])
    }
    _require(
        len(edge_by_digest) == len(edge_digests),
        "duplicate edge quality-message digest",
    )
    ue_digests = _ledger_digests(ledger_path)
    pcap_digests = [str(item["message_sha256"]) for item in rows]
    _require(rows, "no compact quality ACK traversed oaitun_ue1")
    _require(
        Counter(edge_digests) == Counter(pcap_digests) == Counter(ue_digests),
        "edge/UE/oaitun quality ACK digest populations do not reconcile",
    )
    _require(
        all(
            row["destination_ip"] == ue_host
            and int(row["destination_port"]) == int(ue_port)
            and int(row["payload_bytes"]) <= protocol.MAX_WIRE_BYTES
            and row["payload_sha256"] == row["message_sha256"]
            for row in rows
        ),
        "captured quality ACK destination, canonical wire image, or budget drifted",
    )
    for row in rows:
        sent = edge_by_digest[str(row["message_sha256"])]
        local = list(sent.get("socket_local") or ())
        _require(
            len(local) == 2
            and str(row["source_ip"]) == str(local[0])
            and int(row["source_port"]) == int(local[1]),
            "captured quality ACK source endpoint differs from edge send socket",
        )
    csv_path = attempt / "quality_ack_packets.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        fields = tuple(key for key in rows[0] if key != "identity") + ("identity_json",)
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: value for key, value in row.items() if key != "identity"},
                    "identity_json": json.dumps(row["identity"], separators=(",", ":")),
                }
            )
    report = {
        "schema": "scenesense.quality_ack_oaitun_packet_evidence.v1",
        "status": "PASS",
        "interface": "oaitun_ue1",
        "filter_destination": f"{ue_host}:{ue_port}",
        "pcap_packets": int(captured_packets),
        "quality_ack_packets": len(rows),
        "edge_messages": len(edge_digests),
        "ue_messages": len(ue_digests),
        "digest_multisets_equal": True,
        "max_payload_bytes": max(int(row["payload_bytes"]) for row in rows),
        "pcap_sha256": hashlib.sha256(pcap.read_bytes()).hexdigest(),
        "packet_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "capture_dropped_packets": 0,
    }
    # tcpdump prints its kernel drop counter at orderly SIGINT.  Fail closed
    # when the counter cannot be proved or is non-zero.
    log = (attempt / "quality_ack_tcpdump.log").read_text(
        encoding="utf-8", errors="replace"
    )
    dropped_lines = [line for line in log.splitlines() if "packets dropped by kernel" in line]
    _require(len(dropped_lines) == 1, "tcpdump drop counter is unavailable")
    dropped = int(dropped_lines[0].strip().split()[0])
    _require(dropped == 0, f"tcpdump dropped {dropped} quality packets")
    _atomic_json(attempt / "quality_ack_packet_evidence.json", report)
    return report
