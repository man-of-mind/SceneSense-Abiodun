"""Offline packet fixtures; these tests never open a capture or network socket."""
import hashlib
from pathlib import Path
import socket
import struct
import tempfile
import unittest
import uuid

from rl_agent.splitfusion_transport_diagnostic_v1 import packet_capture as P


def ip_packet(payload, *, ident=7, offset=0, more=False, src='10.0.0.2', dst='192.168.70.140'):
    flags = offset // 8 | (0x2000 if more else 0)
    return struct.pack('!BBHHHBBH4s4s', 0x45, 0, 20 + len(payload), ident, flags,
                       64, 17, 0, socket.inet_aton(src), socket.inet_aton(dst)) + payload


def udp(payload, dport=51002):
    return struct.pack('!HHHH', 40123, dport, 8 + len(payload), 0) + payload


def pcap(records, *, linktype=101, endian='<', nanoseconds=False, snaplen=256):
    magic = 0xa1b23c4d if nanoseconds else 0xa1b2c3d4
    data = struct.pack(endian + 'IHHIIII', magic, 2, 4, 0, 0, snaplen, linktype)
    for ts, packet in records:
        if linktype == 1:
            packet = b'\0' * 12 + b'\x08\x00' + packet
        elif linktype == 113:
            packet = b'\0' * 14 + b'\x08\x00' + packet
        elif linktype == 276:
            packet = b'\x08\x00' + b'\0' * 18 + packet
        seconds, fraction = divmod(ts, 10**9)
        if not nanoseconds:
            fraction //= 1000
        captured = packet[:snaplen]
        data += struct.pack(endian + 'IIII', seconds, fraction, len(captured), len(packet)) + captured
    return data


class PacketCaptureTests(unittest.TestCase):
    def parse(self, data):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'fixture.pcap'
            target.write_bytes(data)
            return P.parse_capture(target)

    def test_linktypes_endianness_and_timestamp_precision(self):
        payload = struct.pack('!IHH', 1502, 1, 2) + bytes(range(128))
        for linktype in (101, 228, 1, 113, 276):
            for endian in ('<', '>'):
                with self.subTest(linktype=linktype, endian=endian):
                    result = self.parse(pcap([(1_234_567_890, ip_packet(udp(payload)))],
                        linktype=linktype, endian=endian, nanoseconds=True))
                    row = result['datagrams'][0]
                    self.assertEqual(row['first_wall_ns'], 1_234_567_890)
                    self.assertEqual(row['message_id'], 1502)
                    self.assertEqual(row['chunk_index'], 1)
                    self.assertTrue(row['wire_complete'])

    def test_snaplen_hash_is_prefix_only(self):
        payload = struct.pack('!IHH', 42, 0, 1) + b'a' * 4000
        result = self.parse(pcap([(1_000_000_000, ip_packet(udp(payload)))]))
        row = result['datagrams'][0]
        self.assertFalse(row['full_udp_payload_captured'])
        self.assertNotIn('udp_payload_sha256', row)
        self.assertEqual(row['payload_prefix_bytes'], 128)
        self.assertEqual(row['payload_prefix_sha256'], hashlib.sha256(payload[:128]).hexdigest())
        self.assertTrue(row['wire_complete'])
        self.assertEqual(row['snaplen_truncated_fragments'], 1)

    def test_fragment_wire_coverage_and_out_of_order(self):
        payload = struct.pack('!IHH', 42, 0, 1) + b'a' * 3000
        datagram = udp(payload)
        fragments = [ip_packet(datagram[:1480], more=True),
                     ip_packet(datagram[1480:2960], offset=1480, more=True),
                     ip_packet(datagram[2960:], offset=2960)]
        for order in ((0, 1, 2), (1, 0, 2), (2, 1, 0)):
            result = self.parse(pcap([(1_000_000_000 + n*1000, fragments[i])
                                     for n, i in enumerate(order)]))
            self.assertEqual(len(result['datagrams']), 1)
            row = result['datagrams'][0]
            self.assertTrue(row['wire_complete'])
            self.assertFalse(row['full_udp_payload_captured'])
            self.assertEqual(row['fragment_count'], 3)
            self.assertEqual(row['message_id'], 42)

    def test_missing_fragment_never_claims_complete(self):
        data = udp(struct.pack('!IHH', 42, 0, 1) + b'a'*3000)
        result = self.parse(pcap([(1_000_000_000, ip_packet(data[:1480], more=True)),
                                 (1_001_000_000, ip_packet(data[2960:], offset=2960))]))
        self.assertFalse(result['datagrams'][0]['wire_complete'])

    def test_duplicate_fragment_is_ambiguous(self):
        first = ip_packet(udp(struct.pack('!IHH', 42, 0, 1) + b'a'*3000)[:1480], more=True)
        result = self.parse(pcap([(1_000_000_000, first), (1_001_000_000, first)]))
        self.assertTrue(result['datagrams'][0]['ambiguous'])
        self.assertEqual(P.join_uplink(result, result)['joined'], [])

    def test_feedback_full_digest_and_identity(self):
        session = uuid.UUID('12345678-1234-5678-1234-567812345678')
        body = P.FB.pack(b'R4FB', 1, 1, 2, 0, 0, 5000, 0xffffffff,
                         11, 12, 42, 19, 100, session.bytes, b'a'*32, b'b'*32, .7)
        payload = body + hashlib.sha256(body).digest()
        for linktype in (101, 276):
            result = self.parse(pcap([(1_000_000_000, ip_packet(udp(payload, dport=51014)))],
                                    linktype=linktype))
            row = result['datagrams'][0]
            self.assertTrue(row['feedback_digest_verified'])
            self.assertTrue(row['full_udp_payload_captured'])
            self.assertEqual(row['frame_id'], 42)
            self.assertEqual(row['session_uuid'], str(session))
        bad = payload[:-1] + bytes([payload[-1] ^ 1])
        result = self.parse(pcap([(1_000_000_000, ip_packet(udp(bad, dport=51014)))]))
        self.assertFalse(result['datagrams'][0]['feedback_digest_verified'])

    def test_unique_join_requires_prefix_and_chunk_identity(self):
        packet = ip_packet(udp(struct.pack('!IHH', 42, 0, 1) + b'a' * 3000))
        ue = self.parse(pcap([(1_000_000_000, packet)]))
        edge = self.parse(pcap([(1_050_000_000, packet)], linktype=276))
        joined = P.join_uplink(ue, edge)['joined']
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0]['first_to_first_ms'], 50.)
        self.assertFalse(joined[0]['full_payload_integrity_verified'])
        edge['datagrams'][0]['payload_prefix_sha256'] = 'bad'
        self.assertEqual(P.join_uplink(ue, edge)['joined'], [])

    def test_reused_ip_id_after_complete_is_new_datagram(self):
        packets = [ip_packet(udp(struct.pack('!IHH', frame, 0, 1) + b'a'*128)) for frame in (42, 43)]
        result = self.parse(pcap([(1_000_000_000 + i*1_000_000, p) for i, p in enumerate(packets)]))
        self.assertEqual([d['message_id'] for d in result['datagrams']], [42, 43])

    def test_rejects_truncated_files(self):
        packet = ip_packet(udp(struct.pack('!IHH', 42, 0, 1) + b'a'*128))
        complete = pcap([(1_000_000_000, packet)])
        for data in (b'', complete[:23], complete[:30], complete[:-1]):
            with self.subTest(length=len(data)), self.assertRaises(ValueError):
                self.parse(data)

    def test_commands_are_bounded_and_include_fragment_filter(self):
        commands = P.capture_commands('/tmp/example-capture', edge_pid=123,
                                     ue_ip='10.0.0.2', edge_ip='192.168.70.140')
        self.assertIn('oaitun_ue1', commands['ue'])
        self.assertIn('nsenter', commands['edge'])
        self.assertIn('123', commands['edge'])
        for command in commands.values():
            self.assertIn('--signal=INT', command)
            self.assertIn(str(P.PACKET_LIMIT), command)
            self.assertIn('ip[6:2] & 0x1fff != 0', command[-1])
            self.assertNotIn('-W', command)
        with self.assertRaises(ValueError):
            P.capture_filter('10.0.0.2; anything', '192.168.70.140')
        with self.assertRaises(ValueError):
            P.capture_commands('/tmp/example', edge_pid=1, ue_ip='10.0.0.2', edge_ip='1.1.1.1', duration_s=901)


if __name__ == '__main__':
    unittest.main()
