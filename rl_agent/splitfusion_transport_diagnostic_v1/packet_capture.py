"""Owned, bounded passive captures and a dependency-free classic-pcap parser.

UL captures preserve the production !IHH chunk identity and a 128-byte payload
prefix, not full application payloads. Fragment wire coverage can be proven
from IP lengths even when snaplen truncates bytes; that is not payload integrity.
No command here changes interfaces, routes, radio settings or socket behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import pwd
import re
import signal
import socket
import struct
import subprocess
import time
import uuid

SNAPLEN = 256
PREFIX_BYTES = 128
PACKET_LIMIT = 400_000  # <= 108,800,024 bytes per classic pcap, including headers
FB = struct.Struct('<4sHBBBBHIQQQQQ16s32s32sd')
MAGICS = {b'\xd4\xc3\xb2\xa1': ('<', 1000), b'\xa1\xb2\xc3\xd4': ('>', 1000),
          b'\x4d\x3c\xb2\xa1': ('<', 1), b'\xa1\xb2\x3c\x4d': ('>', 1)}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def process_identity(pid):
    directory = Path('/proc') / str(int(pid))
    return {'pid': int(pid),
            'starttime': directory.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19],
            'cmdline': directory.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()}


def capture_filter(ue_ip, edge_ip, edge_port=51002, feedback_port=51014):
    ue_ip, edge_ip = str(ipaddress.IPv4Address(ue_ip)), str(ipaddress.IPv4Address(edge_ip))
    if not all(0 < int(p) < 65536 for p in (edge_port, feedback_port)):
        raise ValueError('Invalid UDP port')
    # A port-only BPF expression silently loses all noninitial IPv4 fragments.
    return (f'ip and ip proto 17 and host {ue_ip} and host {edge_ip} and '
            f'((ip[6:2] & 0x1fff != 0) or '
            f'(udp dst port {int(edge_port)} or udp dst port {int(feedback_port)}))')


def capture_commands(output, *, edge_pid, ue_ip, edge_ip, edge_port=51002,
                     feedback_port=51014, duration_s=600):
    if not 1 <= int(duration_s) <= 900:
        raise ValueError('Capture duration must be 1..900 seconds')
    bpf = capture_filter(ue_ip, edge_ip, edge_port, feedback_port)
    output = Path(output).resolve()
    try:
        username = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:  # The command sandbox may not expose host NSS through Python.
        username = os.environ.get('LOGNAME', os.environ.get('USER', ''))
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]*\$?', username):
        raise ValueError('Cannot resolve the invoking user for tcpdump privilege drop')
    common = ['tcpdump', '-n', '-p', '-U', '-B', '8192', '-s', str(SNAPLEN),
              '-c', str(PACKET_LIMIT), '-Z', username]
    prefix = ['sudo', '-n', 'timeout', '--signal=INT', '--kill-after=5s', f'{int(duration_s)}s']
    return {
        'ue': prefix + common + ['-i', 'oaitun_ue1', '-w', str(output / 'ue.pcap'), bpf],
        'edge': prefix + ['nsenter', '-t', str(int(edge_pid)), '-n'] + common
                + ['-i', 'any', '-w', str(output / 'edge.pcap'), bpf],
    }


def tcpdump_stats(path):
    text = Path(path).read_text(errors='replace')
    result = {}
    for key, expression in [('captured', r'(\d+) packets? captured'),
                            ('received_by_filter', r'(\d+) packets? received by filter'),
                            ('kernel_drops', r'(\d+) packets? dropped by kernel')]:
        match = re.search(expression, text)
        result[key] = int(match.group(1)) if match else None
    result['listening'] = 'listening on ' in text
    return result


class CaptureSession:
    """Call stop before the edge namespace or the UE tunnel is torn down."""

    def __init__(self, output, processes, identities, streams, record):
        self.output, self.processes = Path(output), processes
        self.identities, self.streams, self.record = identities, streams, record
        self.stopped = False

    def check_health(self):
        for name, proc in self.processes.items():
            if proc.poll() is not None:
                raise RuntimeError(f'{name} capture ended before teardown (code {proc.returncode})')

    def stop(self):
        if self.stopped:
            return json.loads((self.output / 'STOPPED.json').read_text())
        result = {'stopped_wall_ns': time.time_ns(), 'captures': {}}
        errors = []
        for name, proc in self.processes.items():
            forced = False
            try:
                if proc.poll() is None:
                    observed = process_identity(proc.pid)
                    if observed['starttime'] != self.identities[name]['starttime']:
                        raise RuntimeError(f'Capture wrapper PID identity changed: {name}')
                    # Each wrapper owns a new session; sudo may create child groups.
                    # timeout forwards SIGINT to its directly owned command.
                    subprocess.run(['sudo', '-n', 'kill', '-INT', '--', str(-proc.pid)],
                                   check=True, timeout=5)
                try:
                    code = proc.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    forced = True
                    subprocess.run(['sudo', '-n', 'kill', '-TERM', '--', str(-proc.pid)],
                                   check=True, timeout=5)
                    code = proc.wait(timeout=8)
                self.streams[name].close()
                stats = tcpdump_stats(self.output / f'{name}.log')
                result['captures'][name] = {'returncode': code, 'forced_termination': forced,
                    'pcap_bytes': (self.output / f'{name}.pcap').stat().st_size, **stats}
                if code not in (0, 130, -signal.SIGINT):
                    errors.append(f'{name}: abnormal exit or bounded duration expired ({code})')
                if forced or stats['kernel_drops'] != 0 or stats['captured'] is None:
                    errors.append(f'{name}: incomplete flush or capture drops')
                if stats['captured'] == PACKET_LIMIT:
                    errors.append(f'{name}: packet cap reached before explicit stop')
            except BaseException as exc:
                errors.append(f'{name}: {exc!r}')
        result['errors'], result['passed'] = errors, not errors
        write_json(self.output / 'STOPPED.json', result)
        self.stopped = True
        return result


def start_capture(output, *, edge_pid, edge_starttime, ue_ip, edge_ip,
                  edge_port=51002, feedback_port=51014, duration_s=600):
    """Start after edge worker barrier but before route release; wait for listeners.

    ``edge_pid`` is the host PID already verified by host_partition.apply;
    ``edge_starttime`` is the main process pid_starttime from PARTITION_JOURNAL
    (not AFFINITY_APPLIED.before.target.starttime, which belongs to a worker).
    Output is a fresh host directory, never a container-local path.
    """
    output = Path(output).resolve()
    expected = process_identity(edge_pid)
    if expected['starttime'] != str(edge_starttime):
        raise RuntimeError('Verified edge PID has been reused')
    if 'edge_profiled_v1_cpu_partition_v1' not in expected['cmdline']:
        raise RuntimeError('Unexpected edge command identity')
    if not Path('/sys/class/net/oaitun_ue1').exists():
        raise RuntimeError('UE tunnel is not ready')
    output.mkdir(parents=True, exist_ok=False)
    commands = capture_commands(output, edge_pid=edge_pid, ue_ip=ue_ip,
        edge_ip=edge_ip, edge_port=edge_port, feedback_port=feedback_port, duration_s=duration_s)
    record = {'edge_identity': expected, 'commands': commands, 'start_wall_ns': time.time_ns(),
        'duration_s': duration_s, 'snaplen': SNAPLEN, 'packet_limit': PACKET_LIMIT,
        'maximum_pcap_bytes_each': 24 + PACKET_LIMIT * (16 + SNAPLEN),
        'full_uplink_payload_integrity': False, 'prefix_digest_bytes': PREFIX_BYTES}
    write_json(output / 'STARTING.json', record)
    processes, identities, streams = {}, {}, {}
    session = CaptureSession(output, processes, identities, streams, record)
    try:
        for name, command in commands.items():
            if process_identity(edge_pid) != expected:
                raise RuntimeError('Edge identity changed during capture startup')
            streams[name] = (output / f'{name}.log').open('x')
            proc = subprocess.Popen(command, stdout=streams[name], stderr=subprocess.STDOUT,
                                    start_new_session=True)
            processes[name], identities[name] = proc, process_identity(proc.pid)
        deadline = time.monotonic() + 10
        while True:
            if any(proc.poll() is not None for proc in processes.values()):
                raise RuntimeError('Capture exited during startup; inspect capture logs')
            if all(tcpdump_stats(output / f'{name}.log')['listening'] for name in processes):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('tcpdump listening barrier')
            time.sleep(.05)
        record['wrapper_identities'] = identities
        record['ready_wall_ns'] = time.time_ns()
        write_json(output / 'READY.json', record)
        return session
    except BaseException:
        session.stop()
        raise


def _ipv4(packet, linktype):
    if linktype in (101, 228):  # LINKTYPE_RAW, LINKTYPE_IPV4
        return packet
    if linktype == 1:
        if len(packet) < 14:
            raise ValueError('Truncated Ethernet header')
        protocol, offset = struct.unpack_from('!H', packet, 12)[0], 14
        while protocol in (0x8100, 0x88a8):
            if len(packet) < offset + 4:
                raise ValueError('Truncated VLAN header')
            protocol = struct.unpack_from('!H', packet, offset + 2)[0]
            offset += 4
    elif linktype == 113:
        if len(packet) < 16:
            raise ValueError('Truncated Linux SLL header')
        protocol, offset = struct.unpack_from('!H', packet, 14)[0], 16
    elif linktype == 276:
        if len(packet) < 20:
            raise ValueError('Truncated Linux SLL2 header')
        protocol, offset = struct.unpack_from('!H', packet)[0], 20
    else:
        raise ValueError(f'Unsupported linktype {linktype}')
    return packet[offset:] if protocol == 0x0800 else b''


def pcap_packets(path):
    """Stream classic pcap; reject corruption instead of silently truncating."""
    with Path(path).open('rb') as stream:
        header = stream.read(24)
        if len(header) != 24 or header[:4] not in MAGICS:
            raise ValueError('Unsupported or truncated classic pcap header')
        endian, scale = MAGICS[header[:4]]
        _, major, minor, _, _, snaplen, linktype = struct.unpack(endian + 'IHHIIII', header)
        if (major, minor) != (2, 4):
            raise ValueError('Unsupported pcap version')
        index = 0
        while True:
            record = stream.read(16)
            if not record:
                break
            if len(record) != 16:
                raise ValueError('Truncated pcap packet header')
            seconds, fraction, captured, original = struct.unpack(endian + 'IIII', record)
            if captured > snaplen or captured > original or captured > 16 * 1024 * 1024:
                raise ValueError('Invalid pcap packet length')
            packet = stream.read(captured)
            if len(packet) != captured:
                raise ValueError('Truncated pcap packet data')
            if fraction * scale >= 1_000_000_000:
                raise ValueError('Invalid pcap timestamp fraction')
            yield {'packet_index': index, 'wall_ns': seconds * 10**9 + fraction * scale,
                   'captured_bytes': captured, 'original_bytes': original,
                   'linktype': linktype, 'ip': _ipv4(packet, linktype)}
            index += 1


def parse_capture(path, *, edge_port=51002, feedback_port=51014):
    """Return packet and UDP-datagram observations, retaining fragment ambiguity.

    Fragment groups use endpoints, IPv4 ID and a 2-second expiry. Duplicate
    offsets or conflicting lengths are marked ambiguous, excluded from joins.
    Each first-fragment row carries a comparable fixed prefix SHA-256. Only
    complete unfragmented UDP payloads receive udp_payload_sha256.
    """
    packets, groups, active = [], [], {}
    skipped = 0
    for record in pcap_packets(path):
        ip = record.pop('ip')
        if not ip or ip[0] >> 4 != 4:
            skipped += 1
            continue
        if len(ip) < 20:
            raise ValueError('Truncated IPv4 header')
        ihl = (ip[0] & 15) * 4
        if ihl < 20 or len(ip) < ihl:
            raise ValueError('Invalid IPv4 header length')
        if ip[9] != 17:
            skipped += 1
            continue
        total, ident, flags = struct.unpack_from('!HHH', ip, 2)
        if total < ihl:
            raise ValueError('Invalid IPv4 total length')
        offset, more = (flags & 0x1fff) * 8, bool(flags & 0x2000)
        src, dst = socket.inet_ntoa(ip[12:16]), socket.inet_ntoa(ip[16:20])
        ip_payload = ip[ihl:total]
        row = {**record, 'source_ip': src, 'destination_ip': dst, 'ip_id': ident,
               'ip_fragment_offset': offset, 'more_fragments': more,
               'ip_payload_wire_bytes': total - ihl, 'ip_payload_captured_bytes': len(ip_payload)}
        key = (src, dst, ident)
        group = active.get(key)
        if group is None or row['wall_ns'] - group['last_wall_ns'] > 2_000_000_000 or group['wire_complete']:
            group = {'datagram_index': len(groups), 'source_ip': src, 'destination_ip': dst,
                     'ip_id': ident, 'first_wall_ns': row['wall_ns'], 'last_wall_ns': row['wall_ns'],
                     'fragments': [], 'wire_complete': False, 'ambiguous': False}
            groups.append(group)
            active[key] = group
        group['first_wall_ns'] = min(group['first_wall_ns'], row['wall_ns'])
        group['last_wall_ns'] = max(group['last_wall_ns'], row['wall_ns'])
        row['datagram_index'] = group['datagram_index']
        if any(f['ip_fragment_offset'] == offset for f in group['fragments']):
            group['ambiguous'] = True
        group['fragments'].append(row)
        if offset == 0:
            if len(ip_payload) < 8:
                raise ValueError('UDP header not captured')
            sport, dport, udp_length, _ = struct.unpack_from('!HHHH', ip_payload)
            if udp_length < 8:
                raise ValueError('Invalid UDP length')
            payload = ip_payload[8:udp_length]
            direction = 'uplink' if dport == edge_port else 'feedback' if dport == feedback_port else 'other'
            fields = {'source_port': sport, 'destination_port': dport,
                      'udp_length': udp_length, 'direction': direction,
                      'udp_first_fragment_wall_ns': row['wall_ns']}
            prefix_len = min(PREFIX_BYTES, udp_length - 8)
            if len(payload) >= prefix_len:
                fields.update(payload_prefix_bytes=prefix_len,
                              payload_prefix_sha256=hashlib.sha256(payload[:prefix_len]).hexdigest())
            if direction == 'uplink' and len(payload) >= 8:
                message, chunk, count = struct.unpack_from('!IHH', payload)
                fields.update(message_id=message, chunk_index=chunk, chunk_count=count,
                              chunk_header_valid=0 < count <= 4096 and chunk < count)
            full = not more and len(payload) == udp_length - 8
            fields['full_udp_payload_captured'] = full
            if full:
                fields['udp_payload_sha256'] = hashlib.sha256(payload).hexdigest()
            if direction == 'feedback' and payload[:4] == b'R4FB':
                fields['feedback_magic'] = 'R4FB'
                if len(payload) >= FB.size:
                    values = FB.unpack(payload[:FB.size])
                    fields.update(decision_seq=values[8], ticket_seq=values[9], frame_id=values[10],
                                  tensor_seq=values[11], session_uuid=str(uuid.UUID(bytes=values[13])))
                fields['feedback_digest_verified'] = (full and len(payload) == FB.size + 32
                    and hashlib.sha256(payload[:-32]).digest() == payload[-32:])
            row.update(fields)
            group.update(fields)
        ordered = sorted(group['fragments'], key=lambda f: f['ip_fragment_offset'])
        cursor = 0
        for fragment in ordered:
            if fragment['ip_fragment_offset'] != cursor:
                break
            cursor += fragment['ip_payload_wire_bytes']
        else:
            group['wire_complete'] = (not ordered[-1]['more_fragments']
                and cursor == group.get('udp_length') and not group['ambiguous'])
        packets.append(row)
    datagrams = []
    for group in groups:
        fragments = group.pop('fragments')
        group['fragment_count'] = len(fragments)
        group['packet_indices'] = [r['packet_index'] for r in fragments]
        group['snaplen_truncated_fragments'] = sum(
            r['ip_payload_captured_bytes'] < r['ip_payload_wire_bytes'] for r in fragments)
        datagrams.append(group)
    return {'source': str(Path(path).resolve()), 'packet_count': len(packets), 'skipped': skipped,
            'packets': packets, 'datagrams': datagrams,
            'limits': ['Packet times are CLOCK_REALTIME, not application-receipt timestamps.',
                       'Uplink prefix digests prove captured-prefix agreement only.',
                       'IP-ID group joins may be ambiguous under duplication or ID reuse.',
                       'Wire completeness uses IP lengths, not full-payload checksums.']}


def join_uplink(ue, edge):
    """Uniquely join chunk identities and prefix bytes, independent of IP ID/NAT."""
    def key(row):
        if (row.get('direction') != 'uplink' or not row.get('chunk_header_valid')
                or 'payload_prefix_sha256' not in row or row['ambiguous']):
            return None
        return tuple(row[k] for k in ('message_id', 'chunk_index', 'chunk_count',
                                     'udp_length', 'payload_prefix_bytes', 'payload_prefix_sha256'))
    indexes = []
    for capture in (ue, edge):
        index = {}
        for row in capture['datagrams']:
            identity = key(row)
            if identity is not None:
                index.setdefault(identity, []).append(row)
        indexes.append(index)
    joined, nonunique = [], []
    for identity in sorted(indexes[0].keys() | indexes[1].keys()):
        left, right = indexes[0].get(identity, []), indexes[1].get(identity, [])
        if len(left) != 1 or len(right) != 1:
            nonunique.append({'identity': identity, 'ue_count': len(left), 'edge_count': len(right)})
            continue
        a, b = left[0], right[0]
        joined.append({'message_id': a['message_id'], 'chunk_index': a['chunk_index'],
            'chunk_count': a['chunk_count'], 'payload_prefix_sha256': a['payload_prefix_sha256'],
            'ue_first_wall_ns': a['first_wall_ns'], 'edge_first_wall_ns': b['first_wall_ns'],
            'ue_last_wall_ns': a['last_wall_ns'], 'edge_last_wall_ns': b['last_wall_ns'],
            'ue_wire_complete': a['wire_complete'], 'edge_wire_complete': b['wire_complete'],
            'first_to_first_ms': (b['first_wall_ns'] - a['first_wall_ns']) / 1e6,
            'last_to_last_ms': (b['last_wall_ns'] - a['last_wall_ns']) / 1e6,
            'full_payload_integrity_verified': False})
    return {'joined': joined, 'unmatched_or_nonunique': nonunique}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ue', type=Path)
    parser.add_argument('edge', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ue, edge = parse_capture(args.ue), parse_capture(args.edge)
    write_json(args.output / 'ue.json', ue)
    write_json(args.output / 'edge.json', edge)
    write_json(args.output / 'uplink_join.json', join_uplink(ue, edge))


if __name__ == '__main__':
    main()
