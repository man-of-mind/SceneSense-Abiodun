"""Read-only readers for the retained single-UE evidence that the existing loader does not join.

Clock domains (all verified in diagnose.py, never assumed):
- service / UE / edge instants: CLOCK_MONOTONIC_RAW on one host (edge container shares the host boot_id).
- pcap on oaitun_ue1 and the UE T-tracer: CLOCK_REALTIME (kernel rx stamp; softmodem clock_gettime at T()).
- wall -> raw uses the upper envelope of (wall_ns, raw_ns) pairs that the radio-telemetry drain reads back-to-back
  (wall first, then raw: measured wall-raw = true offset - read gap, so the per-second maximum is the estimate).
"""
import bisect
import collections
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np

T_MESSAGES = Path(__file__).resolve().parents[2] / "OAI/openairinterface5g/common/utils/T/T_messages.txt"
DCI_FIELDS = ("direction", "dci_format", "rnti_type", "rnti", "dci_frame", "dci_slot", "sched_frame", "sched_slot",
              "mcs", "mcs_table", "rb_start", "rb_size", "start_symbol", "nr_symbols", "tbs", "harq_pid", "ndi", "rv",
              "round", "qam_mod_order", "target_code_rate", "tpc", "n_cce", "N_cce")
DL = 0


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- clock bridge
class WallToRaw:
    """Piecewise-linear wall->CLOCK_MONOTONIC_RAW map from the per-second upper envelope of drain pairs."""

    def __init__(self, ingest_csvs):
        pairs = []
        for path in ingest_csvs:
            with open(path) as f:
                for line in f:
                    a = line.split(",", 2)
                    pairs.append((int(a[0]), int(a[1])))
        p = np.array(sorted(pairs, key=lambda x: x[1]), dtype=np.int64)
        off = p[:, 0] - p[:, 1]
        sec = p[:, 1] // 1_000_000_000
        t, o = [], []
        for k in np.unique(sec):
            m = np.flatnonzero(sec == k)
            i = m[np.argmax(off[m])]
            t.append(p[i, 1]); o.append(off[i])
        self.raw = np.array(t, dtype=np.int64)
        self.off = np.array(o, dtype=np.int64)
        self.wall = self.raw + self.off
        self.pairs = len(p)
        drift = (self.off[-1] - self.off[0]) / 1e6
        second_diff = np.abs(self.off[1:-1] - (self.off[:-2] + self.off[2:]) / 2) / 1e3
        self.summary = dict(pairs=int(len(p)), envelope_bins=int(len(t)),
                            span_s=round(float((self.raw[-1] - self.raw[0]) / 1e9), 1),
                            wall_minus_raw_drift_ms=round(float(drift), 3),
                            envelope_second_difference_us_p50_p99_max=[round(float(x), 2) for x in
                                                                        np.percentile(second_diff, [50, 99, 100])])

    def __call__(self, wall_ns):
        w = np.asarray(wall_ns, dtype=np.float64)
        return w - np.interp(w, self.wall.astype(np.float64), self.off.astype(np.float64))

    def covers(self, wall_ns):
        return self.wall[0] - 1e9 <= wall_ns <= self.wall[-1] + 1e9


# --------------------------------------------------------------------------- pcap on oaitun_ue1
def pcap_payloads(path):
    """{sha256(UDP payload): [rx wall ns]} for a link-type RAW (raw IPv4) pcap with microsecond stamps."""
    b = Path(path).read_bytes()
    magic, = struct.unpack_from("<I", b, 0)
    if magic != 0xA1B2C3D4:
        raise ValueError(f"unexpected pcap magic {magic:#x} (need little-endian microsecond pcap)")
    linktype, = struct.unpack_from("<I", b, 20)
    out, p, n = collections.defaultdict(list), 24, 0
    while p < len(b):
        s, us, incl, orig = struct.unpack_from("<IIII", b, p)
        data = b[p + 16:p + 16 + incl]
        p += 16 + incl
        if incl != orig or data[0] >> 4 != 4 or data[9] != 17:
            raise ValueError("truncated or non-UDP/IPv4 packet in feedback pcap")
        ihl = (data[0] & 15) * 4
        out[hashlib.sha256(data[ihl + 8:]).hexdigest()].append(s * 1_000_000_000 + us * 1_000)
        n += 1
    return dict(out), dict(packets=n, linktype=linktype, unique_payloads=len(out))


# --------------------------------------------------------------------------- UE T-tracer raw file
def t_ids(path=T_MESSAGES):
    return [m.group(1) for line in open(path) for m in [re.match(r"\s*ID\s*=\s*(\S+)", line)] if m]


def ue_dl_dcis(raw_path, messages=T_MESSAGES):
    """DL DCI grants (NRUE_MAC_DCI_GRANT direction 0) as (wall_ns, fields...); record format
    [int32 len][timespec CLOCK_REALTIME][int32 id][24 x int32]. Whole file must parse to its exact length."""
    ids = t_ids(messages)
    dci = ids.index("NRUE_MAC_DCI_GRANT")
    b = Path(raw_path).read_bytes()
    p, rows, counts = 0, [], collections.Counter()
    while p + 4 <= len(b):
        length, = struct.unpack_from("<i", b, p)
        kind, = struct.unpack_from("<i", b, p + 20)
        counts[ids[kind] if 0 <= kind < len(ids) else str(kind)] += 1
        if kind == dci:
            s, ns = struct.unpack_from("<qq", b, p + 4)
            v = struct.unpack_from("<24i", b, p + 24)
            if v[0] == DL:
                rows.append((s * 1_000_000_000 + ns,) + v)
        p += 4 + length
    if p != len(b):
        raise ValueError(f"T-tracer raw file did not parse to its length ({p} != {len(b)})")
    return np.array(rows, dtype=np.int64), dict(counts)


# --------------------------------------------------------------------------- profiler / GT-service side records
def profile_spans(path, stages=None):
    """All live spans (start, end, stage, frame_id, tid) and marks, sorted by start."""
    spans, marks = [], collections.defaultdict(dict)
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("ph") != "live":
                continue
            if d.get("r") == "span" and (stages is None or d["stage"] in stages):
                spans.append((d["start_raw_ns"], d["end_raw_ns"], d["stage"], d.get("frame_id"), d.get("tid")))
            elif d.get("r") == "mark" and d.get("frame_id") is not None:
                marks[int(d["frame_id"])].setdefault(d["stage"], d["t_raw_ns"])
    spans.sort()
    return spans, marks


class Intervals:
    """Max overlap query of [start, end] spans against a window."""

    def __init__(self, spans):
        self.s = [x[0] for x in spans]
        self.spans = spans
        self.maxlen = max((x[1] - x[0] for x in spans), default=0)

    def overlapping(self, lo, hi):
        i = bisect.bisect_left(self.s, lo - self.maxlen)
        return [x for x in self.spans[i:bisect.bisect_right(self.s, hi)] if x[1] > lo and x[0] < hi]


def gc_events(trace_path):
    """gen2 collections from a GT worker trace: (start_raw_ns, wall_ns, op or None)."""
    if not Path(trace_path).exists():
        return None
    d = json.loads(Path(trace_path).read_bytes())
    return [(g["start_raw_ns"], g["wall_ns"], g.get("op")) for g in d.get("gen2") or []]


def queue_samples(path):
    q = json.loads(Path(path).read_bytes())
    q.sort(key=lambda x: x["t"])
    return q
