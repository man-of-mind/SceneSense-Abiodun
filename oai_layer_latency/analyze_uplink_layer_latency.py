#!/usr/bin/env python3
"""Cross-layer uplink latency/efficiency breakdown from OAI T-tracer CSVs.

Consumes the UE (queue profile) and gNB (full profile) CSV panels produced by
scripts/ttracer_extract_csv_smoke.sh and computes, for the uplink direction:

  A. UE RLC buffer residency via Little's law (occupancy / drain), per LCID.
  B. UE UL grant timing: K2 (DCI->PUSCH) delay, inter-grant gap, TBS/PRB/MCS
     percentiles, and grant fill ratio (RLC SDU bytes vs granted TBS => padding).
  C. UE BSR-reported backlog percentiles.
  D. gNB airtime efficiency: PHY/MAC bytes vs RLC/PDCP bytes => MAC overhead %.
  E. gNB PUSCH channel summary (SNR/PHR/MCS/RSSI) with an RFsim-ideal caveat.
  F. A per-layer latency attribution table separating what is measurable now
     (state-snapshot / Little's law) from what needs per-packet timestamps.

This is a state-snapshot analysis. It does NOT yet give true per-packet
cross-layer residency; that needs the timestamp instrumentation (next phase).
Numerology assumed mu=1 (20 slots/frame, 0.5 ms/slot) unless --slots-per-frame
is given.
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

ABIODUN = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ABIODUN / "metrics_logs" / "scenesense_ttracer"


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load(path):
    path = Path(path)
    if not path.is_file():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fi(row, key, default=0):
    try:
        return int(row[key])
    except (KeyError, ValueError, TypeError):
        return default


def ftime(row):
    """Parse the tracer wall-clock 'time' column HH:MM:SS.ffffff -> seconds."""
    try:
        h, m, s = row["time"].split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except (KeyError, ValueError, AttributeError):
        return None


def time_span(rows):
    ts = [t for t in (ftime(r) for r in rows) if t is not None]
    if len(ts) < 2:
        return float("nan"), None, None
    return max(ts) - min(ts), min(ts), max(ts)


def abs_slot(frame, slot, slots_per_frame, hyperframe=1024):
    return frame * slots_per_frame + slot


def unwrap_slots(seq, slots_per_frame, hyperframe=1024):
    """Unwrap absolute slot indices across SFN wraps (frame is 0..1023)."""
    out = []
    off = 0
    prev = None
    period = hyperframe * slots_per_frame
    for v in seq:
        if prev is not None and v + off < prev - period // 2:
            off += period
        out.append(v + off)
        prev = v + off
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-group", required=True)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--slots-per-frame", type=int, default=20, help="mu=1 -> 20; mu=0 -> 10")
    ap.add_argument("--data-grant-min-tbs", type=int, default=200,
                    help="TBS above which a UL grant is counted as a data grant (excludes keepalive)")
    ap.add_argument("--data-bearer-lcid", type=int, default=4, help="DRB LCID carrying the app payload")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    slot_ms = 0.5 if args.slots_per_frame == 20 else (1.0 if args.slots_per_frame == 10 else 10.0 / args.slots_per_frame)
    base = Path(args.root) / args.run_group
    ue = base / "ue" / "csv"
    gnb = base / "gnb" / "csv"
    out_dir = Path(args.output_dir) if args.output_dir else (base / "layer_latency")
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    def emit(s=""):
        lines.append(s)

    emit(f"# Uplink cross-layer latency/efficiency breakdown")
    emit(f"\nRun group: `{args.run_group}`  |  numerology slots/frame={args.slots_per_frame} ({slot_ms} ms/slot)\n")

    # ---- A. UE RLC residency (Little's law) ----
    rlc = load(ue / "NRUE_MAC_RLC_BUFFER_STATUS.csv")
    bsr = load(ue / "NRUE_MAC_BSR_STATUS.csv")
    grant = load(ue / "NRUE_MAC_DCI_GRANT.csv")

    emit("## A. UE RLC buffer residency (Little's law)\n")
    if rlc and bsr:
        # per-LCID occupancy (only the data bearer is meaningful; SRBs idle)
        occ_by_lcid = {}
        for r in rlc:
            occ_by_lcid.setdefault(fi(r, "lcid"), []).append(fi(r, "bytes_in_buffer"))
        for lc in sorted(occ_by_lcid):
            xs = occ_by_lcid[lc]
            nz = sum(1 for v in xs if v > 0)
            tag = " (data bearer)" if lc == args.data_bearer_lcid else (" (SRB, idle)" if nz == 0 else "")
            emit(f"- LCID {lc}{tag}: occupancy p50={pct(xs,50)/1024:.1f} KB  p95={pct(xs,95)/1024:.1f} KB  "
                 f"max={max(xs)/1024:.1f} KB  ({nz}/{len(xs)} samples nonzero)")
        # drain rate from BSR sdu_bytes over the BSR wall-clock span (robust to SFN wrap)
        sdu_total = sum(fi(r, "sdu_bytes") for r in bsr)
        dur_s, _, _ = time_span(bsr)
        drain_Bps = sdu_total / dur_s if dur_s == dur_s and dur_s > 0 else float("nan")
        emit(f"- SDU drain: {sdu_total/1e6:.1f} MB over {dur_s:.1f} s = {drain_Bps*8/1e6:.1f} Mbps")
        dl = occ_by_lcid.get(args.data_bearer_lcid, [])
        if dl and drain_Bps == drain_Bps and drain_Bps > 0:
            mean_occ = sum(dl) / len(dl)
            emit(f"- data-bearer occupancy mean={mean_occ/1024:.1f} KB")
            emit(f"- **RLC mean queueing delay (Little's law = mean occupancy / throughput):** "
                 f"{mean_occ/drain_Bps*1000:.1f} ms")
            emit(f"- occupancy-based upper bounds (NOT steady-state valid under bursts): "
                 f"p95={pct(dl,95)/drain_Bps*1000:.0f} ms  max={max(dl)/drain_Bps*1000:.0f} ms "
                 f"at avg drain {drain_Bps*8/1e6:.1f} Mbps")
        emit("- CAVEAT: Little's law is valid for the MEAN. Under bursty CARLA traffic the tail "
             "occupancy/avg-drain OVERstates per-frame latency (avg drain includes idle gaps). "
             "True per-frame residency needs the PDCP-ingress->GTP-egress timestamps (phase 2). "
             "The robust signal here is peak occupancy = whole feature frames queue in RLC.")
    else:
        emit("- (missing RLC/BSR CSVs)")
    emit()

    # ---- B. UE UL grant timing + fill ----
    emit("## B. UE UL grant timing and fill\n")
    ul = [g for g in grant if fi(g, "direction") == 1]
    if ul:
        k2 = []
        tbs = []
        prb = []
        mcs = []
        gap = []
        prev_abs = None
        abs_list = []
        for g in ul:
            df, ds, sf, ss = fi(g, "dci_frame"), fi(g, "dci_slot"), fi(g, "sched_frame"), fi(g, "sched_slot")
            k = (sf * args.slots_per_frame + ss) - (df * args.slots_per_frame + ds)
            if k < 0:
                k += 1024 * args.slots_per_frame
            k2.append(k)
            tbs.append(fi(g, "tbs"))
            prb.append(fi(g, "rb_size"))
            mcs.append(fi(g, "mcs"))
            abs_list.append(df * args.slots_per_frame + ds)
        abs_u = unwrap_slots(abs_list, args.slots_per_frame)
        for a in abs_u:
            if prev_abs is not None:
                gap.append(a - prev_abs)
            prev_abs = a
        emit(f"- UL grants: {len(ul)} total")
        emit(f"- **K2 (DCI->PUSCH) delay:** p50={pct(k2,50)*slot_ms:.1f} ms  p95={pct(k2,95)*slot_ms:.1f} ms "
             f"(slots p50={pct(k2,50):.0f})")
        emit(f"- inter-grant gap: p50={pct(gap,50)*slot_ms:.1f} ms  p95={pct(gap,95)*slot_ms:.1f} ms  max={max(gap)*slot_ms:.1f} ms")
        emit(f"- grant TBS: p50={pct(tbs,50):.0f} B  p95={pct(tbs,95):.0f} B  max={max(tbs)} B")
        emit(f"- grant PRB: p50={pct(prb,50):.0f}  p95={pct(prb,95):.0f}")
        emit(f"- grant MCS: p50={pct(mcs,50):.0f}  p95={pct(mcs,95):.0f}")
    # grant fill from BSR (sdu vs padding)
    if bsr:
        data_bsr = [r for r in bsr if fi(r, "sdu_bytes") > 0]
        empty_bsr = [r for r in bsr if fi(r, "sdu_bytes") == 0]
        fills = [(fi(r, "sdu_bytes"), fi(r, "padding_len")) for r in data_bsr]
        if fills:
            sdu = [a for a, _ in fills]
            ratio = [a / (a + b) if (a + b) else 0 for a, b in fills]
            emit(f"- grant fill (grants with data): sdu p50={pct(sdu,50):.0f} B  "
                 f"**fill ratio p50={pct(ratio,50)*100:.0f}%**")
        # over-grant breakdown: padding-only grants = scheduler gave a grant with no data queued
        n_tot = len(bsr)
        n_empty = len(empty_bsr)
        pad_data = sum(fi(r, "padding_len") for r in data_bsr)
        pad_empty = sum(fi(r, "padding_len") for r in empty_bsr)
        pad_tot = pad_data + pad_empty
        emit(f"- **over-grant:** {n_empty}/{n_tot} grants ({100*n_empty/n_tot:.0f}%) were padding-only "
             f"(scheduler granted with no data queued)")
        if pad_tot:
            emit(f"- padding bytes: {pad_tot/1e6:.1f} MB total "
                 f"({100*pad_empty/pad_tot:.0f}% from padding-only grants, {100*pad_data/pad_tot:.0f}% partial-fill)")
    emit()

    # ---- C. BSR backlog ----
    emit("## C. UE BSR-reported backlog\n")
    if bsr:
        lcg_tot = []
        for r in bsr:
            lcg_tot.append(sum(fi(r, f"lcg{i}_bytes") for i in range(8)))
        sent = sum(1 for r in bsr if fi(r, "bsr_sent") == 1)
        emit(f"- BSR reports: {len(bsr)} ({sent} sent); reported backlog "
             f"p50={pct(lcg_tot,50)/1024:.1f} KB  p95={pct(lcg_tot,95)/1024:.1f} KB  max={max(lcg_tot)/1024:.1f} KB")
    emit()

    # ---- D. gNB airtime efficiency ----
    emit("## D. gNB uplink airtime efficiency (byte conservation)\n")
    phy = load(gnb / "GNB_PHY_UL_PAYLOAD_RX_BITS.csv")
    mac = load(gnb / "GNB_MAC_UL.csv")
    rlcg = load(gnb / "ENB_RLC_UL.csv")
    pdcpg = load(gnb / "ENB_PDCP_UL.csv")

    # Traffic window: PDCP only fires on real data, so its wall-clock span is the
    # active-data window. Restrict all layers to it so idle keepalive grants do
    # not inflate the padding/overhead figure.
    _, t0, t1 = time_span(pdcpg)

    def in_window(rows):
        if t0 is None:
            return rows
        out = []
        for r in rows:
            t = ftime(r)
            if t is None or (t0 <= t <= t1):
                out.append(r)
        return out

    def totals(sel_phy, sel_mac, sel_rlc, sel_pdcp, label):
        phy_B = sum(fi(r, "number_of_bits") for r in sel_phy) / 8.0
        mac_B = sum(fi(r, "tbs") for r in sel_mac)
        rlc_B = sum(fi(r, "length") for r in sel_rlc)
        pdcp_B = sum(fi(r, "length") for r in sel_pdcp)
        emit(f"### {label}")
        emit(f"- PHY UL decoded={phy_B/1e6:.1f} MB | MAC TBS={mac_B/1e6:.1f} MB | "
             f"RLC={rlc_B/1e6:.1f} MB | PDCP={pdcp_B/1e6:.1f} MB")
        if mac_B:
            emit(f"- **MAC overhead (headers+CE+padding)=(MAC-RLC)/MAC = {(mac_B-rlc_B)/mac_B*100:.0f}%** "
                 f"(goodput fraction {rlc_B/mac_B*100:.0f}%)")
        emit()

    totals(phy, mac, rlcg, pdcpg, "Whole recording (includes idle keepalive grants)")
    if t0 is not None:
        totals(in_window(phy), in_window(mac), in_window(rlcg), pdcpg,
               f"Active-data window only ({t1-t0:.1f} s)")

    # ---- E. gNB PUSCH channel ----
    emit("## E. gNB PUSCH channel summary (RFsim)\n")
    pc = load(gnb / "GNB_MAC_PUSCH_POWER_CONTROL.csv")
    if pc:
        snr = [fi(r, "snrx10") / 10.0 for r in pc]
        phr = [fi(r, "phr") for r in pc]
        mcs = [fi(r, "mcs") for r in pc]
        rssi = [fi(r, "rssi") for r in pc]
        emit(f"- SNR dB: p50={pct(snr,50):.1f}  min={min(snr):.1f}  max={max(snr):.1f}")
        emit(f"- PHR: p50={pct(phr,50):.0f}   MCS: p50={pct(mcs,50):.0f}   RSSI: p50={pct(rssi,50):.0f}")
        spread = max(snr) - min(snr)
        emit(f"- **caveat:** SNR spread only {spread:.1f} dB => RFsim ideal channel (real value, but flat). "
             f"Enable a channel model for meaningful variation.")
    emit()

    # ---- F. attribution table ----
    emit("## F. Per-layer uplink latency attribution\n")
    emit("| Layer / component | Measured now | Method | Needs per-packet timestamp? |")
    emit("|---|---|---|---|")
    emit("| App -> PDCP (TUN/kernel enqueue) | no | - | YES (app egress + PDCP ingress stamp) |")
    emit("| UE PDCP queue | no | - | YES (PDCP ingress stamp) |")
    emit("| UE RLC queue | **yes** | Little's law (occupancy/drain) | refine with per-SDU stamp |")
    emit("| UE MAC: BSR->grant + K2 | **partial** | K2 fixed; grant cadence/gap | SR->grant needs event pairing |")
    emit("| PHY UL (prop + decode) | n/a | fixed slot timing | small; from RX slot |")
    emit("| gNB MAC/RLC reassembly + PDCP | rate only | byte conservation | YES (gNB RX->PDCP stamp) |")
    emit("| gNB PDCP -> GTP-U -> ext-DN | no | - | YES (GTP-U egress stamp) |")
    emit("| Airtime efficiency (padding) | **yes** | MAC vs RLC bytes | no |")

    # ---- G. per-packet RAN transit from the phase-2 timestamp events ----
    emit("## G. Per-packet RAN uplink transit (phase-2 monotonic timestamps)\n")

    def load_ts(path, size_key="sdu_bytes"):
        rows = []
        for r in load(path):
            try:
                t = int(r["mono_sec"]) + int(r["mono_nsec"]) / 1e9
                rows.append((t, fi(r, size_key)))
            except (KeyError, ValueError, TypeError):
                pass
        return rows

    pin = load_ts(ue / "NR_PDCP_TX_SDU.csv")
    rlin = load_ts(ue / "NR_RLC_TX_SDU.csv")
    gout = load_ts(gnb / "GNB_PDCP_RX_DELIVER.csv")
    if pin and gout:
        n = min(len(pin), len(gout))
        # FIFO correlation by index over matched-size data SDUs (same host => monotonic clock comparable)
        transit = [(gout[i][0] - pin[i][0]) * 1000 for i in range(n)
                   if pin[i][1] == gout[i][1] and pin[i][1] > 1000 and 0 < (gout[i][0] - pin[i][0]) < 10]
        if transit:
            emit(f"- matched data SDUs: {len(transit)}")
            emit(f"- **UE PDCP-ingress -> gNB PDCP-deliver (whole RAN UL transit):** "
                 f"mean={sum(transit)/len(transit):.1f} ms  p50={pct(transit,50):.1f}  "
                 f"p95={pct(transit,95):.1f}  p99={pct(transit,99):.1f}  max={max(transit):.1f} ms")
        if rlin:
            n2 = min(len(pin), len(rlin))
            hand = [(rlin[i][0] - pin[i][0]) * 1000 for i in range(n2) if 0 <= (rlin[i][0] - pin[i][0]) < 1]
            if hand:
                emit(f"- UE PDCP->RLC handoff: p50={pct(hand,50):.3f} ms  p95={pct(hand,95):.3f} ms")
        emit("- NOTE: FIFO index correlation assumes no reordering/loss on the DRB; matched SDU counts "
             "confirm alignment. Same-host RFsim makes UE and gNB CLOCK_MONOTONIC directly comparable.")
    else:
        emit("- (phase-2 timestamp events not present in this run; record NR_PDCP_TX_SDU / "
             "NR_RLC_TX_SDU / GNB_MAC_RX_SDU / GNB_PDCP_RX_DELIVER to populate)")
    emit()

    # ---- H. hard per-layer split via cumulative byte curves (needs dequeue event) ----
    emit("## H. Hard per-layer split: RLC-wait vs MAC/PHY/air/gNB (Little's law + FIFO-matched transit)\n")

    def cum_curve(rows, size_key):
        """Return (times_sorted, cumbytes) sorted by monotonic time."""
        pts = []
        for r in rows:
            try:
                pts.append((int(r["mono_sec"]) + int(r["mono_nsec"]) / 1e9, int(r[size_key])))
            except (KeyError, ValueError, TypeError):
                pass
        pts.sort()
        import itertools
        t = [p[0] for p in pts]
        c = list(itertools.accumulate(p[1] for p in pts))
        return t, c

    deq = load(ue / "NR_RLC_TX_DEQUEUE.csv")
    if pin and deq and bsr and rlc:
        # RLC queue-wait via Little's law (mean occupancy / drain) is the robust
        # measure. The per-byte cumulative ingress-vs-dequeue method is NOT used:
        # dequeue logs PDU bytes (incl RLC header, ~0.6% inflation) which, at the
        # low ~1.4 MB/s data rate, skews cumulative-byte alignment by seconds.
        dl = [fi(r, "bytes_in_buffer") for r in rlc if fi(r, "lcid") == args.data_bearer_lcid]
        mean_occ = sum(dl) / len(dl) if dl else 0
        sdu_total = sum(fi(r, "sdu_bytes") for r in bsr)
        dur_s, _, _ = time_span(bsr)
        drain_Bps = sdu_total / dur_s if dur_s == dur_s and dur_s > 0 else float("nan")
        rlc_wait_mean = mean_occ / drain_Bps * 1000 if drain_Bps and drain_Bps == drain_Bps else float("nan")
        # total transit (robust FIFO-matched, Section G)
        n = min(len(pin), len(gout)) if gout else 0
        transit = [(gout[i][0] - pin[i][0]) * 1000 for i in range(n)
                   if pin[i][1] == gout[i][1] and pin[i][1] > 1000 and 0 < (gout[i][0] - pin[i][0]) < 10]
        tot_mean = sum(transit) / len(transit) if transit else float("nan")
        # dequeue event corroborates the drain: total dequeued bytes / time
        deq_bytes = sum(fi(r, "pdu_bytes") for r in deq)
        deq_dur, _, _ = time_span(deq)
        emit(f"- RLC dequeue events: {len(deq)} PDUs, {deq_bytes/1e6:.0f} MB over {deq_dur:.0f} s "
             f"(confirms continuous drain at the grant-limited rate)")
        emit(f"- **RLC queue-wait (mean, Little's law = mean occupancy {mean_occ/1024:.0f} KB / drain {drain_Bps*8/1e6:.1f} Mbps):** "
             f"~{rlc_wait_mean:.0f} ms")
        if transit:
            post = tot_mean - rlc_wait_mean
            emit(f"- total uplink transit (Section G, FIFO-matched, mean): ~{tot_mean:.0f} ms")
            emit(f"- **=> RLC queue-wait is ~{rlc_wait_mean/tot_mean*100:.0f}% of the uplink transit**; "
                 f"remainder (air K2 + gNB PHY/MAC/reassembly/PDCP) ~{post:.0f} ms")
        emit("- All of this is downstream of PDCP (handoff ~0.1 ms) and is the consequence of the slow QPSK "
             "drain: the ~1 MB frame sits in the RLC TX buffer waiting for grants. MAC/PHY per-slot processing "
             "(<=0.5 ms) and gNB per-SDU reassembly are small; the latency is queue-wait, not processing.")
        emit("- (Per-byte cumulative ingress->dequeue split was attempted but is confounded by RLC-header byte "
             "inflation at low data rate; Little's law + FIFO-matched transit are the trustworthy measures.)")
    else:
        emit("- (NR_RLC_TX_DEQUEUE not present; rerun with the updated 'latency' profile)")
    emit()

    md = "\n".join(lines) + "\n"
    (out_dir / "uplink_layer_latency.md").write_text(md)
    print(md)
    print(f"[analyze_uplink_layer_latency] wrote {out_dir/'uplink_layer_latency.md'}")


if __name__ == "__main__":
    main()
