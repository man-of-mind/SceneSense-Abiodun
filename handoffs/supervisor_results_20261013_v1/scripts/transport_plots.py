"""Read-only, bounded decoding of existing UE T-tracer recordings; PNG-only plots.

OAI record.c layout: int32 length, CLOCK_REALTIME timespec (2 int64),
int32 event id, payload. Decode IDs and integer schemas from each recording's
embedded T_messages configuration, not from mutable OAI source indices.
"""
from __future__ import annotations

import csv
import hashlib
import json
import mmap
import re
import struct
from collections import Counter, defaultdict

import numpy as np
import prepare_plots as P

EVENTS = {"NR_PDCP_TX_SDU": ("pdcp_ingress", "sdu_bytes"),
          "NR_RLC_TX_SDU": ("rlc_ingress", "sdu_bytes"),
          "NR_RLC_TX_DEQUEUE": ("rlc_dequeue", "pdu_bytes")}
RLC, BSR = "NRUE_MAC_RLC_BUFFER_STATUS", "NRUE_MAC_BSR_STATUS"


def schemas(config):
    chunks = re.split(r"(?m)^\s*ID\s*=\s*", config.decode())[1:]
    names, desc = [], {}
    for chunk in chunks:
        name = chunk.splitlines()[0].strip()
        fmt = re.search(r"(?m)^\s*FORMAT\s*=\s*(.*)", chunk)
        fields = [p.strip().split(",") for p in fmt[1].split(":")] if fmt else []
        names.append(name)
        desc[name] = [(kind.strip(), field.strip()) for kind, field in fields]
    return names, desc


def raw_summary(arm, summary):
    path = P.CAPTURES / arm / "capture/cells/a71__favorable_stable/ttracer/ue/ue.raw"
    P.input_path(path)
    start = summary["capture_origin_wall_s"]
    duration = summary["duration_s"]
    full_seconds = int(duration)
    totals = Counter(); counts_all = Counter(); counts_window = Counter()
    bins = {k: np.zeros(full_seconds) for k, _ in EVENTS.values()}
    rlc_samples, bsr_samples = [], []
    identities = defaultdict(set)
    current_key, current_tick, values = None, None, {}
    repeated_lcid, changed_lcid = 0, 0
    first_event, last_event = None, None
    config = bytearray(); ids, descriptions = [], {}
    decoders = {}
    config_blocks = 0

    def close_rlc_group():
        if current_key is not None:
            t = (current_tick / 1e9) - start
            if 0 <= t <= duration:
                rlc_samples.append((t, sum(values.values())))

    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        pos = 0
        while pos < len(raw):
            assert pos + 24 <= len(raw), "Truncated record header"
            length, = struct.unpack_from("<i", raw, pos)
            assert length >= 20 and pos + 4 + length <= len(raw), "Truncated record"
            sec, ns, kind = struct.unpack_from("<qqi", raw, pos + 4)
            assert 0 <= ns < 1_000_000_000
            if kind == -1:
                size, = struct.unpack_from("<i", raw, pos + 24)
                assert size == length - 24
                config.extend(raw[pos + 28:pos + 28 + size])
            elif kind == -2:
                config_blocks += 1
                assert config_blocks == 1, "Unexpected repeated schema"
                ids, descriptions = schemas(config)
                for name in [*EVENTS, RLC, BSR]:
                    assert name in descriptions
                    fields = descriptions[name]
                    assert all(t == "int" for t, _ in fields)
                    decoders[ids.index(name)] = (name, struct.Struct("<" + "i" * len(fields)), {f: i for i, (_, f) in enumerate(fields)})
            else:
                assert ids and 0 <= kind < len(ids), "Unbound event identity"
                name = ids[kind]; counts_all[name] += 1
                stamp = sec * 1_000_000_000 + ns
                first_event = stamp if first_event is None else min(first_event, stamp)
                last_event = stamp if last_event is None else max(last_event, stamp)
                t = stamp / 1e9 - start
                inside = 0 <= t <= duration
                if inside:
                    counts_window[name] += 1
                if kind in decoders:
                    name, unpacker, fields = decoders[kind]
                    assert length == 20 + unpacker.size, (name, length, unpacker.size)
                    v = unpacker.unpack_from(raw, pos + 24)
                    if inside:
                        for key in ["ue_id", "rnti", "rb_id", "lcid"]:
                            if key in fields:
                                identities[f"{name}:{key}"].add(v[fields[key]])
                    if name in EVENTS and inside:
                        key, byte_field = EVENTS[name]
                        nbytes = v[fields[byte_field]]; assert nbytes >= 0
                        totals[key] += nbytes
                        if t < full_seconds:
                            bins[key][int(t)] += nbytes
                    elif name == RLC:
                        key = tuple(v[fields[k]] for k in ["rnti", "ue_id", "frame", "slot"])
                        if key != current_key:
                            close_rlc_group()
                            current_key, current_tick, values = key, stamp, {}
                        lcid, nbytes = v[fields["lcid"]], v[fields["bytes_in_buffer"]]
                        assert nbytes >= 0
                        if lcid in values:
                            repeated_lcid += 1
                            changed_lcid += values[lcid] != nbytes
                        values[lcid] = nbytes
                        current_tick = stamp
                    elif name == BSR and inside:
                        # Residual LCG state is not an exact on-air BSR byte value.
                        bsr_samples.append((t, sum(v[fields[f"lcg{i}_bytes"]] for i in range(8)), v[fields["bsr_sent"]]))
            pos += 4 + length
        assert pos == len(raw)
    # The final open tick is intentionally excluded; a subsequent tick proves completion.
    assert first_event / 1e9 <= start and last_event / 1e9 >= start + duration, "Raw trace does not cover observation window"
    assert all(len(v) == 1 for k, v in identities.items() if k.endswith(":ue_id") or k.endswith(":rnti"))
    for key in EVENTS.values():
        assert counts_window[next(n for n, x in EVENTS.items() if x == key)] > 0
    assert len(rlc_samples) > 0
    raw_meta = {"raw_bytes": path.stat().st_size, "parsed_exactly_to_eof": True,
                "embedded_config_sha256": hashlib.sha256(config).hexdigest(), "schema_blocks": config_blocks,
                "raw_event_counts": dict(counts_all), "window_event_counts": dict(counts_window),
                "identities": {k: sorted(v) for k, v in identities.items()},
                "duplicate_lcid_same_tick": repeated_lcid, "changed_lcid_same_tick": changed_lcid,
                "rlc_completed_ticks": len(rlc_samples), "last_open_tick_excluded": True,
                "direct_bsr_status": "AVAILABLE" if bsr_samples else "NOT_RECORDED",
                "rlc_backlog_bytes_event_quantiles": P.quantiles(v for _, v in rlc_samples)}
    return bins, totals, rlc_samples, bsr_samples, raw_meta


def gnb_buffer(arm, summary, rnti):
    path = P.CAPTURES / arm / "capture/m3_radio/ttracer/live_mcs_with_ingest.csv"
    rows = []
    with P.input_path(path).open() as f:
        reader = csv.reader(f); names = ["wall_ns", "raw_ns"] + next(reader)[2:]
        for v in reader:
            assert len(v) == len(names) and v[0].isdigit()
            r = dict(zip(names, v))
            t = int(r["wall_ns"]) / 1e9 - summary["capture_origin_wall_s"]
            if 0 <= t <= summary["duration_s"]:
                val = int(r["estimated_ul_buffer"])
                rows.append((t, val, int(r["rnti"])))
    assert rows and {r[2] for r in rows} == {rnti}, "gNB and UE identities differ"
    return [(t, v) for t, v, _ in rows]


def bin_samples(samples, n):
    bins = defaultdict(list)
    for t, value in samples:
        if 0 <= t < n:
            bins[int(t)].append(value)
    return [{"elapsed_s": i + .5, "n": len(bins[i]),
             **{f"p{q}_bytes": float(np.percentile(bins[i], q)) if bins[i] else None for q in [50, 99]}}
            for i in range(n)]


def extract():
    summaries = P.read_json(P.OUT / "METRICS.json")
    frame_rows = P.read_csv(P.OUT / "FRAMES.csv")
    data, exported, all_queues, all_stats = {}, [], [], {}
    for arm in P.ARMS:
        s = summaries[arm]; n = int(s["duration_s"])
        bins, totals, rlc, bsr, raw_meta = raw_summary(arm, s)
        rlc_rnti = raw_meta["identities"][f"{RLC}:rnti"]
        assert rlc_rnti and len(rlc_rnti) == 1
        estimated = gnb_buffer(arm, s, rlc_rnti[0])
        frames = [r for r in frame_rows if r["arm"] == arm]
        offered = np.zeros(n)
        for r in frames:
            t = float(r["elapsed_s"])
            if 0 <= t < n:
                offered[int(t)] += int(r["payload_bytes"])
        bins["application_offered"] = offered
        totals["application_offered"] = sum(int(r["payload_bytes"]) for r in frames)
        rates = [{"arm": arm, "bin_start_s": i, "bin_end_s": i + 1,
                  **{f"{k}_Mbps": float(v[i] * 8 / 1e6) for k, v in bins.items()}} for i in range(n)]
        rlc_binned, estimate_binned = bin_samples(rlc, n), bin_samples(estimated, n)
        queues = [{"arm": arm, "elapsed_s": i + .5,
                   "rlc_tick_samples": rlc_binned[i]["n"], "rlc_p50_bytes": rlc_binned[i]["p50_bytes"],
                   "rlc_p99_bytes": rlc_binned[i]["p99_bytes"],
                   "gnb_scheduler_samples": estimate_binned[i]["n"], "gnb_estimate_p50_bytes": estimate_binned[i]["p50_bytes"],
                   "gnb_estimate_p99_bytes": estimate_binned[i]["p99_bytes"]} for i in range(n)]
        stats = {"label": P.LABEL[arm], "full_one_second_bins": n, "duration_s": s["duration_s"],
                 "rates_1s_Mbps": {k: P.quantiles(v * 8 / 1e6) for k, v in bins.items()},
                 "whole_window_mean_Mbps": {k: v * 8 / 1e6 / s["duration_s"] for k, v in totals.items()},
                 "gnb_estimated_buffer_bytes_event_quantiles": P.quantiles(v for _, v in estimated),
                 "trace": raw_meta}
        data[arm] = {"rates": rates, "queues": queues, "stats": stats}
        exported.extend(rates); all_queues.extend(queues); all_stats[arm] = stats
        print(arm, json.dumps({"mean_Mbps": stats["whole_window_mean_Mbps"], "RLC": raw_meta["rlc_backlog_bytes_event_quantiles"],
                               "gnb": stats["gnb_estimated_buffer_bytes_event_quantiles"], "bsr": raw_meta["direct_bsr_status"]}), flush=True)
    P.write_csv("TRANSPORT_RATES_1S.csv", exported)
    P.write_csv("BUFFER_OCCUPANCY_1S.csv", all_queues)
    P.write_json("TRANSPORT_METRICS.json", all_stats)
    return data


def rate_plot(data):
    fig = P.canvas("Application load and RLC data flow", notes=[
        "Rates = bytes in each complete 1-second bin × 8 / 1,000,000. Offered payload uses capture time; stack events use their recorded realtime timestamps.",
        "Layer totals include different headers. RLC dequeue is not delivered application goodput; no per-packet causal matching or bottleneck proof is claimed."])
    styles = [("application_offered", "#162B3B", "--", "Application payload offered"),
              ("pdcp_ingress", "#147DAD", "-", "PDCP ingress"),
              ("rlc_dequeue", "#C26828", "-", "RLC → MAC dequeue")]
    fig.legend(handles=[P.Line2D([], [], color=c, ls=ls, lw=2.5, label=name) for _, c, ls, name in styles],
               loc="center", bbox_to_anchor=(.5, .85), ncol=3)
    for j, arm in enumerate(P.ARMS):
        ax = fig.add_axes([.075 + j * .31, .51, .255, .26])
        rows = data[arm]["rates"]
        for key, color, ls, _ in styles:
            ax.plot([r["bin_start_s"] + .5 for r in rows], [r[key + "_Mbps"] for r in rows], color=color, ls=ls, lw=1.8)
        P.time_axis(ax, "Rate (Mb/s)" if j == 0 else None)
        ax.set_ylim(0, 190); ax.set_yticks([0, 50, 100, 150]); ax.set_title(P.LABEL[arm], color=P.COLOR[arm], pad=18)
        ax.set_xlabel("Elapsed time (s)")
        ax = fig.add_axes([.075 + j * .31, .19, .255, .20])
        for k, (key, color, _, _) in enumerate(styles):
            q = data[arm]["stats"]["rates_1s_Mbps"][key]
            for off, stat, alpha in [(-.16, "p50", 1), (.16, "p99", .45)]:
                ax.bar(k + off, q[stat], width=.27, color=color, alpha=alpha)
                ax.text(k + off, q[stat] + 5, f"{q[stat]:.1f}", ha="center", fontsize=11.5, weight="bold")
        ax.set_xticks(range(3), ["Offered", "PDCP in", "RLC out"])
        P.axis(ax, "1-s rate P50/P99\n(Mb/s)" if j == 0 else None)
        ax.set_ylim(0, 205); ax.set_yticks([0, 50, 100, 150, 200])
        ax.text(.5, 1.07, "P50: solid     P99: pale", transform=ax.transAxes, ha="center", fontsize=12)
    P.save(fig, "11_application_rlc_data_rates.png", "Application offered load is capture-binned SFD4 payload, not a T-tracer application event. PDCP ingress and RLC dequeue are decoded directly from each ue.raw recording. P50/P99 refer to complete one-second rate bins; the final partial second is excluded from bin statistics. RLC ingress is also exported in TRANSPORT_RATES_1S.csv. Protocol-layer byte counts are deliberately not labelled equal to application goodput.")


def queue_plot(data):
    fig = P.canvas("Uplink queues: UE RLC occupancy and the gNB estimate", notes=[
        "Solid: within-second median; shaded: median–P99. UE RLC sums distinct LCIDs per completed tick, before multiplexing. Units are bytes, not bytes/s.",
        "Direct NRUE_MAC_BSR_STATUS was NOT recorded. The gNB estimate is scheduler state, not a recovered BSR report. Symlog preserves measured zero."])
    for j, arm in enumerate(P.ARMS):
        rows = data[arm]["queues"]
        for i, (key, label) in enumerate([("rlc", "UE RLC backlog (MB)"), ("gnb_estimate", "gNB estimated backlog (MB)")]):
            ax = fig.add_axes([.075 + j * .31, .53 - i * .34, .255, .25])
            t = [r["elapsed_s"] for r in rows]
            lo = [r[key + "_p50_bytes"] / 1e6 if r[key + "_p50_bytes"] is not None else np.nan for r in rows]
            hi = [r[key + "_p99_bytes"] / 1e6 if r[key + "_p99_bytes"] is not None else np.nan for r in rows]
            ax.fill_between(t, lo, hi, color=P.COLOR[arm], alpha=.22)
            ax.plot(t, lo, color=P.COLOR[arm], lw=2)
            P.time_axis(ax, label if j == 0 else None)
            ax.set_yscale("symlog", linthresh=.01); ax.set_ylim(0, 60)
            ax.set_yticks([0, .01, .1, 1, 10, 50]); ax.yaxis.set_major_formatter(P.ScalarFormatter()); ax.minorticks_off()
            if i == 0:
                ax.set_title(P.LABEL[arm], color=P.COLOR[arm], pad=24)
            else:
                ax.set_xlabel("Elapsed time (s)")
    P.save(fig, "12_rlc_and_gnb_buffer_time.png", "RLC buffer occupancy is an actual UE-side MAC read of RLC before grant multiplexing. The separate gNB series comes from GNB_MAC_UL_MCS_DECISION.estimated_ul_buffer at telemetry ingest time. Its median sits at 0.30 MB for both static arms despite their very different actual RLC queues; it must not substitute for UE occupancy. Direct post-multiplex BSR/LCG state and BSR report frequency cannot be recovered: NRUE_MAC_BSR_STATUS has zero records in these recordings. Do not interpret occupancy derivatives as throughput.")


def main():
    P.style()
    P.input_path(__file__)
    data = extract()
    rate_plot(data); queue_plot(data)
    assert all(P.sha(path) == digest for path, digest in P.INPUTS.items())
    P.write_json("TRANSPORT_INPUT_SHA256.json", P.INPUTS)
    P.write_json("TRANSPORT_VALIDATION.json", {"inputs_unchanged": True, "raw_schema_from_recording": True,
                 "raw_records_parse_to_eof": True, "no_live_services": True, "figures": P.VALIDATION})
    P.write_json("TRANSPORT_CAPTIONS.json", P.CAPTIONS)
    (P.OUT / "TRANSPORT_READ_ME.txt").write_text(
        "TRANSPORT ADDITION — existing traces, CPU-only\n\n"
        "11: offered application payload, PDCP ingress and RLC dequeue rates and one-second P50/P99.\n"
        "12: actual UE RLC occupancy and separately labelled gNB estimated uplink backlog.\n"
        "Direct BSR events were not enabled in the retained UE recording; BSR report values/rates are unavailable.\n"
        "Do not call the gNB estimate a measured BSR. A rising/falling queue is not a throughput measurement.\n"
        "T-tracer binary IDs and integer layouts are decoded from the embedded configuration of each raw file.\n"
        "UE event timestamps use recorded CLOCK_REALTIME; gNB timestamps are collector ingest.\n"
        "Only complete one-second bins are used for rate quantiles. No binning/interpolation of missing telemetry to zero.\n"
        "RLC tick grouping closes only when the next (RNTI, UE, frame, slot) group begins, summing latest distinct LCIDs.\n"
        "Application rate uses SFD4 bytes at capture time; it measures offered volume, not a packet send timestamp.\n"
        "Reproduce after prepare_plots.py:\n"
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 python3 experiments/splitfusion_supervisor_followup_20261013/transport_plots.py\n")
    P.write_json("OUTPUT_SHA256.json", {p.name: P.sha(p) for p in sorted(P.OUT.iterdir()) if p.is_file() and p.name != "OUTPUT_SHA256.json"})


if __name__ == "__main__":
    main()
