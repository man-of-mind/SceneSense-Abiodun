"""Export editable P50/P95/P99 rows with explicit units and denominators."""
import argparse
import json
from pathlib import Path

import prepare_plots as P
from replot import PACKAGE, records
from export_feedback import write_csv, quantiles


def build(output):
    metrics = json.loads((PACKAGE / "data/METRICS.json").read_text())
    transport = json.loads((PACKAGE / "data/TRANSPORT_METRICS.json").read_text())
    down = json.loads((PACKAGE / "data/DOWNLINK_SUMMARY.json").read_text())
    segments, decisions = records("AOI_SEGMENTS.csv"), records("DECISIONS.csv")
    rows = []

    def add(arm, metric, unit, stats, population, weight="sample", duration=None):
        rows.append({"arm": arm, "configuration": P.LABEL[arm], "metric": metric, "unit": unit,
                     "n_observed": stats.get("n"), "weighting": weight, "observed_duration_s": duration,
                     "p50": stats["p50"], "p95": stats["p95"], "p99": stats["p99"], "population": population})

    for arm in P.ARMS:
        s = metrics[arm]
        for key, unit, definition in [
            ("payload_bytes", "bytes", "All 1000 transmitted frames, including holds and fallbacks"),
            ("prediction_ms", "ms", "Recorded completed policy predictions only; missing outcomes excluded, not zero"),
            ("feedback_ms", "ms", "Recorded policy feedback receipts only; action-open to UE receive, not pure downlink"),
            ("map_install_age_ms", "ms", "Successful map installations, including drain; capture to install"),
            ("radio_snr_db", "dB", "All retained PUSCH transmission events inside capture window"),
            ("radio_mcs", "index", "All retained PUSCH transmission events inside capture window")]:
            add(arm, key, unit, s[key], definition)
        add(arm, "payload_kB", "decimal kB", {k: v / 1000 if k != "n" else v for k, v in s["payload_bytes"].items()},
            "All 1000 transmitted frames, including holds and fallbacks")
        seg = [r for r in segments if r["arm"] == arm]
        add(arm, "map_AoI", "ms", {f"p{q}": P.aoi_quantile(seg, q / 100) * 1000 for q in (50, 95, 99)},
            "After first map install until last transmitted capture; initial no-map interval excluded",
            "continuous time", sum(r["duration_s"] for r in seg))
        dec = [r for r in decisions if r["arm"] == arm]
        for key, unit in [("SI", "Sobel variability"), ("P40", "unitless"), ("q", "fraction"),
                          ("Q_perc", "score"), ("vehicle_recall_completed", "fraction"), ("person_recall_completed", "fraction")]:
            add(arm, key, unit, quantiles(r[key] for r in dec), "Defined per-decision values only; recall quantiles are NOT pooled object recall")
        t = transport[arm]
        for key, stat in t["rates_1s_Mbps"].items():
            add(arm, key, "Mb/s", stat, "Complete 1-second bins; partial final second excluded")
        add(arm, "UE_RLC_occupancy", "bytes", t["trace"]["rlc_backlog_bytes_event_quantiles"],
            "Completed UE ticks; distinct LCIDs summed; not a rate or BSR")
        add(arm, "gNB_UL_buffer_estimate", "bytes", t["gnb_estimated_buffer_bytes_event_quantiles"],
            "Observed scheduler estimates; not a directly measured BSR")
        for key, stat in down[arm]["rates_1s_Mbps"].items():
            add(arm, key, "Mb/s", stat, "Captured feedback-port downlink, complete 1-second bins; not total radio DL throughput")
        add(arm, "logged_controller_reward", "reward", down[arm]["logged_reward_defined_only"],
            "Legacy Run4 logging schema; timeout=-1, evaluator fault excluded; not frozen Run5 training reward")
        add(arm, "R4FB_udp_payload", "bytes", down[arm]["quality_feedback_udp_payload_bytes"],
            "Captured R4FB quality/status messages only; reward scalar resolved locally at UE")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows)
    print(f"Wrote {len(rows)} quantile rows to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PACKAGE / "quantiles_regenerated.csv")
    build(parser.parse_args().output.resolve())
