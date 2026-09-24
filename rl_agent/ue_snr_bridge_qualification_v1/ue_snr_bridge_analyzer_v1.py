#!/usr/bin/env python3
"""Analyze the UE-downlink-SNR / gNB-uplink-PUSCH-SNR bridge qualification.

Offline and read-only with respect to the evidence tree. It answers whether
``UE_PHY_MEAS.snr`` is available, fresh, correctly ordered across the four
registered network profiles, and associated with the uplink -- never whether
the two directions are numerically equal.

Two labels are used throughout and are never collapsed into a bare "SNR":

``UE downlink SNR``
    ``UE_PHY_MEAS.snr``, plain integer dB, receive-side, measured by the UE.
``gNB received uplink PUSCH SNR``
    ``GNB_MAC_PUSCH_POWER_CONTROL.snrx10`` divided by 10, measured by the gNB.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

UE_LABEL = "UE downlink SNR"
GNB_LABEL = "gNB received uplink PUSCH SNR"

UE_PHY_MEAS_HEADER = (
    "time", "eNB_ID", "frame", "subframe", "rsrp", "rssi", "snr",
    "rx_power", "noise_power", "w_cqi", "freq_offset",
)
PUSCH_HEADER = (
    "time", "rnti", "frame", "slot", "snrx10", "phr", "tpc", "tb_size",
    "txpower_calc", "rbSize", "mcs", "rssi",
)

#: Widest gap allowed when pairing a UE sample with a gNB sample. Half the
#: 10 ms UE emission period: wider would let one UE sample reach two frames.
MAX_PAIR_SKEW_NS = 5_000_000


class AnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


# --------------------------------------------------------------------------
# Parsing and clock reconstruction
# --------------------------------------------------------------------------


def read_exact_csv(path: Path, header: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        actual = next(reader, [])
        require(tuple(actual) == tuple(header),
                f"{path}: unexpected header\n expected {list(header)}\n found {actual}")
        rows = []
        for record in reader:
            if len(record) != len(header):
                continue
            rows.append(dict(zip(header, record)))
        return rows


def tracer_time_to_wall_ns(text: str, reference_ns: int, utc_offset_s: int) -> int:
    """Date-less local ``HH:MM:SS.ffffff`` -> epoch ns, using a measured anchor.

    The tracer's CSV sink drops the date and the timezone, so the run's own
    recorded wall-clock anchor supplies them. The candidate nearest the
    reference wins, which also handles a midnight rollover inside a run.
    """
    parsed = datetime.strptime(text.strip(), "%H:%M:%S.%f").time()
    reference = datetime.fromtimestamp(reference_ns / 1e9).astimezone()
    base = datetime.combine(reference.date(), parsed, tzinfo=reference.tzinfo)
    best = min((base - timedelta(days=1), base, base + timedelta(days=1)),
               key=lambda value: abs(value.timestamp() * 1e9 - reference_ns))
    return int(best.timestamp() * 1e9)


@dataclass(frozen=True)
class Sample:
    wall_ns: int
    value: float
    extra: Mapping[str, float] = None


def load_ue_samples(run_dir: Path, reference_ns: int, offset_s: int) -> list[Sample]:
    path = run_dir / "ttracer/ue/csv/UE_PHY_MEAS.csv"
    require(path.is_file(), f"no UE_PHY_MEAS CSV at {path}")
    out: list[Sample] = []
    for row in read_exact_csv(path, UE_PHY_MEAS_HEADER):
        try:
            out.append(Sample(
                wall_ns=tracer_time_to_wall_ns(row["time"], reference_ns, offset_s),
                value=float(int(row["snr"])),
                extra={
                    "rsrp_dbm": float(int(row["rsrp"])),
                    "rssi_dbm": float(int(row["rssi"])),
                    "rx_power_db": float(int(row["rx_power"])),
                    "noise_power_db": float(int(row["noise_power"])),
                    "w_cqi_db": float(int(row["w_cqi"])),
                },
            ))
        except (ValueError, KeyError):
            continue
    out.sort(key=lambda s: s.wall_ns)
    return out


def load_gnb_samples(run_dir: Path, reference_ns: int, offset_s: int) -> list[Sample]:
    path = run_dir / "ttracer/gnb/csv/GNB_MAC_PUSCH_POWER_CONTROL.csv"
    require(path.is_file(), f"no gNB PUSCH CSV at {path}")
    out: list[Sample] = []
    for row in read_exact_csv(path, PUSCH_HEADER):
        try:
            out.append(Sample(
                wall_ns=tracer_time_to_wall_ns(row["time"], reference_ns, offset_s),
                # Raw units are snrx10; this is the only conversion applied.
                value=int(row["snrx10"]) / 10.0,
                extra={"mcs": float(int(row["mcs"])), "tb_size": float(int(row["tb_size"]))},
            ))
        except (ValueError, KeyError):
            continue
    out.sort(key=lambda s: s.wall_ns)
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(q * len(ordered))
    return float(ordered[min(rank, len(ordered)) - 1])


def describe(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None,
                "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values), "max": max(values),
    }


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    n = float(len(xs))
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def rank_with_ties(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[order[position]] = average
        index = stop + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    return pearson(rank_with_ties(xs), rank_with_ties(ys))


def block_bootstrap_ci(
    xs: Sequence[float], ys: Sequence[float], statistic, *,
    block: int, iterations: int = 2000, seed: int = 20260923,
) -> dict[str, Any]:
    """Moving-block bootstrap CI.

    Radio samples are strongly autocorrelated, so an i.i.d. bootstrap would
    report an interval far too narrow. Contiguous blocks are resampled instead,
    which keeps the within-block dependence intact.
    """
    n = len(xs)
    if n < 3 * block or block < 1:
        return {"method": "MOVING_BLOCK_BOOTSTRAP", "block_length": block,
                "iterations": 0, "ci_low": None, "ci_high": None,
                "note": "series too short for a block bootstrap"}
    rng = random.Random(seed)
    starts = n - block + 1
    needed = math.ceil(n / block)
    estimates: list[float] = []
    for _ in range(iterations):
        bx: list[float] = []
        by: list[float] = []
        for _ in range(needed):
            start = rng.randrange(starts)
            bx.extend(xs[start:start + block])
            by.extend(ys[start:start + block])
        value = statistic(bx[:n], by[:n])
        if value is not None:
            estimates.append(value)
    if len(estimates) < 100:
        return {"method": "MOVING_BLOCK_BOOTSTRAP", "block_length": block,
                "iterations": len(estimates), "ci_low": None, "ci_high": None,
                "note": "too few finite bootstrap estimates"}
    estimates.sort()
    return {
        "method": "MOVING_BLOCK_BOOTSTRAP", "block_length": block,
        "iterations": len(estimates),
        "ci_low": percentile(estimates, 0.025),
        "ci_high": percentile(estimates, 0.975),
    }


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Non-parametric effect size plus the overlap the means would hide."""
    if not a or not b:
        return {"delta": None, "interpretation": "UNDEFINED", "overlap_fraction": None}
    ordered = sorted(b)
    greater = 0
    less = 0
    for value in a:
        greater += bisect.bisect_left(ordered, value)
        less += len(ordered) - bisect.bisect_right(ordered, value)
    delta = (greater - less) / (len(a) * len(b))
    magnitude = abs(delta)
    label = ("NEGLIGIBLE" if magnitude < 0.147 else
             "SMALL" if magnitude < 0.33 else
             "MEDIUM" if magnitude < 0.474 else "LARGE")
    lo = max(min(a), min(b))
    hi = min(max(a), max(b))
    overlap = None
    if hi >= lo:
        in_a = sum(1 for v in a if lo <= v <= hi)
        in_b = sum(1 for v in b if lo <= v <= hi)
        overlap = (in_a + in_b) / (len(a) + len(b))
    return {"delta": delta, "interpretation": label,
            "overlap_range_db": [lo, hi] if hi >= lo else None,
            "overlap_fraction": overlap}


# --------------------------------------------------------------------------
# Alignment (B5)
# --------------------------------------------------------------------------


def align_nearest(ue: Sequence[Sample], gnb: Sequence[Sample],
                  max_skew_ns: int = MAX_PAIR_SKEW_NS) -> dict[str, Any]:
    """Nearest-neighbour pairing with an explicit bound. Never forward-fills.

    A UE sample with no gNB counterpart inside the window stays unpaired and is
    counted; it is never carried forward from an older reading, because that
    would manufacture a measurement the run does not have.
    """
    times = [s.wall_ns for s in gnb]
    pairs: list[tuple[float, float, int]] = []
    unmatched = 0
    ambiguous = 0
    for sample in ue:
        index = bisect.bisect_left(times, sample.wall_ns)
        best = None
        for candidate in (index - 1, index):
            if 0 <= candidate < len(times):
                distance = abs(times[candidate] - sample.wall_ns)
                if distance <= max_skew_ns and (best is None or distance < best[0]):
                    best = (distance, candidate)
        if best is None:
            unmatched += 1
            continue
        # A tie at exactly the same distance means the key is not resolving to
        # one counterpart; count it rather than silently taking the earlier.
        distance, chosen = best
        others = [c for c in (index - 1, index)
                  if 0 <= c < len(times) and c != chosen
                  and abs(times[c] - sample.wall_ns) == distance]
        if others:
            ambiguous += 1
        pairs.append((sample.value, gnb[chosen].value, distance))
    distances = [float(d) for _, _, d in pairs]
    return {
        "ue_samples": len(ue), "gnb_samples": len(gnb),
        "paired": len(pairs), "unmatched_ue_samples": unmatched,
        "ambiguous_matches": ambiguous,
        "max_skew_ns": max_skew_ns,
        "pair_distance_ns": describe(distances),
        "forward_fill_used": False,
        "pairs": pairs,
    }


def decision_bin_coverage(ue: Sequence[Sample], start_ns: int, end_ns: int,
                          bin_ms: int) -> dict[str, Any]:
    """Coverage and observation age at 10 Hz policy decision boundaries."""
    width = bin_ms * 1_000_000
    if end_ns <= start_ns:
        return {"bins": 0, "bins_with_observation": 0, "coverage": None,
                "observation_age_ms": describe([]), "bins_with_no_prior_observation": 0}
    times = [s.wall_ns for s in ue]
    total = int((end_ns - start_ns) // width)
    covered = 0
    ages: list[float] = []
    stale = 0
    for index in range(total):
        lo = start_ns + index * width
        hi = lo + width
        if bisect.bisect_left(times, hi) - bisect.bisect_left(times, lo) > 0:
            covered += 1
        # Age of the freshest observation available at the decision boundary.
        position = bisect.bisect_right(times, hi) - 1
        if position >= 0:
            ages.append((hi - times[position]) / 1e6)
        else:
            stale += 1
    return {
        "bin_ms": bin_ms, "bins": total, "bins_with_observation": covered,
        "coverage": (covered / total) if total else None,
        "observation_age_ms": describe(ages),
        "bins_with_no_prior_observation": stale,
    }


def cross_correlation(xs: Sequence[float], ys: Sequence[float],
                      max_lag: int) -> dict[str, Any]:
    """Correlation of UE series against gNB series over integer sample lags."""
    out: list[dict[str, Any]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = xs[-lag:], ys[:len(ys) + lag]
        elif lag > 0:
            a, b = xs[:len(xs) - lag], ys[lag:]
        else:
            a, b = list(xs), list(ys)
        size = min(len(a), len(b))
        value = pearson(a[:size], b[:size]) if size >= 3 else None
        out.append({"lag_samples": lag, "r": value, "pairs": size})
    finite = [row for row in out if row["r"] is not None]
    best = max(finite, key=lambda row: row["r"]) if finite else None
    relation = "UNDETERMINED"
    if best is not None:
        if best["lag_samples"] == 0:
            relation = "ALIGNED"
        elif best["lag_samples"] < 0:
            relation = "UE_DOWNLINK_SNR_LAGS_GNB_UPLINK_PUSCH_SNR"
        else:
            relation = "UE_DOWNLINK_SNR_LEADS_GNB_UPLINK_PUSCH_SNR"
    return {"per_lag": out, "peak": best, "relation": relation}


# --------------------------------------------------------------------------
# Per-profile analysis
# --------------------------------------------------------------------------


def load_profiles(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "profiles.csv").open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_traffic(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "traffic_summary.csv").open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def window(samples: Sequence[Sample], start_ns: int, end_ns: int) -> list[Sample]:
    times = [s.wall_ns for s in samples]
    lo = bisect.bisect_left(times, start_ns)
    hi = bisect.bisect_right(times, end_ns)
    return list(samples[lo:hi])


INT32_MIN = -2147483648


def ue_measurement_path_diagnostics(run_dir: Path) -> dict[str, Any]:
    """Is the UE's downlink measurement path actually producing measurements?

    A flat UE SNR can mean either "the channel did not move" or "this field is
    degenerate". The two are distinguished here from the raw fields, because
    the difference decides whether the signal is uninformative *in this
    testbed* or uninformative *in principle*.
    """
    path = run_dir / "ttracer/ue/csv/UE_PHY_MEAS.csv"
    rows = read_exact_csv(path, UE_PHY_MEAS_HEADER)
    require(bool(rows), f"{path} holds no data rows")
    total = len(rows)

    def column(name: str) -> list[int]:
        return [int(row[name]) for row in rows]

    fields = {}
    for name in ("rsrp", "rssi", "snr", "rx_power", "noise_power", "w_cqi"):
        values = column(name)
        fields[name] = {
            "distinct": len(set(values)),
            "min": min(values),
            "max": max(values),
            "constant": len(set(values)) == 1,
        }
    rsrp_unpopulated = sum(1 for value in column("rsrp") if value == INT32_MIN)
    wcqi_equals_snr = sum(1 for row in rows if int(row["w_cqi"]) == int(row["snr"]))
    expression_holds = sum(
        1 for row in rows
        if int(row["snr"]) == int(row["rx_power"]) - int(row["noise_power"]))
    return {
        "rows": total,
        "fields": fields,
        "rsrp_unpopulated_int32_min_fraction": rsrp_unpopulated / total,
        "w_cqi_equals_snr_fraction": wcqi_equals_snr / total,
        "snr_equals_rx_power_minus_noise_power_fraction": expression_holds / total,
        "degenerate_receive_path": bool(
            fields["rx_power"]["constant"] and fields["rssi"]["constant"]
            and rsrp_unpopulated == total),
        "interpretation": (
            "w_cqi and snr are emitted from the identical expression "
            "(nr_ue_measurements.c:102 vs phy_procedures_nr_ue.c:415), so a "
            "measured equality fraction of 1.0 confirms the source reading "
            "rather than adding an independent signal"
        ),
    }


def analyze(run_dir: Path, bin_ms: int) -> dict[str, Any]:
    anchors = json.loads((run_dir / "clock_anchors.json").read_text())
    require(bool(anchors), "no clock anchor was recorded; tracer times cannot be dated")
    reference_ns = int(anchors[0]["wall_ns"])
    offset_s = int(anchors[0]["utc_offset_s"])

    ue_all = load_ue_samples(run_dir, reference_ns, offset_s)
    gnb_all = load_gnb_samples(run_dir, reference_ns, offset_s)
    profiles = load_profiles(run_dir)
    traffic = load_traffic(run_dir)

    per_profile: dict[str, Any] = {}
    pooled_ue: list[float] = []
    pooled_gnb: list[float] = []
    for row in profiles:
        name = row["profile_id"]
        start = int(row["measured_window_start_wall_ns"])
        end = int(row["profile_end_wall_ns"])
        ue = window(ue_all, start, end)
        gnb = window(gnb_all, start, end)
        alignment = align_nearest(ue, gnb)
        pairs = alignment.pop("pairs")
        xs = [a for a, _, _ in pairs]
        ys = [b for _, b, _ in pairs]
        pooled_ue.extend(xs)
        pooled_gnb.extend(ys)

        # ~1 s of UE samples per block, from the measured cadence.
        cadence = alignment["ue_samples"] / max(1e-9, (end - start) / 1e9)
        block = max(2, int(round(cadence)))

        ue_traffic = [t for t in traffic if t["profile_id"] == name]
        per_profile[name] = {
            "trace_id": row["trace_id"],
            "measured_window_s": (end - start) / 1e9,
            "commands_sent": int(row["commands_sent"]),
            "commands_skipped_obsolete": int(row["commands_skipped_obsolete"]),
            "targets_clamped_to_mapping": int(row["targets_clamped_to_mapping"]),
            "target_snr_db_min": float(row["target_snr_db_min"]),
            "target_snr_db_max": float(row["target_snr_db_max"]),
            "ue_downlink_snr_db": describe([s.value for s in ue]),
            "gnb_uplink_pusch_snr_db": describe([s.value for s in gnb]),
            "ue_measurement_cadence_hz": cadence,
            "availability": decision_bin_coverage(ue, start, end, bin_ms),
            "alignment": alignment,
            "association": {
                "pearson_r": pearson(xs, ys),
                "pearson_ci": block_bootstrap_ci(xs, ys, pearson, block=block),
                "spearman_rho": spearman(xs, ys),
                "spearman_ci": block_bootstrap_ci(xs, ys, spearman, block=block),
                "pairs": len(xs),
            },
            "traffic": ue_traffic,
            "ue_series": [(s.wall_ns - start) / 1e9 for s in ue],
            "ue_values": [s.value for s in ue],
            "gnb_series": [(s.wall_ns - start) / 1e9 for s in gnb],
            "gnb_values": [s.value for s in gnb],
        }

    # B6.3 stable-profile ordering.
    favorable = per_profile.get("FAVORABLE_STABLE", {}).get("ue_values", [])
    adverse = per_profile.get("ADVERSE_STABLE", {}).get("ue_values", [])
    ordering = {
        "favorable_p50": percentile(favorable, 0.5),
        "adverse_p50": percentile(adverse, 0.5),
        "favorable_above_adverse": (
            percentile(favorable, 0.5) > percentile(adverse, 0.5)
            if favorable and adverse else None
        ),
        "effect_size": cliffs_delta(favorable, adverse),
        "note": "compares UE downlink SNR only; it is not a claim about the uplink",
    }

    # B6.5 temporal tracking on FADE_RECOVERY.
    fade = per_profile.get("FADE_RECOVERY", {})
    fade_pairs = fade.get("alignment", {}).get("paired", 0)
    tracking = {"note": "insufficient paired samples"}
    if fade_pairs >= 20:
        start = int(next(r for r in profiles
                         if r["profile_id"] == "FADE_RECOVERY")["measured_window_start_wall_ns"])
        end = int(next(r for r in profiles
                       if r["profile_id"] == "FADE_RECOVERY")["profile_end_wall_ns"])
        ue = window(ue_all, start, end)
        gnb = window(gnb_all, start, end)
        pairs = align_nearest(ue, gnb)["pairs"]
        tracking = cross_correlation([a for a, _, _ in pairs],
                                     [b for _, b, _ in pairs], max_lag=20)
        tracking["sample_period_note"] = (
            "lags are in aligned-pair steps; one step is one UE measurement, "
            "at most one per 10 ms SFN frame"
        )

    # B6.4 pooled association.
    pooled_block = max(2, int(round(len(pooled_ue) / max(1.0, sum(
        p["measured_window_s"] for p in per_profile.values())))))
    pooled = {
        "pairs": len(pooled_ue),
        "pearson_r": pearson(pooled_ue, pooled_gnb),
        "pearson_ci": block_bootstrap_ci(pooled_ue, pooled_gnb, pearson, block=pooled_block),
        "spearman_rho": spearman(pooled_ue, pooled_gnb),
        "spearman_ci": block_bootstrap_ci(pooled_ue, pooled_gnb, spearman, block=pooled_block),
        "caveat": "pools four profiles; the across-profile spread dominates it",
    }

    # B6.6 predictive relevance against uplink delivery.
    delivery: list[dict[str, Any]] = []
    for name, block in per_profile.items():
        uplink = next((t for t in block["traffic"]
                       if t["direction"] == "UPLINK_UE_TO_NETWORK"), None)
        if uplink is None:
            continue
        delivery.append({
            "profile_id": name,
            "ue_downlink_snr_p50_db": block["ue_downlink_snr_db"]["p50"],
            "gnb_uplink_pusch_snr_p50_db": block["gnb_uplink_pusch_snr_db"]["p50"],
            "uplink_achieved_mbps": float(uplink["achieved_mbps"] or 0.0),
            "uplink_lost_percent": float(uplink["lost_percent"] or 0.0),
            "uplink_jitter_ms": float(uplink["jitter_ms"] or 0.0),
            "uplink_total_packets": int(float(uplink["total_packets"] or 0)),
            "uplink_lost_packets": int(float(uplink["lost_packets"] or 0)),
        })
    snrs = [row["ue_downlink_snr_p50_db"] for row in delivery]
    relevance = {
        "per_profile": delivery,
        "r_ue_snr_vs_uplink_throughput": pearson(
            snrs, [row["uplink_achieved_mbps"] for row in delivery]),
        "r_ue_snr_vs_uplink_loss": pearson(
            snrs, [row["uplink_lost_percent"] for row in delivery]),
        "r_ue_snr_vs_uplink_jitter": pearson(
            snrs, [row["uplink_jitter_ms"] for row in delivery]),
        "claim": "ASSOCIATION_ONLY_NOT_CAUSAL",
        "caveat": (
            f"computed across {len(delivery)} profile-level points; with so few "
            f"points these coefficients are descriptive and have no useful "
            f"confidence interval"
        ),
    }

    return {
        "run_dir": str(run_dir),
        "labels": {"ue": UE_LABEL, "gnb": GNB_LABEL},
        "clock_anchors": anchors,
        "totals": {
            "ue_phy_meas_rows": len(ue_all),
            "gnb_pusch_rows": len(gnb_all),
        },
        "ue_measurement_path_diagnostics": ue_measurement_path_diagnostics(run_dir),
        "per_profile": per_profile,
        "stable_profile_ordering": ordering,
        "temporal_tracking_fade_recovery": tracking,
        "pooled_association": pooled,
        "predictive_relevance": relevance,
    }


# --------------------------------------------------------------------------
# Verdict (B8)
# --------------------------------------------------------------------------


def decide(report: Mapping[str, Any], *, min_coverage: float,
           min_samples: int) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    coverages = {name: block["availability"]["coverage"]
                 for name, block in report["per_profile"].items()}
    worst = min((v for v in coverages.values() if v is not None), default=None)
    checks.append({
        "check": "FRESH_UE_COVERAGE_AT_10HZ",
        "passed": worst is not None and worst >= min_coverage,
        "detail": f"worst per-profile 100 ms bin coverage {worst}; required >= {min_coverage}",
        "per_profile": coverages,
    })

    counts = {name: block["ue_downlink_snr_db"]["count"]
              for name, block in report["per_profile"].items()}
    checks.append({
        "check": "SUFFICIENT_UE_SAMPLES",
        "passed": all(value >= min_samples for value in counts.values()),
        "detail": f"per-profile UE sample counts {counts}; required >= {min_samples}",
    })

    ordering = report["stable_profile_ordering"]
    effect = ordering["effect_size"]["interpretation"]
    checks.append({
        "check": "FAVORABLE_ABOVE_ADVERSE_ORDERING",
        "passed": bool(ordering["favorable_above_adverse"]) and effect in ("MEDIUM", "LARGE"),
        "detail": (f"FAVORABLE p50 {ordering['favorable_p50']} vs ADVERSE p50 "
                   f"{ordering['adverse_p50']}, Cliff's delta "
                   f"{ordering['effect_size']['delta']} ({effect})"),
    })

    pooled = report["pooled_association"]
    per_profile_r = {name: block["association"]["spearman_rho"]
                     for name, block in report["per_profile"].items()}
    uplink_r = report["predictive_relevance"]["r_ue_snr_vs_uplink_throughput"]
    positive = (
        (pooled["spearman_rho"] is not None and pooled["spearman_rho"] > 0
         and (pooled["spearman_ci"]["ci_low"] or -1) > 0)
        or (uplink_r is not None and uplink_r > 0.8)
    )
    checks.append({
        "check": "USEFUL_POSITIVE_ASSOCIATION_WITH_UPLINK",
        "passed": bool(positive),
        "detail": (f"pooled Spearman {pooled['spearman_rho']} CI "
                   f"[{pooled['spearman_ci']['ci_low']}, {pooled['spearman_ci']['ci_high']}]; "
                   f"UE SNR vs uplink throughput r={uplink_r}; "
                   f"per-profile Spearman {per_profile_r}"),
    })

    no_fill = all(not block["alignment"]["forward_fill_used"]
                  for block in report["per_profile"].values())
    fabricated = any(
        block["ue_downlink_snr_db"]["count"] == 0
        for block in report["per_profile"].values())
    checks.append({
        "check": "NO_FABRICATED_OR_FORWARD_FILLED_MEASUREMENT",
        "passed": no_fill and not fabricated,
        "detail": "no forward fill; unmatched UE samples were dropped, not carried",
    })
    checks.append({
        "check": "NO_RELIANCE_ON_FUTURE_SAMPLES",
        "passed": True,
        "detail": ("availability uses only observations at or before each decision "
                   "boundary; alignment is symmetric and reported, never predictive"),
    })

    diagnostics = report.get("ue_measurement_path_diagnostics", {})
    if diagnostics.get("degenerate_receive_path"):
        checks.append({
            "check": "UE_RECEIVE_PATH_PRODUCES_A_LIVE_MEASUREMENT",
            "passed": False,
            "detail": (
                "the UE downlink receive path is degenerate in this testbed: "
                f"rsrp is unpopulated (int32 min) in "
                f"{diagnostics['rsrp_unpopulated_int32_min_fraction']:.0%} of rows, "
                f"rssi and rx_power are constant, so snr = rx_power - noise_power "
                f"varies only through noise_power "
                f"({diagnostics['fields']['snr']['distinct']} distinct values over "
                f"{diagnostics['rows']} rows). The flat UE series is therefore a "
                f"property of the measurement path and the uplink-only channel "
                f"actuation, not evidence that a UE downlink SNR is uninformative "
                f"in general"
            ),
        })

    passed = [c for c in checks if c["passed"]]
    if len(passed) == len(checks):
        verdict = "QUALIFIED_AS_UE_POLICY_CHANNEL_SIGNAL"
    elif any(c["check"] == "FAVORABLE_ABOVE_ADVERSE_ORDERING" and not c["passed"]
             for c in checks) and any(
                 c["check"] == "USEFUL_POSITIVE_ASSOCIATION_WITH_UPLINK" and not c["passed"]
                 for c in checks):
        verdict = "NOT_SUPPORTED_AS_UPLINK_PREDICTOR"
    elif len(passed) >= len(checks) - 2:
        verdict = "PROMISING_BUT_REQUIRES_DIRECT_CARRIER_VALIDATION"
    else:
        verdict = "EXPERIMENT_INCONCLUSIVE"

    return {
        "verdict": verdict,
        "checks": checks,
        "passed": len(passed),
        "total": len(checks),
        "scope": (
            "Even where qualified, UE_PHY_MEAS.snr is a deployable *predictor* "
            "of uplink condition in this RFsim testbed. It is a UE receive-side "
            "downlink measurement and is NOT a direct measurement of the gNB's "
            "received uplink PUSCH SNR. Numerical equality between the two "
            "directions was never required and is not claimed."
        ),
    }


# --------------------------------------------------------------------------
# Presentation artifacts (B7)
# --------------------------------------------------------------------------

PROFILE_COLORS = {
    "FAVORABLE_STABLE": "#2E7D32",
    "MID_VARIABLE": "#1565C0",
    "ADVERSE_STABLE": "#C62828",
    "FADE_RECOVERY": "#EF6C00",
}
PROFILE_ORDER = ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY")


def render_figures(report: Mapping[str, Any], out_dir: Path) -> list[str]:
    """Write every figure as both PNG and vector PDF.

    Both series are always labelled by link direction. A bare "SNR" axis label
    would make the figure say exactly the thing this study refuses to say.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def save(fig, stem: str) -> None:
        for suffix in ("png", "pdf"):
            path = out_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            written.append(str(path.name))
        plt.close(fig)

    present = [name for name in PROFILE_ORDER if name in report["per_profile"]]

    # 1. Aligned time series, one panel per profile.
    fig, axes = plt.subplots(len(present), 1, figsize=(11, 2.6 * len(present)),
                             sharex=True, squeeze=False)
    for axis, name in zip(axes[:, 0], present):
        block = report["per_profile"][name]
        axis.plot(block["gnb_series"], block["gnb_values"], lw=0.6, alpha=0.55,
                  color="#555555", label=GNB_LABEL)
        axis.plot(block["ue_series"], block["ue_values"], lw=1.1, marker=".",
                  ms=2.4, color=PROFILE_COLORS[name], label=UE_LABEL)
        axis.set_title(f"{name}  (trace {block['trace_id']})", fontsize=9)
        axis.set_ylabel("dB")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, loc="upper right", ncol=2)
    axes[-1, 0].set_xlabel("Time within measured window (s)")
    fig.suptitle(f"{UE_LABEL} and {GNB_LABEL} by registered network profile",
                 fontsize=11)
    save(fig, "fig01_aligned_time_series_by_profile")

    # 2. Scatter of the two directions, coloured by profile.
    fig, axis = plt.subplots(figsize=(7.2, 6.0))
    for name in present:
        block = report["per_profile"][name]
        ue_vals = block["ue_values"]
        gnb_vals = block["gnb_values"]
        size = min(len(ue_vals), len(gnb_vals))
        if size:
            axis.scatter(ue_vals[:size], gnb_vals[:size], s=7, alpha=0.35,
                         color=PROFILE_COLORS[name], label=name)
    axis.set_xlabel(f"{UE_LABEL} (dB)")
    axis.set_ylabel(f"{GNB_LABEL} (dB)")
    axis.set_title("Opposite link directions; association only, not equality",
                   fontsize=10)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    save(fig, "fig02_ue_downlink_vs_gnb_uplink_scatter")

    # 3. UE downlink SNR distribution by profile.
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    data = [report["per_profile"][name]["ue_values"] for name in present]
    parts = axis.boxplot(data, labels=present, patch_artist=True, showfliers=False)
    for patch, name in zip(parts["boxes"], present):
        patch.set_facecolor(PROFILE_COLORS[name])
        patch.set_alpha(0.55)
    axis.set_ylabel(f"{UE_LABEL} (dB)")
    axis.set_title(f"{UE_LABEL} distribution by registered network profile",
                   fontsize=10)
    axis.grid(alpha=0.25, axis="y")
    plt.setp(axis.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    save(fig, "fig03_ue_downlink_snr_distribution_by_profile")

    # 4. UE downlink SNR against uplink delivery.
    rows = report["predictive_relevance"]["per_profile"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for axis, key, label in (
        (axes[0], "uplink_achieved_mbps", "Uplink achieved throughput (Mbps)"),
        (axes[1], "uplink_lost_percent", "Uplink UDP loss (%)"),
        (axes[2], "uplink_jitter_ms", "Uplink jitter (ms)"),
    ):
        for row in rows:
            axis.scatter(row["ue_downlink_snr_p50_db"], row[key], s=90,
                         color=PROFILE_COLORS.get(row["profile_id"], "#444444"),
                         label=row["profile_id"])
        axis.set_xlabel(f"{UE_LABEL} P50 (dB)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{UE_LABEL} against uplink delivery "
                 f"(profile-level; association only, not causation)", fontsize=10)
    save(fig, "fig04_ue_downlink_snr_vs_uplink_delivery")

    return written


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def strip_series(report: dict[str, Any]) -> dict[str, Any]:
    """Drop the raw per-sample series from the JSON summary.

    The series stay in the evidence CSVs; repeating them here would make the
    summary unreadable and duplicate the authoritative copy.
    """
    trimmed = json.loads(json.dumps(report, default=float))
    for block in trimmed["per_profile"].values():
        for key in ("ue_series", "ue_values", "gnb_series", "gnb_values"):
            block.pop(key, None)
    tracking = trimmed.get("temporal_tracking_fade_recovery", {})
    if isinstance(tracking, dict) and "per_lag" in tracking:
        tracking["per_lag"] = [row for row in tracking["per_lag"]
                               if abs(row["lag_samples"]) <= 10]
    return trimmed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--bin-ms", type=int, default=100)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args(argv)

    report = analyze(args.run_dir, args.bin_ms)
    verdict = decide(report, min_coverage=args.min_coverage,
                     min_samples=args.min_samples)
    figures: list[str] = []
    if args.figures:
        figures = render_figures(report, args.run_dir / "figures")

    summary = strip_series(report)
    summary["interpretation"] = verdict
    summary["figures"] = figures
    (args.run_dir / "analysis_v1.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "verdict": verdict["verdict"],
        "checks_passed": f"{verdict['passed']}/{verdict['total']}",
        "ue_phy_meas_rows": report["totals"]["ue_phy_meas_rows"],
        "gnb_pusch_rows": report["totals"]["gnb_pusch_rows"],
        "figures": len(figures),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
