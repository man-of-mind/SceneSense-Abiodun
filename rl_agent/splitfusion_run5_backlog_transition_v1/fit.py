"""Rate/access-delay fit on RLC busy periods (plan: FITTING_PLAN.json). The observed next-decision backlog is never used."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from . import sessions as S
from .transition import RateTableV1

BINS = ((0, 10), (11, 15), (16, 19), (20, 23), (24, 28))


def busy_periods(name: str) -> list[dict]:
    rel, cell_rel, part = S.SESSIONS[name]
    cell = S.PROFILER / cell_rel
    ticks = S.load_rlc_ticks(cell)
    pdcp = S.load_pdcp(cell)
    pt = [p[0] for p in pdcp]
    cum = [0]
    for _, b in pdcp:
        cum.append(cum[-1] + b)
    frames = S.load_frames(cell)
    dec = S.load_decisions(rel)
    sends = sorted((f["first_send"], k) for k, f in enumerate(frames))
    st = [s[0] for s in sends]
    # next decision's MCS for each frame index
    next_dec_mcs = {}
    nxt = None
    for k in range(len(frames) - 1, -1, -1):
        f = frames[k]
        next_dec_mcs[k] = nxt
        if f["kind"] == "POLICY_DECISION" and f["frame_id"] in dec:
            nxt = dec[f["frame_id"]].get("prior_ul_mcs")
    out, i, n = [], 0, len(ticks)
    while i < n:
        if ticks[i].bytes > 0 and (i == 0 or ticks[i - 1].bytes == 0):
            j = i
            gap = False
            while j < n and ticks[j].bytes > 0:
                if j > i and ticks[j].source_raw_ns - ticks[j - 1].source_raw_ns > 5_000_000:
                    gap = True
                j += 1
            if j >= n:
                break
            start, end = ticks[i].source_raw_ns, ticks[j].source_raw_ns
            if ticks[j].source_raw_ns - ticks[j - 1].source_raw_ns > 5_000_000:
                gap = True
            a = bisect.bisect_left(pt, start - 2_000_000)
            b = bisect.bisect_left(pt, end)
            R = cum[b] - cum[a]
            k0 = bisect.bisect_left(st, start - 2_000_000)
            frame = frames[sends[k0][1]] if k0 < len(sends) and sends[k0][0] <= start else None
            if gap or R < 5000 or frame is None or a >= len(pt):
                i = j
                continue
            kidx = sends[k0][1]
            if frame["kind"] == "POLICY_DECISION":
                mcs = dec.get(frame["frame_id"], {}).get("prior_ul_mcs")
            elif frame["kind"] == "POLICY_HOLD":
                mcs = next_dec_mcs.get(kidx)
            else:
                mcs = None
            if mcs is not None:
                T = (end - pdcp[a][0]) / 1e6
                out.append(dict(session=name, partition=part, kind=frame["kind"], frame_id=frame["frame_id"], mcs=int(mcs),
                                R=R, T=T))
            i = j
        else:
            i += 1
    return out


def fit(periods: list[dict]) -> RateTableV1:
    hi = [p for p in periods if p["mcs"] >= 20]
    X = np.column_stack([np.ones(len(hi)), [p["R"] for p in hi]])
    beta, *_ = np.linalg.lstsq(X, np.array([p["T"] for p in hi]), rcond=None)
    d0 = float(beta[0])
    rows = []
    for lo, hi_ in BINS:
        vals = [p["R"] / (p["T"] - d0) for p in periods if lo <= p["mcs"] <= hi_ and p["T"] - d0 > 1.0]
        rows.append([lo, hi_, len(vals), float(statistics.median(vals)) if vals else None])
    sparse = []
    k = 0
    while k < len(rows):                               # merge bins with < 20 periods into the next higher bin
        if rows[k][2] < 20 and k + 1 < len(rows):
            sparse.append(f"{rows[k][0]}-{rows[k][1]}")
            lo = rows[k][0]
            vals = [p["R"] / (p["T"] - d0) for p in periods if lo <= p["mcs"] <= rows[k + 1][1] and p["T"] - d0 > 1.0]
            rows[k + 1] = [lo, rows[k + 1][1], len(vals), float(statistics.median(vals))]
            rows.pop(k)
        else:
            k += 1
    # weighted pool-adjacent-violators: non-decreasing in MCS
    blocks = [[r[3], r[2], [i]] for i, r in enumerate(rows)]
    changed = True
    while changed:
        changed = False
        for b in range(len(blocks) - 1):
            if blocks[b][0] > blocks[b + 1][0]:
                w = blocks[b][1] + blocks[b + 1][1]
                v = (blocks[b][0] * blocks[b][1] + blocks[b + 1][0] * blocks[b + 1][1]) / w
                blocks[b] = [v, w, blocks[b][2] + blocks[b + 1][2]]
                blocks.pop(b + 1)
                changed = True
                break
    rate = [None] * len(rows)
    for v, _, idx in blocks:
        for i in idx:
            rate[i] = v
    return RateTableV1(tuple((r[0], r[1]) for r in rows), tuple(rate), d0, tuple(sparse)), rows


def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    allp = []
    for name in S.SESSIONS:
        p = busy_periods(name)
        allp += p
        print(f"{name:14s} busy periods {len(p)} (hold-started {sum(1 for x in p if x['kind']=='POLICY_HOLD')}); MCS<=10: {sum(1 for x in p if x['mcs']<=10)}, 11-15: {sum(1 for x in p if 11<=x['mcs']<=15)}")
    fitp = [p for p in allp if p["partition"] == "FIT"]
    table, rows = fit(fitp)
    loso = {}
    for name, v in S.SESSIONS.items():
        if v[2] != "FIT":
            continue
        t, _ = fit([p for p in fitp if p["session"] != name])
        loso[name] = t.to_dict()
    print("ALL-FIT table:", json.dumps(table.to_dict()), "\nbin rows (lo, hi, n, median r):", rows)
    for name, t in loso.items():
        print(f"  LOSO -{name:14s} d0 {t['d0_ms']:.2f} rates {[round(r) for r in t['rate_bytes_per_ms']]} bins {t['bins']}")
    with open(out / "BUSY_PERIODS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allp[0]))
        w.writeheader()
        w.writerows(allp)
    (out / "FIT.json").write_text(json.dumps(dict(all_fit=table.to_dict(), bin_rows=rows, loso=loso,
                                                  n_fit_periods=len(fitp)), indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    run(Path(p.parse_args().out))
