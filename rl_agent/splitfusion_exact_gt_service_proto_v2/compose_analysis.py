"""Post-hoc analysis of a composed replay (read-only on its evidence). Writes ANALYSIS.json.

* Attribution of source-delivery tail: was the prep worker busy when the freeze arrived?
* RECONSTRUCTED live receipt (kept separate from measurements): measured result-at-edge in the
  replay + that frame's recorded v11 edge feedback-emit span and emit-to-UE-receipt downlink.
"""
import argparse, bisect, json, math
from pathlib import Path
from . import parity_common as P, timeline as TL


def stats(v):
    v = sorted(x for x in v if x is not None)
    n = len(v)
    return dict(n=n, p50=round(v[(n - 1) // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2, 3),
                p95=round(v[max(0, math.ceil(.95 * n) - 1)], 3), max=round(v[-1], 3), min=round(v[0], 3))


def run(name):
    out = P.EVIDENCE_ROOT / name
    rows = [json.loads(l) for l in open(out / "FRAMES.jsonl")]
    ue = {r["frame_id"]: r for r in json.loads((out / "UE_RECORDS.json").read_bytes())["records"]}
    samples = json.loads((out / "QUEUE_SAMPLES.json").read_bytes())
    times = [s["t"] for s in samples]
    spans, marks = TL._profile(TL.EDGE_PROFILE)
    uspans, umarks = TL._profile(TL.UE_PROFILE)
    busy, delivery_busy, delivery_idle, recon = 0, [], [], []
    for r in rows:
        ao = ue[r["frame_id"]]["action_open_raw_ns"]
        arrival = ao + int(r["freeze_received"] * 1e6)
        i = bisect.bisect_left(times, arrival)
        before = samples[max(0, i - 1)]
        delivery = r["source_decoded"] - r["freeze_received"]
        prep_busy = before["prep_queue"] > 0 or before["label"] in ("prep",) and False
        # A prep request in flight is visible as prep_open (previous frame unresolved) or a queued item.
        flag = before["prep_queue"] > 0 or before["prep_retained"] > 1
        (delivery_busy if flag else delivery_idle).append(delivery)
        busy += flag
        f = r["frame_id"]
        emit = spans[f].get("edge.feedback_emit")
        receipt = umarks[f].get("ue.feedback_receipt")
        if emit and receipt:
            recon.append(r["result_received_at_edge"] + (emit[1] - emit[0]) / 1e6 + (receipt - emit[1]) / 1e6)
    analysis = dict(
        delivery_ms_prep_busy_or_backlog=stats(delivery_busy) if delivery_busy else dict(n=0),
        delivery_ms_prep_idle=stats(delivery_idle), frames_with_prep_backlog_at_freeze=busy,
        reconstructed_ue_receipt_ms=dict(stats(recon), timely_le_170=sum(x <= 170 for x in recon),
            basis="measured replay result-at-edge + recorded v11 edge.feedback_emit span + recorded emit->UE receipt"),
        note="Reconstructed values are NOT measurements of a live run.")
    (out / "ANALYSIS.json").open("x").write(json.dumps(analysis, indent=1, sort_keys=True))
    print(json.dumps(analysis, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--name", required=True); run(p.parse_args().name)
