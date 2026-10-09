"""Per-base-chain live hold timing for the environment (ms after the chain's own action open), from existing evidence.

chain id ``<source>#<frame_id>`` with source valid03 (pilot_valid03), gcv2, retention.
* held_first_send: the chain's POLICY_HOLD frame first send (None if the next frame is not its hold)
* sample: the next decision's (policy or fallback) selected RLC sample instant = open' - backlog_age'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from . import sessions as S

SOURCES = {"valid03": "pilot_valid03", "gcv2": "gcv2", "retention": "retention"}
MAX_GAP_NS = 600_000_000


def build() -> dict:
    table, summary = {}, {}
    for source, name in SOURCES.items():
        frames = S.load_frames(S.PROFILER / S.SESSIONS[name][1])
        n_ok = 0
        for k, f in enumerate(frames):
            if f["kind"] != "POLICY_DECISION":
                continue
            held = frames[k + 1] if k + 1 < len(frames) and frames[k + 1]["kind"] == "POLICY_HOLD" else None
            nxt = next((g for g in frames[k + 1:] if g["kind"] in ("POLICY_DECISION", "FALLBACK") and g["open"] is not None
                        and g["age"] is not None), None)
            entry = dict(first_send=(f["first_send"] - f["open"]) / 1e6,
                         held_first_send=None if held is None else (held["first_send"] - f["open"]) / 1e6,
                         sample=None)
            if nxt is not None and 0 < nxt["open"] - f["open"] <= MAX_GAP_NS:
                entry["sample"] = (nxt["open"] - nxt["age"] - f["open"]) / 1e6
            if entry["held_first_send"] is not None and entry["sample"] is not None:
                n_ok += 1
            table[f"{source}#{f['frame_id']}"] = entry
        vals = [v for k_, v in table.items() if k_.startswith(source + "#") and v["sample"] is not None and v["held_first_send"] is not None]
        summary[source] = dict(chains=sum(1 for k_ in table if k_.startswith(source + "#")), complete=n_ok,
                               held_first_send_p50=statistics.median(v["held_first_send"] for v in vals),
                               sample_p50=statistics.median(v["sample"] for v in vals),
                               held_to_sample_p50=statistics.median(v["sample"] - v["held_first_send"] for v in vals))
    medians = {s: dict(held_offset_after_first_send=statistics.median(v["held_first_send"] - v["first_send"] for k_, v in table.items()
                                                                       if k_.startswith(s + "#") and v["held_first_send"] is not None),
                       sample=summary[s]["sample_p50"]) for s in SOURCES}
    return dict(schema="splitfusion.run5_backlog_transition.chain_timing.v1", chains=table, summary=summary, pool_medians=medians)


def write(out: Path) -> str:
    out.mkdir(parents=True, exist_ok=False)
    doc = build()
    text = json.dumps(doc, indent=1, sort_keys=True)
    (out / "CHAIN_TIMING.json").write_text(text)
    print(json.dumps(doc["summary"], indent=1), json.dumps(doc["pool_medians"]))
    digest = hashlib.sha256(text.encode()).hexdigest()
    print("CHAIN_TIMING.json sha256", digest)
    return digest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    write(Path(p.parse_args().out))
