"""Byte-identity of GT outputs, original vs candidate accounting (offline replay, in-process, CPU only).

The live service embeds monotonic clock readings in the source evidence (``source_synchronization.observed_raw_ns``,
``sources/frozen_world/snapshot_observed_raw_ns``), so ready-packet/semantic digests differ between ANY two live runs.
Offline replay mode uses the recorded clocks instead, so two runs on identical inputs must be byte-identical.

Two complete runs of the prep worker's engine class (``DeferredProducerEngine`` via ``parity_common.DeferredDriver``,
producer V2 + tilt-aware AVO switches installed, as in the prep worker) over the 49-frame retained corpus with the same
accept/hold/unsent schedule: arm ``original`` (nothing else installed) and arm ``candidate`` (accounting switch
installed).  Per frame the ready packet bytes, identity/semantic digests, commit result, history digests before/after,
and every archived file must be byte-identical; the engine/owner logical byte accounting (``owner.payload_bytes`` per
retained frame, sampled after every action) must be equal.

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 4 /usr/bin/python3 -B -m \\
      rl_agent.splitfusion_gt_storage_accounting_v3.output_parity --out <new dir>
"""
import argparse
import hashlib
import json
from pathlib import Path

from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN
from rl_agent.splitfusion_numeric_gt_producer_v2 import worker_switch as PV2
from rl_agent.splitfusion_avo_tilt_v1 import switch as TILT

from . import accounting as A, switch
from rl_agent.splitfusion_gt_storage_accounting_v1.process_parity import action_for


def _h(value):
    return hashlib.sha256(value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def one_arm(corpus, candidate):
    PV2.install(); TILT.install()
    if candidate:
        switch.install()
    rows = []
    try:
        driver = P.DeferredDriver(corpus)
        try:
            for i, case in enumerate(corpus.cases):
                r = driver.run(case, action=action_for(i))
                owner = driver.engine._owner
                row = dict(frame_id=case.frame_id, action=r["action"], history_before=r["history_before"],
                           history_after=r["history_after"], owner_payload_bytes=owner.snapshot()["payload_bytes"],
                           owner_rows=sorted((k, v.payload_bytes) for k, v in owner._frames.items()))
                if r["action"] == "accept":
                    row.update(packet=_h(r["packet"]), identity=_h(r["identity_bytes"]), semantic=r["semantic_sha256"],
                               commit=_h(r["commit"]), archive={k: _h(v) for k, v in r["archive_raw"].items()},
                               archive_descriptor=_h(r["archive_descriptor"]))
                rows.append(row)
            snapshot = driver.engine.snapshot()
        finally:
            driver.close()
        accounting_was = OWN.payload_storage_bytes
        session_counters = switch.session().snapshot() if candidate else None
    finally:
        switch.uninstall(); TILT.uninstall(); PV2.uninstall()
    return dict(rows=rows, engine_retired=snapshot["retired"], engine_issued=snapshot["issued"],
                session=session_counters,
                accounting_function=(f"{accounting_was.__module__}.{accounting_was.__qualname__}" if hasattr(accounting_was, "__qualname__") else f"{type(accounting_was).__module__}.{type(accounting_was).__qualname__}"))


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    P.environment_ok()
    if not switch.is_original(OWN.payload_storage_bytes):
        raise RuntimeError("original arm would not run the original ownership.payload_storage_bytes")
    corpus = P.Corpus(49)
    original = one_arm(corpus, candidate=False)
    candidate = one_arm(corpus, candidate=True)
    mismatches = [dict(frame_id=a["frame_id"], keys=[k for k in set(a) | set(b) if a.get(k) != b.get(k)])
                  for a, b in zip(original["rows"], candidate["rows"]) if a != b]
    summary = dict(schema="splitfusion.gt_storage_accounting_v3.output_parity.v1", mode="OFFLINE_REPLAY_RECORDED_CLOCKS",
                   frames=len(original["rows"]), accepted=sum(r["action"] == "accept" for r in original["rows"]),
                   holds=sum(r["action"] == "hold" for r in original["rows"]),
                   unsent=sum(r["action"] == "unsent" for r in original["rows"]),
                   identical_frames=len(original["rows"]) - len(mismatches), mismatches=mismatches,
                   engine=dict(original=[original["engine_issued"], original["engine_retired"]],
                               candidate=[candidate["engine_issued"], candidate["engine_retired"]]),
                   accounting_functions=[original["accounting_function"], candidate["accounting_function"]],
                   switches_restored=not switch.installed() and not TILT.installed() and not PV2.installed(),
                   candidate_version=A.VERSION, candidate_session=candidate["session"])
    (out / "ROWS.json").write_text(json.dumps(dict(original=original, candidate=candidate), indent=1) + "\n")
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    used = candidate["session"]["snapshot_fast"] > 0 and candidate["session"]["owner_shortcut_hits"] > 0
    return (not mismatches and summary["switches_restored"] and used
            and original["accounting_function"] != candidate["accounting_function"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    raise SystemExit(0 if run(p.parse_args().out) else 1)
