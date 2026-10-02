#!/usr/bin/env python3
"""Offline parity of the thread-cap arms before any live capture.

1. **Actor replay.**  The 420 recorded actor inputs of the frozen 1,000-frame
   capture (``agent.actor`` spans: 21 features plus the live decision digests)
   are replayed through the registered frozen actor in a fresh process per arm
   (arm A: baseline environment; arm B: ``OMP_NUM_THREADS=1 MKL_NUM_THREADS=1``,
   both with ``CUDA_VISIBLE_DEVICES=`` and no ``PYTHONPATH``).  Per-arm outputs
   (mode, q_e4, state and head digests, latency) are retained, then compared
   arm-to-arm and each arm to the live record.
2. **Existing parity checks.**  The registered bit-exact and codec-identity
   test modules run unchanged, once per arm environment, with their own
   tolerances (exact equality; nothing is relaxed).

The ``ue.profile.jsonl`` the inputs come from is hash-checked against the
frozen capture record first.  Everything is written create-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from . import threadcap_v1 as TC

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
SCHEMA = "scenesense.single_gpu_profiler.threadcap_offline_parity.v1"
FROZEN_RECORD = PACKAGE / "CAPTURE_1000_FROZEN_EXPORT.json"
UE_PROFILE_RELPATH = ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
                      "20261001T231343Z_run4_profiling_capture_1000/cells/a71__favorable_stable/"
                      "profiler/ue.profile.jsonl")
PARITY_TEST_FILES = (
    "rl_agent/splitfusion_sensor_optimization_v1/tests/test_optimized_stages.py",
    "rl_agent/splitfusion_sensor_optimization_v1/tests/test_wired_path_equivalence.py",
    "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/test_phase3_continuous_execution_v2.py",
)
DIGEST_KEYS = ("decided_mode_id", "decided_q_e4", "state_sha256", "heads_sha256")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def recorded_inputs(root: Path = ROOT) -> list[dict[str, Any]]:
    """The 420 live actor inputs, after verifying the source file's registered hash."""
    record = json.loads(FROZEN_RECORD.read_text(encoding="utf-8"))
    expected = record["evidence_sha256"][UE_PROFILE_RELPATH]
    path = root / UE_PROFILE_RELPATH
    TC.require(sha256_file(path) == expected, "frozen ue.profile.jsonl hash differs from record")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if '"stage":"agent.actor"' not in line:
                continue
            record = json.loads(line)
            if record.get("r") == "span" and record.get("stage") == "agent.actor":
                rows.append({"seq": record["seq"], "frame_id": record.get("frame_id"),
                             "features": [float(v) for v in record["features"]],
                             **{k: record.get(k) for k in DIGEST_KEYS}})
    rows.sort(key=lambda r: r["seq"])
    return rows


def replay(arm: str, out: Path) -> dict[str, Any]:
    """Replay in this process (the caller sets the environment); write JSONL."""
    import torch  # noqa: F401 - loaded after the environment is set

    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import frozen_actor_v2 as FA

    actor = FA.load_registered_actor()
    inputs = recorded_inputs()
    header = {"record": "header", "arm": arm, "schema": SCHEMA,
              "actor_boundary_sha256": actor.boundary_sha256,
              "settings": TC.settings_snapshot("replay", arm=arm), "inputs": len(inputs)}
    latencies = []
    with Path(out).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True, default=str) + "\n")
        for row in inputs:
            started = time.perf_counter_ns()
            decision = actor.act_on_vector(tuple(row["features"]))
            latency_us = (time.perf_counter_ns() - started) / 1e3
            latencies.append(latency_us)
            handle.write(json.dumps({
                "record": "decision", "seq": row["seq"], "frame_id": row["frame_id"],
                "decided_mode_id": int(decision.mode_id), "decided_q_e4": int(decision.q_e4),
                "state_sha256": decision.state_sha256, "heads_sha256": decision.heads_sha256,
                "latency_us": latency_us}, sort_keys=True) + "\n")
    ordered = sorted(latencies)
    return {"arm": arm, "replayed": len(inputs), "torch_intra_op_threads":
            header["settings"].get("torch", {}).get("intra_op_threads"),
            "latency_us": {"p50": ordered[len(ordered) // 2], "p95": ordered[int(len(ordered) * 0.95)],
                           "max": ordered[-1]} if ordered else None}


def _arm_env(arm: str, *, cuda_visible: Optional[str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for key in TC.CAP_VARIABLES:
        env.pop(key, None)                       # baseline on this host: unset
    env.update(TC.ARMS[arm])
    if cuda_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    return env


def _load_rows(path: Path) -> tuple[dict, list[dict]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return rows[0], [r for r in rows[1:] if r.get("record") == "decision"]


def compare_replays(a_path: Path, b_path: Path, inputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    header_a, rows_a = _load_rows(a_path)
    header_b, rows_b = _load_rows(b_path)
    by_seq = {r["seq"]: r for r in inputs}

    def digest(row: Mapping[str, Any]) -> tuple:
        return tuple(row.get(k) for k in DIGEST_KEYS)

    def versus_live(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        mism = [r["seq"] for r in rows if digest(r) != digest(by_seq[r["seq"]])]
        return {"compared": len(rows), "mismatches": len(mism), "examples": mism[:10]}

    ab = [(x["seq"], digest(x), digest(y)) for x, y in zip(rows_a, rows_b) if digest(x) != digest(y)]
    action_only = [s for s, x, y in ab if x[:2] != y[:2]]
    return {"a_rows": len(rows_a), "b_rows": len(rows_b),
            "a_vs_b": {"compared": min(len(rows_a), len(rows_b)), "mismatches": len(ab),
                       "action_mismatches_mode_or_q": len(action_only), "examples": ab[:5]},
            "a_vs_live": versus_live(rows_a), "b_vs_live": versus_live(rows_b),
            "settings": {"A": _settings_summary(header_a), "B": _settings_summary(header_b)}}


def _settings_summary(header: Mapping[str, Any]) -> dict[str, Any]:
    settings = header.get("settings") or {}
    torch = settings.get("torch") if isinstance(settings.get("torch"), Mapping) else {}
    return {"env": {k: (settings.get("environment") or {}).get(k) for k in TC.CAP_VARIABLES},
            "torch_intra_op_threads": torch.get("intra_op_threads"),
            "torch_inter_op_threads": torch.get("inter_op_threads"),
            "openmp_max_threads": settings.get("openmp_max_threads"),
            "threadpools": settings.get("threadpools"), "cap_effective": settings.get("cap_effective"),
            "actor_boundary_sha256": header.get("actor_boundary_sha256")}


def run_tests(arm: str, out: Path) -> dict[str, Any]:
    env = _arm_env(arm, cuda_visible=None)       # the CUDA bit-exact tests need the device
    with Path(out).open("x", encoding="utf-8") as handle:
        completed = subprocess.run([sys.executable, "-m", "unittest", "-v", *PARITY_TEST_FILES],
                                   cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   timeout=1800, check=False)
    text = Path(out).read_text(encoding="utf-8")
    summary = next((line for line in reversed(text.splitlines()) if line.startswith("Ran ")), "")
    verdict = next((line for line in reversed(text.splitlines())
                    if line.startswith(("OK", "FAILED"))), "")
    return {"arm": arm, "returncode": completed.returncode, "ran": summary, "verdict": verdict,
            "skipped": text.count("... skipped"), "passed": completed.returncode == 0,
            "modules": list(PARITY_TEST_FILES)}


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-arm", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.replay_arm:
        summary = replay(args.replay_arm, args.out)
        print(json.dumps(summary, sort_keys=True))
        return 0
    TC.require(args.run and args.out_dir is not None, "--run --out-dir required")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    inputs = recorded_inputs()
    result: dict[str, Any] = {"schema": SCHEMA, "inputs": len(inputs),
                              "ue_profile_sha256": sha256_file(ROOT / UE_PROFILE_RELPATH),
                              "replay": {}, "tests": {}}
    for arm in ("A", "B"):
        out = out_dir / f"actor_replay_{arm}.jsonl"
        completed = subprocess.run(
            [sys.executable, "-m", "rl_agent.splitfusion_single_gpu_profiler_v1."
             "offline_parity_threadcap_v1", "--replay-arm", arm, "--out", str(out)],
            cwd=str(ROOT), env=_arm_env(arm, cuda_visible=""), capture_output=True, text=True,
            timeout=900, check=False)
        (out_dir / f"actor_replay_{arm}.stdout.txt").write_text(completed.stdout + completed.stderr)
        TC.require(completed.returncode == 0, f"replay {arm} failed: {completed.stderr[-400:]}")
        result["replay"][arm] = json.loads(completed.stdout.strip().splitlines()[-1])
        result["tests"][arm] = run_tests(arm, out_dir / f"parity_tests_{arm}.log")
    result["comparison"] = compare_replays(out_dir / "actor_replay_A.jsonl",
                                           out_dir / "actor_replay_B.jsonl", inputs)
    comparison = result["comparison"]
    result["verdict"] = {
        "actor_identity_a_vs_b": comparison["a_vs_b"]["mismatches"] == 0,
        "actor_identity_vs_live": (comparison["a_vs_live"]["mismatches"] == 0
                                   and comparison["b_vs_live"]["mismatches"] == 0),
        "parity_tests_both_arms": all(t["passed"] for t in result["tests"].values()),
        "cap_effective_in_b": comparison["settings"]["B"]["cap_effective"] is True,
        "cap_absent_in_a": comparison["settings"]["A"]["cap_effective"] is False}
    result["passed"] = all(result["verdict"].values())
    result["files_sha256"] = {p.name: sha256_file(p) for p in sorted(out_dir.iterdir())}
    with (out_dir / "OFFLINE_PARITY_RESULT.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
    print(json.dumps({"passed": result["passed"], "verdict": result["verdict"],
                      "replay": result["replay"], "tests": {k: v["ran"] + " " + v["verdict"]
                                                            for k, v in result["tests"].items()}},
                     indent=1, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
