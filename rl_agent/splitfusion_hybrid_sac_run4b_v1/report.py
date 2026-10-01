#!/usr/bin/env python3
"""Run-4B learning report and hash-bound evidence index (read-only).

Usage: python -m rl_agent.splitfusion_hybrid_sac_run4b_v1.report RUN_DIR
where RUN_DIR holds ``smoke/``, ``resume_test/`` and ``campaign/``.
Writes textual evidence into this package's ``evidence/`` directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

from . import runner as R

FAMILIES = ("noAE", "AE128", "AE64", "AE32")
PACKAGE = Path(__file__).resolve().parent


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1) + 0.5))]


def _entropy(counts: dict[Any, int]) -> float:
    total = sum(counts.values())
    return -sum(c / total * math.log(c / total) for c in counts.values() if c)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def window_stats(decisions: list[dict], metrics: list[dict]) -> dict[str, Any]:
    rewards = [d["reward"] for d in decisions]
    timely = [d for d in decisions if d["terminal"] == "TIMELY_SUCCESS"]
    latency = [d["operational_latency_ns"] / 1e6 for d in timely]
    modes: dict[int, int] = {}
    families = {f: 0 for f in FAMILIES}
    for d in decisions:
        modes[d["mode_id"]] = modes.get(d["mode_id"], 0) + 1
        families[d["family"]] += 1
    q = [d["q_e4"] / 9800.0 for d in decisions]
    q_mean = sum(q) / len(q)
    q_std = math.sqrt(sum((v - q_mean) ** 2 for v in q) / len(q))
    finite = all(math.isfinite(float(v)) for m in metrics for v in m.values())
    return {
        "decisions": len(decisions),
        "reward_mean": sum(rewards) / len(rewards),
        "timely_fraction": len(timely) / len(decisions),
        "timeouts_or_failures": len(decisions) - len(timely),
        "q_perc_training_only_mean_timely": (
            sum(d["q_perc_training_only"] for d in timely) / len(timely)
            if timely else None),
        "operational_latency_ms_timely": {
            "p50": _pct(latency, 0.50), "p95": _pct(latency, 0.95),
            "p99": _pct(latency, 0.99)},
        "family_shares": {f: c / len(decisions) for f, c in families.items()},
        "distinct_modes": len(modes),
        "executed_mode_entropy_nats": _entropy(modes),
        "q_normalized_mean": q_mean, "q_normalized_std": q_std,
        "updates": len(metrics),
        "policy_discrete_entropy_mean": (
            sum(m["discrete_entropy"] for m in metrics) / len(metrics)
            if metrics else None),
        "critic_loss_mean": (sum(m["critic_loss"] for m in metrics)
                             / len(metrics) if metrics else None),
        "critic_grad_norm_max": max((m["critic_grad_norm"] for m in metrics),
                                    default=None),
        "actor_grad_norm_max": max((m["actor_grad_norm"] for m in metrics),
                                   default=None),
        "all_metrics_finite": finite,
    }


def seed_report(seed_dir: Path) -> dict[str, Any]:
    decisions = _jsonl(seed_dir / "DECISIONS.jsonl")
    metrics = _jsonl(seed_dir / "UPDATE_METRICS.jsonl")
    events = _jsonl(seed_dir / "CHECKPOINT_EVENTS.jsonl")
    windows = []
    previous_update, previous_decisions = 0, R.WARMUP
    windows.append({"window": "warmup_288", "checkpoint_update": 0,
                    **window_stats(decisions[:R.WARMUP], []),
                    "policy_panel_families": events[0]["policy_panel"][
                        "family_probability_mean"]})
    for event in events[1:]:
        update, count = event["update"], event["decision_count"]
        windows.append({
            "window": f"updates_{previous_update + 1}_{update}",
            "checkpoint_update": update,
            **window_stats(decisions[previous_decisions:count],
                           metrics[previous_update:update]),
            "under_observed_share": event["window_family"][
                "under_observed_share"],
            "policy_panel_families": event["policy_panel"][
                "family_probability_mean"],
            "state_fingerprint": event["state_fingerprint"],
            "manifest_sha256": event["manifest_sha256"]})
        previous_update, previous_decisions = update, count
    return {"seed": int(seed_dir.name.split("_")[1]),
            "final_update": len(metrics), "decisions": len(decisions),
            "windows": windows}


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown(document: dict[str, Any]) -> str:
    lines = [
        "# Run-4B learning report (offline modeled, exploratory)",
        "",
        "Scope: offline modeled training under "
        "`EXPLORATORY_POOLED_FAMILY_TRANSFER_ASSUMPTION`; not deployment "
        "evidence. Q_perc is a training-reward input only. Latency is the "
        "operational `A+S+T+E+D` total.",
        "",
        f"Campaign status: **{document['campaign_status']}**. Pre-registered "
        f"live actor: seed 43 / update 10,000, actor SHA-256 "
        f"`{document.get('live_actor_sha256')}`.",
        "",
    ]
    for seed in document["seeds"]:
        lines += [f"## Seed {seed['seed']}", "",
                  "| Window | Reward | Timely | Timeouts | Q_perc (timely) | "
                  "L_op p50/p95 ms | noAE / AE128 / AE64 / AE32 | Modes | "
                  "Mode H | q std | Policy H | Finite |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for w in seed["windows"]:
            lat = w["operational_latency_ms_timely"]
            fam = w["family_shares"]
            lines.append(
                f"| {w['window']} | {_fmt(w['reward_mean'])} | "
                f"{_fmt(w['timely_fraction'])} | {w['timeouts_or_failures']} | "
                f"{_fmt(w['q_perc_training_only_mean_timely'])} | "
                f"{_fmt(lat['p50'], 1)}/{_fmt(lat['p95'], 1)} | "
                + " / ".join(_fmt(fam[f], 2) for f in FAMILIES)
                + f" | {w['distinct_modes']} | "
                f"{_fmt(w['executed_mode_entropy_nats'], 2)} | "
                f"{_fmt(w['q_normalized_std'], 3)} | "
                f"{_fmt(w['policy_discrete_entropy_mean'], 2)} | "
                f"{w['all_metrics_finite']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(run_dir: str) -> int:
    run = Path(run_dir)
    campaign = run / "campaign"
    complete = campaign / "CAMPAIGN_COMPLETE.json"
    status_doc = json.loads((complete if complete.exists()
                             else campaign / "CAMPAIGN_HALTED.json").read_text())
    seeds = [seed_report(campaign / f"seed_{s}") for s in R.SEEDS
             if (campaign / f"seed_{s}" / "DECISIONS.jsonl").exists()]
    smoke = seed_report(run / "smoke" / "seed_17")
    exports = status_doc.get("actor_exports", {})
    document = {
        "schema": "splitfusion.run4b.learning_report.v1",
        "campaign_status": status_doc["status"],
        "code_commit": status_doc["code_commit"],
        "live_actor_sha256": exports.get("43", {}).get(
            "actor_state_dict_sha256"),
        "smoke_seed_17": smoke, "seeds": seeds,
    }
    evidence = PACKAGE / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "LEARNING_REPORT.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n")
    (evidence / "LEARNING_REPORT.md").write_text(markdown(document))
    for source, name in (
            (run / "smoke" / "SMOKE_500_GATE.json", "SMOKE_500_GATE.json"),
            (run / "smoke" / "seed_17" / "PREFLIGHT_288.json",
             "SMOKE_PREFLIGHT_288.json"),
            (run / "resume_test" / "RESUME_EQUIVALENCE.json",
             "RESUME_EQUIVALENCE.json"),
            (campaign / "CAMPAIGN_START.json", "CAMPAIGN_START.json"),
            (complete if complete.exists() else
             campaign / "CAMPAIGN_HALTED.json", complete.name if
             complete.exists() else "CAMPAIGN_HALTED.json")):
        shutil.copyfile(source, evidence / name)
    for seed in R.SEEDS:
        events = campaign / f"seed_{seed}" / "CHECKPOINT_EVENTS.jsonl"
        if events.exists():
            shutil.copyfile(events, evidence / f"seed_{seed}_CHECKPOINT_EVENTS.jsonl")
        export = campaign / f"seed_{seed}" / "final_actor" / "ACTOR_EXPORT.json"
        if export.exists():
            shutil.copyfile(export, evidence / f"seed_{seed}_ACTOR_EXPORT.json")
    index = {str(p.relative_to(run)): {"sha256": _sha(p), "bytes": p.stat().st_size}
             for p in sorted(run.rglob("*")) if p.is_file()}
    (evidence / "PAYLOAD_INDEX.json").write_text(json.dumps(
        {"schema": "splitfusion.run4b.payload_index.v1",
         "run_dir": str(run.resolve().relative_to(R.ROOT)),
         "files": index}, indent=2, sort_keys=True) + "\n")
    print(evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
