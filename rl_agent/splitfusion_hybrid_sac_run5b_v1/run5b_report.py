#!/usr/bin/env python3
"""Run-5B learning report and hash-bound evidence index (read-only).

Per-window statistics reuse Run-4B's ``report.window_stats``/``seed_report``
unchanged.  Added for Run 5B: SNR response (outcomes and executed family by
causal-SNR tercile in the final 2,000-update window; controlled and shuffled
SNR actor diagnostics on each final replay), checkpoint COMMITTED hashes and
the actor exports.  No held-scene or live validation.

    python -m rl_agent.splitfusion_hybrid_sac_run5b_v1.run5b_report CAMPAIGN_RUNS_DIR
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

from rl_agent.splitfusion_hybrid_sac_run4b_v1 import report as RP4

from . import run5b_checks as K
from . import run5b_runner as R

PACKAGE = Path(__file__).resolve().parent
FAMILIES = RP4.FAMILIES


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snr_terciles(decisions: list[dict]) -> dict[str, Any]:
    ordered = sorted(d["snr_db"] for d in decisions)
    cuts = (ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3])
    groups = {"low": [], "mid": [], "high": []}
    for d in decisions:
        key = "low" if d["snr_db"] < cuts[0] else "mid" if d["snr_db"] < cuts[1] else "high"
        groups[key].append(d)
    out = {"cuts_db": list(cuts)}
    for key, rows in groups.items():
        timely = [d for d in rows if d["terminal"] == "TIMELY_SUCCESS"]
        out[key] = {
            "decisions": len(rows),
            "snr_db_mean": sum(d["snr_db"] for d in rows) / len(rows),
            "reward_mean": sum(d["reward"] for d in rows) / len(rows),
            "timely_fraction": len(timely) / len(rows),
            "q_perc_timely_mean": (sum(d["q_perc_training_only"] for d in timely) / len(timely)
                                   if timely else None),
            "mcs_mean": sum(d["prior_ul_mcs"] for d in rows) / len(rows),
            "q_normalized_mean": sum(d["q_e4"] / 9800.0 for d in rows) / len(rows),
            "family_shares": {f: sum(d["family"] == f for d in rows) / len(rows)
                              for f in FAMILIES}}
    return out


def build(runs: Path) -> dict[str, Any]:
    campaign = runs / "campaign_three_seed_10000_v1"
    status_path = campaign / "CAMPAIGN_COMPLETE.json"
    status = json.loads((status_path if status_path.exists()
                         else campaign / "CAMPAIGN_HALTED.json").read_text())
    torch.set_num_threads(R.THREADS)
    sources = R.E.load_sources(R.EVIDENCE_ROOT)
    seeds = []
    for seed in R.SEEDS:
        seed_dir = campaign / f"seed_{seed}"
        report = RP4.seed_report(seed_dir)
        decisions = RP4._jsonl(seed_dir / "DECISIONS.jsonl")
        final = R.restore_runner(seed_dir / "checkpoints" / R.bundle_name(R.FINAL_UPDATE),
                                 sources, seed)
        events = RP4._jsonl(seed_dir / "CHECKPOINT_EVENTS.jsonl")
        report.update({
            "snr_response_final_2000_updates": snr_terciles(decisions[-8000:]),
            "snr_actor_diagnostics_final": K.snr_diagnostics(final, seed),
            "profile_segments_audit_only": final.env.profile_balance(),
            "checkpoints": [{"update": e["update"], "bundle": e["bundle"],
                             "committed_manifest_sha256": e["manifest_sha256"],
                             "state_fingerprint": e["state_fingerprint"]} for e in events],
            "actor_export": status.get("actor_exports", {}).get(str(seed))})
        seeds.append(report)
    exports = status.get("actor_exports", {})
    return {"schema": "splitfusion.run5b.learning_report.v1",
            "campaign_status": status["status"], "code_commit": status["code_commit"],
            "registration_sha256": R.REG.sealed_sha256(),
            "joint_channel_binding_sha256": sources.binding_sha256,
            "operational_latency_provider_sha256": sources.run4b.provider.binding_sha256,
            "live_actor_sha256": exports.get("43", {}).get("actor_state_dict_sha256"),
            "live_actor": {"seed": 43, "update": 10_000,
                           "actor_state_dict_sha256": exports.get("43", {}).get(
                               "actor_state_dict_sha256"),
                           "actor_tree_sha256": exports.get("43", {}).get("actor_tree_sha256")},
            "fresh_process_actor_verification": {
                k: v for k, v in status.get("fresh_process_actor_verification", {}).items()
                if k != "actors"},
            "smoke_seed_17": RP4.seed_report(runs / "smoke_seed17_v2" / "seed_17"),
            "seeds": seeds}


def markdown(document: dict[str, Any]) -> str:
    text = RP4.markdown(document).replace("# Run-4B learning report", "# Run-5B learning report")
    lines = [text, "## SNR response (final 2,000 updates, causal-SNR terciles)", "",
             "| Seed | Tercile | SNR dB | MCS | Reward | Timely | Q_perc (timely) | q | "
             "noAE / AE128 / AE64 / AE32 |", "|---|---|---|---|---|---|---|---|---|"]
    for seed in document["seeds"]:
        terciles = seed["snr_response_final_2000_updates"]
        for key in ("low", "mid", "high"):
            t = terciles[key]
            lines.append(f"| {seed['seed']} | {key} | {t['snr_db_mean']:.1f} | "
                         f"{t['mcs_mean']:.1f} | {t['reward_mean']:.3f} | "
                         f"{t['timely_fraction']:.3f} | {t['q_perc_timely_mean']:.3f} | "
                         f"{t['q_normalized_mean']:.3f} | "
                         + " / ".join(f"{t['family_shares'][f]:.2f}" for f in FAMILIES) + " |")
    lines += ["", "| Seed | Controlled 0→1 mode TV | argmax change | Δq_e4 | "
              "Shuffled mode TV | argmax change | abs Δq_e4 |", "|---|---|---|---|---|---|---|"]
    for seed in document["seeds"]:
        s = seed["snr_actor_diagnostics_final"]
        lines.append(f"| {seed['seed']} | {s['controlled_sweep_0_to_1_mode_tv_mean']:.3f} | "
                     f"{s['controlled_sweep_argmax_mode_change_rate']:.3f} | "
                     f"{s['controlled_sweep_q_e4_delta_mean']:.0f} | "
                     f"{s['shuffled_mode_tv_mean']:.3f} | "
                     f"{s['shuffled_argmax_mode_change_rate']:.3f} | "
                     f"{s['shuffled_abs_q_e4_delta_mean']:.0f} |")
    return "\n".join(lines) + "\n"


def main(runs_dir: str) -> int:
    runs = Path(runs_dir)
    document = build(runs)
    evidence = PACKAGE / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "LEARNING_REPORT.json").write_text(json.dumps(document, indent=2,
                                                              sort_keys=True) + "\n")
    (evidence / "LEARNING_REPORT.md").write_text(markdown(document))
    campaign = runs / "campaign_three_seed_10000_v1"
    copies = [(runs / "smoke_seed17_v2" / "SMOKE_500_GATE.json", "SMOKE_500_GATE.json"),
              (runs / "smoke_seed17_v2" / "seed_17" / "PREFLIGHT_288.json",
               "SMOKE_PREFLIGHT_288.json"),
              (runs / "smoke_seed17_attempt1_checker_dtype_defect" / "SMOKE_500_GATE.json",
               "SMOKE_ATTEMPT1_CHECKER_DEFECT_GATE.json"),
              (runs / "resume_seed17_v2" / "RESUME_EQUIVALENCE.json", "RESUME_EQUIVALENCE.json"),
              (campaign / "CAMPAIGN_START.json", "CAMPAIGN_START.json")]
    for name in ("CAMPAIGN_COMPLETE.json", "CAMPAIGN_HALTED.json"):
        if (campaign / name).exists():
            copies.append((campaign / name, name))
    for seed in R.SEEDS:
        copies.append((campaign / f"seed_{seed}" / "CHECKPOINT_EVENTS.jsonl",
                       f"seed_{seed}_CHECKPOINT_EVENTS.jsonl"))
        export = campaign / f"seed_{seed}" / "final_actor" / "ACTOR_EXPORT.json"
        if export.exists():
            copies.append((export, f"seed_{seed}_ACTOR_EXPORT.json"))
    for source, name in copies:
        shutil.copyfile(source, evidence / name)
    index = {str(p.relative_to(runs)): {"sha256": _sha(p), "bytes": p.stat().st_size}
             for p in sorted(runs.rglob("*")) if p.is_file()}
    (evidence / "PAYLOAD_INDEX.json").write_text(json.dumps(
        {"schema": "splitfusion.run5b.payload_index.v1",
         "runs_dir": str(runs.resolve().relative_to(R.ROOT)), "in_git": False,
         "files": index}, indent=2, sort_keys=True) + "\n")
    print(evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
