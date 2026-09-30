#!/usr/bin/env python3
"""Read-only diagnosis of the Run-5 held-scene evaluation v1 (no rerun, no training).

The evaluator persisted, per decision, the executed action/outcome, its
expected reward and the restricted-oracle maxima - but not the 72 individual
anchor scores or the full 22-D state.  Both are reconstructed here offline:

* the actor-independent draws (retained residual, transport success) are
  regenerated from their registered per-context RNG streams;
* each anchor is scored by the evaluator's own ``score_action`` on the row's
  stored scene/backlog/MCS plus those draws;
* the 22-D state is rebuilt from the stored scene descriptors, MCS, backlog,
  SNR and the previous row's action/outcome with the training scaling.

The reconstruction is admitted only if it reproduces, for every analysed row,
the stored executed reward, expected reward, oracle maxima and refusal count,
and if the frozen actors reproduce every recorded action from the rebuilt
state.  Nothing in the evaluation output directory is written or changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract as AC
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import quantize_q_e4
from rl_agent.ue_production_transport_model_v2 import artifact_v2 as A2
from rl_agent.ue_production_transport_model_v2 import collector_v1 as CV

from . import run5_bundle as B
from . import run5_channel as J
from . import run5_collector as RC
from . import run5_evaluator as EV
from . import run5_held_scene_partition as H
from . import run5_models as RM
from . import run5_preregistration as PR
from . import run5_snr_v2 as SNR

PACKAGE = Path(__file__).resolve().parent
OUTPUT = PACKAGE / "evaluation_runs" / "heldscene_eval_v1"
OUTPUT_MANIFEST_SHA256 = "cba1a71bbbeaedf14a92095e1e397f6fe44e4cda1a3975d6f840338c47fd7825"
FINAL = ("run5_seed17_u10000", "run5_seed29_u10000", "run5_seed43_u10000")
RUN4, FIXED = "run4_seed43_u10000", "fixed_mode11_q3000"
FIXED_ACTION = (11, 3000)
TRANSITION_DB = 3.0   # diagnostic stratification threshold for |dSNR| between decisions


class DiagnosisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosisError(message)


def verify_outputs() -> dict:
    manifest_path = OUTPUT / "OUTPUT_MANIFEST.json"
    require(B.sha256_bytes(manifest_path.read_bytes()) == OUTPUT_MANIFEST_SHA256,
            "output manifest differs from the sealed hash")
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        require(B.sha256_bytes((OUTPUT / entry["path"]).read_bytes()) == entry["sha256"],
                f"{entry['path']} changed")
    return manifest


def load_rows() -> dict[tuple, list[dict]]:
    by = defaultdict(list)
    for path in sorted((OUTPUT / "rows").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            by[(row["profile"], row["validation_seed"], row["policy_id"])].append(row)
    for rows in by.values():
        rows.sort(key=lambda r: r["k"])
    return by


# ---------------------------------------------------------------------------
# Exact reconstruction
# ---------------------------------------------------------------------------


class _Context:
    __slots__ = ("reward_scene_key", "backlog_bytes", "prior_ul_mcs",
                 "retained_residual_ns", "success_draw")


class _Shim:
    """Just the attributes ``EV.score_action`` reads."""

    def __init__(self, catalog, model, reserve):
        self.catalog, self.model, self.actor_reserve_ns = catalog, model, reserve
        self.context = _Context()


def exogenous_draws(shared, profile: str, seed: int, n: int) -> list[tuple[int, float]]:
    label = f"{profile}:{seed}"
    residual = random.Random(J.derive_seed(seed, f"eval-residual:{label}"))
    transport = random.Random(J.derive_seed(seed, f"eval-transport:{label}"))
    return [(residual.choice(shared["retained_residuals"]), transport.random()) for _ in range(n)]


def detailed_score(shim: _Shim, mode: int, q: int) -> dict[str, Any]:
    """``EV.score_action`` plus its internal terms, from the identical formulas."""
    c = shim.context
    draw = shim.catalog.draw(c.reward_scene_key, mode_id=mode, q_e4=q)
    try:
        prediction = shim.model.predict(pre_enqueue_backlog_bytes=float(c.backlog_bytes),
                                        wire_bytes=int(draw.wire_bytes),
                                        prior_ul_mcs=int(c.prior_ul_mcs))
    except A2.OutOfSupport:
        return {"in_support": False, "q_perc": float(draw.q_perc), "wire": draw.wire_bytes}
    composed = (c.retained_residual_ns + int(round(CV.SEND_SPAN_NS_PER_BYTE * draw.wire_bytes))
                + int(round(prediction.conditional_latency_ms * 1e6)) + shim.actor_reserve_ns)
    within = composed <= EV.DEADLINE_NS
    latency_ms = composed / 1_000_000.0
    success_reward = (float(draw.q_perc) - R4.REWARD_LATENCY_WEIGHT * (latency_ms / R4.REWARD_DEADLINE_MS)
                      if within else R4.REGISTERED_FAILURE_REWARD)
    p = prediction.on_time_probability
    return {"in_support": True, "q_perc": float(draw.q_perc), "wire": draw.wire_bytes,
            "p": p, "composed_ms": latency_ms, "within_deadline": within,
            "residual_ms": c.retained_residual_ns / 1e6,
            "expected": p * success_reward + (1.0 - p) * R4.REGISTERED_FAILURE_REWARD,
            "realized": success_reward if c.success_draw < p else R4.REGISTERED_FAILURE_REWARD}


class Reconstruction:
    def __init__(self, evidence_root: Path):
        self.shared = RC.build_shared_sources(EV.ARTIFACT, evidence_root)
        partition = H.load_sealed_partition(evidence_root)
        self.held = H.HeldSceneCatalogV1(evidence_root, partition["eligible_keys"])
        self.anchors = EV.anchors()
        values = [self.shared["catalog"].scene_descriptors(k)[0] for k in self.shared["catalog"].keys]
        mean = sum(values) / len(values)
        self.camera_center = float(mean)
        self.camera_scale = float(max(1e-6, math.sqrt(sum((v - mean) ** 2 for v in values)
                                                      / len(values))))
        self.backlog_scale = RC.CV.BACKLOG_LOG1P_SCALE

    def shim(self) -> _Shim:
        return _Shim(self.held, self.shared["model"], self.shared["actor_reserve_ns"])

    def state(self, row: dict, previous: dict | None) -> tuple[float, ...]:
        camera, radar = self.held.scene_descriptors(row["scene"])
        values = [(camera - self.camera_center) / self.camera_scale, float(radar),
                  (int(row["mcs"]) - 0) / float(28),
                  math.log1p(int(row["backlog_bytes"])) / float(self.backlog_scale)]
        one_hot = [0.0] * 12
        if previous is None:
            values += one_hot + [0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            one_hot[previous["mode_id"]] = 1.0
            success = previous["terminal"] == "SUCCESS"
            values += one_hot + [previous["q_e4"] / float(AC.Q_E4_MAX),
                                 float(previous["delivered_q_perc"]) if success else 0.0,
                                 float(previous["latency_ms"]) / R4.REWARD_DEADLINE_MS
                                 if success else 0.0, 1.0, 1.0 if success else 0.0]
        values.append(SNR.scale_snr_db(row["snr_db"]))
        return tuple(float(v) for v in values)

    def score_trajectory(self, rows: list[dict], profile: str, seed: int) -> list[dict]:
        draws = exogenous_draws(self.shared, profile, seed, len(rows))
        shim, out = self.shim(), []
        for row, (residual, success) in zip(rows, draws):
            c = shim.context
            c.reward_scene_key, c.backlog_bytes, c.prior_ul_mcs = row["scene"], row["backlog_bytes"], row["mcs"]
            c.retained_residual_ns, c.success_draw = residual, success
            anchors = {a: detailed_score(shim, *a) for a in self.anchors}
            own = detailed_score(shim, row["mode_id"], row["q_e4"])
            official = EV.score_action(shim, row["mode_id"], row["q_e4"])
            supported = [s for s in anchors.values() if s["in_support"]]
            require(own["in_support"] and own["expected"] == row["expected_reward"]
                    and own["realized"] == row["reward"] == official.realized
                    and official.expected == own["expected"], "executed-action reconstruction differs")
            require(max(s["expected"] for s in supported) == row["oracle_expected_reward"]
                    and max(s["realized"] for s in supported) == row["oracle_realized_reward"]
                    and len(anchors) - len(supported) == row["oracle_refused_anchors"],
                    "oracle reconstruction differs")
            out.append({"row": row, "own": own, "anchors": anchors,
                        "fixed": anchors[FIXED_ACTION] if FIXED_ACTION in anchors
                        else detailed_score(shim, *FIXED_ACTION)})
        return out


# ---------------------------------------------------------------------------
# Phase 2: exact reward decomposition
# ---------------------------------------------------------------------------


def decompose(rows: list[dict]) -> dict[str, Any]:
    quality = latency = timeout = 0.0
    exact = 0
    for r in rows:
        if r["terminal"] == "SUCCESS":
            q = float(r["delivered_q_perc"])
            lat = -R4.REWARD_LATENCY_WEIGHT * (float(r["latency_ms"]) / R4.REWARD_DEADLINE_MS)
            exact += int(q + lat == r["reward"])
            quality += q
            latency += lat
        else:
            exact += int(r["reward"] == R4.REGISTERED_FAILURE_REWARD)
            timeout += r["reward"]
    n = len(rows)
    total = math.fsum(r["reward"] for r in rows) / n
    parts = {"delivered_quality": quality / n, "latency_penalty": latency / n,
             "timeout_or_failure": timeout / n, "action_switch_penalty": 0.0, "other_terms": 0.0}
    return {**parts, "reward_mean": total, "rows": n, "rows_reproduced_exactly": exact,
            "sum_of_components": math.fsum(parts.values()),
            "abs_residual": abs(math.fsum(parts.values()) - total)}


# ---------------------------------------------------------------------------
# Phase 3: mode vs q
# ---------------------------------------------------------------------------


def mode_q_regret(scored: list[dict]) -> dict[str, Any]:
    within, mode_part, total, fixed_gap, signed_q, comparable = [], [], [], [], [], []
    best_q_lower = best_q_higher = best_q_equal = 0
    for s in scored:
        row, own = s["row"], s["own"]
        mode_anchors = {q: v for (m, q), v in s["anchors"].items() if m == row["mode_id"] and v["in_support"]}
        best_mode_q, best_mode = max(mode_anchors.items(), key=lambda kv: kv[1]["expected"])
        glob = max(v["expected"] for v in s["anchors"].values() if v["in_support"])
        within.append(best_mode["expected"] - own["expected"])
        mode_part.append(glob - best_mode["expected"])
        total.append(glob - own["expected"])
        fixed_gap.append(s["fixed"]["expected"] - own["expected"])
        signed_q.append(row["q_e4"] - best_mode_q)
        best_q_lower += int(best_mode_q < row["q_e4"])
        best_q_higher += int(best_mode_q > row["q_e4"])
        best_q_equal += int(best_mode_q == row["q_e4"])
        nearest = min(sorted({q for _, q in s["anchors"]}), key=lambda q: abs(q - row["q_e4"]))
        at_q = [v["expected"] for (m, q), v in s["anchors"].items() if q == nearest and v["in_support"]]
        own_at_q = s["anchors"].get((row["mode_id"], nearest))
        if at_q and own_at_q and own_at_q["in_support"]:
            comparable.append(max(at_q) - own_at_q["expected"])
    n = len(scored)
    mean = statistics.fmean
    return {"decisions": n,
            "label": "ONE_STEP_RESTRICTED_ANCHOR_DIAGNOSTIC (not a continuous or sequential oracle)",
            "executed_expected_reward_mean": mean(s["own"]["expected"] for s in scored),
            "within_mode_q_opportunity_mean": mean(within),
            "discrete_mode_opportunity_mean": mean(mode_part),
            "total_restricted_regret_mean": mean(total),
            "additivity_residual": abs(mean(within) + mean(mode_part) - mean(total)),
            "fixed_mode11_q3000_minus_executed_mean": mean(fixed_gap),
            "best_mode_at_nearest_registered_q_opportunity_mean": mean(comparable) if comparable else None,
            "signed_q_minus_best_within_mode_anchor_mean": mean(signed_q),
            "signed_q_minus_best_within_mode_anchor_median": statistics.median(signed_q),
            "best_within_mode_q_is_lower_fraction": best_q_lower / n,
            "best_within_mode_q_is_higher_fraction": best_q_higher / n,
            "best_within_mode_q_equals_executed_fraction": best_q_equal / n,
            "executed_q_mean": mean(s["row"]["q_e4"] for s in scored) / 10000.0}


def why_fixed(scored_by_policy: dict[str, list[dict]]) -> dict[str, Any]:
    out = {}
    for pid, scored in scored_by_policy.items():
        rows = [s["row"] for s in scored]
        timeouts = [s for s in scored if s["row"]["terminal"] != "SUCCESS"]
        out[pid] = {
            "mean_wire_bytes": statistics.fmean(s["own"]["wire"] for s in scored),
            "mean_on_time_probability": statistics.fmean(s["own"]["p"] for s in scored),
            "mean_composed_latency_ms": statistics.fmean(s["own"]["composed_ms"] for s in scored),
            "mean_residual_ms": statistics.fmean(s["own"]["residual_ms"] for s in scored),
            "timeouts": len(timeouts),
            "timeouts_from_transport_draw": sum(1 for s in timeouts if s["own"]["within_deadline"]),
            "timeouts_from_composed_latency_over_170ms": sum(1 for s in timeouts
                                                             if not s["own"]["within_deadline"]),
            "executed_q_perc_mean": statistics.fmean(r["executed_q_perc"] for r in rows)}
    return out


def mode_quality_table(recon: Reconstruction) -> list[dict[str, Any]]:
    """Held-scene Q_perc and wire bytes of every registered anchor (scene means)."""
    table = []
    for mode, q in recon.anchors:
        draws = [recon.held.draw(k, mode_id=mode, q_e4=q) for k in recon.held.keys]
        table.append({"mode_id": mode, "q_e4": q,
                      "q_perc_mean": statistics.fmean(d.q_perc for d in draws),
                      "wire_bytes_mean": statistics.fmean(d.wire_bytes for d in draws)})
    return table


# ---------------------------------------------------------------------------
# Phase 4: actor / critic / entropy
# ---------------------------------------------------------------------------


def load_final(seed: int):
    bundle = B.verify_bundle(EV.CAMPAIGN / f"seed_{seed}" / "checkpoints" / "checkpoint_010000")
    actor, critics = RM.build_run5_models(actor_seed=0, critic_seed=0)
    actor.load_state_dict(B.torch_from_bytes(bundle.payload("actor_state_dict.pt")), strict=True)
    state = B.torch_from_bytes(bundle.payload("training_state.pt"))
    critics.load_state_dict({**state["online_critics"], **state["target_critics"]}, strict=True)
    actor.eval()
    critics.eval()
    return actor, critics


def per_mode_q(actor, states: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        heads = actor(states)
        lower, upper = actor.active_q_e4_bounds()
        z = 0.5 * (torch.tanh(heads.mean) + 1.0)
        q = (lower.float() + (upper - lower).float() * z) / 10000.0
        return quantize_q_e4(q), torch.softmax(heads.logits, dim=-1)


def critic_q(critics, states: torch.Tensor, modes: torch.Tensor, q_e4: torch.Tensor) -> torch.Tensor:
    one_hot = torch.nn.functional.one_hot(modes, 12).to(torch.float32)
    with torch.no_grad():
        q1, q2 = critics.q_values(states, one_hot, q_e4.to(torch.float32) / float(AC.Q_E4_MAX))
    return torch.minimum(q1, q2)


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(x):                        # average ranks for ties
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        start = 0
        while start < len(order):
            end = start
            while end + 1 < len(order) and x[order[end + 1]] == x[order[start]]:
                end += 1
            for position in range(start, end + 1):
                r[order[position]] = (start + end) / 2.0
            start = end + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def actor_critic(recon: Reconstruction, seed: int, trajectories: dict, scored: dict) -> dict[str, Any]:
    actor, critics = load_final(seed)
    pid = f"run5_seed{seed}_u10000"
    states, rows, scores = [], [], []
    for (profile, vseed, policy), trajectory in trajectories.items():
        if policy != pid:
            continue
        previous = None
        for row, s in zip(trajectory, scored[(profile, vseed, policy)]):
            states.append(recon.state(row, previous))
            rows.append(row)
            scores.append(s)
            previous = row
    x = torch.tensor(states, dtype=torch.float32)
    # Batch-1 inference, exactly as the evaluator acted (batched matmul can move
    # a float by one ulp and flip a quantized q_e4 by one unit).
    reproduced, q_rows, p_rows = 0, [], []
    with torch.no_grad():
        for i, r in enumerate(rows):
            execution = actor.deterministic_execution(x[i:i + 1])
            reproduced += int(int(execution.mode_index[0]) == r["mode_id"]
                              and int(execution.q_e4[0]) == r["q_e4"])
            q_one, p_one = per_mode_q(actor, x[i:i + 1])
            q_rows.append(q_one[0])
            p_rows.append(p_one[0])
    require(reproduced == len(rows), f"state reconstruction reproduces {reproduced}/{len(rows)} actions")
    q_modes, probs = torch.stack(q_rows), torch.stack(p_rows)
    n = len(rows)
    critic_all = torch.stack([critic_q(critics, x, torch.full((n,), m), q_modes[:, m])
                              for m in range(12)], dim=1)
    critic_mode = critic_all.argmax(dim=1)
    actor_mode = torch.tensor([r["mode_id"] for r in rows])
    follows = (critic_mode == actor_mode).float().mean().item()
    one_step_actor, one_step_critic = [], []
    shim = recon.shim()
    # one-step value of the critic-preferred mode at the actor's own q for that mode
    draws = {}
    for (profile, vseed, policy) in trajectories:
        if policy == pid:
            draws[(profile, vseed)] = exogenous_draws(recon.shared, profile, vseed, 255)
    index = 0
    for (profile, vseed, policy), trajectory in trajectories.items():
        if policy != pid:
            continue
        for k, row in enumerate(trajectory):
            c = shim.context
            c.reward_scene_key, c.backlog_bytes, c.prior_ul_mcs = row["scene"], row["backlog_bytes"], row["mcs"]
            c.retained_residual_ns, c.success_draw = draws[(profile, vseed)][k]
            m = int(critic_mode[index])
            alt = detailed_score(shim, m, int(q_modes[index, m]))
            one_step_actor.append(scores[index]["own"]["expected"])
            one_step_critic.append(alt["expected"] if alt["in_support"] else None)
            index += 1
    valid = [(a, b) for a, b in zip(one_step_actor, one_step_critic) if b is not None]
    # critic vs one-step ranking over admissible anchors (ranking diagnostic only)
    rhos, top1 = [], 0
    for i, s in enumerate(scores):
        supported = [(a, v) for a, v in s["anchors"].items() if v["in_support"]]
        modes = torch.tensor([a[0] for a, _ in supported])
        qs = torch.tensor([a[1] for a, _ in supported])
        values = critic_q(critics, x[i].repeat(len(supported), 1), modes, qs).tolist()
        one_step = [v["expected"] for _, v in supported]
        rhos.append(spearman(values, one_step))
        top1 += int(max(range(len(values)), key=values.__getitem__)
                    == max(range(len(one_step)), key=one_step.__getitem__))
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
    gap = (critic_all.max(dim=1).values - critic_all.gather(1, actor_mode[:, None]).squeeze(1))
    alpha_d, alpha_c = PR.CONFIG.alpha_d, PR.CONFIG.alpha_c
    return {"decisions": n, "state_reconstruction_actions_reproduced": reproduced,
            "actor_follows_critic_mode_rate": follows,
            "critic_q_gap_best_minus_actor_mode_mean": gap.mean().item(),
            "critic_q_gap_nonzero_mean": gap[gap > 0].mean().item() if bool((gap > 0).any()) else 0.0,
            "one_step_expected_actor_mode_mean": statistics.fmean(a for a, _ in valid),
            "one_step_expected_critic_mode_mean": statistics.fmean(b for _, b in valid),
            "critic_mode_evaluable": len(valid),
            "ranking_label": "RANKING_DIAGNOSTIC_ONLY (critic soft return vs one-step reward; not Bellman calibration)",
            "critic_vs_one_step_spearman_mean": statistics.fmean(r for r in rhos if not math.isnan(r)),
            "critic_top1_equals_one_step_top1_rate": top1 / n,
            "discrete_entropy_mean_nats": entropy.mean().item(),
            "alpha_d": alpha_d, "alpha_c": alpha_c,
            "alpha_d_times_entropy_mean": alpha_d * entropy.mean().item(),
            "actor_mode_probability_mean": probs.gather(1, actor_mode[:, None]).mean().item(),
            "max_mode_probability_mean": probs.max(dim=1).values.mean().item()}


# ---------------------------------------------------------------------------
# Phase 5: SNR
# ---------------------------------------------------------------------------


def snr_analysis(trajectories: dict) -> dict[str, Any]:
    out = {}
    for seed in (17, 29, 43):
        true_id, shuf_id = f"run5_seed{seed}_u10000", f"run5_shuffled_snr_seed{seed}_u10000"
        strata = defaultdict(lambda: {"n": 0, "mode_dis": 0, "abs_dq": 0.0, "dr": 0.0,
                                      "probe_dis": 0})
        mcs, snr = [], []
        for (profile, vseed, policy), rows in trajectories.items():
            if policy != true_id:
                continue
            shuf = trajectories[(profile, vseed, shuf_id)]
            for k, (a, b) in enumerate(zip(rows, shuf)):
                region = ("TRANSITION" if k and abs(a["snr_db"] - rows[k - 1]["snr_db"]) >= TRANSITION_DB
                          else "STEADY")
                for key in ("ALL", profile, region):
                    s = strata[key]
                    s["n"] += 1
                    s["mode_dis"] += int(a["mode_id"] != b["mode_id"])
                    s["abs_dq"] += abs(a["q_e4"] - b["q_e4"])
                    s["dr"] += b["reward"] - a["reward"]
                    s["probe_dis"] += int(a["shuffled_probe"]["mode"] != a["mode_id"])
                mcs.append(a["mcs"])
                snr.append(a["snr_db"])
        mm, ms = statistics.fmean(mcs), statistics.fmean(snr)
        cov = sum((x - mm) * (y - ms) for x, y in zip(mcs, snr))
        corr = cov / math.sqrt(sum((x - mm) ** 2 for x in mcs) * sum((y - ms) ** 2 for y in snr))
        out[str(seed)] = {"strata": {k: {"decisions": v["n"],
                                          "closed_loop_mode_disagreement": v["mode_dis"] / v["n"],
                                          "closed_loop_abs_q_e4_change_mean": v["abs_dq"] / v["n"],
                                          "closed_loop_reward_change_mean": v["dr"] / v["n"],
                                          "one_step_probe_mode_disagreement": v["probe_dis"] / v["n"]}
                                      for k, v in sorted(strata.items())},
                          "corr_snr_mcs": corr, "r2_snr_on_mcs_linear": corr ** 2}
    return {"transition_threshold_db": TRANSITION_DB, "per_seed": out,
            "scope": "modeled held-scene evaluation only; no intrinsic claim about SNR"}


# ---------------------------------------------------------------------------


def run(evidence_root: Path) -> dict[str, Any]:
    torch.set_num_threads(4)
    verify_outputs()
    trajectories = load_rows()
    recon = Reconstruction(evidence_root)
    analysed = FINAL + (RUN4, FIXED)
    scored = {key: recon.score_trajectory(rows, key[0], key[1])
              for key, rows in trajectories.items() if key[2] in analysed}
    by_policy = {pid: [s for key, v in scored.items() if key[2] == pid for s in v] for pid in analysed}
    phase2 = {pid: {"overall": decompose([s["row"] for s in by_policy[pid]]),
                    **{profile: decompose([s["row"] for key, v in scored.items()
                                           if key[2] == pid and key[0] == profile for s in v])
                       for profile in EV.PROFILES}} for pid in analysed}
    phase3 = {pid: mode_q_regret(by_policy[pid]) for pid in analysed}
    phase4 = {str(seed): actor_critic(recon, seed, trajectories, scored) for seed in (17, 29, 43)}
    run4_reproduced = 0
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import frozen_actor_v2 as FA
    frozen = FA.load_registered_actor(evidence_root)
    total_run4 = 0
    for (profile, vseed, policy), rows in trajectories.items():
        if policy != RUN4:
            continue
        previous = None
        for row in rows:
            decision = frozen.act_on_vector(recon.state(row, previous)[:21])
            run4_reproduced += int((decision.mode_id, decision.q_e4) == (row["mode_id"], row["q_e4"]))
            total_run4 += 1
            previous = row
    require(run4_reproduced == total_run4, "Run-4 state reconstruction differs")
    return {"schema": "splitfusion.run5.heldscene_diagnosis.v1",
            "read_only": True, "evaluation_output_manifest_sha256": OUTPUT_MANIFEST_SHA256,
            "reconstruction": {
                "note": ("per-anchor scores and 22-D states were not persisted by the evaluator; "
                         "they were rebuilt offline and admitted only after exact reproduction"),
                "rows_scored": sum(len(v) for v in scored.values()),
                "anchor_scores_rebuilt": sum(len(v) for v in scored.values()) * len(recon.anchors),
                "executed_reward_oracle_and_refusals_reproduced_exactly": True,
                "run4_actions_reproduced": f"{run4_reproduced}/{total_run4}"},
            "phase2_reward_decomposition": phase2,
            "phase2_why_fixed": why_fixed(by_policy),
            "phase2_anchor_quality_payload_held_scenes": mode_quality_table(recon),
            "phase3_mode_vs_q": phase3,
            "phase4_actor_critic_entropy": phase4,
            "phase5_snr": snr_analysis(trajectories)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PACKAGE / "RUN5_HELDSCENE_DIAGNOSIS.json")
    args = parser.parse_args(argv)
    require(not args.output.exists(), "diagnosis output is create-only")
    document = run(args.evidence_root.resolve())
    with args.output.open("x") as handle:
        json.dump(document, handle, indent=1, sort_keys=True)
        handle.write("\n")
    verify_outputs()
    print(json.dumps({"written": str(args.output), "sha256": B.sha256_bytes(args.output.read_bytes())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
