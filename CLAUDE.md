# abiodun/ — project state for Claude (repo-tracked; survives ~/.claude cache wipes)

Abiodun Ganiyu (IDCC). Research: an **instrumented, network- and safety-aware multi-modal cooperative-perception
system** (RGB+radar split inference, CARLA 0.10 / Town10HD_Opt → OAI 5G edge over RFsim). This file is the durable
index — the authoritative detail lives in the docs below (read them before acting on their topic).

> **Direction change (2026-08-14): this is no longer an RL project.** Learned control was falsified for the
> evaluated contract (see status below). The Month-6 deliverable is an **end-to-end system paper**. Read
> `rl_agent/FORMULATION_AND_RELATED_WORK.md` §8 before any paper, planning, or controller work — it holds the
> thesis, contributions C1-C4, banked-vs-pending status, limitations, critical path, and the open venue/OTA decision.

## Conventions (do not violate)
- Work only in `abiodun/`; never edit top-level shared scripts (copy into an `abiodun/` subfolder).
- GT convention = **actor origin** offline, and offline knob-matrix numbers are the accuracy anchor (~0.95 m
  loc floor), never a loose-matcher live number.
- **Do NOT export `PYTHONPATH` for a CARLA client** (front/back/loopback) — it shadows `abiodun/` with the
  stale `neu_collab/` copy → `UDPMessageSocket … remote_host`. Analysis/eval scripts that only `import
  carla` are fine.
- T-tracer byte-compares `T_messages.txt` vs the compiled copy → rebuild BOTH softmodems after any edit.
- Don't kill other users' CARLA/OAI; reuse a running server; check `/proc/loadavg` + `docker ps` first.
- Deployed entropy codec = **zstd** (lossless; accuracy codec-invariant).
- Be systematic; validate the pipeline before writing findings; never rescue broken data with "relative
  patterns still hold"; don't lower score gates to force a pass.

## Authoritative docs (source of truth)
- **RL design (LOCKED):** `rl_agent/AGENT_CONSTRAINTS.md` — §9 = state/action/reward synthesis; §8 = the
  density+segmentation finding; §1–§6 = the staleness/latency/FPS bounds.
- **Knob matrix (accuracy↔knob↔payload, transport-invariant):** `rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md`.
- **Density+seg study (DONE, 9/9 gates):** `rl_agent/density_knob/DENSITY_KNOB_RESULTS.md` — ROI drop
  destroys segmentation; seg-safe knob is `ae32/u4/ROI0 ≈ 90 KB`, density-invariant.
- **Channel sweep (DONE):** `channel_condition_sweep/CHANNEL_SWEEP_RESULTS.md` and `combined_surface.csv` —
  106PRB OAI channelmod AWGN surface used by the table-driven policy environment.
- **OAI MCS policy work:** `oai_mcs_policy_track2/` — SINR-driven UL MCS implemented at
  `OAI/…/gNB_scheduler_ulsch.c:2027`; fixes the clean-channel MCS-7 sparse-window artifact.
- **Uplink-only architecture + staleness redo:** `uplink_only_spatial_map_pipeline/`,
  `staleness/uplink_only_latency_budget/`.
- **Papers:** `SCAN_AI_03_13_26_2.pdf` (single-UE foundation), `V2X_for_AD.pdf` (CoDriving).

## Current status / next (2026-08-12 local / 2026-08-13 UTC)
- Density+seg study, RL design lock, measured loopback latency, and the **channel-condition sweep: DONE**
  (`channel_condition_sweep/CHANNEL_SWEEP_RESULTS.md` + `combined_surface.csv` + `plots/`). Knee: 1 MB→clear
  only, 400 KB→to 15.6 dB, 90 KB seg-safe floor→every rung. `AGENT_CONSTRAINTS §9.1` holds the measured
  channel_state + `payload_budget=capacity/fps` rule.
- The native-10-Hz advisor-rich v5 corpus is accepted. Collection completed 24/24 runs; verification
  `data_collection/experiments/policy_corpus_advisor_rich_v5/20260813_045142_full/verification/20260813_061952`
  is `PASS` on structural controller-corpus gates after excluding only impact run `pcarv5_mixed_va01` (23/24
  retained). Recall is report-only: held-out pedestrian <=12 m is 67.87% and vehicle <=25 m is 67.20%, with
  trajectory-grouped CIs. Accepted-run radar density is 19,404.5/frame; traffic and cleanup are clean.
- Freshness re-score `freshness_rescore/20260813_062203` has no QC exclusions and confirms both slow pedestrian
  and sustained >=10 m/s vehicle regimes, with 54.43% GT-seeded mapped freshness pressure.
- The authoritative reward-v5 pre-RL ladder is
  `rl_agent/policy/experiments/controller_ladder/20260813_063514`. Greedy reward is 0.19655 and MPC is 0.19834
  with identical 91.13% matched-safe rate; they disagree on only 2.54% of finite frames. See
  `data_collection/EVALUATION_CONTRACT_DECISION_V5.md`.

## RL decision: NO-GO (2026-08-14) — three independent gates, scope stated honestly
1. **Single-UE ladder:** greedy ~= MPC (above), bootstrap interval covers zero.
2. **Expanded action space** (`rl_agent/policy/experiments/expanded_action_gate/20260813_233947_pdt`):
   greedy 0.192625 vs oracle 0.195290 = **+1.383%**, below the registered +5%/+0.01 bar →
   `EXPANDED_SURROGATE_NO_GO_STOP`. **Scope:** the oracle is **one-step, on greedy-visited states, matched-support**
   (`policy/expanded_gate.py:581`) — it does **NOT** bound sequential policies. Claim only *"no useful one-step
   headroom within the static-quality, queue-free, matched-support contract."*
3. **Multi-UE contention** (`rl_agent/multiue_oai/`): measured N=2 shows **no coordination gap** — the MAC scheduler
   already sits at the capacity ceiling (6.090 vs 6.077 Mbps). Corrected N=50/100 screen: 0/216 cells survive
   (`STOP_CHEAP_NO`). DG-B / N=4 / the campaign / the ladder / RL are all **stopped**.

**Untested, therefore NOT falsified:** scene-conditioned knob selection (`policy/shield.py:49 profile_quality()`
takes no observation → the Phase-1 hypothesis was unrepresentable, not rejected); queue-coupled surrogates;
calibrated LOCAL actions; Phase-2 object-selective map sharing. Reopening RL requires a *new* pre-registered gap on
an expanded contract, not a retune of these gates.

## Current work (2026-08-14)
- **Positive result:** load-shaping deadline-feasibility frontier — 7/200 feasible cells at 400 KiB → 58/200 at
  90 KB → 89/200 at 49.4 KB (even the smallest payload leaves ~55% infeasible).
- **Binding gap: C2, the transport-conditioned cooperation gain, does not exist yet.** The 1.40 m two-view
  triangulation result is groundwork only — static egos, oracle association, **no OAI transport**.
- **Critical path:** Phase-2 recipient-specific map sharing integrated end-to-end (= multi-vehicle integration when
  scoped to one helper + one recipient). ~7-10 weeks, 9-12 with contingency. OTA is a parallel venue risk.
- **In flight (desk-only):** Task A argmax-stability/rank-reversal screen (a detection-only null is **INCONCLUSIVE**,
  not a closure — per-frame segmentation metrics are missing and seg is the most profile-sensitive term);
  Task B vulnerable-object shield guardrails (protect only *observed* pedestrians); Task C exact budgeted
  enumerator + lambda-RDO supported-hull lookup (report action agreement / reward gap / duality gap; the AoI
  heuristic is **not** a Whittle index).
- `SCENESENSE_MONTHLY_CHECKLIST.md` still predates these results and needs a reconciliation pass.

> The `~/.claude` memory cache was wiped by a retention cleanup on 2026-08-03 (harness, not us). This
> repo-tracked file exists so project state is never lost that way again. Keep it updated as work advances.
