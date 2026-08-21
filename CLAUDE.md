# abiodun/ — project state for Claude (repo-tracked; survives ~/.claude cache wipes)

Abiodun Ganiyu (IDCC). Research: an **instrumented, network- and safety-aware multi-modal cooperative-perception
system** (RGB+radar split inference, CARLA 0.10 / Town10HD_Opt → OAI 5G edge over RFsim). This file is the durable
index — the authoritative detail lives in the docs below (read them before acting on their topic).

> **Current execution reset (2026-08-20): split-profile characterization first.**
> The sole current authority is `rl_agent/UE_AGENT_EXECUTION_CHECKLIST_V2.md`.
> The active experiment starts from all 72 measured split-profile anchors and
> characterizes them on one repeatable route under four saved time-varying OAI
> traces.
> SKIP, LOCAL, continuous-q promotion, and policy training follow only after
> the split table and simple baselines. Helper-recipient work is parked in
> `phase2_map_sharing/PARKED_STATUS_2026-08-19.md`. Learned control remains
> conditional on a residual gap after simple baselines. External reviews are
> advisory, not approval gates; scope changes require an explicit
> Abiodun–Codex decision.

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
- **Current execution authority:** `rl_agent/UE_AGENT_EXECUTION_CHECKLIST_V2.md`.
- **Current split experiment plan:** `rl_agent/UE_SPLIT_ONLY_EXPERIMENT_PLAN_V2.md`.
- **Current UE diagram:** `rl_agent/state_diagram.md`.
- **Historical measured RL design:** `rl_agent/AGENT_CONSTRAINTS.md` — §9 =
  frozen Phase-1 state/action/reward synthesis; §8 = the density+segmentation
  finding; §1–§6 = the staleness/latency/FPS bounds.
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

## Current status / next (2026-08-19)
- Density+seg study, RL design lock, measured loopback latency, and the **channel-condition sweep: DONE**
  (`channel_condition_sweep/CHANNEL_SWEEP_RESULTS.md` + `combined_surface.csv` + `plots/`). Knee: 1 MB→clear
  only, 400 KB→to 15.6 dB, 90 KB seg-safe floor→every rung. `AGENT_CONSTRAINTS §9.1` holds the measured
  channel_state + `payload_budget=capacity/fps` rule.
- The native-10-Hz advisor-rich v5 corpus is accepted for perception QA, workload characterization, and legacy
  matched-support replay. It is **not** a paired helper-recipient causal-control corpus and cannot measure C2
  warning lead. Collection completed 24/24 runs; verification
  `data_collection/experiments/policy_corpus_advisor_rich_v5/20260813_045142_full/verification/20260813_061952`
  is `PASS` on structural controller-corpus gates after excluding only impact run `pcarv5_mixed_va01` (23/24
  retained). Recall is report-only: held-out pedestrian <=12 m is 67.87% and vehicle <=25 m is 67.20%, with
  trajectory-grouped CIs. Accepted-run radar density is 19,404.5/frame; traffic and cleanup are clean.
- Freshness re-score `freshness_rescore/20260813_062203` has no QC exclusions and confirms both slow pedestrian
  and sustained >=10 m/s vehicle regimes, with 54.43% GT-seeded mapped freshness pressure.
- The completed reward-v5 Phase-1 ladder artifact is
  `rl_agent/policy/experiments/controller_ladder/20260813_063514`. Greedy reward is 0.19655 and MPC is 0.19834
  with identical 91.13% matched-safe rate; they disagree on only 2.54% of finite frames. A 2026-08-14 audit found
  same-frame post-tail observation leakage and GT-assisted track construction. These numbers are therefore a
  **noncausal matched-support upper-bound study**, not deployable controller evidence. See
  `rl_agent/RL_JOURNEY_REPORT.md` and `data_collection/EVALUATION_CONTRACT_DECISION_V5.md`.

## RL decision (2026-08-14) — static NO-GO retained; dynamic decision reopened
1. **Single-UE ladder:** greedy ~= MPC (above), bootstrap interval covers zero, but the observation is noncausal.
   This does **not** close the deployable dynamic-controller question.
2. **Expanded action space** (`rl_agent/policy/experiments/expanded_action_gate/20260813_233947_pdt`):
   greedy 0.192625 vs oracle 0.195290 = **+1.383%**, below the registered +5%/+0.01 bar →
   `EXPANDED_SURROGATE_NO_GO_STOP`. **Scope:** the oracle is **one-step, on greedy-visited states, matched-support**
   (`policy/expanded_gate.py:581`) and uses the same noncausal state — it does **NOT** bound causal sequential
   policies. Claim only *"no useful one-step headroom within the noncausal, static-quality, queue-free,
   matched-support contract."*
3. **Multi-UE contention** (`rl_agent/multiue_oai/`): measured N=2 shows **no coordination gap** — the MAC scheduler
   already sits at the capacity ceiling (6.090 vs 6.077 Mbps). Corrected N=50/100 screen: 0/216 cells survive
   (`STOP_CHEAP_NO`). DG-B / N=4 / the campaign / the ladder / RL are all **stopped**.

**Still valid:** the static measured-table profile-selection result and Task C's full-36-profile scalar analysis.
**Untested, therefore NOT falsified:** a causal dynamic controller; scene-conditioned knob selection (`policy/shield.py:49 profile_quality()`
takes no observation → the Phase-1 hypothesis was unrepresentable, not rejected); queue-coupled surrogates;
calibrated LOCAL actions; Phase-2 object-selective map sharing. Reopening RL requires a *new* pre-registered gap on
an expanded contract, not a retune of these gates.

## Current work (scope-reset 2026-08-20)
- **Binding path:** follow
  `rl_agent/UE_AGENT_EXECUTION_CHECKLIST_V2.md`. Qualify the repeatable route,
  calibrate the 100-ms OAI SNR-trace actuator, validate authoritative map
  feedback, then run the 4x4 pilot before separately authorizing the 72x4
  split-profile characterization. SKIP/LOCAL and policy training remain later.
- **Phase 2 parked:** recipient-specific map sharing, warning, factor
  realization, exact-16, and paired CARLA/OAI work are preserved but inactive.
  They resume only under the conditions in
  `phase2_map_sharing/PARKED_STATUS_2026-08-19.md`.
- **Positive result:** load-shaping deadline-feasibility frontier — 7/200 feasible cells at 400 KiB → 58/200 at
  90 KB → 89/200 at 49.4 KB (even the smallest payload leaves ~55% infeasible).
- **Banked later-stage gap: C2, the transport-conditioned cooperation gain, does not exist yet.** The 1.40 m two-view
  triangulation result is groundwork only — static egos, oracle association, **no OAI transport**.
- **Superseded ordering:** the earlier 7-10 week helper-recipient critical path
  remains a paper-stage estimate, not the current implementation sequence.
- **Task A complete:** the exact 36-profile/1,683-frame screen includes the already-available per-frame
  segmentation metrics and finds no practical class/range reversal under the registered gate. True occlusion and
  cyclists remain outside scope. Artifact: `rl_agent/contextual_knob/experiments/20260814_214749`.
- **Task B complete:** hard observed-vulnerable no-skip + low-confidence ROI0 clamp, with an explicit C1-conflict
  flag. Paired replay improves matched-safe rate +1.63 pp at a finite-reward cost of -0.0477 and +1.10 Mbps;
  detector misses are not protected. The rule/contract is valid, but those empirical deltas inherit the noncausal
  replay caveat. Artifact: `rl_agent/policy/experiments/vulnerable_guardrail/20260814_215337`.
- **Task C complete:** full-36 lambda-RDO agrees with exact enumeration at 80.56% of payload breakpoints (max
  utility gap 0.011686; max duality gap 0.017359). The retained-catalog runtime ladder agrees 100% with zero reward
  gap only inside the noncausal matched-support replay. The AoI baseline is explicitly index-inspired, not
  Whittle. Artifact:
  `rl_agent/policy/experiments/task_c/20260814_220006`.
- `SCENESENSE_MONTHLY_CHECKLIST.md` is reconciled through 2026-08-17. **Phase-2 contract plumbing exists:**
  `phase2_map_sharing/` passes synthetic contract validation for recipient isolation, causal hazard-only selection,
  association, warning provenance, exact byte accounting, and production-header chunk reassembly. This is plumbing,
  not C2 evidence. Its adapter also passes the existing two-stream recordings (26 fresh accepted, 11 stale rejected),
  which lack synchronized hazard truth. The `phase2_paired_causal_v1` spec and v2 collector/replay/verifier path are
  now integrated offline: strict schema/unknown-field rejection, object+recipient uncertainty propagation,
  pre-capture causal audit, physically separate truth, source-local tracking, quota-bounded raw/logit retention,
  isolated three-arm replay, and the nine fail-fast gates. The reviewed legal-route two-trajectory pilot is complete
  at `data_collection/experiments/phase2_paired_causal_v1/20260817_181354_pilot`; authoritative
  `evaluation_v4` / `verification_v4` pass all nine structural/computability gates. The evaluation-only
  `hazard_adjudication_v2` also passes: positive hazard truth uses the matched benign no-yield recipient trajectory,
  while realized stopping remains explicitly non-policy-attributable. The v1 adjudication is superseded due to the
  intervention paradox. The deterministic `phase2_suite_ab_v1` candidate now fixes Suite A=designed and Suite
  B=naturalistic across 210 independent groups / 330 world trajectories with hashed 20/20/60 assignments,
  conditional power and tiered retention. The signalized-corner and parked-van-midblock pedestrian geometries are
  visually accepted and route-hash frozen. Full collection, OAI evaluation, and RL remain unauthorized until three
  pending geometries and two paired routes are visually accepted and the 15-trajectory calibration audit plus
  registered simulation-power gate pass. The renderer gate is now resolved operationally: every primary Phase-2
  design row locks explicit CARLA `Epic` (`-quality-level=Epic`). Low survives only as the already-captured labelled
  stress condition; no future Low collection is authorized. The frozen <=12 m dense weighted comparison was
  inconclusive because it had zero near-pedestrian support, so do not claim Epic statistically dominated every
  class or retroactively relabel the unrecorded M-prime training renderer.

> The `~/.claude` memory cache was wiped by a retention cleanup on 2026-08-03 (harness, not us). This
> repo-tracked file exists so project state is never lost that way again. Keep it updated as work advances.
