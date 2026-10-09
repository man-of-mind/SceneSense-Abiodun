# Calibration gaps: corrected training-environment candidate v1

Every gap below blocks a training launch. Some are enforced at run time by `UncalibratedTermError`. Others are listed by `CalibrationSetV1.launch_blockers()`, and an approval must waive each of those explicitly. No gap was filled by assumption.

## Run-time blockers

### 1. Action space (`action_stratum`; blockers `CONTINUOUS_Q_EXACT_POINTS_ONLY`, `MODES_UNMEASURED`)

**What is uncovered:**
- The actors choose 12 modes × continuous q.
- Chains exist only at measured (mode, q_e4) points.
- B's screen adds 7 fixed actions. The live100 binding adds 45 actor-chosen points with one chain each.

**What is missing:**
- The q-dependence of codec (C), edge (F) and worker costs within a mode.
- A validated interpolation rule.

Seven fixed actions are not coverage of the space.

**Options:**
- Measure at least 2–3 q per mode and validate a within-mode rule.
- Or decide jointly to restrict the actor to measured points. That changes the action space and needs its own approval.

### 2. Channel profiles (`action_stratum` / `channel_profile`; blocker `PROFILES_UNMEASURED`)

**Measured:** FAVORABLE_STABLE only.

**Unmeasured:** MID_VARIABLE, ADVERSE_STABLE and FADE_RECOVERY have no chains. Favourable results are never transferred to them.

**Run-4:** the unlabeled Run-4 MCS Markov channel (MCS 8–28) is admitted only where its MCS lies inside the stratum's measured MCS range.

### 3. Uplink transfer (`E_uplink`; blocker `UPLINK_TRANSFER_UNVALIDATED`)

**What is used:** the pinned T_v2b model moves only E, and only inside a scope bound to its validation evidence (`CALCULATIONS.json`).

**What the evidence supports:**
- Median error +0.9 ms (n=497) and +2.2 ms (n=48).
- Per-frame rank agreement ρ = 0.22 and 0.32 only.
- FAVORABLE only. The declared MCS scope is 24–28, narrower than the observed 9–28, because most frames are at MCS 26.

**Known inconsistency:** validation used payload bytes, while the environment uses UDP-application bytes. The difference is about 0.1%.

**Needed:** per-profile uplink validation, ideally from the screen's own E across 7 payloads.

### 4. Scene conditioning (`scene_conditioning`)

**How it works:** chains are drawn by k-nearest neighbour on raw camera SI and radar P40, within the stratum's measured support. Margin is 0 in the proposed binding.

**Mismatch:**
- Training scenes come from one FIT episode (`canonical_v3_05_val_30_30`).
- Chains are Route-B live frames.
- GT preparation (G) and edge cost may depend on actor count and occlusion, which SI/P40 do not capture.

**Pending:** the fraction of FIT scenes inside each stratum's support is reported only after B delivers.

### 5. Worker reorder costs (`worker_request_costs`)

**Problem:** a chain measures only its realised order. If the uplink transfer flips the order, the opposite-role costs come from the nearest same-stratum chain with the opposite order.

**Why it matters:** prediction-first frames are rare (5/48 live). If a stratum has none, a flipped decision is uncalibrated.

## Launch blockers outside the sampler

### 6. Fallbacks (`FALLBACK_RATE_AND_MECHANISM_UNCALIBRATED`)

**Live behaviour:** a fallback opens no ticket and keeps the previous outcome.
- live100: 3 startup fallbacks and 1 stale-SI fallback.
- Run-5 capture: 5 fallbacks.

**Why it is not represented:** the unchanged `Run4SequentialEnvironmentV1` cannot carry a previous outcome across a non-decision opportunity. Representing it needs a versioned environment wrapper plus a bound rate.

**What is implemented:** parity of the encoding only (holds and fallbacks create no state).

### 7. Pipeline version

**Constraint:** chains bind one runtime by pipeline identity.

**Consequence:** any runtime-overhead change invalidates them and requires re-measurement. Candidates are the UE input function (15.6 ms against 6.2 ms legacy), the post-prediction path (21.4 ms), and the edge queue handoff.

**Pooling rule:** live100 and the screen are pooled only if their identity records are byte-identical.

### 8. Undefined-quality rate

**Training:** follows the grid. 36/512 FIT scenes are undefined in every cell (7%), so undefined quality is scene-level and action-independent.

**Live:** 5/48 (live100) and 65/497 (Run-5 capture). These are different scene distributions; the rate is not calibrated to live.

### 9. Training cadence versus live

**Inherited:** the next action-open is ≥ 210 ms after the previous open (from `collector_v1`). Live it is 200 ms.

**Effect:** chains embed held-frame contention only at the live cadence. The effect on the next decision's backlog and contention is unmeasured.

### 10. Cold start

**Policy:** cold frames are excluded by binding policy.

**Gap:** live processes are warm after the first two decisions. Session breaks after faults are not process restarts. No cold penalty is modelled.

### 11. Trainer wiring (intentionally absent)

**What exists:** only `FIXTURE` mode, which is labelled synthetic mechanics and not replay-exportable. `CALIBRATED` raises `LaunchBlockedError` until `APPROVAL.json` names the binding, the candidate source digest and every blocker.

**What is not done:** no replay export, no `ModeledCompositeTrainingIssuerV1` disclosures and no campaign runner were written.

### 12. Inherited quality caveats (unchanged by design)

- JPEG training input against raw live RGB.
- Linear off-anchor q interpolation (its leave-one-q-out qualification was not located).
- Single-episode scene coverage.
- Live conservative conflict detector against historical manual quarantine.

### 13. Exact-GT service queue: RESOLVED in composition v2 (`SERVICE_QUEUE_V2`)

**What the model now holds:**
- **GT path, as separate boundaries:** freeze hand-off; then FREEZE (preparation), BIND, CONFIRM_SEND and PUBLISH, each with its own dispatch wait, send, execution and reply. GT is ready at the service at the PUBLISH reply.
- **Score channel:** one serial channel. PREDICTION (priority 0) beats TARGET (priority 1) when both are pending. Each request has its own dispatch, send, execution and reply. Other frames' score work (carry-in) blocks the channel. The environment regenerates carry-over from each decision's post-scoring jobs.

**Evidence:** all 48 + 334 recorded chains (live100 and the 7-arm screen, cold included) reproduce every boundary to ≤ 3e-14 ms.
- The earlier "anomalies" were the PUBLISH reply arriving after the prediction (1861) and cold carry-in from the previous decision (1362, 1292, 1317, 1351).
- Chains that violate the queue rule are rejected.

**Still uncalibrated:**
- CONFIRM_SEND waiting on a late UE send never occurred in the data. Composition raises `gt_confirm_send_dependency` if a counterfactual would need it.
- The prep channel is not queued across frames. Its measured dispatch waits are ≤ 1.1 ms and are kept as segments.

### 14. Uplink evidence outside the validated wire scope (reported only; scope not widened)

Across the 7 screen arms (11–312 KB, MCS 16–28), T_v2b measured minus modelled has a median between −2.5 and +2.8 ms. The P5/P95 spread is about −18 to +32 ms.

This supports a decision to widen the declared scope, which is 96.6–403.5 KB and MCS 24–28. Widening is not done. Until it is, transfers for the four minimum-payload actions raise `E_uplink`.

### 15. GT-path floor (runtime, not policy)

GT readiness is 127–133 ms median for every action:
- freeze hand-off ≈ 22 ms
- FREEZE preparation ≈ 85–90 ms
- BIND ≈ 16 ms

Adding the TARGET scoring request (≈ 13–17 ms) and the return (≈ 6 ms) gives a receipt floor of about 150 ms median, even at 11 KB. No action choice removes it.

## Pending quality-design review (retained; not performed, no reward change)

**Scope:** class-specific within-tolerance recall and localisation tails. The current reference tolerances are vehicle 1.0 m and person 1.2 m, the same values as Q_perc's τ.

**What it tests:** whether Q_perc's median-error form, sqrt(recall·exp(−median/τ)), hides the tails and misses that matter for safety.

**Why it is not blocking here:** it is independent of the latency candidate. The reward, quality targets and state stay unchanged until it is reviewed and jointly decided.
