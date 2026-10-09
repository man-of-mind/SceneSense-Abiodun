# Run-5 SNR pre-training report (2026-09-29)

**Status: `TRAINING_BLOCKED_PENDING_REMAINING_USER_CONTEXT`.**
No training, CARLA, OAI, RFsim, Docker or CUDA was run. Baseline commit `42bca44`;
branch `run5-snr-scaffold-v1` (worktree `../abiodun_run5_wt`); nothing pushed. All
changes are under `rl_agent/splitfusion_hybrid_sac_run5_v1/`; Run 4 and every
existing experiment are byte-for-byte unchanged.

## 1. Headline

The retained evidence **cannot identify a direct SNR effect** on deadline
probability, conditional latency, successor backlog or reward error once MCS,
backlog and wire bytes are controlled. Verdict:
`SNR_DIRECT_EFFECT_NOT_IDENTIFIED_BY_RETAINED_EVIDENCE`. SNR stays in the Run-5
**state** (feature 22) and must **not** be inserted into the outcome model.

The causal join itself is clean: 2,700/2,700 decisions, zero future joins, zero
cross-cell/session joins, zero non-ACKed/clamped/target-less values, an
independent brute-force re-join agrees on every row, and two runs are identical.

## 2. What was built

| File | Content |
|---|---|
| `run5_state_contract.py` | 22-feature schema; `UlSnrProxyObservationV1`; `UlSnrProxyProviderV1` protocol; `RfsimEffectiveSnrProviderV1`; SNR scaling/freshness bindings (no defaults); `guard_run5_state_for_action`; `build_run5_policy_features` |
| `run5_models.py` | Run-5 model binding (Run-4 config, `state_dim` 21→22 only); `load_run5_model_state`; 21-D refusal |
| `retained_snr_audit.py` | hashed-source causal join, baseline reproduction, one pre-registered comparator |
| `RETAINED_SNR_RESIDUAL_AUDIT.json`, `retained_snr_join.csv` | audit outputs (create-only, deterministic) |
| `test_run5_state_contract.py` (18), `test_run5_models.py` (9), `test_retained_snr_audit.py` (13) | 40 CPU-only tests, all passing |

**Schema.** Positions 0–20 come from calling the unchanged Run-4
`guard_state_for_action` and `build_policy_features`, then copying the result.
Tests compare IEEE-754 bit patterns against Run 4 for genesis, previous-success,
previous-failure and previous-timeout states and for 200 randomized states.
Position 21 is `effective_external_ul_snr_proxy_scaled`, which is
`(snr_db − center_db)/scale_db`. Reward, action space, the 170-ms deadline,
transitions, previous-outcome fields and Hybrid-SAC hyper-parameters are
imported from Run 4, not restated.

**Label.** The label is `SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB`: the simulator's
commanded effective UL SNR, not a UE-measured SNR. gNB PUSCH SNR has no path
into the module and is verifier-only.

**Fail-closed rules.** Each of the following raises the Run-4
`ExternalFallbackRequired`:

- missing, invalid or `None` SNR;
- stale SNR (age > bound; the bound itself is inclusive);
- SNR outside the closed support range;
- a foreign session, UE or clock;
- SNR that became available after state commit;
- any Run-4 slot failure.

A missing observation cannot carry a numeric value, so zero-filling is not
representable.

**RFsim provider.** The allow-list reader takes only send/ACK timestamps,
status, clamp flag and `target_snr_db`. It drops `profile_id`, `step_index`,
`reason` and `commanded_noise_power_db`, and ignores any trace or Markov field.
It selects the latest command whose ACK is at or before state commit, which is
strictly before action-open. If that latest effective command is clamped,
errored or has no target (for example the clean restore), the result is
**invalid**. An older value is never brought back, because the channel has
already left it. A newer command that was sent but not yet ACKed is ignored, as
the frozen rule requires, and is flagged as guard-only metadata.

**Checkpoints.** The following are all refused:

- 21-D actor or critic tensors, under either binding;
- a Run-4 binding, even when the tensors are 22-D;
- a relabelled Run-4 binding;
- zero-padded Run-4 weights under a forged Run-5 binding (the SNR input column
  is identically zero);
- any foreign width;
- any durable checkpoint directory. Only the manifest is read, never
  `torch.load`. Run 5 has no durable format yet.

## 3. Retained-evidence audit

**Population.** The data are
`experiments/ue_production_queue_capture_v1/20260929_causal_join_v2b`: 12 cold
cells, 6 FIT and 6 VALIDATION, each a `FAVORABLE_STABLE` or `ADVERSE_STABLE`
trace. That is 2,700 decisions at 5 Hz. The 50 source artifacts are
SHA-256-hashed in the audit JSON; they include every `command_log.json`, the
decision table, the v2b evaluation, the model, and the code.

**Join.** A decision uses a command only if its ACK is strictly before
`frame_open` (= action-open). This is the same strict cutoff the v2 join used
for backlog and MCS.

| Gate | Result |
|---|---|
| Coverage | 2,700 / 2,700 |
| Future joins | 0 |
| Cross-cell / cross-session joins | 0 (sessions time-disjoint; selected ACK never in a sibling log) |
| Non-ACKed / clamped / target-less | 0 / 0 / 0 |
| Independent reference join disagreements | 0 |
| Profile/trace/noise fields exposed | 0 |
| Deterministic (2 full runs) | identical |

**Baseline reproduction.** The frozen `model_v2.grouped_cross_validation` on the
1,350 FIT rows reproduces `20260929_model_v2b/EVALUATION_V2.json` exactly: all 6
folds and the pooled block are dict-equal.

**Comparator.** One comparator was fixed and committed (`43ecdad`) before any
outcome was examined. It uses the same family, seed and optimizer as the
baseline, plus `snr_db/30`:

- in the deadline head, with a positive-direction sign constraint;
- in the latency head, with a negative-direction sign constraint;
- in the queue head, as a non-negative per-dB service slope about the MCS-bin
  median.

The decision rule was also fixed in advance. SNR improves a metric only if the
pooled held-out value is strictly lower **and** it is strictly lower in ≥5 of the
6 held-out FIT cells. A head's effect counts as identified only if the metric
improves **and** the head's SNR coefficient is >0 in ≥5 of 6 folds.

Pooled held-out results over 6 grouped folds (lower is better):

| Question (primary metric) | Baseline | + SNR | Δ | Cells improved |
|---|---|---|---|---|
| Deadline probability (Brier) | 0.01981 | 0.02049 | +3.4 % | 0 / 6 |
| Conditional latency (\|err\| P50, ms) | 7.317 | 7.268 | −0.7 % | 1 / 6 |
| Successor backlog (NMAE) | 0.01250 | 0.01338 | +7.0 % | 0 / 6 |
| Reward (expected-transport-reward MAE) | 0.08655 | 0.08752 | +1.1 % | 1 / 6 |

Secondary metrics, all reported:

| Metric | Baseline | + SNR | Cells improved |
|---|---|---|---|
| False-success rate | 0.90 % | 1.03 % | 0 / 6 |
| Latency \|err\| P95 (ms) | 33.34 | 31.90 | 2 / 6 |
| Reward error P50 | 0.01076 | 0.01069 | 1 / 6 |
| Reward error P95 | 0.0490 | 0.0469 | 2 / 6 |

The pooled P95 gains are not consistent across cells and do not meet the rule.

Per held-out cell (the per-cell values are in the JSON). SNR loses on Brier in
all 6 cells and on queue NMAE in all 6. Latency P50 improves only in
`adverse_stable__perm2`.

**Why the effect cannot be identified.** The SNR coefficient is positive in 6/6
folds in every head, yet held-out prediction gets worse. SNR is mostly a
relabelled MCS:

- corr(SNR, prior MCS) = 0.979 on FIT;
- R²(SNR ~ MCS, backlog, bytes) = 0.958, so VIF = 23.7;
- residual SNR SD is 1.24 dB;
- adding SNR lowers the MCS deadline weight from about 5.8 to about 4.2.

Every cell of a profile replays the **same** target trace on a shared epoch. SNR
is therefore a function of (trace, elapsed step), and only two traces carry the
identification. The descriptive VALIDATION score was computed once, after both
models were frozen, and used for no decision. It shows the same pattern: Brier
0.02518→0.02581 and NMAE 0.00732→0.00768, with latency P50 5.97→5.95.

## 4. Findings that need an explicit decision

1. **The proxy lags by one step.** In 2,335/2,700 decisions (86.5 %) the next
   RFsim command ACKs within 1 ms *after* action-open. The trace grid and frame
   open share one epoch, so the channel that carries the transmission is usually
   the command applied about 0.1–1 ms after the decision. The causal proxy is
   therefore the previous 100-ms target. That is correct causally, but it means
   the retained data measure a *lagged* SNR. Offsetting the actuator phase is a
   design choice and was not made.
2. **Commands in flight at action-open.** This happened in 87 decisions (3.2 %),
   where a command was sent before open and ACKed after it. The frozen rule used
   the previous ACK in those cases. They are counted and flagged, not refused.
3. **The freshness bound for a hold-until-replaced actuator is unresolved.** The
   runner sends a command only when the quantized noise changes, and the prime
   is held about 6 s before the replay. Ages: P50 99.8 ms, P95 199.8 ms,
   max 6.33 s; 354/2,700 exceed 150 ms. A literal age bound would reject values
   that are still in force. `SnrProxyFreshnessV1` therefore has no default. A
   keep-alive re-ACK or an equivalent rule is needed before a bound can be set.
4. **Scaling and support are unbound.** Retained support spans 5.54–24.49 dB over
   two traces. The Run-5 traces, the four saved OAI traces, may differ, so no
   production `SnrProxyScalingV1` was chosen.
5. **The live clock domain differs.** Retained ACKs are host `CLOCK_MONOTONIC`,
   while the Run-4 live contract clock is `CLOCK_MONOTONIC_RAW`. A live RFsim
   provider adapter must stamp ACKs in the decision clock. It is not implemented
   (live work is out of scope).
6. **The proxy is the commanded target, not the applied level.** RFsim applies a
   0.25-dB-quantized noise command. As specified, the proxy is the target of the
   effective command, not an inversion of the applied noise.

## 5. Explicitly not done

- No RL training, model-family or threshold search, or selection on validation
  data.
- No change to the outcome model; SNR is not inserted, because it has no
  support.
- No empty-scene outcome design has been chosen.
- No live adapter, CARLA, OAI, RFsim, Docker or CUDA was used.

**Test note.** `ue_production_transport_model_v2` test
`test_reference_is_the_verified_oai_ceiling` errors *in the worktree only*,
because the untracked `OAI/` tree is absent there. It passes in the main
checkout. It is unrelated to Run 5. All Run-4 contract, model and state-adapter
tests pass in the worktree.

`TRAINING_BLOCKED_PENDING_REMAINING_USER_CONTEXT`
