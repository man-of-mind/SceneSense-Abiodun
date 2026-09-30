# Run-5 seed-17 / update-500 smoke and recovery report (2026-09-29)

**Status: `RUN5_SMOKE_AND_RECOVERY_PASS__AWAITING_DEEP_TRAINING_AUTHORIZATION`.**

This was a CPU-only modeled smoke. It is not a policy-performance, convergence or
deployment claim. The 3 × 10,000 campaign was not launched, and `--mode deep`
refuses to start without an authorization file that does not exist.

- **Where:** worktree `../abiodun_run5_wt`, branch `run5-snr-scaffold-v1`. Nothing
  was pushed.
- **Scope of changes:** everything is inside `rl_agent/splitfusion_hybrid_sac_run5_v1/`.
  There are 0 changed lines outside it relative to `42bca44`, and the main checkout
  was not touched.
- **Not used:** CARLA, OAI, RFsim, Docker, the map service, CUDA, and any live
  experiment.

## 1. Commits

| Commit | Content |
|---|---|
| `5358872` | Snapshot of the paused Run-5 files, preserved before hardening. `run5_smoke.py` was later superseded; it lives in this commit. |
| `602136b` | Hardening: native 22-input binding, 4-profile channel, shared SNR lease interface, atomic bundles, resume, signals, and the prospective preregistration. |
| `d9e9eaf` | Preregistration amendment A1 and reseal. The sealed preregistration is `RUN5_TRAINING_PREREGISTRATION.json`, SHA-256 `d6b91ffe…7936`. |

**Amendment A1.** The first smoke launch refused at the cold-host gate. `nvidia-smi`
lists two persistent host services as compute apps:

- `gnome-remote-desktop-daemon`, up 85 days;
- `nvidia-cuda-mps-server`, up 35 days.

GPU utilization was 1% and no training step ran; the logs are in
`smoke_runs/refused_attempt_1/`. The amendment allow-lists exactly those two
process names, and only while GPU utilization is at most 5% with no other compute
app. The CARLA, OAI, RFsim, Phase-6 and container checks are unchanged, and no
config or design value changed. A drift test shows the seal refuses any unsealed
source change.

## 2. Scientific binding

- **State (22 features).**
  - Positions 0–20 come from the unchanged Run-4 guard and feature builder, and a
    bit-level check against the Run-4 vector runs on every state.
  - Position 21 is `effective_external_ul_snr_proxy_scaled = (snr_db − 5.5)/19`.
  - Support is the registered `[5.5, 24.5]` dB. Anything outside it triggers the
    external fallback; nothing is clipped.
- **Models.**
  - Actor and twin critics are native 22-input networks. The hyper-parameters come
    from the Run-5 preregistration and are asserted equal to Run 4's.
  - The Run-5 runtime binding contains no Run-4 model-binding, model-schema or
    smoke-preregistration identity; a test asserts this.
  - Refused, each by a test:
    - an ordinary Run-4 checkpoint;
    - a relabelled Run-4 binding;
    - a 21-D or zero-padded Run-4 actor;
    - feature order or count drift;
    - a changed scaling identity;
    - a foreign preregistration.
- **SNR semantics.**
  - The value is the latest ACKed, unclamped target that is active at state commit
    and held by a live controller heartbeat.
  - Modeled training and the future live adapter share one `observe()` and one
    guard. The live adapter stamps `CLOCK_MONOTONIC_RAW` itself; the modeled twin
    accepts only explicit times on the virtual RAW timeline. Neither has a clock
    bridge.
  - The next command, future samples, profile labels, the hidden state and gNB
    PUSCH SNR never enter the state.
  - Tests cover the call order (kernel resolved → channel advanced) and features
    being frozen before the step.
- **Reward (unchanged).**
  - Success: `Q_perc − 0.25·latency_ms/170`.
  - Registered delivery failure or timeout: −1.
  - Infrastructure or evaluator fault: excluded.
  - SNR is not a reward term. All 2,288 smoke rewards match the formula exactly,
    and immediate outcomes are bit-identical when only the SNR changes.
- **Previous action and outcome.** The encoding matches the prior record exactly
  on every transition (0 mismatches). It is non-zero after success and timeout,
  and after a registered delivery failure (contract-level test).
- **Channel.**
  - All four registered profiles are used in balanced 4-segment blocks; the smoke
    had 4/4/4/4 segments. The profile never appears in the state.
  - FAVORABLE_STABLE and ADVERSE_STABLE use the accepted SNR-tilted kernel under an
    explicit `PROFILE_TRANSFER_UNVALIDATED` label, because it was fitted on
    MID_VARIABLE and FADE_RECOVERY.
  - Training and validation channel seeds are derived independently.
- **Eligible GT.** The Run-4 FIT scene treatment is kept: undefined-Q_perc scenes
  are excluded, and every result is conditional on eligible GT. The modeled smoke
  had 0 session rollovers; live Option-C rollover counting is registered for later.

## 3. Crash consistency

- **Bundle layout.** `checkpoint_NNNNNN/` holds `event.json`,
  `training_state.pt` (online and target critics, both Adam states, all five
  generators), `actor_state_dict.pt`, `channel_state.json`, `manifest.json` and
  `COMMITTED`.
- **Publication order.** Staging directory → fsync each file → manifest → marker →
  fsync the directory → rename → fsync the parent → atomic `LATEST` update.
  Readers refuse missing markers, hash or size mismatches, extra members,
  symlinks, and corrupt or ambiguous candidates, and they ignore staging
  directories.
- **LATEST lag.** If the pointer lags a newer verified bundle (a crash between
  rename and pointer update), resume repairs it and reports the repair. A
  missing, forged or ahead-of-evidence pointer is refused.
- **Replay and collector continuation.** These are rebuilt by re-executing the
  recorded actions through the environment. There is no gradient replay (a test
  patches `update_once` to fail). The rebuild is verified against every boundary
  digest.
- **Signals and ledgers.**
  - SIGINT/SIGTERM finish the current update, commit a non-candidate `emergency_`
    bundle, and exit with 75.
  - `metrics.jsonl`, `decisions.jsonl` and `runs.jsonl` are keyed, append-only
    ledgers. After a resume, recomputed rows are verified rather than
    duplicated, and a torn last line is dropped.
- **Checkpoint cadence.** Deep recovery bundles fall at 0, 100, 250 and then every
  500 updates to 10,000, so the maximum gap is 500. The smoke used 0/100/250/500.
- **Tests.**
  - Fault injection at every mkdir, write, fsync, rename and replace, plus
    simulated ENOSPC. No final-looking bundle ever appears without weights.
  - Corruption or removal of every member.
  - Stale pointers and incomplete staging directories.
  - Separate-process 250→500 resume and SIGTERM-then-resume.

## 4. Smoke results (seed 17, update 500, 2,288 decisions)

All registered gates pass:

| Gate | Result |
|---|---|
| Cold host (amended gate) | pass: load 0.45, GPU 1%, only the allow-listed daemons |
| Losses, gradients and parameters finite (500/500 updates) | pass |
| All 12 modes chosen by the **post-warm-up actor** | pass: counts 87/126/18/64/97/108/295/252/166/223/79/485 |
| Continuous q non-degenerate | pass: 1,762 distinct q_e4, range 22–9788 |
| State features vary | pass |
| Reward equals the registered formula | pass (2,288/2,288) |
| Current SNR never a future sample | pass (0 tick violations, 0 value collisions) |
| Four profiles balanced | pass (4/4/4/4 segments) |
| All four boundary bundles verify | pass |
| Fresh-process `weights_only` actor load | pass |

**Exploration accounting.** The 288 warm-up decisions are the deterministic
stratified schedule (12 modes × 6 q strata × 4) and are reported separately. The
post-warm-up actor made 2,000 decisions:

- empirical mode entropy 2.233 nats, against a maximum of 2.485;
- mean policy discrete entropy 1.858 nats.

**Per-feature variation (distinct values / span):**

| Feature | Distinct | Span |
|---|---|---|
| Camera SI | 473 | 4.42 |
| Radar P40 | 473 | 0.45 |
| MCS | 20 | 0.68 |
| Backlog | 138 | 0.71 |
| SNR | 2,288 | 0.99 |
| Previous mode | 12 | — |
| Previous q | 1,962 | — |
| Previous Q_perc | 1,571 | — |
| Previous latency | 1,868 | — |
| Previous presence | {0, 1} | — |
| Previous success | {0, 1} | — |

**Update metrics:**

| Update | Critic loss | Actor loss | Batch reward mean | Discrete entropy | Mean executed q |
|---|---|---|---|---|---|
| 100 | 0.551 | −0.440 | 0.111 | 1.93 | 0.699 |
| 250 | 0.516 | −0.512 | 0.062 | 1.52 | 0.679 |
| 500 | 0.505 | −0.807 | 0.119 | 1.55 | 0.770 |

The post-warm-up success rate was 0.805 (warm-up 0.892), and the mean reward was
0.109. These numbers are descriptive only.

**Controlled SNR response.** Sweeping only feature 21 from 5.5 dB to 24.5 dB on
the observed states:

- expected executed q changes by +0.0042 (95% CI [0.0031, 0.0052]);
- expected two-tensor ingress changes by −8,724 B (CI [−9,431, −7,997]);
- the mode distribution moves by a mean total-variation distance of 0.130.

This is reported without a direction gate. At 500 updates it shows that the
actor reads the feature, not that it uses it well.

**Shuffled-SNR control.** With an audit-only, in-support, temporally shuffled
SNR input to the actor and critic:

- the actor's expected q moves by 0.0063 (mean absolute change);
- the mode distribution moves by a total-variation distance of 0.040;
- the critic's value for the executed action moves by 0.044 (mean absolute change).

## 5. Recovery evidence

`smoke_runs/RESUME_EQUIVALENCE_250_TO_500.json` compares two runs:

- an uninterrupted process running 0→500;
- a run that stopped at 250 (exit 75), then a **new process** that ran
  `--resume` from `checkpoint_000250` (`LATEST_CURRENT`) through 500.

Everything compared is bit-identical:

- the files of all four bundles, byte for byte;
- actor, online critics, target critics, both optimizers and all generators;
- the channel state and the replay and boundary digests;
- `metrics.jsonl` and `decisions.jsonl`;
- the final actor tree hash.

The unit suite separately proves SIGTERM → emergency bundle → resume →
bit-identical 500, and refusal of a resume from a corrupted bundle.

## 6. Tests

| Suite | Result |
|---|---|
| Run-5 package | 93/93 pass, including the 5 separate-process campaign tests (~3 min) |
| Run-4 core (`splitfusion_hybrid_sac_run4_v1`) | 448/448 pass |
| Checkpoint sidecar | 22/22 pass |
| Transport model v2 | 43/43 pass |
| `live_route_b_v2` | 64/69 pass; 5 errors are environmental (below) |

The Run-4 suites were run from a clean `git archive 42bca44` in the scratchpad,
with the untracked evidence linked in read-only. In the worktree alone, those
suites fail only because untracked evidence and the `OAI/` tree are absent.

In `live_route_b_v2`, the 5 errors are also environmental. Their pinned model
checkpoint resolves outside the scratch root, and in the worktree it is absent.
No Run-4 file changed.

## 7. Disk preflight and deep-campaign notes

- **Smoke footprint.** The conservative smoke estimate was 82.6 MB. The actual
  footprint is 13 MB for both smoke runs; bundles are 0.6–1.9 MB.
- **Deep estimate.** The conservative model gives 4.06 GB for three seeds, and
  8.11 GB required with reserve. Free space is **9.7 GB on a volume that is 100%
  used**, so the check passes today with little margin. The calibrated footprint
  is about 1 GB. Before any deep launch, confirm or add headroom (another user
  filling the disk would make the launch preflight refuse).
- **Resume cost.** At update 10,000, resume re-executes about 40k recorded actions
  through the environment. That takes roughly 7 minutes, against about 40 minutes
  for gradient replay.
- **Deep-validation diagnostics.** These are preregistered: the same held-out
  contexts for Run 4 and Run 5, reward, deadline rate, latency, Q_perc, mode and q
  distribution, oracle regret, and true SNR versus the shuffled audit input,
  across all seeds and evaluation checkpoints. The selection rule is fixed: the
  update-10,000 actor, with no validation-based selection and emergency bundles
  never eligible. The evaluator itself is not implemented yet; that is a later
  stage.

`RUN5_SMOKE_AND_RECOVERY_PASS__AWAITING_DEEP_TRAINING_AUTHORIZATION`
