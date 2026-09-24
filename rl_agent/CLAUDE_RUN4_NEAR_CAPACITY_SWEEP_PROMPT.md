# Claude prompt: Run-4 near-capacity UE MCS/backlog sweep

Implement and run a new bounded SplitFusion near-capacity UE MCS/backlog
qualification. Do not modify, rerun, reinterpret, or overwrite
`rl_agent/experiments/ue_mcs_backlog_calibration_v1/20260924_131015`.

## Scientific objective

Measure the policy-relevant queue-capacity transition using exact registered
actions 70, 69, and 68 while preserving causal UE-local prior-MCS and
pre-enqueue-backlog semantics. This is a follow-up to the existing
`INCONCLUSIVE` campaign, not a replacement for it.

The earlier payloads (6,229 B, 263,507 B, and 880,567 B) leave the important
20--160 KiB control region unmeasured: the two upper points are already
saturation regimes. Do not interpolate a fitted kernel across that gap.

## Immutable action authority

- Catalog JSON:
  `rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json`
- Catalog JSON SHA-256:
  `07e0690f8a55bdd6068b8b283d14b7e165ccbf44742dd0a9568cfdd5dcac54c3`
- Catalog CSV SHA-256:
  `0512cb39982178e8c7c96a65ed26e272b3aa3a5aec0020a8dd9cf1cdb6696fbb`
- Shared checkpoint SHA-256:
  `e2f867757e8db0620316c092264ac7eb53d12bb5ef66ed14475eb40693d1f271`
- Low: action 70, `split_ae32_uint4_q9000`, 28,109 B, one 60,000-B chunk,
  2.25 Mbps at 10 FPS.
- Middle: action 69, `split_ae32_uint4_q7000`, 81,087 B, two chunks,
  6.49 Mbps at 10 FPS.
- High: action 68, `split_ae32_uint4_q5000`, 129,707 B, three chunks,
  10.38 Mbps at 10 FPS.

Refuse any identity, payload, checkpoint, catalog, or chunk-count mismatch.

Create a new additive implementation/version and a new create-only experiment
root, for example:

`rl_agent/experiments/ue_mcs_backlog_near_capacity_v1/<timestamp>/`

Never write inside `20260924_131015`. Before and after this task, hash its:

- `manifest.json`;
- `INTEGRITY_AMENDMENT_DERIVED_V1_SUPERSEDED.md`;
- `VERIFIER_INPUT_MANIFEST_V2.json`;
- `analysis_v2.json`;
- `decisions_v2.csv`.

Prove all five remain byte-identical. Preserve its verdict as `INCONCLUSIVE`.
Do not relabel it accepted. Its bound corrected-v2 result is 4/7 scientific
checks and 12/13 structural gates; do not repeat the earlier superseded 5/7 or
12/12 claim.

## Frozen experiment design

- Profiles: `FAVORABLE_STABLE` and `ADVERSE_STABLE`.
- Three tiers: action 70 low, action 69 middle, action 68 high.
- 150 decisions per block at 10 FPS; 450 decisions per cell.
- Run all six payload permutations under each profile:
  `L-M-H`, `L-H-M`, `M-L-H`, `M-H-L`, `H-L-M`, `H-M-L`.
- Total: 12 cells and 5,400 tagged decisions.
- Predeclare `L-M-H`, `M-H-L`, and `H-L-M` as FIT cells and their three
  reversed orders as blocked VALIDATION cells, separately within each profile.
- Randomize/interleave physical cell execution using one pinned seed. Never
  move cells between partitions after observing results.

This is the minimum balanced design: every payload occupies every block
position under each channel, and every transition direction is represented.

## Phase A -- audit and preregistration only

Do not launch any live component in Phase A.

1. Audit and reuse the already-qualified runner, sender, receiver, causal join,
   route proof, and teardown logic only where hashes and semantics still match.
   Prefer an additive package/wrapper over editing the completed v1 evidence
   path.
   Record the complete starting `git status`. Preserve every pre-existing
   dirty/untracked path byte-for-byte, including work being produced by other
   agents. Stage only an explicit allow-list of files owned by this task;
   never use `git add -A`, reset, stash, clean, checkout, pull, rebase, or push.
2. Write and commit the complete preregistration before collection.
3. Bind the exact actions, all permutations, fit/validation cells,
   randomization seed, timestamps, endpoints, ports, radio profiles, frame
   counts, causal joins, gates, retry policy, and output schema.
4. Add focused tests proving the design balance, catalog identities,
   create-only behavior, and old-evidence hash guards.
5. Report starting/final HEAD, exact files, tests, unresolved assumptions, and
   expected runtime.
6. Stop and print exactly `AWAITING_GO_LIVE_NEAR_CAPACITY_SWEEP`. Do not begin
   Phase B without explicit authorization.

## Mandatory preflight and live requirements

- No CARLA, CUDA, perception model, model checkpoint load, or map server.
- Do not edit or rebuild OAI.
- Destination must be non-host-local and route from the UE namespace through
  `oaitun_ue1`. Refuse any host-local/IP-rule bypass.
- Prove traffic reaches UE PDCP/RLC; `NR_PDCP_TX_SDU` must be nonzero.
- The policy-side MCS is the latest strictly prior UE-decoded round-0/new-data
  granted/final table-0 MCS from UL DCI.
- gNB selected/final MCS is verifier-only provenance, never an actor input.
- Backlog is raw UE RLC bytes sampled strictly before the current payload
  enqueue.
- Preserve real MCS 0 and backlog 0. Missing observations remain explicitly
  missing and are never forward-filled or converted to zero.
- Retain full prior-grant identity. UE-side MCS coverage must be 100%. The gNB
  final-MCS join is verifier-only: require at least 99% unique match coverage
  within every cell, zero ambiguity, and zero UE-versus-gNB mismatches among
  matched rows. Explicitly exclude unmatched verifier rows from model fitting;
  never impute, forward-fill, or reinterpret them as mismatches.
- Collect per-100-ms new-data TBS service, next backlog, every enqueue/sender
  terminal, chunk arrival, and complete-reassembly latency.
- Use the measured same-event wall/monotonic clock bridge. Its residual P95
  must be no more than 1 microsecond.
- Exact decision, terminal, and chunk accounting is mandatory.
- Read back radio state before and after every cell.
- Tear down all task-created processes, containers, namespaces/tunnels, and
  restore `noise_power_dB=-50` after every failure and at final completion.
- Finish with a cold-host proof.

Stop without changing criteria or silently repairing evidence if:

- any authority/hash/action identity differs;
- traffic bypasses the UE radio path;
- an observation occurs at or after its action enqueue;
- missing is converted to zero;
- UE-side MCS is missing, the verifier-only gNB match falls below 99% in any
  cell, or any matched grant is ambiguous or disagrees with gNB final MCS;
- terminal/chunk accounting fails;
- an output would overwrite an existing path;
- any protected `20260924_131015` file changes;
- teardown or RF restoration fails.

If an engineering defect prevents collection, preserve the failed attempt,
repair only the proven defect, obtain authorization under the predeclared retry
policy, and never mix different code revisions inside one scientific run.

## Immutable evidence requirements

- Raw evidence is create-only and hash-manifested.
- Analysis generations are create-only; never overwrite an earlier analysis.
- Record source commit, effective configs, commands, clock domains, route
  proof, and SHA-256/size for every retained file.
- Do not force-add large ignored raw traces unless explicitly authorized.
- Generate a final verifier-input manifest binding raw evidence, source commit,
  analysis, figures, and all protected-old-evidence hashes.

## Preregistered analysis and model gates

Use whole cells for fitting/validation; never randomly split individual rows.

Report:

- backlog/service/transient trajectories by action, profile, block position,
  and transition direction;
- MCS load invariance and channel separation;
- pre-enqueue backlog, per-interval service bytes, next backlog, complete
  reassembly, uplink latency, and 170-ms transport outcomes;
- payload+backlog versus payload+backlog+MCS on blocked validation cells;
- next-backlog error, latency error, Brier score, false-success rate, and
  monotonicity violations.

Prospectively enforce these gates:

1. 12/12 cells and exactly 5,400 terminal outcomes.
2. 100% UE-side MCS coverage; at least 99% unique verifier-only gNB provenance
   coverage in every cell; no ambiguity and no UE-final-MCS mismatch. Exclude
   the unmatched verifier rows and report their identities and distribution.
3. Near-capacity queue-ceiling censoring below 1%; otherwise stop and redesign
   rather than fit through a censored region.
4. Validation next-backlog normalized median absolute error no more than 10%,
   and at least 20% better than a backlog-persistence baseline.
5. Per-cell validation uplink-latency P50 error no more than 17 ms and P95
   error no more than 34 ms.
6. Validation 170-ms false-success rate no more than 5% and Brier score no more
   than 0.15.
7. Adding MCS must not worsen backlog-only validation Brier by more than 0.01,
   and matched near-boundary MCS contrasts must have the physically correct
   direction.
8. At fixed other inputs, increasing payload or backlog must never improve
   predicted delivery/latency; increasing MCS must never worsen it inside
   measured support.

Do not multiply reward by an admission probability. Do not expose the hidden
profile identity to the actor. Do not claim kernel acceptance merely because
collection gates pass. Preserve a failed or `INCONCLUSIVE` result honestly.

## Intended later queue/reward semantics (do not train in this task)

For each transmitted tensor:

\[
B_{t+1}=\min\!\left(B_{\max},\max(0,B_t+P_t-S_t)\right),
\]

where \(B_t\) is strictly pre-enqueue UE backlog, \(P_t\) is the actual
reward-requested or held-frame payload, and \(S_t\) is measured new-data
service over the following 100 ms. Queue overflow must be explicit, never
silently clipped.

The reward-requested tensor succeeds only after complete UDP reassembly and
full action-open-to-quality-feedback latency \(L\le170\) ms:

\[
r=\begin{cases}
Q_{\mathrm{perc}}-0.25L/170,&\text{success},\\
-1,&\text{delivery failure, service failure, or timeout}.
\end{cases}
\]

Held tensors reuse the exact action, request no reward, and still enter the
queue recurrence.

Do not fit or train the Hybrid-SAC agent during this task. Expected Phase-B
runtime after authorization is approximately 25--40 minutes including
qualification, all 12 cells, teardown, and create-only analysis.
