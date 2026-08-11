# Track A implementation contract

Status: frozen for the SPLIT+SKIP surrogate/oracle experiments. This contract implements the decisions in
`collab/REVIEW_NOTES.md`, including the 2026-08-10 post-calibration review. It does not authorize LOCAL, RL
training, CARLA, or OAI runs.

## 1. Clock, scheduler, and delivery ordering

- The environment advances on a fixed 20 Hz clock (`dt=0.05 s`).
- A SPLIT action selects one profile and a target FPS from `{2, 5, 10, 15, 20}`. A rate accumulator schedules
  actual captures. Changing the selected SPLIT action or selecting SKIP resets the accumulator; this prevents
  fractional credit from one profile/FPS being spent by another.
- Captured frames remain in an event queue until their modeled publish time. Multiple frames may be in flight.
- A contribution is installed only when its capture timestamp is newer than the object's installed capture
  timestamp. Late/out-of-order arrivals cannot overwrite fresher map state.
- `scheduler_credit`, active schedule, and observable in-flight summaries are controller state. Hidden delivery
  truth stays in the environment.

## 2. Per-object map state and post-action task utility

The map is keyed by `(episode_id, actor_id)`. Each installed contribution stores source, capture/publish time,
confidence, `profile_id`, and a task-quality snapshot. For a candidate outcome:

- delivery installs the selected profile quality for objects captured in that frame;
- drop or SKIP retains each object's prior valid contribution and quality;
- a newly present object without a contribution is `unobserved`; SKIP cannot make its localization risk finite;
- departed objects are removed from the scored current-object set;
- an empty dynamic scene has `G=0` and post-action task utility `0`.

`U_task` is the mean post-action map quality across currently present objects, not credit for merely selecting a
profile. Thus a dropped transmission cannot earn the selected profile's accuracy.

## 3. Reward v4 and pilot defaults

Reward v4 is authoritative. Localization is structural through the tail-risk shield and the small normalized
margin; it is not repeated as the old `-0.50*loc_error/epsilon` utility term.

```text
R_inner_expected = w_task * E[U_task_post]
                   - lambda_prb * C_PRB
                   - lambda_roi * C_ROI
                   - lambda_switch * C_switch
                   - w_E * E_expected / epsilon
```

Pilot defaults are declared in `configs/track_a_pilot.yaml`: task metric proportions `0.50/0.25/0.25`,
`w_task=1`, `w_E=0.05`, `lambda_prb=1`, `lambda_roi=0.5`, and `lambda_switch=0.1`. References are the
best-achievable measured matrix values. `C_PRB=offered_rate/true_capacity` is environment-side realized cost;
the deployable controller never observes true current capacity. ROI cost is normalized `roi_q/0.5`. Switching
cost is one only when the top-level mode changes. Low/high one-at-a-time variants are pre-registered in config
and must run before the 12-condition advisor sweep.

## 4. Channel and latency projection

Channel rungs are keyed by condition/MCS, never exact floating SNR:

| rung | MCS | nominal capacity |
|---|---:|---:|
| clear | 28 | 37 Mbps |
| mild | 24 | 28 Mbps |
| mid | 19 | 20 Mbps |
| strong | 9 | 10 Mbps |

The episode samples true capacity within the documented +/-30% band. Delivery is the agreed sharp threshold:
`payload_bits * target_fps <= true_capacity`; there is no smooth delivery interpolation across the congestion
knee. The C1 mask uses only `pessimism_factor * lagged_noisy_capacity_estimate`.

The 90 KB row is the latency anchor. Let `p` be profile payload KiB, `C_r` nominal rung capacity in Mbps, and
`d_compute=(front_ms+back_ms)-(24.7+12.1)` relative to `ae32__uint4__roi0.0`:

```text
serialization_slope_ms_per_KiB = 8.192 / C_r
L_p50 = measured_90KB_capture_to_map_p50[r]
        + d_compute + (p-90) * serialization_slope * p50_factor[r]
L_p95 = (136.07-13.36) + measured_90KB_front_to_edge_p95[r]
        + d_compute + (p-90) * serialization_slope * p95_factor[r]
```

Factors are frozen in config. Clear/mild p50 and p95 factors come from the stable 90-to-400 KB cells; mid/strong
tail factors conservatively inherit/scale the last stable behavior rather than interpolate through collapsed
400 KB cells. Latencies are floored at the profile compute time. Every outcome records whether it is a measured
90 KB anchor or a payload/FPS projection. Only the 90 KB, roughly 6-8 FPS source cells are measured.

The observation-based shield evaluates deterministic capacity multipliers around `s_obs`, forms outcome-level
`G=max_j(e_j)` first, then computes expected and p95 risk and `B=E_hat_risk+k*sigma_hat`. Phase 1 fixes
`ucb_k=0` and `c1_pessimism_factor=0.70`: the honest basis is the hard C1 mask plus deterministic p95
localization tail. At that C1 floor, ensemble sigma is numerical zero for C1-admitted/raw-safe/selected actions,
though it can be nonzero for C1-rejected candidates. The 0.70 value matches the modeled -30% capacity floor as
a conservative engineering convention; it is not statistically calibrated. Residual/conformal calibration is
deferred to live validation. All current scoring remains surrogate validation only.

For a delivered candidate, C2 covers the **entire interval**, not only the fresh state after publication. For
an object with a prior map contribution, use the worse of (a) the old contribution's error immediately before
the scheduled publish and (b) the new contribution's error at publish. For a newly observed/unmapped object,
use the current observation propagated across `time_to_capture + capture_to_map_latency` as the recovery risk.
This is AoI event accounting—not a second `1/FPS` term—and prevents 2 FPS from appearing safe merely because
its eventual delivered frame is fresh.

## 5. Replay and observation hygiene

- Discover paired `*_object_ground_truth.csv` and `*_object_predictions.csv` files; reject empty/header-only
  ground-truth traces.
- Normalize repeated runs into scenario families and split families, never frames, into train/validation/test.
- Apply the measured M-prime validity gate used by the staleness/model-validation studies:
  `in_camera_frustum AND range <= configured_range`; off-FOV actors are logged as out of scope for this
  single-view Track A result, not counted as shield failures. Resample in-scope positions to the 20 Hz clock
  with a bounded interpolation gap.
- Ground truth drives hidden scene dynamics and evaluation. The deployable oracle receives matched predictions,
  tracker-derived speed, declared noise/uncertainty, lagged channel estimates, scheduler state, and observable
  send feedback. Pending contributions are keyed only by objects present in the prediction/tracker observation
  at capture time; hidden GT must never seed map state. Only the clairvoyant oracle receives current truth for
  action selection and upper-bound evaluation.
- Replay matching mirrors the validated staleness convention: use actor `origin_x/origin_y` when present,
  prediction score at least 0.20, greedy one-to-one same-class association, and a 5 m gate. Using logged
  bounding-box centers would reintroduce the known 1-1.3 m GT-convention error.
- The observable tracker may bridge matched-prediction gaps for at most 0.50 s. Report two safety views:
  **tracked-object C2** (the localization domain used by the staleness study) and strict **end-to-end GT
  exposure** (which also counts never-observed in-scope objects). Report observation coverage/object recall
  separately so upstream perception misses are not misattributed to the channel/AoI shield.
- The current real replay contains vehicles only. Pedestrian results require a separately labelled synthetic
  stress trace and cannot be presented as real-replay validation.

## 6. Preferred-core segmentation tiers

The configurable segmentation floor is a preference tier, not a hard safety claim:

- `preferred_core_kib=90`: both ROI0 profiles (90 and 129 KB) are core; sub-90 ROI profiles are degraded.
- `preferred_core_kib=129`: only the 129 KB ROI0 profile is core; 90 KB and sub-90 profiles are degraded.
- The controller considers degraded profiles only when no core action is C1-admissible and shield-safe. Their use
  is flagged and pays measured quality loss plus ROI cost where applicable.
- A separate strict-floor diagnostic masks below-floor actions and reports the resulting infeasibility; it does
  not replace graceful degradation in the primary controller.

## 7. Gates before the 12-condition sweep

1. Contract and resolved config committed with source hashes.
2. Canonical seven-profile action catalog generated; flattened Track A catalog is 35 SPLIT actions + SKIP.
3. Contract tests pass, including drops, new objects, out-of-order arrivals, C1 misses, graceful degradation,
   OOD fallback, and hidden-truth separation.
4. Four hand-checkable deterministic episodes pass their invariants.
5. The `epsilon=2.0`, 90 KB preferred-core, 25 m pilot completes and its reward/risk decomposition is reviewed.

Only after all five gates pass may safety calibration begin. Before the 3 epsilon x 2 preferred-core x 2 range
sweep, separately characterize `ucb_k` and the C1 pessimism factor at the fixed pilot point using:

- the raw `{B<=epsilon}` shield-safe set before preferred-core/reward narrowing;
- conditional false-admit and false-reject rates with explicit counts and descriptive Wilson intervals;
- schedule-selection and actual capture-attempt rates as distinct quantities;
- tracked, strict end-to-end, and finite/non-sentinel true-scored rewards as separately labelled quantities;
- common latency random numbers indexed by episode/control tick, alongside the already paired channel seeds;
- an explicit identifiability check on raw safe sets and selected actions for each swept axis.

The completed calibration was reviewed and found flat/non-identifiable for operating-point selection. The
approved follow-on experiments keep `(ucb_k=0, c1_pessimism_factor=0.70)` fixed and independently run:

1. estimator quality: telemetry lag `{0,1,2,4}` x capacity-estimate noise `{0,0.05,0.10}` at the pilot point;
2. reward one-at-a-time robustness: baseline plus low/high `w_error`, `lambda_prb`, and `w_task`;
3. advisor characterization: epsilon `{1.5,2.0,2.5}` x preferred core `{90,129}` KiB x range `{25,40}` m.

All three use the same replay split, paired channel seeds, and latency common random numbers indexed by episode
and control tick. The 40 m cells are explicitly extrapolative. Per-epsilon over-budget/feasibility is the
advisor sweep's headline result; no script automatically chooses advisor-pending values.

## 8. Pre-RL controller-ladder execution contract

- The fixed schedule, explicit threshold rule, observation-based greedy oracle, contextual bandit, and MPC all
  receive the same observable `Observation` and the same `ShieldDecision`. Their selected action must belong to
  that decision's `candidate_action_ids`; the common runner rejects any bypass.
- The rule uses declared capacity, map-age/risk, and speed thresholds and does not fit data or enumerate the
  full expected reward. Greedy is the separately labelled one-step expected-reward oracle.
- The disjoint LinUCB controller fits only on the grouped training split. Its environment feedback is the
  selected action's matched/tracked-object expected reward, supplied only after selection; evaluation is
  frozen. True channel capacity and replay truth never enter its context features.
- MPC replans from observable state each 20 Hz tick. Phase-1 planning propagates Markov expected capacity,
  uses the modal rung for latency, and holds observed object kinematics constant; scheduler-credit rules and
  planned contribution arrivals are explicit. Existing
  in-flight traffic is available only as the deployable summary and cannot secretly install hidden objects in
  the planner. Every future branch is re-masked and re-shielded by the shared implementation.
- Controller comparisons use paired channel seeds and per-tick latency random numbers. A verified corrected-
  vehicle corpus root and episode-level grouped split manifest are mandatory; the runner refuses the legacy
  replay as a headline input. It also requires the verifier's `PASS` manifest and checks the full-batch mode,
  corpus identity, and recorded batch/config/split hashes before loading replay.
- A one-episode `--scaffold-smoke` is labelled plumbing validation. Only a complete configured run is labelled
  a surrogate controller evaluation. Neither is live safety validation, and no DQN/SAC/PPO result exists yet.
