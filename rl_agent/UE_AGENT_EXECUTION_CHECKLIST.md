# UE-side inference controller — locked execution checklist

> **SUPERSEDED FOR CURRENT EXECUTION (2026-08-20).** This checklist preserves
> the earlier lean-controller and candidate-filtering sequence. Use
> [`UE_AGENT_EXECUTION_CHECKLIST_V2.md`](UE_AGENT_EXECUTION_CHECKLIST_V2.md)
> as the current execution authority. In particular, the active design exposes
> all 72 measured split-profile anchors subject to technical smoke, uses a
> repeatable live route and
> time-varying saved SNR traces, and requires authoritative map-install
> feedback before the collection sweep.

**Status:** LOCKED CURRENT EXECUTION AUTHORITY (2026-08-20)

**Owners:** Abiodun and Codex.

This checklist is the sole execution path for the current agent milestone.
External reviews may supply ideas, but they are not approval gates and cannot
silently change this sequence. A scope or design change requires an explicit
Abiodun–Codex decision recorded here with its reason.

> **Immediate evidence sequence (2026-08-20).** Following the supervisor
> discussion, the next deliverable is the design-only split-profile baseline in
> `UE_SPLIT_ONLY_EXPERIMENT_PLAN.md`. The concise discussion handoff is
> `UE_SPLIT_ONLY_SUPERVISOR_DELIVERABLE.md`, with the 16-row planning surface in
> `UE_SPLIT_ONLY_SUPERVISOR_COMBINATIONS_V1.csv`. The reuse-only Stage-A audit now covers
> the existing 72-profile quality/payload pool, four network regimes, and
> staleness evidence. The approved floor yields a review proposal with 26
> normal aggregate candidates plus one separately typed provisional rescue;
> this is not `N=27` and not a final action catalog. New 10-Hz runs remain
> unauthorized. Candidate/equivalence review may propose a bounded run only;
> execution requires a separate explicit Abiodun--Codex authorization. Any
> later authorized measurements
> vary only registered `SPLIT(profile)` bundles and calibrated network regimes
> while input/task schema, hardware allocation, input frames, replay rate, and
> map path stay fixed. A bundle binds its model/checkpoint and compression path;
> effects are attributed to that bundle. It has no SKIP,
> LOCAL, radar-derived decision state, reward, or learned policy. The lean
> seven-scalar `SKIP/LOCAL/SPLIT` controller below remains the later target and
> is derived only after the split-only data sheet is reviewed. This sequencing
> correction narrows the first experiment; it does not reopen parked Phase 2.

The immutable evidence audit is
`experiments/ue_split_stage_a_v1/20260820_024055_review`; the approved floor is
`decisions/ue_split_object_map_v1_floor_v1.yaml`. The candidate-proposal bundle
must end in `CANDIDATE_REVIEW_REQUIRED`, with no run, training, or final-freeze
authority.

The current objective is:

> At each eligible sensing opportunity, a single UE uses only causally
> available UE, map, compute, and lagged-network state to choose whether and
> where perception runs, so its own contribution keeps the edge spatial map
> within a speed- and hazard-dependent freshness envelope whenever feasible,
> while minimizing radio and UE-compute cost.

## Scope lock

### Included now

- A single vehicle UE and a single edge spatial-map endpoint.
- Unconditional RGB+radar seven-channel capture on every sensor frame, followed
  by one genuine pre-model decision from the exact lean state:
  SKIP_INFERENCE, LOCAL_INFER, or SPLIT_FEATURE(profile).
- LOCAL and SPLIT both execute the common UE front after selection; no
  current-frame front/tail semantic output is part of the v1 decision state.
- Measured compression knobs for SPLIT, measured sustainable compute for
  LOCAL, and a fixed compact LOCAL-result upload.
- Edge-install freshness, current radar risk, ego speed, network feasibility,
  compute feasibility, task quality, and graceful degradation.
- Rule, exact greedy/enumerator, bandit, and MPC baselines.
- Discrete RL only if a pre-registered sequential gap survives those baselines.

### Fixed for this milestone

- Every valid LOCAL or SPLIT result follows one fixed publish-all path to the
  same edge map. Publication is not a learned action.
- LOCAL inference itself is network-independent, but the current milestone
  gives it controller credit only when its measured compact result is accepted
  by the edge map (expected near 2 KB, to be verified). Onboard completion is
  logged as timing provenance, not as an occlusion, warning, braking, or reward
  endpoint.
- SPLIT sends a measured compressed feature profile and the edge completes the
  back-half inference.
- SKIP_INFERENCE discards only the captured frame from model processing. It
  does not stop or throttle the sensors; the next seven-channel frame is still
  captured normally. It generates no new perception result and therefore ages
  the edge map.
- CARLA truth and future information remain evaluation-only.

### OBJECT_MAP_V1 service lock

The current split service requires vehicle/pedestrian class, confidence,
predicted actor-reference world XY, source/capture identity, and a
valid-empty-versus-missing-update distinction. `profile_id` is required
wire/provenance metadata, not an object semantic. The world position is not a
segmentation-mask centroid. Dimensions, yaw, parked state, and radar-support
fields remain best-effort until separately validated.

Object detection and world-XY localization are primary. Segmentation mIoU,
vehicle IoU, and person IoU are secondary offline evaluation metrics and never
veto catalog admission in v1. Their numerical reward weight is not yet frozen.
The normal experimental floor and provisional rescue are hash-bound in
`decisions/ue_split_object_map_v1_floor_v1.yaml`; those offline values are
catalog-development evidence, not live/deployment certification.

`DEGRADED_RESCUE` is a separate action tier. It may be considered only when no
normal action is physically/network feasible, remains subject to the network
mask, emits service debt, and never counts as normal-quality success. It does
not relax the normal floor or become an extra hidden reward mode.

### Parked for Phase 2

- Helper/recipient pairs and recipient-specific state.
- Object-selective or hazard-selective publication.
- Recipient-installed helper-track gain and cooperative-warning lead.
- Matched positive/benign recipient trajectories.
- The factor-realization grid and exact-16 collection.
- Warning-to-braking, stopping-distance, and cooperative actuation outcomes.
- Multi-UE contention and learned inter-vehicle map sharing.

The parked work is indexed in
../phase2_map_sharing/PARKED_STATUS_2026-08-19.md. It is preserved, not
discarded, and it does not block this checklist.

## Checkbox and evidence discipline

- [x] means the acceptance condition passed and an evidence path/date is
  recorded beside the item.
- [ ] means pending. Implementation existing somewhere in the repository does
  not by itself close an item.
- At most one unchecked stage may be active.
- A stage never launches the next stage automatically.
- No long CARLA/OAI run begins before the design, static checks, and smallest
  smoke for that exact path pass.
- After a repeated failure, stop and inspect the complete causal path before
  relaunching.

### Mandatory drift test before and after every task

Record one sentence answering each question:

1. Which UE state, action, transition, reward, constraint, or evaluation
   question does this task answer?
2. What downstream UE-controller decision will its result change?
3. Can existing evidence or a cheaper table-driven check answer it?
4. Does it introduce a helper, recipient, publication policy, warning lead, or
   vehicle actuation? If yes, park it in Phase 2.
5. What is the timebox and stop condition?

If questions 1 and 2 do not have specific answers, the task does not enter the
current plan. “Potentially useful later” is insufficient.

## Locked UE control loop

The detailed Mermaid source is state_diagram.md. The decision timing is:

1. **Unconditional capture:** the synchronized RGB+radar seven-channel sensor
   frame is captured regardless of the previous or current controller action.
2. **Lean causal state, after capture but before the perception model:** derive
   exactly seven learned-controller scalars plus validity guards. Optional
   signals are not silently appended.
3. **Single decision:** choose genuine SKIP_INFERENCE, LOCAL_INFER, or
   SPLIT_FEATURE(profile). Hard masks and the freshness service shield run
   before policy preference.
4. **Common front, only for LOCAL or SPLIT:** run the UE front backbone on that
   already captured frame, then execute the selected local or edge back half.
5. **Action outcome:** LOCAL attempts one immediate compact-result upload;
   SPLIT uploads the selected feature and uses the edge back half.
6. **Fixed map update:** the newest valid result installed at the edge updates
   this UE's map contribution. Only a received accepted-install ACK advances
   the UE's known freshness state.
7. **Next observation:** delivery, latency, compute completion, and map events
   may update only the seven derived scalars and validity guards, and only after
   their availability time. They do not become additional policy inputs.

Important semantic rule: sensor capture is not inference and is never skipped.
Current front output cannot justify SKIP_INFERENCE because running the front
has already incurred model processing. If a future design uses a front output
to abstain, name that action SKIP_BACK_HALF_UPDATE and account for the sunk
front cost; it is not part of v1.

## Causal state contract

### Exact v1 learned-controller vector

The v1 policy receives exactly these seven scalars:

1. `freshness_slack_s` — derived from the newest accepted edge-install ACK;
2. `radar_risk` — one frozen scalar derived from the current aligned raw-radar
   sample (the exact reducer and sign convention are a Stage-1 decision);
3. `ego_speed_mps`;
4. `ul_capacity_lcb_bps` — one lagged pessimistic capacity scalar with
   estimation uncertainty already folded into it;
5. `in_flight_age_s`;
6. `local_compute_slack_ms`; and
7. `time_since_last_processed_s`.

Sensor validity/alignment, ACK validity, common-front readiness, no-in-flight,
and support flags are guards or masks, not extra learned features. Raw tracks,
telemetry, and action tables may be internal sources used to derive the seven
values; they are not silently exposed as additional policy inputs. Stage 1
freezes the reducer, sign, units, clipping, missing-value sentinel, and
multi-item aggregation for every scalar, including oldest in-flight age.

The sole new feedback interface required by v1 is an edge installation ACK:
`{capture_id, capture_timestamp, edge_install_at|null, accepted/status}`. A
rejected response has `accepted=false` and `edge_install_at=null`. The UE also
logs `ack_received_at`. An accepted ACK may advance known freshness at any
later consuming decision satisfying `ack_received_at <= decision_at`, subject
to newer-capture-ID ordering. Rejection becomes observable only when its NACK
arrives; ACK/NACK loss becomes observable only at the declared timeout. Neither
event triggers an application resend, and neither advances freshness.

### Optional ablation, not base state

Low-resolution Spatial Information (Sobel edge/texture complexity) and
Temporal Information (consecutive-capture luminance change) may be evaluated
offline on existing retained frames. They are not semantic object density.
High validated visual activity may eventually veto SKIP; low SI/TI can never
authorize it. A successful test remains a separately named `v1+SI/TI`
counterfactual ablation and does not expand the seven-scalar v1 state without
an explicit Abiodun-Codex re-lock. Retain the ablation only if a held-out paired
test shows incremental safe decision value with negligible measured stage
latency and no additional critical misses. No new CARLA collection is
authorized for this ablation.

### Denylist

- Current post-tail detections, confidence, track identity, or map quality
  before selecting the action that produces them.
- Current front output, current object-head objectness, or an auxiliary learned
  urgency score; none is part of v1.
- GT actor IDs/association, future actor motion, future SNR/capacity, or future
  delivery.
- Scenario ID, trajectory ID, absolute episode time, authored hazard schedule,
  or manual-driver label.
- Speed limit, route/road ID, junction/crosswalk distance, curvature,
  sight/occluder geometry, stopping distance, or CARLA actor/scenario geometry.
  These remain excluded until a concrete deployable source and incremental
  controller value are demonstrated.
- Shadow unchosen-action results.
- Any helper or recipient state.

Every consumed field must satisfy available_at <= decision_at for its consuming
stage.

## LOCAL completion, outbound delivery, and install ACK

LOCAL produces two timestamps, but only one current-milestone service outcome:

1. `local_result_available_at` records when local computation completed; it is
   timing provenance and may support a later onboard application.
2. `edge_install_at`, confirmed by an accepted ACK, is the only event that may
   refresh this UE's edge-map contribution and earn current reward credit.

The compact result is expected to be near 2 KB, so successful delivery should
remain likely even under severe measured channel conditions. That expectation
is a hypothesis, not an exemption from measurement. Stage 3 must measure
delivery, queueing, and tail latency across the existing SNR rungs and at least
one transient outage/recovery trace.

There is no controller-visible or application-level frame, feature, or result
buffer and no store-and-forward retry of an old frame. Every outbound LOCAL or
SPLIT action gets one immediate enqueue attempt. Immediate backpressure or
failure drops that current application item; the next captured frame proceeds
normally. If the common front cannot begin within the current frame's deadline,
LOCAL and SPLIT are physically masked and the frame is dropped through SKIP.
The edge rejects stale/out-of-order capture IDs. Lower-layer HARQ/RLC queues or
retransmissions may still occur and must be measured, but they are transport
behavior rather than an agent buffer or action. An unacknowledged result earns
no map-freshness credit.

## Latency evidence and percentile policy

No input feature is free. Use a monotonic clock and log alignment, state
extraction, policy, front, branch, transport, edge processing, install, and ACK
stages separately. For every tested condition report sample count, p50, p90,
p95 with a confidence interval, and the configured-deadline miss count/rate;
maximum is always reported and p99 is reported when sample support is adequate.

The operational gate is the deadline-miss evidence, not p95 in isolation.
A single slow maximum is not called p95 and is not blamed on the controller
unless its stage timing increased. For example, with 99 samples at 45 ms and
one at 70 ms, the 70 ms point affects the maximum/extreme tail, not normally
the p95. With only 20 observations, one point already represents 5%, so p95 is
too unstable for a firm gate. A firm per-condition p95 requires at least 400
steady-state samples across at least three independent runs; cheaper offline
compute microbenchmarks target at least 1,000. Below 400, p95 is explicitly
provisional rather than a pass/fail statistic.

The working engineering target for extra state extraction plus policy is
p95 <= 2 ms; p95 above 5 ms triggers review, not automatic rejection. Optional
features use paired same-frame on/off timing. Warm-up/cold-start is declared
and reported separately. A slow sample is excluded only for a predeclared,
logged invalid event such as a clock discontinuity, simulator pause, debugger
interruption, or instrumentation failure; inclusive sensitivity is retained.
Normal OS contention remains part of integrated-system latency. Results are
stratified by action and operating regime rather than pooled across unlike
conditions. Use a 95% moving/block-bootstrap interval for temporally correlated
latency and miss/delivery rates. Wilson 95% intervals are appropriate only for
independent Bernoulli trials, not correlated per-frame network outcomes. Record
the block and quantile conventions in the resolved experiment config.

## Map freshness and graceful degradation

For evaluation/internal map bookkeeping, object j's canonical age is

AoI(j,t) = t - capture_time(newest valid own contribution for j installed at
the edge map).

An initial service model is

e_j(a,t) = sqrt(b_j(a)^2 + (v_j * AoI'_j(a,t))^2)

and therefore

AoI_max,j(a) =
sqrt(max(epsilon_j^2 - b_j(a)^2, 0)) / max(v_j, v_floor).

Here b_j(a) is the measured action/profile localization floor and AoI'_j is the
post-action age under completion, delivery, drop, or SKIP. Before each decision
a deterministic, timed reducer collapses the internal records, uncertainty,
and service envelope into the single `freshness_slack_s` scalar. Neither the
policy nor the live mask/shield consumes per-object age, speed, class, range,
or covariance as additional state. Stage 1 freezes that reducer and its
empty/new-object/invalid rules. A single epsilon/service reference and its
offline sensitivity remain a checklist decision; 2.0 m is provisional, not a
deployment guarantee.

“Always fresh” means satisfying the declared service envelope whenever a
physically feasible action can do so. If none can:

1. admit only physically feasible actions;
2. choose the action with minimum predicted safety/freshness debt;
3. report the infeasible state and violated constraint; and
4. do not punish the controller as though a safe action existed.

SKIP does not create a sensing blind interval because a fresh seven-channel
capture still reaches the gate on the next sensor frame. The remaining safety
question is narrower: the gate's current-frame representation must retain
enough information to recognize when processing is required. Sensor capture
alone is not proof that a semantic hazard was recognized. Until the gate's
false-negative behavior is validated, Stage 1 may retain a conservative
maximum consecutive-SKIP or maximum edge-update interval. That is a processing
fallback, not a sensing/probe schedule.

## Constraint precedence

To avoid collision with the paper's C1–C4 contribution labels, the UE
constraints are named U0–U4:

1. **U0 — causal and measurement validity:** deny unavailable, GT, future, and
   unsupported fields/actions.
2. **U1 — physical feasibility:** mask SPLIT actions above the pessimistic
   feature-uplink budget and mask LOCAL when the common front/local back half
   cannot start and finish within measured compute capacity. SPLIT is also
   masked when the shared common front cannot start. LOCAL's one-shot compact upload is independently subject to the
   measured delivery/deadline model; an unacknowledged result does not refresh
   the edge map.
3. **U2 — freshness service:** use the one reduced freshness-slack state to
   meet the frozen edge-map service envelope when achievable; otherwise invoke
   explicit graceful degradation. Per-object/class/range diagnostics remain in
   the evaluation plane.
4. **U3 — perception validity and quality:** use only measured profiles and
   validity regions. Normal actions must satisfy the hash-bound
   `OBJECT_MAP_V1` object-quality floor. A separately labelled rescue may be
   used only under its registered no-normal-feasible contract. Segmentation is
   recorded as a secondary diagnostic and is not a v1 eligibility veto.
5. **U4 — efficiency:** optimize airtime/PRB occupancy, UE compute, and
   switching only inside U0–U3.

Safety constraints cannot be traded away by reward weights.

## Reward starting contract

The historical reward-v5 weights are not the current frozen utility. For
`OBJECT_MAP_V1`, pedestrian/vehicle detection and world-XY localization are
primary; segmentation is secondary. No numerical task-utility weights are
frozen until Stage 1 defines metric normalization and one-at-a-time
sensitivity. Catalog quality gates remain outside the reward and cannot be
traded away by learned preferences.

The current reward design must:

- score resulting installed-map quality, not selected-profile quality after a
  drop;
- give LOCAL current-milestone utility only after accepted edge installation;
  local completion remains timing provenance and receives no occlusion,
  warning, braking, or onboard-safety reward;
- include a small normalized freshness/deadline margin inside the hard service
  structure;
- charge measured network airtime/PRB use, UE compute, and a small switching
  cost;
- contain no blanket SKIP or arbitrary LOCAL penalty;
- penalize the causal consequences of abstention through reduced freshness
  slack, service debt, and lost task utility; and
- keep stopping distance/collision outside this milestone because this agent
  does not control braking.

Exact epsilon values, cost weights, references, and sensitivity ranges are not
locked merely by this checklist; Stage 1 locks them before implementation.

---

## Stage 0 — scope reset and parking

- [x] **UE-0.1:** Reconfirm the immediate problem as one UE selecting
  SPLIT/LOCAL/SKIP for one edge map.
  Evidence: Abiodun–Codex scope decision, 2026-08-19.
- [x] **UE-0.2:** Park helper/recipient Phase-2 work non-destructively.
  Evidence: ../phase2_map_sharing/PARKED_STATUS_2026-08-19.md.
- [x] **UE-0.3:** Establish this file as the sole current execution authority.
  Evidence: current-work pointers reconciled on 2026-08-19.
- [x] **UE-0.4:** Preserve the old dynamic ladder only as noncausal historical
  evidence while retaining its static measured-table results.
  Evidence: CLAUDE.md and POLICY_KICKOFF.md scope banners.

**Stage-0 acceptance:** a contributor can identify the current UE milestone
without being routed into recipient-specific collection.

**Do not proceed if:** a current task needs a recipient, peer contribution,
warning endpoint, or learned publication action.

## Stage 1 — freeze the UE contract before code

- [ ] **UE-1.1:** Freeze the frame/event clock at the measured 10 Hz sensor
  contract, with unconditional capture before every action. V1 has no FPS
  action: its realized processed/update rate emerges from per-frame SKIP,
  LOCAL, and SPLIT choices; repeated frames never receive fresh timestamps.
- [ ] **UE-1.2:** Freeze the exact aligned raw-radar representation used to
  derive `radar_risk`, all seven scalar reducers/sentinels, extraction cost,
  single-decision timing, the latency percentile/deadline-miss policy, and any
  conservative maximum-consecutive-SKIP fallback. Front/object-head urgency is
  excluded from v1; SI/TI is ablation only.
- [ ] **UE-1.3:** Freeze the state allowlist/denylist, geometry exclusions,
  validity masks, and timestamp audit.
- [ ] **UE-1.4:** Freeze the accepted-install ACK, map-install transitions,
  newer-capture-wins ordering, AoI/new-object initialization, and one-shot,
  no-application-buffer semantics for both LOCAL results and SPLIT features,
  including NACK and ACK-timeout availability.
- [ ] **UE-1.5:** Freeze U0–U4, the single v1 freshness-slack reducer/service
  reference, offline epsilon sensitivity, and graceful-degradation semantics.
- [ ] **UE-1.6:** Freeze the reward equation, references, costs, and
  one-at-a-time sensitivity ranges.
- [ ] **UE-1.7:** Work the four canonical hand traces: an update is due under a
  good channel and SPLIT is selected; bad channel with quiet radar and positive
  freshness slack permits SKIP; bad channel with expiring freshness/radar risk,
  infeasible SPLIT, feasible local compute, and feasible compact delivery
  selects LOCAL; and no action can meet the service target, invoking the
  explicit minimum-debt graceful-degradation rule. Include an ACK timeout/drop
  subcase without creating a fifth policy scenario.

**Deliverables:** versioned UE controller contract, final diagram, hand-trace
table, and explicit unresolved-measurement list.

**Accept when:** every observation precedes its consuming action, every action
has defined transitions, and all four hand traces behave sensibly.

**Stop if:** any state value exists only because the selected action already
ran.

## Stage 2 — audit reusable evidence and expose only real gaps

- [x] **UE-2.1:** Validate schema, unit, clock, and provenance consistency for
  the 72-profile evidence pool, four-regime channel surface, and staleness
  evidence. Evidence:
  `experiments/ue_split_stage_a_v1/20260820_024055_review`, 2026-08-20.
- [x] **UE-2.2a:** Publish the validated reuse-only
  `CANDIDATE_REVIEW_REQUIRED` proposal: 26 normal aggregate candidates plus one
  separately typed provisional rescue, with every measurement unauthorized.
  Evidence:
  `experiments/ue_split_catalog_proposal_v1/20260820_042414_candidate`,
  2026-08-20.
- [ ] **UE-2.2b:** Review the proposal, resolve the bounded difficult-object and
  equivalence/catalog-budget decisions, then freeze final `N_normal`,
  `N_rescue`, and `N_total` without claiming unsupported profiles are
  selectable.
- [ ] **UE-2.3:** Inventory existing local-compute evidence and distinguish
  target-device measurements from desktop proxies.
- [ ] **UE-2.4:** Audit existing v5 traces for truly causal pre-action fields.
  Missing signals remain missing; GT does not repair them.
- [ ] **UE-2.5:** Run the optional low-resolution SI/TI paired retained-frame
  `v1+SI/TI` counterfactual ablation; it stays separate from base v1 even if it
  adds held-out safe-decision value at negligible measured cost. Promotion
  requires a later explicit re-lock.
- [ ] **UE-2.6:** Produce a matrix marking each input as reusable, needs
  recalibration, missing, or Phase-2-only.

**Deliverable:** UE_EVIDENCE_AND_GAPS.md plus machine-readable source metadata.

**Accept when:** every modeled state, action, transition, and reward term has a
traceable measured source or an explicit unsupported flag.

**No new corpus is authorized by Stage 2.**

## Stage 3 — measure the minimal LOCAL action table

- [ ] **UE-3.1:** Freeze the compact LOCAL-result schema and byte-accounting
  boundary, using approximately 2 KB only as the hypothesis to verify. If it
  carries objects only, LOCAL earns no new edge-map segmentation credit.
- [ ] **UE-3.2:** Measure full-local stage latency using the frozen percentile
  policy (sample count, p50/p90/p95+CI, deadline misses, supported p99/max),
  sustainable FPS, and compute occupancy on the intended UE compute target or
  label the proxy.
- [ ] **UE-3.3:** Measure compact-result bytes versus object count/content.
- [ ] **UE-3.4:** Compare LOCAL and SPLIT quality on identical retained inputs:
  segmentation, pedestrian recall, vehicle recall, and localization.
- [ ] **UE-3.5:** Measure or validly interpolate compact-result delivery and
  capture-to-map latency over the existing four OAI channel rungs using shaped
  traffic; include delivery/drop, lower-layer queue/retransmission latency, and an
  outage-to-recovery trace. No CARLA is required.
- [ ] **UE-3.6:** Report `local_result_available_at`, `edge_install_at`,
  `ack_received_at`, and ACK status so channel variation cannot be mistaken for
  local-compute variation or unconfirmed delivery for installation.
- [ ] **UE-3.7:** Validate one-shot send/drop, no application retry or stored
  old-frame upload, ACK timeout handling, and stale/out-of-order ID rejection.
- [ ] **UE-3.8:** Remove LOCAL if its compute/delivery envelope is not
  empirically supported.
  Energy is report-only unless a trustworthy meter already exists.

**Deliverable:** versioned LOCAL_ACTION_TABLE.csv and explanatory report.

**Accept when:** every retained LOCAL action has measured/bounded compute,
payload, quality, delivery, and map-install support.

**Stop if:** LOCAL feasibility is being inferred from a desired action set
rather than measurements.

## Stage 4 — design and build the causal table-driven surrogate

- [ ] **UE-4.1:** Write the transition and logging design before code.
- [ ] **UE-4.2:** Compose the four measured channel rungs into steady,
  good-to-bad, fade, burst, and recovery traces. Do not add Sionna or more rungs
  until a policy-sensitive result requires them.
- [ ] **UE-4.3:** Define evaluation-only balanced decision-opportunity strata:
  empty/new arrival, slow/fast vehicle, pedestrian/cyclist where evidence
  exists, fresh/stale map, and free/loaded local compute. Labels remain hidden
  truth and never expand the seven-scalar observation.
- [ ] **UE-4.4:** Keep a separately reported naturalistic mixture; never use
  the balanced set alone as the headline denominator.
- [ ] **UE-4.5:** Generate observations causally and keep truth in a separate
  evaluation stream.
- [ ] **UE-4.6:** Implement unconditional capture, action-dependent completion,
  one-shot delivery/drop, delayed/out-of-order install and ACK, SKIP of
  inference only, new-object arrival, and graceful degradation.
- [ ] **UE-4.7:** Use grouped scenario/trace splits, deterministic seeds, and
  structured episode logs.
- [ ] **UE-4.8:** Test that sensing continues across SKIP and in-flight actions,
  invalid/unsupported reducer inputs conservatively mask SKIP, repeated old
  frames cannot reset freshness, and unchosen-action outcomes never enter
  state.

**Deliverables:** surrogate design, config, tests, and small auditable smoke.

**Accept when:** transition invariants pass and representative manual traces
match the Stage-1 equations.

**Stop if:** the surrogate requires helper/recipient data or hidden truth in
the policy state.

## Stage 5 — simplest-controller ladder and RL decision

All controllers receive the identical causal state, action catalog, masks, and
service shield.

- [ ] **UE-5.1:** Pre-register metrics, uncertainty method, and the minimum
  meaningful sequential gap before examining ladder results.
- [ ] **UE-5.2:** Evaluate a shielded hand rule and exact one-step
  enumerator/greedy.
- [ ] **UE-5.3:** Evaluate contextual bandit only if state-conditioned
  one-step selection remains useful.
- [ ] **UE-5.4:** Evaluate short-horizon MPC.
- [ ] **UE-5.5:** Report task utility, per-class freshness compliance,
  worst-object error, deadline debt, graceful-degradation rate, PRB/bytes,
  compute use, switching, action distribution, ACK/drop/in-flight behavior, and
  current-frame gate behavior on both
  balanced and naturalistic holdouts.
- [ ] **UE-5.6:** Check degenerate always-SKIP, always-LOCAL,
  always-smallest-profile, and repeated-frame strategies.
- [ ] **UE-5.7:** Run registered reward/constraint sensitivity.

**RL gate:**

- [ ] If MPC does not expose a meaningful sequential advantage over the best
  causal simple/myopic controller, stop and ship/report the simplest UE
  controller. This is an RL NO-GO, not a project failure.
- [ ] If a reproducible held-out temporal gap remains, authorize a discrete
  learned controller.
- [ ] A sparse/unbalanced environment or detection-only null is inconclusive,
  never a dynamic-controller NO-GO.

**Deliverable:** UE_CONTROLLER_LADDER_DECISION.md.

**Stop if:** RL is being justified by algorithm preference rather than a
measured future consequence.

## Stage 6 — learned controller only if Stage 5 authorizes it

- [ ] **UE-6.1:** Use a discrete masked method such as DQN or masked PPO; do not
  begin with continuous SAC for a measured categorical action catalog.
- [ ] **UE-6.2:** Train offline/asynchronously from the surrogate with the same
  U0–U4 structure and grouped splits.
- [ ] **UE-6.3:** Compare against rule, exact greedy, bandit, and MPC.
- [ ] **UE-6.4:** Require no service/safety regression and a held-out benefit
  on the pre-registered sequential endpoint.
- [ ] **UE-6.5:** Test unseen channel transitions, speeds, object arrivals, and
  compute loads.

**Deliverable:** RL feasibility report and versioned config/checkpoint bundle.

**Stop if:** benefit disappears under grouped holdout, sensitivity, or causal
replay.

## Stage 7 — bounded single-UE live validation

- [ ] **UE-7.1:** Use one UE and an already validated CARLA geometry. No paired
  helper/recipient corpus. The scenario setup is evaluation context and never
  a policy-state geometry feature.
- [ ] **UE-7.2:** First prove pre-action timestamps and the causal feature
  allowlist in a tiny smoke.
- [ ] **UE-7.3:** Exercise each retained action and at least one
  fade/recovery trace.
- [ ] **UE-7.4:** Confirm LOCAL/SPLIT provenance, fixed map install, AoI
  transitions, uninterrupted capture, local completion timing, one-shot
  publish/drop/ACK semantics, stale-ID rejection, and network/compute masks.
- [ ] **UE-7.5:** Compare live outcomes against surrogate uncertainty bounds.
- [ ] **UE-7.6:** Run longer validation detached and self-logging only after
  the smoke passes.

**Deliverable:** single-UE live-validation report.

**Stop if:** validation begins requiring recipient knowledge, cooperative
warning, or a new paired corpus.

## Stage 8 — Phase-2 resumption is a separate decision

- [ ] Resume helper/recipient map sharing only after the UE milestone has a
  reviewed Stage-5 decision and bounded Stage-7 validation, and only after
  Abiodun explicitly reauthorizes Phase 2.

The immediate split-only task is **UE-2.2b candidate/evidence review**. The
first pending full-controller design task remains **UE-1.1**. No controller
implementation, CARLA/OAI run, new corpus, or RL training is authorized before
Stage 1 is complete and the applicable later-stage gate is explicitly opened.
