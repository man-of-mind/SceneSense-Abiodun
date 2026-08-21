# UE agent execution checklist v2

**Status:** LOCKED CURRENT EXECUTION AUTHORITY (2026-08-20)

**Owners:** Abiodun and Codex

**Companion plan:**
[`UE_SPLIT_ONLY_EXPERIMENT_PLAN_V2.md`](UE_SPLIT_ONLY_EXPERIMENT_PLAN_V2.md)

**Supersedes:** `UE_AGENT_EXECUTION_CHECKLIST.md` for current execution. The
older checklist remains historical design evidence.

## Current direction

The immediate work is a single-UE split-profile characterization experiment.
It starts from all 72 measured discrete split anchors, admits each one after a
technical wire-path smoke, and exercises it on the same repeatable CARLA route
under four saved time-varying OAI network traces. It measures the resulting
perception, latency, freshness, and resource outcomes.

The policy must eventually learn which actions violate or satisfy the service
contract. The historical 26-pass subset and provisional rescue are annotations,
not a hand-selected action catalog. The four representative profiles used in
the integration pilot are measurement anchors, not the policy action space.

Stage 1 route qualification and UE-A1--A4 are complete. The successor
technical registry freezes all 72 registered profiles as technically valid,
records zero genuine technical failures, and applies no quality-derived mask.
UE-N1 has frozen the OAI uplink actuator and observation interface without
claiming a numeric calibration. The first active task is now **UE-N2: implement
and run the bounded persistent-Telnet 100-ms actuator smoke**. The full 72 x 4
sweep, continuous-q study, and policy training are not yet authorized.

## Scope and drift guard

### In scope now

- one UE and its own contribution to one edge spatial map;
- 10-Hz always-on RGB+radar capture;
- 72 measured fixed-profile split candidates for characterization, subject
  only to technical-validity smoke;
- one repeatable Town10HD route and simple deterministic scene;
- four pre-generated 100-ms network trace families;
- OAI target/command/achieved-radio logging;
- authoritative install feedback and local timeout;
- fixed local compute with telemetry; and
- later offline policy learning from the resulting causal table.

### Parked

- helper/recipient pairs and learned map-sharing policy;
- cooperative occlusion reasoning, warning, braking, or actuation;
- publication selection;
- multi-UE contention;
- `LOCAL_INFER` and `SKIP_INFERENCE` measurements until the split baseline is
  complete; and
- any claim that this UE discovers a truly occluded object it cannot sense.

### Mandatory question before every task

> Does this task directly implement, measure, validate, or evaluate the
> single-UE inference-placement controller?

If the answer is not specific, park the task. Also record:

1. the exact downstream controller decision the result can change;
2. whether existing evidence or a cheaper offline check is sufficient;
3. the timebox and stop condition; and
4. whether it introduces helper/recipient logic or vehicle actuation.

At most one runtime/state-changing collection stage may be active. Independent
read-only analysis may proceed only in a separate immutable output and may not
silently change the active contract.

## Locked experiment semantics

### Action registry

- Discrete baseline: `4 model families x 3 quantizers x 6 measured q values =
  72 actions`.
- The codec is a fixed system parameter, not an agent action: UE-to-edge
  feature payloads use `zstd` level 3, while the separate edge-to-map JSON
  packet keeps `zlib` level 1.
- This split is deliberate. On the paired 36-profile offline matrix, zlib's
  median payload advantage was only about 1.18%; the independent loopback A/B
  found content-dependent payload ordering but faster zstd transport/decode in
  all 36 profiles. The complete six-q, 72-action evidence is zstd-3.
- All 72 remain available if their exact checkpoint, codec, feature schema,
  decoder, and wire path are technically valid.
- Recall, precision, localization, segmentation, payload, latency, or AoI
  outcomes do not pre-filter the action registry.
- Only technical invalidity may hard-mask an action, with an explicit recorded
  reason.
- Network trace is an environment condition, not an action.
- A later hybrid candidate has 12 categorical model/quantizer branches plus
  continuous q; it is not another way to count the 72 actions.

### Service and utility

- Protect vehicle/pedestrian coverage first.
- World-XY localization is the primary task utility.
- Capture-to-install latency, AoI, deadline misses, and stable queueing are
  explicit service outcomes.
- Segmentation and validated dimension/footprint quality are secondary values
  when resources permit.
- Payload, PRB/airtime, retransmissions, and compute are efficiency costs.
- No reward directly favors or penalizes AE family, quantizer, or q.
- Ground truth scores the chosen action afterward; it never enters pre-action
  state.

The provisional quality references remain evaluation/constraint labels:

```text
vehicle recall >= 0.90       pedestrian recall >= 0.85
vehicle precision >= 0.49    pedestrian precision >= 0.61
vehicle XY MAE <= 0.90 m     pedestrian XY MAE <= 1.20 m
false positives/frame <= 1.45
```

They are quality-preservation references, not autonomous-driving safety limits
and not action masks.

### Segmentation and dimensions

The object head already supplies class, confidence, world location,
best-effort dimensions/yaw, and an optional 2-D box. A semantic mask supplies
image-plane class pixels and silhouette, not direct 3-D dimensions. A
mask-assisted footprint requires instance association, calibrated camera pose,
depth/range or ground-plane assumptions, and visibility/occlusion handling.

If evaluated, compare object-head dimensions, class-template dimensions, and a
mask-assisted footprint against CARLA truth, including added processing time.
Segmentation IoU is not used as a proxy for dimension accuracy.

### Capture and feedback

- Frame `k+1` is captured 100 ms after frame `k` at nominal 10 Hz even if
  frame `k` is still processing or awaiting feedback.
- `100 ms` is the frame/service cadence, not a stop-and-wait period.
- Only an authoritative `ACK_INSTALLED` advances the UE's known map state.
- `NACK_REJECTED` requires the edge/map to know and reject a capture ID.
- `NACK_REASSEMBLY_TIMEOUT` requires identifiable partial reception.
- If no response arrives, the UE emits local `TIMEOUT_NO_ACK`; a map cannot
  NACK a frame it never knew existed.
- A late accepted ACK retains the original deadline/timeout violation but may
  refresh known state if its accepted capture is newer.
- No application resend or old-result buffer is added.

### SNR trace and policy visibility

- Pre-generate, hash, and replay the desired-achieved-PUSCH-SNR trace at
  100-ms granularity.
- Calibrate `desired_achieved_pusch_snr_db` to the RFsim
  `commanded_noise_power_db` control and measure achieved PUSCH SNR,
  scheduler/EMA SNR, MCS, genuine available UL HARQ/CRC evidence, and queue
  service.
- The current scheduler saturation threshold is `24.5 dB`; the attach-safe
  lower achieved-SNR bound remains a calibration result.
- The future trace and target value are experiment truth, not policy input.
- Desired SNR and commanded noise are experiment-control/evaluation fields,
  never policy state. gNB collector time is not UE-policy availability. A
  later policy may use a radio observation only after a measured UE-visible
  feedback path proves `policy_observation_available_monotonic_ns <=
  decision_cutoff_monotonic_ns` in the same RAN/control epoch.

### Network-profile interpretation

A Gaussian is a bell-shaped distribution around a mean. Independent Gaussian
draws forget the preceding 100-ms value. The v2 trace generator uses persistent
states, so a good or poor period lasts several frames:

```text
GOOD/MID/POOR state
    -> usually remains, sometimes switches
    -> emits bounded correlated Gaussian variation around that state's mean
```

The four families are `FAVORABLE_STABLE`, `MID_VARIABLE`, `ADVERSE_STABLE`,
and `FADE_RECOVERY`. Their exact means, variances, correlations, transition
probabilities, and lower bound are frozen only after the OAI actuator
calibration.

## Stage 0 — scope and design reset

- [x] **UE-0.1:** Park helper/recipient map-sharing work as Phase 2.
- [x] **UE-0.2:** Freeze one-UE-to-one-map scope.
- [x] **UE-0.3:** Freeze all 72 measured profiles as the initial discrete
  action registry, subject only to technical validity.
- [x] **UE-0.4:** Reclassify the 26 normal-reference passers and provisional
  rescue as descriptive outcome analysis rather than a reduced action catalog.
- [x] **UE-0.5:** Freeze localization/coverage as primary, segmentation and
  dimensions as secondary, and forbid direct knob rewards.
- [x] **UE-0.6:** Distinguish the discrete-72 baseline from the later
  12-branch-plus-continuous-q hypothesis.

**Evidence:** Abiodun-supervisor-Codex design decision and
`UE_SPLIT_ONLY_EXPERIMENT_PLAN_V2.md`, 2026-08-20.

## Stage 1 — repeatable route qualification

- [x] **UE-R1:** Freeze one closed Town10HD route file/hash, spawn pose, target
  speed, direct route controller, completion radius, heading tolerance, and
  cleanup behavior. Traffic Manager is disabled for this qualifier. Use
  `data_collection/routes/town10hd_opt_advisor_safe_perimeter_loop_v3.progress.csv`.
- [x] **UE-R2:** Freeze the route-qualification scene as ego-only: zero ambient
  vehicles, walkers, blockers, model sensors, OAI, and map path. Allow only the
  ego, its collision sensor, and the chase spectator.
- [x] **UE-R3:** Freeze route-only output schema and predeclared pass/fail
  tolerances before running CARLA.
- [x] **UE-R4:** Run exactly three independently spawned one-lap route-only
  trials while Abiodun watches the full chase view. Report route length,
  duration, return position/heading error, control cadence, collisions, stalls,
  divergence, and actor cleanup.
- [x] **UE-R5:** Record Abiodun's per-trial visual review and require both the
  frozen machine gates and manual review to pass. Machine success alone remains
  `REVIEW_REQUIRED`.
- [x] **UE-R6:** Freeze the loop duration at 62.4 s and the resulting network
  trace length at 624 values on a 100-ms cadence.

**Accept when:** all three independent one-lap trials meet the frozen route,
control, safety, cleanup, and manual-review contract. Do not tune tolerances
after seeing a failed run merely to pass it.

**UE-R1--R3 evidence:** `UE_ROUTE_QUALIFICATION_CONTRACT_V1.md`,
`configs/ue_route_qualification_v1.json`, and
`UE_ROUTE_QUALIFICATION_V1.md`, frozen before the live trials on 2026-08-20.

**UE-R4--R6 evidence:**
`experiments/ue_route_qualification_v1/20260821_003523_624322/`. All three
trials completed the 338.023-m route in 62.4 s with zero duration spread and
zero collisions; every frozen machine gate and all eight manual checks passed.
The final create-only authority is `ROUTE_QUALIFIED.json`.

## Stage 2 — 72-action technical registry

- [x] **UE-A1:** Verify each profile's checkpoint, model family, quantizer, q,
  codec, feature schema, and edge decoder binding.
- [x] **UE-A2:** Run the smallest fixed-profile local/wire smoke needed to
  prove every anchor can load, encode, serialize, decode, and reach the fixed
  map schema.
- [x] **UE-A3:** Record every genuine technical failure. Do not remove an
  action because its offline quality or payload is unattractive.
- [x] **UE-A4:** Hash the 72-row technical action registry.

**Accept when:** every action is either technically valid or has a reproducible
technical-invalid reason. No quality-derived mask is present.

**UE-A1 evidence:** `registries/ue_split_profile_registry_v1/`. The create-only
bundle contains exactly 72 unfiltered static bindings. All checkpoint,
integrated-AE, codec, feature-schema, decoder-evidence, host-path, and
edge-container-path declarations passed; the registry CSV SHA-256 is
`9542adc8e014960bf8876e87cdbd9783f8911140fbc37820605ce7dd69e23722`.
Every row remains `REGISTERED_PENDING_SMOKE`: wire identity, mismatch
rejection, observed feature shapes, and current edge-launcher decoder-override
propagation are explicitly deferred to UE-A2. This is not a technical-validity
freeze and applies no quality mask.

**UE-A2--A3 evidence:**
`experiments/ue_a2_technical_smoke_v1/20260820_cuda_model_smoke_02/`. The
create-only `UE_A2_PASSED.json` records 72/72 technically valid profiles, four
strict front loads, four strict edge loads, four backbone encodes, 24 q/AE
paths, 72 finite tail/map paths, and 72/72 actual localhost-UDP round trips.
All 34 injected contract mismatches were rejected before decode/map use, the
source seals were unchanged, and no quality mask was applied. UE-A3 records
zero genuine technical failures; no action was removed. The earlier `_01`
bundle is superseded and is not authority.

**UE-A4 evidence:** `registries/ue_split_technical_registry_v1/`. The
create-only `UE_A4_TECHNICAL_REGISTRY_FROZEN.json` freezes 72/72 technically
valid rows, zero invalid rows, and `quality_mask_count=0`. The registry CSV
SHA-256 is
`6de6e88e6c03abcef4a907dc9bea367938f99cc34f0161497df9901f840daec4`;
the manifest SHA-256 is
`ea044dcc31632f3729f9ddae11311ab980598c6120f1aacad09034bd32698128`.
The immutable UE-A1 registry remains the operational identity source consumed
by the A2-smoked runtime; UE-A4 is the authoritative technical-evidence
successor and does not silently retarget that runtime.

## Stage 3 — OAI SNR actuator calibration and trace freeze

- [x] **UE-N1:** Freeze the physical injection and observation interface:
  saved desired achieved PUSCH SNR to a future calibrated RFsim noise command
  to achieved PUSCH/scheduler observations. Do not claim a numeric mapping.
- [ ] **UE-N2 (bounded smoke captured; full envelope remains open):** Implement persistent Telnet, recheck the effective clean
  `-50` configuration/runtime seals, and replay a short trace at 100-ms
  cadence. Log scheduled/send/response-ACK timing brackets, estimated
  first-effect lag, instantaneous/EMA achieved SNR, MCS, TBS/grant, genuine
  available UL HARQ/CRC evidence, BSR/backlog, and collector-ingest time. Do
  not invent a channel-command application timestamp.
- [ ] **UE-N3:** Verify the deployed scheduler's `24.5-dB` MCS-28 boundary and
  calibrate the attach-safe lower achieved-SNR bound.
- [ ] **UE-N4:** Quantify target-to-achieved error, lag, command jitter,
  scheduler smoothing, MCS occupancy, attachment, and queue behavior.
- [ ] **UE-N5:** Freeze four trace families with bounded parameters, transition
  matrices, dwell times, correlations, seeds, and hashes.
- [ ] **UE-N6:** Compare the correlated traces with an IID bounded-Gaussian
  control using jump size, autocorrelation, and fade/recovery duration.

**Accept when:** the target/command/achieved distinction is empirically
validated and every intended trace is reproducible, bounded, attach-safe, and
occupies its intended channel behavior.

**UE-N1 evidence:** `registries/ue_n1_oai_ul_actuator_interface_v2/`. The
create-only `UE_N1_INTERFACE_V2_FROZEN.json` records
`FROZEN_INTERFACE_ONLY`, no runtime/socket/CARLA/OAI execution, no numeric
mapping or bounds, and UE-N2 as the next item. It freezes the gNB-side
single-UE `rfsimu_channel_ue0.noise_power_dB` actuator, dynamic model-index
resolution, persistent-Telnet requirement, 100-ms monotonic/no-catch-up
schedule, ACK timing brackets, raw-event envelope, and causal policy boundary.
The manifest SHA-256 is
`0a53d754fc8e16d291dc63fe971f2749e3c0965b385b88910229fdc002a18987`.
The immutable v1 bundle is superseded pre-final evidence and is not authority.

**UE-N2 bounded-smoke evidence (2026-08-21):**
`experiments/ue_n2_oai_ul_calibration_smoke_v1/20260821_meeting_smoke_04/`
records `UE_N2_SMOKE_CAPTURED_PARTIAL_EVIDENCE`. The single-UE physical run
sent 120/120 persistent-Telnet commands at 100-ms cadence, received every
response before the next boundary, delivered 141/141 shaped UDP frames, and
verified clean `-50` restoration plus cold teardown. Median achieved
PUSCH-SNR/MCS pairs for commands `-10/-8/-5/-4` were respectively
`19.5/24`, `16.0/19`, `10.0/12`, and `8.5/9`. Handler-bracket p50/p95/max was
`0.123/0.409/1.048 ms`. The checkbox deliberately remains open because the
stock tracer does not provide the complete timestamp envelope needed for a
causal first-effect estimate, and direct UL BLER remains unresolved. These
limitations do not promote a numeric bound or universal calibration. Meeting
brief: `UE_N2_MEETING_BRIEF_20260821.md`.

## Stage 4 — map-install feedback and asynchronous timing

- [ ] **UE-F1:** Implement authoritative `ACK_INSTALLED` only after accepted
  map installation; do not reuse the current pre-publisher tiny ACK.
- [ ] **UE-F2:** Implement `NACK_REJECTED`, identifiable
  `NACK_REASSEMBLY_TIMEOUT`, and UE-local `TIMEOUT_NO_ACK` without resend.
- [ ] **UE-F3:** Freeze capture period, 100-ms service deadline, map
  acceptance-expiry policy, and separate ACK timeout.
- [ ] **UE-F4:** Test accepted, rejected, partial-reassembly, publisher/map
  loss, ACK loss, late ACK, duplicate, stale, and out-of-order cases.
- [ ] **UE-F5:** Verify frame `k+1` capture never waits for frame `k` feedback.
- [ ] **UE-F6:** Verify a timeout followed by a late accepted ACK retains the
  violation but refreshes known state only when newer.

**Accept when:** feedback is causally authoritative, all terminal/late paths are
unambiguous, and the 10-Hz producer is never blocked.

## Stage 5 — representative integration pilot

- [ ] **UE-P1:** Freeze four payload/quality-diverse pilot anchors. Label them
  measurement anchors, not a reduced policy catalog.
- [ ] **UE-P2:** Freeze resolved configs, one common route/scene seed, the four
  common trace hashes, warm-up, timeout, overload stop, reset, and output
  contracts.
- [ ] **UE-P3:** Run `4 actions x 4 trace families` with unique cell/stream IDs
  and clean map/queue/actor state between cells.
- [ ] **UE-P4:** Validate route pairing, target/achieved SNR, timing joins,
  feedback, map AoI, perception metrics, radar/compute telemetry, and atomic
  terminal records.
- [ ] **UE-P5:** Review action-order drift, thermal effects, variance,
  overload behavior, and failure recovery.
- [ ] **UE-P6:** Make a separate explicit Abiodun-Codex go/no-go decision for
  the full 72 x 4 sweep.

**Accept when:** all 16 pilot cells are complete or have scientifically valid
failure records, with no source/config drift and no hidden manual repair.

## Stage 6 — full discrete characterization

- [ ] **UE-C1:** Counterbalance the 72-action order within/between trace blocks.
- [ ] **UE-C2:** Run the authorized `72 x 4 = 288` base cells with create-only,
  resumable outputs and fixed paired inputs.
- [ ] **UE-C3:** Preserve bounded overload failures as outcomes; do not replace
  a difficult cell with inference from another cell.
- [ ] **UE-C4:** Audit exact matrix coverage, hashes, resets, timestamps, and
  mutually exclusive `COMPLETED`/`FAILED` terminals.
- [ ] **UE-C5:** Estimate variance and replicate boundary or high-variance
  cells first; freeze any broader replicate plan separately.
- [ ] **UE-C6:** Assemble the per-frame/per-cell table for baseline and policy
  design.

**Accept when:** all 288 base cells have direct evidence and provenance, or an
explicit registered failure outcome, for the same route/scene/trace contract.

## Stage 7 — bounded continuous-q study

- [ ] **UE-Q1:** Freeze `0 <= q <= 0.8` as the normal continuous study range;
  keep measured `0.9/0.98` as discrete extrapolation anchors and forbid `q=1`.
- [ ] **UE-Q2:** Select unseen intermediate q values without looking at their
  outcomes and evaluate them on identical frames.
- [ ] **UE-Q3:** Measure interpolation error and monotonicity for payload,
  recall, precision, XY, dimensions, and segmentation.
- [ ] **UE-Q4:** Compare the 72-action discrete baseline with a hybrid action
  of 12 categorical branches plus continuous q.
- [ ] **UE-Q5:** Promote continuous q only if it adds reproducible utility or
  constraint success beyond discrete anchors.

**Accept when:** continuous-q outcomes are measured rather than synthesized,
and the algorithm matches the mixed action structure.

## Stage 8 — causal controller table and baseline ladder

- [ ] **UE-B1:** Freeze the live causal state from availability-timestamped
  fields only: lagged achieved network state, map freshness/feedback, aligned
  radar summary, ego speed, in-flight state, fixed/later compute headroom, and
  prior outcomes.
- [ ] **UE-B2:** Keep target/future SNR, GT, route/scenario ID, and unchosen
  action outcomes out of policy state.
- [ ] **UE-B3:** Define coverage, localization, freshness/deadline, secondary
  segmentation/dimensions, and resource costs on comparable scales.
- [ ] **UE-B4:** Compare exact rule/enumerator, greedy/contextual bandit, and
  MPC before authorizing RL.
- [ ] **UE-B5:** Report normal constraint success, degraded service, violation
  rate, AoI, localization with coverage, secondary quality, PRB, compute,
  action distribution, and switch behavior.
- [ ] **UE-B6:** Authorize a learned policy only if a held-out sequential or
  contextual advantage survives the simpler baselines.

## Stage 9 — later action-set expansion

- [ ] **UE-L1:** Measure full-local compute, quality, compact-result bytes, and
  compact-result delivery before adding `LOCAL_INFER`.
- [ ] **UE-L2:** Define genuine pre-model `SKIP_INFERENCE` using causal radar,
  freshness, speed, and prior state without front/tail leakage.
- [ ] **UE-L3:** Extend the action space to `SKIP/LOCAL/SPLIT` only after both
  actions have measured causal transitions and the split baseline is stable.
- [ ] **UE-L4:** Validate profile-switch/model-residency latency and memory
  before claiming per-frame live selection across model families.

## Stage 10 — bounded live policy validation

- [ ] **UE-V1:** Run only a small single-UE live validation on the qualified
  route after the offline/controller gate.
- [ ] **UE-V2:** Verify causal timing, action execution, queue behavior,
  install feedback, freshness, perception, and graceful degradation.
- [ ] **UE-V3:** Compare the learned controller with all registered simple
  baselines on held-out network traces/routes.

## Stage 11 — Phase-2 resumption

- [ ] Resume helper/recipient map sharing only through a new explicit scope
  decision after the UE controller milestone is complete.

## Immediate handoff

The next task is **UE-N2**: implement the persistent-Telnet player and run only
the bounded 100-ms actuator smoke. Recheck the effective channel configuration
and binary/telemetry seals, then record command send/ACK brackets and measured
radio response; do not begin the 72 x 4 collection, continuous-q promotion, or
policy training.
