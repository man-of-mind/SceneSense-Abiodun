# UE split-profile characterization plan v2

**Status:** DESIGN LOCKED; Stages 1 and 2 plus UE-N1 are complete; continue at
UE-N2.

**Decision date:** 2026-08-20

**Owners:** Abiodun and Codex. Supervisor feedback informed this revision.

**Supersedes:** `UE_SPLIT_ONLY_EXPERIMENT_PLAN.md` for current execution. The
old plan and its Stage-A/candidate artifacts remain immutable historical
evidence; they are not the active action-catalog decision.

## 1. Big picture

The current milestone is intentionally narrow:

> A single vehicle UE captures RGB and radar continuously and sends its own
> split-inference feature representation to one edge spatial map. We want to
> measure how each supported split configuration behaves under repeatable
> scene motion and time-varying OAI channel conditions, then use those outcomes
> to design the UE policy.

The UE does not reason about helper vehicles, recipients, cooperative
occlusion, warning, or braking. Those remain parked for Phase 2.

This revision changes the experimental question from “which small catalog
should we hand-select?” to:

> Can a policy learn which technically supported split action gives the best
> localization, freshness, and resource outcome in the current causal context,
> while protecting object coverage and using segmentation/dimension quality as
> secondary value?

The experiment therefore exposes all supported actions and records their
consequences. It does not reward or penalize an AE family, quantizer, or ROI
drop fraction directly.

## 2. Scope lock

### Included in this collection

- one UE, one edge tail, and one edge spatial-map endpoint;
- always-on aligned three-camera-channel plus four-radar-channel capture at a
  nominal 10 Hz;
- all 72 measured discrete split-profile anchors;
- one fixed split action for each complete characterization loop;
- one versioned repeatable Town10HD closed route;
- a deliberately simple scene with reduced NPCs, fixed blockers, and a small
  deterministic pedestrian component;
- four saved, seeded, temporally correlated target-SNR trace families;
- target-SNR actuation through OAI plus achieved-radio readback;
- asynchronous map-install feedback and UE-local ACK timeout;
- causal per-frame radar activity and fixed-compute telemetry for later policy
  design; and
- localization, coverage, freshness, transport, resource, segmentation, and
  optional dimension/footprint outcomes.

### Fixed in the first collection

- UE and edge hardware allocation;
- local compute background-load policy;
- model input, class definitions, preprocessing, split interface, lossless
  `zstd` level-3 UE-to-edge feature wire, `zlib` level-1 edge-to-map
  `OBJECT_MAP_V1` JSON packet, map schema, and publish-all behavior;
- route file, spawn, route controller, scene seeds, and actor choreography;
- capture cadence and one-action-per-loop behavior;
- OAI PRB/TDD configuration and SINR-driven scheduler; and
- timestamp, timeout, reset, logging, and output contracts.

### Deferred

- `SKIP_INFERENCE` and `LOCAL_INFER` actions;
- per-frame live switching among the four checkpoint families;
- variable local-compute-load experiments;
- actual RL training or online adaptation;
- multi-UE contention;
- helper/recipient state, selective sharing, cooperative occlusion, warning,
  braking, or controlled recipient outcomes; and
- a claim that any threshold is an autonomous-driving safety guarantee.

## 3. Output service and quality hierarchy

The service remains `OBJECT_MAP_V1`. A useful installed object update contains:

- source/capture identity and capture time;
- valid-empty versus missing-update status;
- vehicle or pedestrian class;
- confidence; and
- predicted actor-reference world XY.

The object head already predicts class, confidence, local/world location,
best-effort dimensions and yaw, and an optional 2-D box. The semantic head
produces a dense `{background, vehicle, person}` mask, but the current map
publisher sends only mask class counts; the map does not currently fuse the
dense mask.

### Segmentation and dimensions

The supervisor's mask/dimension idea is retained as a testable secondary
hypothesis, with a precise interpretation:

- a semantic mask directly provides image-plane class pixels, silhouette,
  area, and pixel extents;
- it does not provide instance identity when same-class objects overlap;
- a pole-camera mask can support a mask-assisted 2-D footprint or approximate
  metric extent only after object association plus calibrated camera pose,
  depth/range or ground-plane assumptions, visibility, and occlusion handling;
  and
- a monocular semantic mask alone does not uniquely recover 3-D length, width,
  and height.

The active integrated-training configuration also gives location more weight
than dimensions: center `4.0`, location `1.5`, dimensions `0.6`, and yaw
`0.3`; checkpoint selection uses a location-plus-downweighted-dimension
criterion. This supports measuring dimension quality explicitly instead of
assuming it from segmentation IoU.

If the dimension hypothesis is evaluated, compare three alternatives against
CARLA truth: object-head dimensions, simple class-template dimensions, and a
calibration-aware mask-assisted footprint. Report added processing latency.
Until that test passes, dimensions remain best-effort secondary metadata.

### Outcome priority

1. **Coverage protection:** vehicle/pedestrian recall and valid update delivery
   prevent a policy from appearing accurate by missing difficult objects.
2. **Primary utility:** time-aligned world-XY localization quality.
3. **Service performance:** capture-to-map-install latency, AoI, deadline
   misses, and stable queue behavior.
4. **Secondary utility:** segmentation IoU and validated dimension/footprint
   quality when conditions permit.
5. **Efficiency:** payload, PRB/airtime, retransmissions, and compute cost.

Latency is logged directly. It is not treated as fully implicit in
localization error because a late result for a slow or static object may have
small XY error while still being stale.

The existing reference guardrails remain outcome labels and constraint
signals, not action-catalog filters:

| Metric | Provisional normal-service reference |
|---|---:|
| Vehicle recall | >= 0.90 |
| Pedestrian recall | >= 0.85 |
| Vehicle precision | >= 0.49 |
| Pedestrian precision | >= 0.61 |
| Vehicle world-XY MAE | <= 0.90 m |
| Pedestrian world-XY MAE | <= 1.20 m |
| False positives/frame | <= 1.45 |

They are empirical quality-preservation references derived from the measured
AE baseline envelope, not safety limits. The historical 26-pass analysis and
provisional rescue profile remain useful annotations. They do not remove an
otherwise technically valid action. A later controller must report normal
constraint success, degraded service, and violation rate separately.

## 4. Action-space contract

### 4.1 Discrete baseline

The initial split action registry contains the 72 measured anchors:

```text
4 model families {no-AE, AE-32, AE-64, AE-128}
x 3 quantizers {uint8, uint6, uint4}
x 6 measured feature-drop fractions q {0.00, 0.30, 0.50, 0.70, 0.90, 0.98}
= 72 discrete actions
```

The codec is frozen rather than selected by the agent. The UE-to-edge feature
envelope uses `zstd` level 3; the independent edge-to-map JSON packet retains
`zlib` level 1. The older 36-profile offline matrix made zlib about 1.18%
smaller at the median, but the paired runtime A/B showed content-dependent
payload ordering and faster zstd transport/decompression in every profile.
Only zstd-3 has complete evidence for all six q anchors. Changing the feature
codec therefore creates a new experiment contract; it is not an implicit A4
edit.

All 72 are exposed to offline policy learning if their exact checkpoint,
feature schema, codec, decoder, and wire path pass a technical smoke. Quality,
payload, or latency outcomes annotate the action; they do not pre-solve the
policy by masking it. Only a genuinely missing, incompatible, or failed bundle
may be technically masked, with an explicit reason.

The current runtime may launch one fixed profile at a time. That is sufficient
for the characterization loops. A later per-frame controller additionally
requires a resident model/codec registry, profile-tagged wire contract,
matching edge decoder selection, warm-up, and switch-cost measurement.

### 4.2 Continuous-q study

Continuous q is a separate hybrid-action hypothesis, not another description
of the 72 actions:

```text
discrete baseline: 72 exact model x quantizer x measured-q actions
hybrid candidate: 12 model x quantizer branches + one continuous q
```

The normal continuous study range is `0 <= q <= 0.8`, matching training
exposure. The measured `q=0.90` and `q=0.98` rows remain useful discrete
extrapolation/degraded anchors; `q=1` is invalid because it removes every
feature cell.

The rank-drop mechanism is finite and stepwise, and payload/quality need not
interpolate smoothly. Before a continuous policy is promoted:

- evaluate held-out q values between the six measured anchors on identical
  frames;
- measure interpolation error, monotonicity, payload, recall, XY error,
  segmentation, and dimension outcomes;
- compare against the 72-action discrete baseline; and
- use a hybrid-action algorithm rather than pretending ordinary continuous SAC
  or TD3 directly chooses the categorical model/quantizer branch.

No reward term directly rewards or punishes q.

## 5. Repeatable route and scene

Use one versioned closed route rather than attempting to traverse every CARLA
road-graph waypoint. The initial candidate is:

`data_collection/routes/town10hd_opt_advisor_safe_perimeter_loop_v3.progress.csv`

It contains 85 progress points. Its route file and all runtime controls must be
hashed in the resolved experiment config.

Before any network/profile matrix:

1. freeze the spawn pose, target speed, route controller, completion radius,
   heading tolerance, Traffic Manager seed, actor definitions, and cleanup;
2. qualify the route first with only the ego, collision sensor, and chase
   spectator—no perception model, OAI, map path, ambient traffic, blocker, or
   pedestrian;
3. run three independently spawned one-lap trials with full manual viewing;
4. require both the frozen machine gates and Abiodun's visual review to pass;
5. only after the route passes, freeze the reduced background traffic, fixed
   blockers, and deterministic pedestrian events for the later integration
   pilot, triggered from route progress rather than wall time; and
6. report route length, duration, return error, control cadence, stalls,
   divergence, collisions, and actor cleanup.

The measured loop duration determines the number of 100-ms samples in each
saved SNR trace. Route-progress and scenario-phase fields are logged so small
run-to-run timing differences can be aligned during analysis.

## 6. Network trace design

### 6.1 What Gaussian and Markov mean here

A Gaussian distribution is a bell-shaped rule saying values near a mean occur
more often than values far from it. Drawing an independent Gaussian value
every 100 ms forgets the previous value; it can jump from good to poor and back
immediately.

A Markov mixture adds persistent channel states such as `GOOD`, `MID`, and
`POOR`:

```text
current state --usually stays, occasionally switches--> next state
       |                                                   |
       +---- bounded Gaussian variation around its mean ---+
```

“Markov” means the next state depends on the current state. “Mixture” means the
trace uses more than one bell-shaped state distribution. This keeps a fade or
recovery present for several frames and lets queues build and drain naturally.
It extends, rather than rejects, the supervisor's Gaussian proposal.

At a 100-ms step, a state's expected dwell duration is:

```text
expected dwell seconds = 0.1 / (1 - probability of staying)
```

For example, a stay probability of `0.95` produces an average dwell of about
two seconds. These parameters have an understandable physical meaning, but
they are not called realistic until compared with a measured or Sionna trace.

### 6.2 Four trace families

Freeze four versioned families after actuator calibration:

1. `FAVORABLE_STABLE`: mostly upper-band service with narrow correlated
   variation and rare brief fades;
2. `MID_VARIABLE`: middle-band service with wider correlated variation and
   both upward and downward excursions;
3. `ADVERSE_STABLE`: mostly lower-band service with narrow variation and rare
   recovery;
4. `FADE_RECOVERY`: explicit multi-second switching among favorable and
   adverse states.

Each profile records bounds, state means/variances, within-state correlation,
transition matrix, expected dwell times, clipping rule, seed, and trace hash.
The traces are generated before collection at 100-ms granularity. The same
route, scene seed, and exact trace are replayed for every compared split
action. New complete trace seeds define replicates.

Keep an IID bounded-Gaussian trace as a diagnostic control. Compare its jump
size, autocorrelation, and fade dwell times with the Markov trace rather than
assuming the more complex generator is automatically better.

A later SCAN/Sionna-derived position-indexed trace is a held-out spatial test,
not another training profile and not a prerequisite for this first matrix.

## 7. Saved SNR trace into OAI

The intended physical path is:

```text
saved desired_achieved_pusch_snr_db trace at 100-ms steps
    -> trace player
    -> calibrated OAI RFsim commanded_noise_power_db
    -> achieved PUSCH SNR
    -> scheduler SNR estimate/EMA
    -> selected MCS, transport service, queue, and delivery outcome
```

Yes, this follows the same broad pattern as reading a precomputed SNR sequence
while the vehicle moves. The desired trace is not itself authoritative radio
truth. The OAI RFsim noise command must be calibrated against achieved PUSCH
SNR, scheduler state, MCS, genuinely available UL HARQ/CRC evidence, and
throughput.

Under the current one-layer scheduler table, MCS 28 begins at `24.5 dB`, not
`24.4 dB`. Above `24.5 dB`, additional SNR does not unlock a higher MCS under
the fixed current configuration. The table's MCS-0 threshold is `-1.0 dB`, but
that does not prove the UE remains attached or offers useful service there.
The operational lower bound is therefore `TBD_CALIBRATION`. Historical
evidence demonstrates approximately `8.2 dB` achieved service in the poorest
attach-safe rung; calibration may expand the lower bound only if attachment,
BLER, and queue behavior remain valid.

For every 100-ms trace step log:

- `trace_id`, `trace_index`, and `desired_achieved_pusch_snr_db`;
- `commanded_noise_power_db`, scheduled/send time, and response-ACK receipt
  time; the ACK is an upper timing bracket, not an application timestamp;
- achieved instantaneous PUSCH SNR;
- scheduler/EMA SNR;
- MCS, TBS/grant, PRBs, genuine available UL HARQ/CRC evidence, RLC/BSR, and
  backlog; and
- source-event and collector-ingest times plus explicit missingness. A future
  policy additionally requires a separately measured UE-visible availability
  time.

The scheduler estimate is smoothed, so a command need not produce an immediate
equal observed SNR. The desired trace and commanded noise are
evaluation/control truth, not policy state. gNB collector ingest is not
UE-policy availability. A later policy may consume only an observation
delivered through a measured UE-visible feedback path with
`policy_observation_available_monotonic_ns <= decision_cutoff_monotonic_ns` in
the same RAN/control epoch.

A direct scheduler-SNR injection, if used as a diagnostic, must be labelled
`SCHEDULER_EMULATION`: it scripts MCS/capacity and does not by itself reproduce
PHY fading/loss. It cannot replace the physical channel-actuation result.

## 8. 100-ms cycle, install ACK, and timeout

The nominal 10-Hz sensor clock gives a 100-ms service cadence, not a
stop-and-wait protocol:

```text
capture k -> choose fixed/current action -> front -> uplink -> tail -> map install
capture k+1 occurs 100 ms later whether or not k has completed
```

Keep three different timing concepts:

1. `frame_period_ms = 100` for capture/action opportunities;
2. `service_deadline_ms = 100` for the desired capture-to-install outcome; and
3. `ack_timeout_ms`, calibrated separately and normally greater than 100 ms,
   for declaring feedback absent.

Only an authoritative accepted-install ACK may refresh the UE's known map
state. The existing tiny edge ACK, emitted before the asynchronous map
publisher actually sends/installs the result, is not sufficient.

Use these outcomes:

- `ACK_INSTALLED`: the map accepted a specific capture and returns capture ID,
  capture time, install time, and status;
- `NACK_REJECTED`: the edge/map knew the capture ID but rejected it;
- `NACK_REASSEMBLY_TIMEOUT`: some identifiable chunks arrived but edge
  reassembly expired; and
- `TIMEOUT_NO_ACK`: the UE's local watchdog observed no feedback by the ACK
  timeout.

The map cannot NACK a frame it never knew existed. That case, or lost feedback,
is represented by the UE-local timeout; no separate reliable frame-announcement
protocol is added. There is no application resend of an old frame.

An install ACK arriving after 100 ms is `LATE_ACCEPTED`: the original deadline
miss remains recorded, but the ACK may advance known freshness if its accepted
capture is newer than the current known install. An ACK after a prior local
timeout is recorded as both timeout and late accepted. Captures and decisions
never wait for that ACK.

## 9. Base matrix and staged execution

The base characterization matrix is:

```text
72 discrete actions x 4 network trace families = 288 base cells per trace seed
```

These are experiment cells, not 288 policy actions. Each cell holds one split
action fixed for one complete route and replays the same route/scene/network
trace as its paired cells.

Do not start with an unattended 288-cell run. Use this gate sequence:

### Stage 0 — design freeze

- This plan and the execution checklist are authoritative.
- Historical 26-pass/one-rescue evidence is preserved as outcome annotation,
  not an action filter.
- No long CARLA/OAI run is authorized by the document alone.

### Stage 1 — route qualification

**Completed 2026-08-20:** three independent 338.023-m laps passed every frozen
machine gate and Abiodun's eight-check manual review. Each lap lasted 62.4 s,
so one saved network trace contains 624 values at 100-ms cadence. Final evidence
is under `experiments/ue_route_qualification_v1/20260821_003523_624322/`.

- implement/configure the ego-only closed route, chase view, lap detector, and
  deterministic cleanup;
- run three independently spawned one-lap route-only trials;
- require both machine gates and Abiodun's complete manual review;
- freeze measured duration and route acceptance; and
- calculate the exact trace length.

### Stage 2 — 72-action technical registry

**UE-A1 completed 2026-08-20:** the create-only static registry under
`registries/ue_split_profile_registry_v1/` binds all 72 unfiltered actions to
their checkpoint, integrated AE, quantizer, q, codec, feature schema,
evidence-compatible decoder settings, and distinct host/edge checkpoint paths.
All rows remain `REGISTERED_PENDING_SMOKE`; no quality mask or technical-valid
claim was applied.

**UE-A2--A3 completed 2026-08-20:** the authoritative create-only bundle under
`experiments/ue_a2_technical_smoke_v1/20260820_cuda_model_smoke_02/` proves
72/72 strict load, feature/codec, finite tail/map, and actual localhost-UDP
paths. All 34 injected mismatches were rejected before decode/map use. No
technical failure or quality mask was recorded; the earlier `_01` bundle is
superseded.

**UE-A4 completed 2026-08-20:** the create-only successor under
`registries/ue_split_technical_registry_v1/` freezes all 72 actions as
technically valid with zero invalid rows and no quality-derived mask. Its CSV
SHA-256 is
`6de6e88e6c03abcef4a907dc9bea367938f99cc34f0161497df9901f840daec4` and
its manifest SHA-256 is
`ea044dcc31632f3729f9ddae11311ab980598c6120f1aacad09034bd32698128`.
It preserves UE-A1 operational identity while certifying it against A2 `_02`;
it does not silently retarget the runtime.

- add fail-closed profile/schema/checkpoint/codec identity to the wire;
- make the current edge launcher propagate every registry-bound decoder option;
- run the smallest fixed-profile load/encode/serialize/decode/map-schema smokes;
- record every genuine technical failure without filtering for quality or
  payload preference; and
- hash the final technical-valid/invalid 72-row registry.

### Stage 3 — OAI actuator calibration

- **UE-N1 completed 2026-08-20:** the create-only final v2 interface bundle at
  `registries/ue_n1_oai_ul_actuator_interface_v2/` freezes the single-UE,
  gNB-side RFsim actuator and observation semantics without numeric
  calibration, bounds, runtime edits, or execution. Manifest SHA-256:
  `0a53d754fc8e16d291dc63fe971f2749e3c0965b385b88910229fdc002a18987`.
  The immutable v1 bundle is superseded pre-final evidence.
- verify one saved trace can update the channel actuator at 100-ms cadence;
- calibrate target to achieved SNR across attach-safe points;
- freeze lower/upper operational bounds; and
- validate command jitter, lag, MCS occupancy, BLER, and attachment.

### Stage 4 — install-feedback smoke

- implement authoritative map-install ACK/NACK and UE-local timeout;
- run injected accepted, rejected, partial-reassembly, loss, late-ACK,
  duplicate, stale, and out-of-order cases; and
- verify capture remains asynchronous at 10 Hz.

### Stage 5 — representative integration pilot

- select four payload/quality-diverse actions as measurement anchors, not as
  the policy catalog;
- run `4 actions x 4 trace families` with one fixed seed;
- reset map, queues, streams, and actors between cells;
- inspect timing, target/achieved SNR, route pairing, output completeness, and
  overload stop behavior; and
- separately decide whether the full base sweep is authorized.

### Stage 6 — resumable base sweep

- run the authorized 72 x 4 cells;
- counterbalance action order so temperature/drift does not align with one
  family or q value;
- use unique stream/cell/replicate IDs and create-only outputs;
- stop overload cells at a predeclared bounded condition while retaining their
  failure evidence;
- write an atomic per-cell terminal record; and
- resume only missing/failed cells without overwriting completed evidence.

### Stage 7 — continuous-q study

- run only after the discrete anchors and held-out-q contract are frozen;
- begin offline/on retained identical inputs where possible;
- compare discrete-72 and hybrid-12-plus-q formulations; and
- keep it separate from the 288-cell base count.

### Stage 8 — data sheet and controller design

- assemble per-frame and per-cell outcomes;
- define constrained rule/greedy baselines before RL;
- decide how radar activity, freshness, achieved network state, and later local
  compute enter the causal state; and
- freeze reward/constraint normalization only after observing metric scales.

## 10. Measurements

### Identity and pairing

- experiment, cell, replicate, stream, route, trace, profile, checkpoint,
  codec, config, seed, capture, and frame IDs/hashes;
- route progress, scenario phase, actor seed, and reset generation; and
- target action and whether the row is direct, composed, or diagnostic.

### Timing and feedback

- capture, decision cutoff/start/done, front start/done, serialization, send,
  edge receive, tail done, map install, ACK emit/receive, deadline, timeout,
  and late-ACK timestamps;
- capture-to-install and capture-to-ACK latency;
- deadline miss, timeout, rejection, loss, stale/out-of-order, and valid-empty
  outcomes; and
- p50, p90, p95, maximum, per-replicate rates, and correlation-aware
  uncertainty.

### OAI and transport

- target, command, instantaneous achieved, and scheduler/EMA SNR;
- target-to-achieved error, lag, cross-correlation, trace autocorrelation,
  state dwell times, and transition counts;
- MCS, TBS, grants, PRBs, BLER, HARQ/RLC, BSR/backlog, queue slope and recovery;
- application payload, chunks, on-wire bytes, delivery, retransmission, and
  drop/reassembly outcomes.

### Perception and map

- vehicle/pedestrian TP, FP, FN, recall, precision, valid-empty rate, and
  coverage;
- source-time and time-aligned world-XY error, always paired with coverage;
- installed-map AoI and time above each provisional freshness budget;
- object-head dimension error and box/footprint overlap when available;
- segmentation macro/vehicle/person IoU as secondary evaluation; and
- held-prior consequences when an update is delayed or missing.

### Context and cost

- ego speed from CARLA actor velocity;
- aligned raw-radar validity plus the frozen cheap activity/risk summary and
  reducer latency;
- front/encode/tail/map processing time and fixed local-compute headroom; and
- CPU/GPU utilization, temperature, and memory diagnostics where available.

Radar and compute are logged contextual variables in this fixed-action sweep;
they do not choose the loop's action. A later policy may consume only the
causally available version of a validated summary.

## 11. Output contract

Every run uses one YAML config and writes a new timestamped directory with:

```text
resolved_config.yaml
manifest.json
route_contract.json
network_profile.json
snr_trace.csv
per_frame_metrics.csv
radio_trace.csv
map_feedback.csv
perception_metrics.csv
RESULTS_SUMMARY.json
COMPLETED.json or FAILED.json
```

The manifest binds every input/output hash, code revision, dirty-tree status,
model/checkpoint, route, trace, seed, schema, and row count. Terminal files are
atomic and mutually exclusive. Existing experiment directories are never
mutated or overwritten.

At minimum, `snr_trace.csv` contains:

```text
trace_id,trace_index,scheduled_offset_ns,desired_achieved_pusch_snr_db,state,seed
```

At minimum, the UE-N2 command log contains:

```text
ran_epoch_id,control_session_id,trace_id,trace_index,commanded_noise_power_db,
scheduled_monotonic_ns,send_monotonic_ns,response_received_monotonic_ns,status
```

At minimum, `map_feedback.csv` contains:

```text
experiment_id,cell_id,stream_id,capture_id,capture_at,service_deadline_at,
edge_install_at,feedback_emit_at,feedback_received_at,ack_timeout_at,status,
accepted,late,timeout_seen,rejection_reason
```

At minimum, `per_frame_metrics.csv` joins the action, route progress, causal
context, radio outcome, processing stages, install outcome, and evaluation-only
quality fields without exposing future/GT fields to the policy.

## 12. Acceptance gates

### Design and technical registry

- [x] UE-only scope and Phase-2 parking are explicit.
- [x] Discrete 72-action and hybrid continuous-q formulations are distinct.
- [x] Quality references are outcome constraints, not action prefilters.
- [x] Localization/coverage are primary and segmentation/dimensions secondary.
- [x] All 72 anchors have hash-bound, unfiltered static UE/edge declarations.
- [ ] Every one of the 72 anchors passes a fixed-profile technical wire-path
  smoke or has a registered technical failure reason.

### Route

- [x] Route/config hashes are frozen.
- [x] Three loops complete without false completion, unresolved collision,
  deadlock, divergence, or actor leakage.
- [x] Return position/heading, 10-Hz cadence, duration variability, and event
  behavior meet predeclared tolerances.

### Network traces

- [ ] Exact injection point is frozen.
- [ ] Attach-safe lower bound and `24.5-dB` scheduler saturation boundary are
  verified on the deployed build.
- [ ] Four trace parameter sets, seeds, hashes, dwell behavior, and bounds pass.
- [ ] Target-to-achieved error/lag and 100-ms command jitter are reported.

### Feedback and timing

- [ ] True map-install ACK is distinguished from enqueue/publish ACK.
- [ ] Rejection, reassembly timeout, no-feedback timeout, late ACK,
  duplicate/stale/out-of-order, and lost-feedback paths pass.
- [ ] Frame `k+1` capture never waits for frame `k` feedback.
- [ ] The 100-ms service deadline and separate ACK timeout are frozen from
  evidence before the pilot.

### Collection

- [ ] The 4 x 4 pilot passes completeness, reset, pairing, queue, and failure
  gates.
- [ ] Abiodun and Codex explicitly authorize the full 72 x 4 sweep after pilot
  review.
- [ ] Full runs are resumable, counterbalanced, create-only, and fail closed.
- [ ] No cell is silently replaced by monotonic inference or a proxy action.

### Interpretation

- [ ] Training uses no future SNR, GT, or realized post-action outcome as
  pre-action input.
- [ ] Localization is never reported without coverage/recall.
- [ ] Normal, degraded, and violation outcomes remain distinct.
- [ ] The 72-action discrete baseline is compared before continuous-q or RL is
  claimed useful.

## 13. Remaining decisions

No further conceptual decision is required before the UE-N2 bounded actuator
smoke. The following numerical/interface values are intentionally resolved by
short staged evidence rather than guessed now:

1. attach-safe achieved-SNR lower bound;
2. replicated desired-achieved-PUSCH-SNR-to-commanded-noise calibration (the
   four-point UE-N2 smoke is provisional evidence, not a promoted mapping);
3. exact four network-profile means, variances, correlations, transition
   probabilities, and seeds;
4. map acceptance-expiry and UE ACK-timeout values;
5. the four representative pilot actions;
6. replicate count after pilot variance is known; and
7. whether mask-assisted dimensions justify their added complexity.

The bounded UE-N2 physical smoke is captured as partial evidence under
`experiments/ue_n2_oai_ul_calibration_smoke_v1/20260821_meeting_smoke_04/`.
Its four commands produced monotone median achieved PUSCH SNR values of
`19.5`, `16.0`, `10.0`, and `8.5 dB`; all 120 responses met the 100-ms cadence
and clean teardown passed. The immediate next task is UE-N3: replicate and
order-check the mapping, verify the deployed 24.5-dB MCS-28 boundary, and
calibrate the attach-safe lower achieved-SNR bound. The stock-tracer
first-effect timestamp limitation remains explicit. The full 72 x 4 sweep,
continuous-q experiment, and policy training remain separate later
authorizations.
