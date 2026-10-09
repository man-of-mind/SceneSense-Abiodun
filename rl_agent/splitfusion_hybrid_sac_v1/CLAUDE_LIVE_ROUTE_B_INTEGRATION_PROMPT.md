# Claude task: exact Hybrid-SAC live Route-B integration

You are implementing the control-plane and runtime seams needed to validate the
trained SplitFusion Hybrid-SAC policy on one chronological live Route-B loop.
Work in phases and **stop after every phase for Codex review**. Do not proceed
to the next phase merely because tests pass.

## Read first

Read `CLAUDE.md` completely, then inspect the following without modifying them:

- `rl_agent/ue_route_b_split_cell_adapter_v1.py`
- `rl_agent/splitfusion_hybrid_sac_v1/hybrid_sac_models.py`
- `rl_agent/splitfusion_hybrid_sac_v1/empirical_contextual_run3_runner.py`
- `rl_agent/splitfusion_hybrid_sac_v1/run_empirical_contextual_run3.py`
- `rl_agent/splitfusion_hybrid_sac_v1/state_reward_transition_contract.py`
- `rl_agent/splitfusion_hybrid_sac_v1/reward_ticket_controller.py`
- `rl_agent/splitfusion_hybrid_sac_v1/protocol_v2_contract.py`
- `rl_agent/splitfusion_hybrid_sac_v1/scene_descriptors.py`
- `rl_agent/splitfusion_supervisor_analysis_v1/profiled_sensor_stages.py`
- `rl_agent/splitfusion_supervisor_analysis_v1/profiled_direct_adapter_v1.py`
- `rl_agent/splitfusion_scene_descriptor_live_v1/`
- `rl_agent/splitfusion_quality_feedback_probe_v1/`
- `rl_agent/splitfusion_direct_edge_map_v1/`
- `rl_agent/splitfusion_live_dispatch_v1/dynamic_execution_contract.py`

Training campaign:

`experiments/splitfusion_hybrid_sac_run3_training_v1/20260922_three_seed_10000_updates_optimized_v1`

The preregistered pilot actor is **seed 17, update 10,000**. This is the first
registered seed and was chosen before live results; do not substitute a seed
based on live performance.

## Non-negotiable safety and scope

1. Work only under `abiodun/`. Preserve every existing dirty path. Never reset,
   overwrite, delete, stage, or edit user-owned changes.
2. Use an additive package:
   `rl_agent/splitfusion_hybrid_sac_live_route_b_v1/`.
   Do not mutate SHA-pinned production files. Import and wrap them.
3. Do not launch CARLA, OAI, RFsim, Docker, CUDA, the map server, or a live
   network process in implementation phases. Do not export `PYTHONPATH`.
4. CPU-only unit/integration fixtures are allowed. Assert that CUDA remains
   uninitialized.
5. Do not push. Use one focused commit per approved phase. Report starting and
   final HEAD, exact paths, tests, hashes, and remaining limitations.
6. Create-only evidence. Never overwrite any prior campaign or failed run.
7. Do not lower a gate, reinterpret a failure, fabricate unavailable radio
   telemetry, or silently fall back to a catalog action.
8. This task is **live CARLA/testbed closed-loop validation**, not online
   learning and not a claim of real-world deployability. Exact reward quality
   uses privileged CARLA ground truth that will not exist in deployment. Actor
   weights remain frozen and no optimizer/replay update runs in the live
   process.

## Scientific contract

The live Route-B frame rate remains approximately 10 FPS. The policy may make
decisions less frequently:

- A decision frame sets `reward_requested=true`. The ticket object is created
  when that action is committed to the frame, but its absolute deadline is the
  source-frame capture timestamp plus 200 ms.
- While that ticket is unresolved, subsequent perception frames reuse the last
  executed action and set `reward_requested=false`.
- `k_min=2`: at least two tensors use an action before a new decision.
- Success, service timeout, delivery failure, evaluator fault, late orphan and
  duplicate feedback retain distinct terminal identities.
- A service/delivery timeout is penalized by the registered reward. An
  infrastructure/evaluator fault is excluded, not blamed on the policy.
- Late feedback is recorded but can never close or relabel a newer ticket.

Registered realized reward:

```text
timely success:  Q_perc - 0.25 * latency_ms / 200
delivery failure or service timeout: -1
infrastructure/evaluator fault: excluded
```

The live latency in that equation is exactly:

```text
L_t = same-clock UE quality-ACK receipt time
      - source-frame capture time
```

Run-3's fixed-stage latency already includes the approximately 30 ms sensor
compute interval, so do not restart the clock at action selection, tensor
concatenation, first UDP send, edge receipt or tail completion. A valid reward
ACK received at or before `capture_time + 200 ms` is timely (`L_t <= 200 ms`).
The ticket may be created later at action commit, but it inherits that absolute
capture-anchored deadline. Bind both timestamps to one validated clock domain;
never form this latency by adding stage medians or subtracting unrelated clock
domains. Define stale-before-send, reassembly/delivery failure, evaluation
failure and deadline expiry as separate terminal events.

Do not multiply the runtime reward by an admission probability. Probabilities
belong only to offline analysis; live reward uses the realized terminal event.

The map path and reward path are parallel:

- Tail inference/post-processing proceeds directly to the edge map.
- Exact CARLA quality evaluation runs in a separate bounded worker.
- A compact **reward ACK** returns frame identity, executed action identity,
  perception quality, capture-to-ACK latency and terminal status to the UE
  after privileged evaluation.
- A distinct **map/install ACK** is emitted only after map installation.
- Map installation is logged separately and is not on the critical reward-ACK
  path. The reward ACK may precede or follow map installation; do not impose
  map-before-reward-ACK ordering and do not let either worker block the other.
- Only `reward_requested=true` frames enter the privileged evaluation worker.
  Reused-action frames still follow the complete perception-to-map path but do
  not enter the reward queue and never produce a reward ACK.

## Important training-distribution limitation

Run-3 was trained on independent one-step genesis observations. Therefore the
first live pilot must not invent a chained previous-feedback state that the
actor never saw during training. Build the 31-D state with the same registered
feature order and genesis/unavailable semantics for previous-outcome fields,
while recording ticket feedback separately for analysis and decision gating.
Label this explicitly:

`GENESIS_MATCHED_FROZEN_POLICY_LIVE_PILOT_NOT_SEQUENTIAL_ONLINE_ADAPTATION`

Do not claim that reward feedback changes the frozen actor's weights. A later
sequential retraining phase may use the previous-outcome features.

## Exact continuous-q requirement

The policy is hybrid:

- discrete mode: 12 `(family, quantizer)` combinations;
- continuous `q`, quantized by the registered half-up rule to `q_e4` in the
  declared executable interval.

Do **not** snap `q_e4` to the six historical anchors in the primary path. Reuse
`DynamicExecutionContract` and implement an authoritative execution identity
containing at least:

- `mode_id`, family, quantizer;
- exact integer `q_e4` and selective keep/drop counts;
- execution-bundle SHA-256;
- anchor `action_id` only when the exact pair is a registered anchor, otherwise
  explicit null/non-anchor status.

The ROI operation remains selective according to the existing learned saliency
ordering; it is not random cell dropping.

Before any live launch, prove at the appropriate boundary:

- in CPU-only Phase 3, every anchor reproduces its existing catalog identity,
  quantization, keep/drop counts, serialization and deterministic codec
  fixtures bit-for-bit;
- in CPU-only Phase 3, representative off-anchor values preserve identity and
  registered keep/drop counts through the additive protocol/codec seam;
- UE encoder and edge decoder agree on family, quantizer and `q_e4`;
- no component fabricates an anchor `action_id` for an off-anchor action;
- serialized feature and feedback messages remain within their registered
  bounds;
- live CUDA/model-output parity is not claimed by CPU fixtures. Qualify every
  dynamic action actually exercised in Phase 6, report the exercised
  mode/`q_e4` coverage honestly, and do not claim unselected-mode coverage.

## Live causal state requirements

Construct exactly the 31 actor fields in `POLICY_FEATURE_ORDER`, in exact
element-for-element order with the registered normalization, feature-count and
binding hashes. Import that authoritative tuple; do not create a second local
feature order, and assert exact tuple and hash equality before inference. The
required source groups are:

- camera SI and radar P40 from the already validated sensor-preparation seam.
  P40 must use the current non-overlapping 100 ms radar sweep; do not substitute
  the model's rolling 200 ms radar aggregation window;
- decision-time OAI SNR, MCS and BSR, with source timestamps, ages and same-epoch
  provenance;
- registered normalization and freshness gates;
- genesis/unavailable previous-action/outcome fields and their masks exactly as
  registered for this first pilot.

Never replace missing radio telemetry with zero. If a required measurement is
missing, stale, cross-epoch or unbound, fail closed and log the reason. If the
current live OAI collectors cannot supply a causal decision-time observation,
stop the phase and report the exact blocker rather than inventing one.
Decision-time policy state may use only measurements already observable at the
UE at that instant. It must never contain the requested network-profile label,
future SNR/profile trace, post-run gNB summaries, a future ACK, or any value
derived from later frames. Log profile actuation separately for evidence, not
as policy input.

Actor inference must be batch size 1, CPU float32, deterministic mode argmax
plus the selected mode's conditional-mean `q`. Record policy inference latency,
checkpoint/file hashes and actor-state hash.

## Required per-frame evidence

Every perception frame must retain:

- session, frame, tensor, decision, ticket and action-hold identities;
- whether it opened a decision or reused an action;
- `reward_requested`;
- all 31 state features plus raw SI/P40/SNR/MCS/BSR, timestamps and ages;
- proposed mode/q and exact executed mode/q_e4/action identity;
- policy inference time;
- payload bytes and datagram count;
- send, reassembly, edge receipt, tail, evaluation, ACK and timeout timestamps;
- separate reward-ACK and map/install-ACK timestamps and identities;
- reassembly/admission/timeout/supersession/late-orphan outcome;
- vehicle/person segmentation IoU;
- vehicle/person localization recall, XY error and footprint IoU;
- validity/status for every class metric; missing person support is null, never
  zero;
- `Q_seg`, `Q_loc`, `Q_perc` and exact realized reward breakdown;
- direct map publication/install timestamps and AoI as a separate diagnostic.

## Phase gates

### Phase 1 — audit and frozen-policy runtime (no live launch)

Create the additive package skeleton and implement:

- hash-verified Run3 checkpoint loader fixed to seed 17/update 10,000, including
  its campaign manifest, checkpoint digest, model configuration, state order,
  reward and environment/source bindings;
- batch-1 deterministic actor adapter;
- exact action identity output;
- 31-D state-builder interface with freshness/provenance validation;
- create-only evidence schemas and analyzer skeleton.

Use synthetic inputs to prove deterministic repeated output, exact feature
order, no identifiers leaking into policy tensors, no ambient RNG mutation and
no CUDA initialization. Stop and report.

### Phase 2 — causal telemetry and scene-state seams (no live launch)

Add wrappers for the already validated SI/P40 computation and a causal
decision-time OAI telemetry provider. If causal SNR/MCS/BSR cannot be obtained
from the current interface without changing pinned OAI code, stop with a
measured/line-specific blocker and propose the narrowest sidecar. Test missing,
stale and cross-epoch rejection. Prove that target profile labels, future trace
samples and post-run summaries cannot enter the policy tensor. Prove P40 uses
only the current 100 ms sweep. Stop and report.

### Phase 3 — exact continuous execution path (no live launch)

Wire `mode_id,q_e4` through an additive SFD1/UE/edge seam. With CPU-only
fixtures, prove all-anchor protocol/identity/keep-drop/serialization parity and
representative off-anchor identity plus deterministic codec cases. Do not call
this CUDA model-output parity or live runtime qualification. No CARLA or CUDA.
Stop and report.

### Phase 4 — ticket, action hold and early quality ACK (no live launch)

Connect `RewardTicketController` to synthetic frame dispatch and synthetic ACK
receive. Implement the 200 ms timeout, `k_min=2`, action reuse,
`reward_requested`, duplicate idempotence and late-orphan rules. Ticket creation
occurs at action commit, while its deadline remains `capture_time + 200 ms`.

Preserve quality-feedback v1 unchanged for prior evidence, but add a versioned
off-anchor-capable reward-ACK protocol. Its authenticated identity must contain
at least `session_uuid`, controller lineage, `decision_seq`, `ticket_id`, frame
and tensor identities, `mode_id`, exact `q_e4`, execution-bundle SHA-256,
`reward_requested`, and a nullable anchor `action_id`. It must never require or
fabricate an anchor/profile identity for an off-anchor action. Return quality,
capture-to-ACK latency and the exact terminal class. Prove anchor-semantic parity
with v1 where both apply.

Keep direct map publication and its install ACK independent from the reward ACK.
Test every terminal class, exact reward, queue saturation/evaluator fault,
reward-requested-frame supersession, duplicate ACK and late orphan. Then run a
composed CPU-only dry run connecting state builder -> frozen actor -> continuous
execution identity -> synthetic transport/edge -> independent fake map and
quality workers -> reward ACK/ticket closure. Prove exact terminal accounting,
no cross-frame reward attachment, no deadlock, and that reused-action frames
reach the fake map without entering evaluation. Stop and report.

### Phase 5 — launch and teardown plan only

Produce, but do not execute, guarded commands, resolved paths and preflight
gates for the two separately authorized phases below. Use fresh create-only
output directories and parent-owned teardown. Include live clock qualification,
checkpoint/source/hash binding, actor-state hash, CUDA/model preload, causal
telemetry freshness, radio restoration and final host-cold checks. Stop and
report. **Do not launch either run.**

### Phase 6 — 300-frame live qualification (separate authorization required)

Do not enter this phase without explicit Codex authorization after Phase 5
review. Launch exactly one 300-transmitted-frame qualification under
`FAVORABLE_STABLE`, using seed 17/update 10,000 and a fresh directory.

Hard acceptance gates:

- exact capture/terminal/ticket/reward-ACK/map-install accounting;
- no action, frame, ticket, controller-lineage or execution-bundle mismatch;
- correct nullable anchor identity for every exercised off-anchor action;
- same-clock capture-to-reward-ACK latency with `<= 200 ms` timely semantics;
- no record-bearing map data over the radio;
- map/install ACK only after install, with no ordering requirement between map
  install and the independent reward ACK;
- valid metric coverage, bounded reward-worker behavior and no cross-frame
  reward attachment;
- report all exercised modes/`q_e4` values and do not claim coverage of modes
  the deterministic policy did not select;
- clean teardown, radio restoration/read-back and host-cold proof.

Live CUDA/model parity is claimed only for the exercised dynamic actions and
only if the corresponding evidence passes. Stop after analysis and report. Do
not proceed automatically to Phase 7.

### Phase 7 — complete FADE_RECOVERY Route-B loop (new authorization required)

Do not enter this phase without a second explicit Codex authorization after all
Phase-6 gates pass. Run one complete chronological Route-B loop under
`FADE_RECOVERY`, using the same frozen seed-17/update-10,000 actor and a fresh
directory. Apply all Phase-6 accounting, identity, clock, ACK separation,
metric, teardown and host-cold gates over the full route.

Additionally prove from timestamps and dataflow that the actor consumed only
causal UE-visible OAI observations: the requested profile label and future
FADE_RECOVERY trace must never enter the actor state. Report chronological
per-frame action, q, latency, quality, terminal, ACK and independent map-install
behavior without presenting the privileged CARLA reward as a deployable
real-world oracle. Stop, analyze and report; do not launch another run.

## Reporting requirements at every stop

Report:

- starting/final HEAD and commit;
- files created/modified;
- why each change is additive and does not invalidate prior evidence;
- focused and full test results;
- any assumption disproved by implementation;
- exact unresolved blockers;
- confirmation that CARLA/OAI/Docker/CUDA were not launched and existing dirty
  paths were untouched.
