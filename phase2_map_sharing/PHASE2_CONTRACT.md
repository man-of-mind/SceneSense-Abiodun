# Phase-2 recipient-specific map-sharing contract

Status: v1 implementation/scaffold contract, 2026-08-14. The local core and
synthetic acceptance fixture may run offline. It is **not** the collection or
deployable safety contract: schema v1 lacks covariance/process-noise semantics,
and the accepted v5 corpus lacks causal paired observations. CARLA/OAI work is
gated by `PHASE2_PAIRED_CAUSAL_CORPUS_SPEC.md` and its reviewed pilot.

## Claim and smallest sound path

One helper/source publishes object evidence to one named recipient ego. The
recipient map associates the evidence without using CARLA actor IDs, rejects
stale/out-of-order contributions, retains source/capture/publish provenance,
and emits an explainable collision warning. The first C2 comparison is paired:

1. ego-only evidence;
2. periodic/send-everything helper sharing;
3. compact hazard-only sharing.

The key outcome is **marginal actionable warning lead**:

`lead_gain_s = first_warning_time_ego_only - first_warning_time_cooperative`

A helper earns cooperation credit only if a delivered contribution advances a
recipient warning on the same truth trajectory. Also report false-warning rate,
missed-hazard rate, stale-update rejection, bytes per advanced warning, map AoI
at warning, and warning provenance. Localization alone is supporting evidence,
not the C2 endpoint.

## Runtime/evaluation separation

Runtime payloads contain source-local track hints, class, world pose/velocity,
confidence, occlusion state, a transparent recipient-hazard score, timestamps,
profile/payload provenance, and a **recipient ID**.
CARLA actor IDs and future truth are forbidden from the runtime envelope. They
may exist only in a separate evaluation stream for scoring.

The initial association baseline is class-consistent nearest predicted XY with
one-to-one gating and persistent canonical track IDs. It is intentionally
simple and must be compared with explicit target-ID/oracle association only as
an evaluation ceiling, never used to generate the deployable warning.

The hazard-only baseline is causal **as a post-inference publication rule**: it predicts closest approach from
the helper's causal object estimates and the named recipient's current state.
It does not select with CARLA truth or a future collision label. The unit of
cooperation is therefore an **intent-conditioned map contribution**—evidence
useful to a particular recipient—rather than a globally salient object.

This publication choice must not be conflated with the **pre-inference placement**
choice (`LOCAL_INFER`, `SPLIT_FEATURE`, or `SKIP_INFERENCE`). A local result may
subsequently use `PUBLISH_ALL`, `PUBLISH_HAZARD_SUBSET`, or `SKIP_PUBLICATION`;
choosing placement after seeing that result would already have paid its compute cost.

## Warning contract

For each current track, predict constant-velocity closest approach to the
recipient over a configured horizon. Warn only when:

- time to closest approach is within the horizon;
- closest-approach distance is below the class-specific safety radius;
- the observation/map age is within TTL; and
- confidence clears the declared floor.

Every warning records canonical track ID, class, recipient, first-warning time,
TTC, closest-approach distance, evidence sources, capture/publish timestamps,
map AoI, and whether evidence was ego-only, helper-only, or multi-source.
This is a deterministic warning baseline, not a learned driving policy.

## Freshness and ordering invariants

- `published_at_s >= captured_at_s` and `received_at_s >= published_at_s`.
- Sequence numbers are strictly increasing per `(source, recipient)`.
- Older capture timestamps cannot overwrite a newer track state.
- Contributions addressed to another recipient cannot enter the map.
- Tracks expire after TTL; expired evidence cannot cause warnings.
- Source time, map-install time, and warning time remain separate in logs.

## Schema-v2 requirement before a pilot

`scenesense.map_contribution.v1` remains immutable for its checked-in plumbing
artifacts. Before live collection, define and review `v2` with per-object
position/velocity covariance, object measurement time, motion-model ID,
process-noise parameters/validity horizon, and the complete inference/publication
provenance and timestamp chain. The recipient must propagate uncertainty under
the declared motion model. A configured uncertainty floor is not a calibrated
safety model. The complete required fields and gates are canonical in
`PHASE2_PAIRED_CAUSAL_CORPUS_SPEC.md`.

## Step 3 — canonical local path after the pilot design gate

1. Implement the reviewed v2 schema and causal state allowlist without changing v1.
2. Run only the two-trajectory pilot: one controlled positive occlusion/hazard
   and one matched benign negative.
3. Prove pre-action availability, unfiltered detections, causal source-local
   tracking, raw-sensor alignment, separate truth, and C2 computability.
4. Stop for human review. Only a reviewed PASS authorizes the full paired
   designed and naturalistic suites.

The checked-in synthetic fixture validates plumbing only; it is not C2 evidence.
Its application-byte counts are exact canonical JSON lengths, with UDP/IP chunk
overhead reported separately; no illustrative payload sizes are substituted.
The adapter also passes the existing recorded two-stream snapshots (37 eligible,
26 accepted, 11 correctly rejected above the 1 s age gate; 106 pedestrian and
91 vehicle observations). Those recordings have no synchronized hazard truth,
so they validate integration but cannot estimate warning lead.

## Step 4 — two-UE OAI RFsim insertion

Keep the contribution JSON identical. Reuse the existing chunked UDP transport
and route helper traffic through its discovered UE tunnel address; do not create
a second wire protocol. The local codec already uses the production `!IHH`
header and validates out-of-order reassembly; live tunnel routing and receiver
integration remain Step 4. Log at source, OAI sender, map receiver, map install,
and recipient warning:

- contribution/sequence/recipient IDs;
- capture, enqueue, first-byte, last-byte/reassembly, install, and warning time;
- application bytes/chunks, tunnel interface/IP, QFI/DRB if visible;
- gNB/UE MCS, PRB/TBS, HARQ/RLC/PDCP counters over the same causal window.

Acceptance requires identical local/OAI contribution semantics, both tunnels
stable, no cross-recipient leakage, and a decomposable helper-capture-to-warning
latency. RFsim transport success alone is not C2.

## Step 5 — warning/override evaluation

Start with warning only. Compare ego-only, send-everything, and hazard-only over
the separately reported pre-registered designed-opportunity and naturalistic
paired suites. Pre-register lead-time, false-warning, missed-hazard, payload,
uncertainty, and stale-map gates before CARLA runs.
Only after warning correctness is established may an override apply braking;
override evaluation then adds avoided collision/near-miss, minimum clearance,
braking latency, and unnecessary-intervention rate.

## Learning gate

No map-sharing RL is authorized by this scaffold. First establish C2 on the
causal paired corpus, then add periodic,
send-everything, hazard-only, deadline-aware, and object-priority baselines. A
genuine Whittle baseline is considered only after objects are demonstrated to
be meaningful scheduling arms and indexability is analyzed. DQN/MARL is opened
only by a pre-registered residual sequential gap.
