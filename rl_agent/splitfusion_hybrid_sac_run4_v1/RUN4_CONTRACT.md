# Run-4 minimal state, reward, and transition contract

Status: provisional Run-4 semi-empirical training contract. The state, reward,
hybrid actor/critics, replay buffer, trainer, causal decision-cycle kernel,
exploration gate, and persistent-runner mechanics are implemented and tested.
Production training remains fail-closed until the exact 273-PRB calibration,
state scaling/freshness limits, fitted queue/latency model, composite verifier,
and smoke diagnostic panel are measured, accepted, and hash-bound. Nothing in
this document claims live-deployment qualification.

## Actor state: exactly 21 values

The order is immutable:

1. `camera_si_scaled`
2. `radar_p40`
3. `prior_ul_mcs_normalized`
4. `pre_action_rlc_backlog_log1p_scaled`
5. twelve previous-joint-mode one-hot values, mode 0 through mode 11
6. `prev_q_normalized`
7. `prev_quality_qperc`
8. `prev_latency_normalized`
9. `prev_present`
10. `prev_success`

Camera SI and `log1p(backlog)` scaling parameters have no production defaults.
A versioned `EmpiricalScalingV2`, including its evidence hash, must be supplied
before vectorization. `radar_p40` is already the frozen unitless `[0,1]`
proximity descriptor. The prior UL MCS is normalized over the registered table-0
wire domain `[0,28]`. Previous `q` is normalized by the frozen wire maximum,
and successful previous latency is normalized by the 170 ms reward deadline.

Only `decision_seq=0` is episode genesis and only that state may omit a previous
outcome; all previous-decision fields are then zero and `prev_present=0`. Every
later sequence must carry the immediately preceding resolved outcome. After
success, the exact previous action, quality, and latency
are required and `prev_present=prev_success=1`. After a registered failure or
timeout, the previous action remains required, quality and latency must be
absent, and only those two vector positions become numeric zero under the
explicit `prev_present=1, prev_success=0` mask.

There is deliberately no `prev_reward` feature. For success it is exactly
recoverable as

```text
prev_quality_qperc - 0.25 * prev_latency_normalized
```

and for a present registered failure/timeout it is exactly `-1`. Adding the
same value again would be redundant.

TBS, grants other than the registered prior-MCS value, network profile, gNB
PUSCH SNR, measurement age, validity, timestamps, frame identifiers, FPS, and
current/future outcomes are not actor features.

## External causal guard

Every camera, radar, UE-MCS, and RLC observation carries typed sample identity,
source, source timestamp, availability timestamp, clock domain, semantic kind,
observer, link direction, and validity. A versioned, evidence-hashed freshness
policy is also mandatory. Its production limits remain unresolved until they
are empirically registered; this module supplies none.

Before actor invocation, the external guard proves

```text
source timestamp <= availability timestamp <= state commit < action open
```

on one clock, checks same-session/same-UE identity, checks freshness, and checks
the expected measurement semantics. In particular:

- camera SI and radar P40 must carry the exact same current-scene
  `SampleIdentity`; UE MCS and RLC samples may remain asynchronous;
- `prior_ul_mcs` must be the latest strictly prior, UE-decoded, round-0/new-data
  granted/final UL DCI MCS under table 0 and the registered SINR-driven
  scheduler. It is the value actually decoded by the UE after scheduler-side
  constraints, not a fabricated UE SNR or an intermediate lookup value; the typed
  grant evidence hash includes table, HARQ round, NDI, grant identity,
  scheduler policy, and the strict-prior selection rule;
- gNB-received uplink PUSCH SNR, UE downlink SNR, and retransmission-grant MCS
  cannot occupy that slot;
- RLC backlog must be the UE's pre-action uplink queue value;
- missing or stale MCS/backlog raises `ExternalFallbackRequired`;
- missing values are never zero-filled. A valid measured backlog of zero remains
  distinguishable from a missing backlog.

Metadata remains outside the actor vector. It controls whether the actor may be
called at all.

## Reward

The sole learning-reward clock is action-open to feedback. Its deadline is
exactly 170.0 ms, inclusive:

```text
success, latency <= 170 ms:
    reward = q_perc - 0.25 * (latency_ms / 170.0)

registered delivery failure, registered service failure, or timeout:
    reward = -1

infrastructure fault or evaluator fault:
    excluded; no learning transition
```

A delivered result at 170.001 ms is a timeout; one at exactly 170.000 ms is a
success. There is no admission-probability, switch, payload, or aggression term.

Every successful row must carry a direct, same-frame
`action_open_to_feedback_ns` measurement, its own evidence digest, and fitted
numeric support. This authoritative total alone drives the reward and the
inclusive 170 ms deadline. It is never reconstructed from component
percentiles or independently sampled stages.

The following exact six-stage decomposition is optional diagnostic evidence:

1. `ue_action_path_ns`: action open to first feature-datagram send. It includes
   7-channel concatenation, front inference, AE/quantization, compression, and
   datagram construction; sensor callback/preparation precedes the reward clock.
2. `feature_uplink_ns`: first feature-datagram send to complete edge reassembly.
   It includes the remaining UE send loop, radio path, and reassembly.
3. `edge_pre_model_ns`: complete edge reassembly to model dispatch. It includes
   latest-only scheduler wait, decompression, unpack/dequantization, AE decode,
   and input reconstruction.
4. `model_tail_ns`: model dispatch to model-ready completion.
5. `post_model_feedback_preparation_ns`: model ready to the actual compact
   feedback socket-send call. It includes required post-processing, p025
   filtering, serialization, evaluator/ground-truth wait and scoring, feedback
   encoding, and local send preparation.
6. `feedback_downlink_ns`: feedback socket-send call to UE receipt.

When that diagnostic exists, all six nonnegative components, all six separate
supports, and all six evidence digests must exist together. The registered
boundary order makes the components contiguous, and their exact integer sum
must equal the authoritative total. A missing boundary makes only the
decomposition absent; it does not invalidate an independently bound total.
Absence is represented as absent/`None`, never as six fabricated zeros. Map
service follows the feedback boundary and is outside this reward clock.

Durable sequential-kernel v2 latency records are incompatible with v3 because
they lack the mandatory total and total-evidence fields. Legacy
`edge_decompression_ns` and `quality_evaluation_ns` records are also
incompatible because those labels described only sub-operations, not the
complete registered intervals.

The approximately 200 ms sensor-start-to-feedback system path is a separately
logged diagnostic. It is not substituted for the 170 ms action-open reward
clock and is not an actor feature.

## One training step is one feedback-gated decision cycle

A training transition is not a fixed image/FPS step. It starts when one policy
decision opens and ends when that decision obtains feedback or times out. The
advance is virtual and semi-Markov: training performs no literal wall-clock
sleep.

The chosen action governs the reward-requested tensor and every held tensor
until closure. A closed hold contains at least two transmitted tensors under
the nominal 10 Hz transmission contract. Only the earliest tensor requests a
reward. Every later held tensor reuses the action, requests no reward, and its
offered payload is still counted in total queue input. Each tensor labels that
payload and binds its provenance as either an exact measured action node or a
same-scene modeled interpolation. The latter remains a finite float when the
provider returns one: the contract neither rounds it nor calls it measured.
The contract does not synthesize queue service.

The exact positive elapsed virtual time is bound to

```text
cycle_end_timestamp - current_action_open_timestamp
```

For a continuing transition, `cycle_end_timestamp` must equal the next real
decision's action-open timestamp on the same clock. It cannot precede the
current reward closure. A hold of duration `d` occupies `d` complete nominal
transmission periods, so the successor action cannot open earlier than
`current_action_open + d * 100 ms`. This is a semi-Markov decision-cycle timing
binding, not a claim that scene frames are contiguous.

The caller supplies `gamma`, the exact duration, and the stored discount. The
contract admits the transition only when

```text
discount == gamma ** duration
```

The successor must be the next real decision in the same session and UE, and
its previous-outcome record must exactly bind the current action and reward
resolution. Scene sample identifiers are not required to be numerically
adjacent. This avoids inventing temporal continuity in a semi-empirical data
source while preserving genuine decision/outcome succession.

Episode boundaries are explicit:

- `CONTINUES` requires that real successor and retains `gamma ** duration` as
  the bootstrap discount;
- `TERMINATED` and `TRUNCATED` require no successor and force the bootstrap
  discount to zero;
- a radio/service timeout is a reward outcome, not an episode boundary, and can
  therefore remain `CONTINUES`.

This prevents the Run-3 failure mode in which every decision was relabelled as
terminal while still allowing a trace end to stop bootstrapping honestly.

## Integrity and scope

Feature, reward, transition, and aggregate contract descriptors each have a
schema ID, version, and canonical SHA-256. Scaling and freshness specifications,
guarded states, reward resolutions, feature vectors, and transitions carry
their relevant bindings. Derived records fail closed unless produced by their
validating factory.

Importing `run4_contract.py` performs no filesystem access, RNG operation,
network access, process launch, CARLA/OAI/CUDA call, or runtime mutation.

The adversarial tests in `test_run4_contract.py` cover feature order/count,
forbidden leakage, causal timestamps, freshness and missingness, link-direction
separation, pre-action backlog ordering, previous-result masks, reward boundary
values, excluded faults, minimum hold duration, held payload accounting,
semi-Markov discounting, non-adjacent scene samples, and import purity.
