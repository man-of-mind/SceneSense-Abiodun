# Phase-6 addendum 11: 300-frame timing characterization (prospective)

- Base commit: `cc8ad99`.
- Implementation: v10, unchanged.
- Claim scope: SYSTEMS_INTEGRATION_QUALIFICATION_ONLY.
- Classification: **TIMING_CHARACTERIZATION_NOT_QUALIFICATION**. This is not a
  policy-performance PASS and not a deployment qualification.
- Registered before any new Phase-6 evidence was collected. The
  machine-readable version is `phase6_timing_characterization_addendum_11.json`.
  It binds all 167 v8, v9 and v10 evidence files by SHA-256.

## Unchanged

The following are the same as in v10:

- the frozen Run-4 actor;
- the 21-D state;
- the reward formula;
- the 170-ms action-open timeout;
- GT semantics and the optimized 40-m builder;
- transport, map and feedback identities;
- k_min = 2, the fallback and the scheduler.

## Added instrumentation

One record-only CARLA trace was added: the instant of every world tick and
every RGB receipt, on the raw clock. It gives the CARLA side of the
GPU-overlap measurement and does not change behaviour.

## Two independent measurements

1. **Frozen policy deadline**: UE feedback receipt minus action-open must be at
   most 170 ms. This is the contract that decides reward and timeout.
2. **System KPI**: UE feedback receipt minus sensor capture must be at most
   200 ms. This is report-only. It never changes a reward, a timeout or a gate.

A reward ticket with no feedback counts as a measured miss for both. It is
never censored.

A third report checks whether each reward arrived before the **actual next
eligible decision**. That decision is the plan instant of the first frame
planned after the ticket sent k_min = 2 tensors, taken from the recording. No
sensor period is assumed.

## What is reported

For every sent frame, uncensored:

- capture to action-open (for hold frames, capture to the plan instant);
- action-open to the first and last datagram;
- uplink completion;
- edge queue and edge compute;
- map installation;
- GT ready;
- evaluator time;
- feedback emission and UE receipt;
- the next eligible decision instant;
- mode, q_e4, payload and action-reuse count;
- GPU overlap between the UE front, the edge tail and CARLA rendering.

Latencies are summarized as P50/P90/P95/P99/max (nearest-rank). Counts and
fractions are given for:

- feedback within 170 ms of action-open;
- feedback within 200 ms of capture;
- feedback before the next eligible decision;
- map installation latency;
- timeouts, late feedback and exclusions.

## Hard integration requirements

Timing is characterized, not gated. Timeouts are measured outcomes and never
abort the run. The integration requirements are hard:

- no identity mismatch and no reward attached to the wrong frame;
- no duplicate or conflicting feedback;
- no deadlock;
- correct map installation and supersession handling;
- exact terminal accounting.

## Run

- One run with a transmitted budget of 300 and no stop-after-decisions.
- No retry. The timeout is not changed and nothing is retrained.
- Afterwards, preserve and hash the evidence, then stop for review.
