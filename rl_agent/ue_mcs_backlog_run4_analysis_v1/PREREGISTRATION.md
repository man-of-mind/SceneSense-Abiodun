# Run-4 UE queue/service analysis preregistration

Status: **frozen before live collection**

Capture authority: commit `d7d50b8df0cb723c1298eea5cce5addc85d280f2`

Claim boundary: **offline physical UE queue/service calibration only**.

This package analyzes a future, create-only 12-cell capture. It does not
authorize that capture, inspect an outcome, train a policy, endorse an action,
or turn 1,200-byte calibration packetization into production latency evidence.

## 1. Units and causal transition

The capture emits at 10 Hz, but the controller acts at 5 Hz. The primary row
is therefore a 200-ms cycle:

```text
decision t (even index) -> held frame t+1 -> successor state t+2
```

The held frame must reuse the exact mode and q. Both frames' exact byte loads
enter the queue recurrence. A 450-frame cell yields 224 closed cycles; the
last even decision has no in-cell successor and is reported as `UNCLOSED`, not
zero-filled. Across 12 cells the expected population is 5,400 raw terminals
and 2,688 closed cycles (1,344 FIT, 1,344 VALIDATION).

Service is measured in the UE queue domain:

- `S0`: `NR_RLC_TX_DEQUEUE` bytes in `[decision_t, decision_t+1)`;
- `S1`: dequeue bytes in `[decision_t+1, decision_t+2)`;
- ingress is reconstructed from same-event PDCP/RLC evidence, not catalog
  application bytes;
- pre-action backlog and prior new-data round-0/table-0 UE MCS must be strictly
  earlier than `decision_t` and no older than 100 ms;
- the successor backlog is the last valid pre-enqueue RLC sample strictly
  before `decision_t+2`.

All joins use the same-event `NR_PDCP_TX_SDU` wall/monotonic bridge. Residual
P95 above 1 microsecond refuses the cell. CSV schema/row-width errors,
ambiguous identity, wrong UE/RNTI/bearer, negative intervals, and unexplained
byte residuals are failures, never silently dropped rows.

## 2. Inputs, targets and hidden information

The only model inputs are:

1. pre-action RLC backlog bytes;
2. prior UE new-data round-0/table-0 UL MCS;
3. decision-frame bytes;
4. held-frame bytes.

`profile_id`, target SNR, gNB MCS and the MCS observed after the decision are
audit-only. In particular, the saved network-profile label is never a feature,
fit key or back-off key.

FIT retains an empirical conditional 100-ms service process and its joint
two-step atoms. The registered bins and back-off order are fixed in
`contract.py`. Whole cells are FIT or VALIDATION; validation cannot choose
bins, back-off, monotone projection, scaling, or a coupling alternative.

Each empirical atom carries `S0`, `S1`, next backlog, a FIFO byte-cohort
clearance offset (or explicit non-clearance), and an integer weight. Sampling
requires an externally supplied exact integer draw; no global RNG is used.

## 3. Byte domains

The live calibration uses a 1,200-byte payload plus a 24-byte SSBURST header.
Those bytes qualify the UE queue/service process only. They do not qualify
production delivery, reassembly or transport latency.

Production actions use the frozen `!IHH` contract:

```text
n_datagrams = ceil(total_transmitted_bytes / 12492)
udp_application_bytes = total_transmitted_bytes + 8 * n_datagrams
```

Thus 12,492 bytes maps to 12,500 bytes and 12,493 maps to 12,509 bytes. The
three production authority sources and this boundary behavior are pinned in
`contract.py`. Any residual IP/UDP/RLC byte difference is measured and
reported; calibration-specific chunk overhead is never learned as if it were
an action effect.

For byte-domain auditing, adding the ordinary 8-byte UDP and 20-byte IPv4
headers makes a full production packet 12,528 bytes before any fragmentation,
well above the registered 1,500-byte path MTU. In contrast, a full calibration
packet is 1,252 bytes and MTU-safe. Therefore equal `payload_bytes` values do
**not** establish equal PDCP/RLC ingress. Calling them equal is a blocker.

The queue model can be fitted in directly observed RLC bytes, but a production
prediction is refused until a sealed `ByteDomainProofV1` reconciles, per
decision, sender -> PDCP -> RLC bytes in both packetizations; observes rather
than assumes production fragmentation; publishes the conversion/residual
support; and proves that packetization-specific delivery and latency were not
projected into the reward.

Every prediction states:

- `CALIBRATION_QUEUE_SERVICE_DYNAMICS_ONLY`;
- `SINGLE_UE_RADIO_CONFIGURATION_ONLY`;
- `PROFILE_TRANSFER_UNVALIDATED`;
- `PACKETIZATION_TRANSFER_NOT_USED_FOR_LATENCY`;
- `MODE_TRANSFER_UNVALIDATED` for modes other than measured AE64/UINT8;
- `PAYLOAD_INTERPOLATION_UNVALIDATED` for in-range off-anchor bytes.

Payloads outside the measured pair-byte range are refused rather than
extrapolated.

## 4. Frozen FIT transformations

`backlog_log1p_scale` is the nearest-rank P99 of
`log1p(pre_action_backlog_bytes)` over accepted **FIT whole-cell primary
cycles only**. It must be finite and positive. The artifact publishes the
population identities/count, one-based rank, raw-population digest,
ordered-value digest and scale. Values are not clipped; normalized values
above 1 remain visible. Validation never influences this derivation.

The external freshness policy is one tensor period (100 ms) for camera SI,
radar P40, prior UE MCS and pre-action RLC backlog. Missing/stale means an
external fallback, never zero-fill. Calibration can measure only MCS/backlog
coverage; SI/P40 are explicitly `NOT_PRESENT_IN_NETWORK_ONLY_CALIBRATION`.
Coverage is reported at the frozen sensitivity grid 50/75/100/125/150/200 ms,
but the operational 100-ms bound is not selected or changed from validation.

## 5. Validation gates

The frozen numerical gates are:

1. 12 sealed cells, 5,400 exact raw terminals and 2,688 registered cycles;
2. 100% causal UE-MCS coverage at 100 ms, >=99% unique gNB provenance in every
   cell, zero ambiguity and zero UE/gNB final-MCS mismatch;
3. queue-ceiling censoring below 1%;
4. next-backlog NMAE <=10% and >=20% improvement over persistence;
5. per-cell RLC cohort-clearance P50 error <=17 ms and P95 error <=34 ms;
6. queue-clearance-within-170-ms false success <=5% and Brier <=0.15;
7. adding MCS worsens Brier by no more than 0.01 and higher MCS has the correct
   physical direction;
8. zero monotonicity violations for predicted clearance probability/latency
   inside measured support.

Gates 5/6 concern **UE RLC cohort clearance**, not complete feature delivery
or end-to-end feedback. A queue clearing inside 170 ms is necessary but not
sufficient for the full action to meet 170 ms.

## 6. Composite coupling decision

Two alternatives, and only these, are admissible:

**A — `STATE_TRANSITION_ONLY`.** The model updates the next backlog. It is not
added to the 288 feature-uplink total. This is scientifically usable for queue
dynamics, but it leaves the state-to-reward latency path unresolved and
therefore **blocks Run-4 training readiness**.

**B — `RESIDUALIZED_SERVICE_TO_LATENCY`.** The service process determines
whether/when the decision byte cohort clears. It may affect reward latency only
after a separate exact proof identifies the overlapping segment in the 288
uplink component, replaces that segment, and joins only a nonnegative disjoint
residual. The proof must also preserve the production delivery population and
bind production packetization and row identities. Blindly adding service time
to the inclusive 288 total is forbidden.

The eventual report seals the raw manifest digest, canonical d=2 row digest,
FIT/VALIDATION cell and row digests, fitted service-table digest, backlog-scale
and freshness-policy digests, every gate result, coupling decision/proof, and
an independent re-parser/verifier result. Analysis outputs are create-only and
outside the immutable raw-evidence tree; JSON/CSV are used, never pickle.
