# Run-4 target-radio dynamic prior-MCS qualification

## Why this bounded capture is necessary

The previous dynamic UE-decoded MCS evidence was measured on 40 MHz / 106 PRB
/ 7D2U. Run 4 and the 288-cell action surface use
`OAI_N78_100MHZ_273PRB_4D5U_V1`. TDD cadence and bandwidth change grant timing,
so the legacy sequence may demonstrate mechanics but cannot calibrate the
Run-4 transition kernel. The corrected near-capacity experiment covers stable
favorable/adverse conditions; it does not supply the dynamic ordering needed
from `MID_VARIABLE` and `FADE_RECOVERY`.

This package captures only that missing evidence. It does not launch CARLA,
CUDA, an edge model, a map service, or another 288-cell sweep.

## Frozen design

- Radio: n78, 100 MHz, 273 PRB, numerology 1, 4D5U, one UE, 5QI 6,
  `SCENESENSE_MCS_POLICY=sinr`.
- Profiles: the first 300 registered 100-ms samples of `MID_VARIABLE` and
  `FADE_RECOVERY`, with the original trace IDs, seeds and hashes.
- Traffic: action 68's measured median payload, 129,707 bytes (three production
  SSBURST chunks), at 10 Hz. At 10.37656 Mbps this is an observability probe,
  not an action label and not a capacity experiment.
- Grid: traffic action-open and RF replay share one future monotonic boundary.
  A changed RF command is due 10 ms before its action-open boundary and must be
  acknowledged before that boundary. Obsolete commands are never burst.
- Fit: decision indices 0-209. Internal validation: 210-299. The boundary is an
  explicit episode reset.
- Semi-Markov compatibility: duration two selects index `t+2`, exactly 200 ms
  later. Transitions 208/209 and 298/299 are reset/terminal rather than crossing
  a partition or profile boundary.

## Causal MCS rule

The actor-visible value is the UE-decoded UL DCI MCS from the latest event
strictly before the canonical action-open boundary, restricted to uplink,
table 0 and HARQ round 0. NDI may be either 0 or 1: new data is the round-0 NDI
toggle, not the literal value one. The gNB trace is verifier-only.

Every row retains the source grant timestamp, schedule identity, NDI, age and
provenance digest. No prior event is `MISSING`; an event older than 200 ms is
`STALE`. Both carry a null policy value. MCS zero remains a valid measured MCS
and is never confused with missingness. No missing/stale value is imputed or
forward-filled. A still-fresh grant may legitimately be selected by adjacent
decisions, but its unchanged identity and increasing age remain explicit.

Target SNR, profile identity, frame/decision ID, timestamps and all gNB values
are verifier metadata. Only `{"prior_ul_mcs_index": n}` from a `VALID` row may
enter the actor state.

## Gates fixed before collection

- exact 300-row action grid and profile schedule per profile;
- no sender schedule miss, socket drop, RF clamp, RF skip or late RF command;
- same-event PDCP wall/monotonic bridge P95 residual at most 1 microsecond;
- at least 90% valid MCS observations in each fit/validation partition;
- at least three distinct valid MCS values per profile;
- at least 99% UE-to-gNB grant provenance coverage and zero UE/final-MCS
  mismatch;
- exact 200-ms duration-two successors, never across a reset;
- radio route proven through UE PDCP and the non-host-local ext-DN;
- RFsim restored/read back at -50 dB and the host returned application-cold.

Failure preserves the create-only attempt and authorizes no splice or partial
reuse. One measured realization per profile is sufficient for initial kernel
fit/internal validation only; external Route-B live validation remains
required before deployment claims.

## Commands

Offline, no external process:

```bash
/usr/bin/python3 -m rl_agent.ue_dynamic_mcs_273prb_v1.runner --validate-only
```

Live, only after the corrected near-capacity run has fully torn down and an
operator explicitly authorizes this network-only capture:

```bash
/usr/bin/python3 -m \
  rl_agent.ue_dynamic_mcs_273prb_v1.runner \
  --execute AUTHORIZE_RUN4_DYNAMIC_MCS_273PRB_CAPTURE_V1 \
  --output rl_agent/experiments/ue_dynamic_mcs_273prb_v1/<NEW_RUN_ID>
```

The scientific replay itself is exactly 60 seconds (2 x 300 x 100 ms), plus
8 seconds of registered warmup, 4 seconds of sender lead and 10 seconds of
receiver tail. Two fresh RAN attach/teardown cycles and T-tracer extraction are
host-dependent; expected wall time is 8-12 minutes, with a 30-minute outer
operator expectation. Do not wrap the command in GNU `timeout`: an external
SIGTERM could pre-empt the runner's RF/core cleanup. The launcher owns temporary
CN5G startup; the runner requires an initially cold CN/RAN and removes the core
again at the end.
