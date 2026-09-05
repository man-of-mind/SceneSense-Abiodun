# SplitFusion Phase-14B four-profile replay implementation

Status: `IMPLEMENTED_NOT_EXECUTED`

Phase 14B is a prospective qualification of the measured Phase-14A
`OAI_N78_100MHZ_273PRB_4D5U_V1` target-SNR mapping. It does not inherit the
old 40-MHz/106-PRB replay as evidence and it does not authorize the 16-cell or
288-cell campaign.

## Bound evidence

The runner refuses drift in the committed Phase-14A calibration terminal,
manifest, mapping, calibration summary, and anchor table. The principal hashes
are:

- mapping: `841ee69e53d7325570652a0f4baa7ae7f554a2204c4e775ce963a064cabd2677`
- calibration summary: `02ccdcfb22c0bfa8e16c213a41f935427196327a938d91a92610f3ba83ddb328`
- anchor table: `279027290439a2fce3600675b0057df6bcc36a65cf4ae8400cdb4ed148a0829b`
- calibration manifest: `5f14b43dedf7aa947598442ffd2823dea155476ad396b6665fc32e37ddb96091`
- calibration terminal: `2e2aae9578e174ba7bc6663596337b83fcfbede2c24416b2be31797c5ee7e99a`

The requested campaign-binding hash
`1708018018060f7fd2c69ed8fb1cb557126ca183fe01e593641997a39cdce043`
is the immutable Phase-14A implementation blob at commit
`c53224b8b95aaf983643832004113b07eafe6fb1`. Later launch/preflight repairs
changed only the launcher and calibration-runner hashes. Phase 14B verifies
that historical blob, verifies that its non-provenance content equals the
current binding, and separately seals the current topology-amended binding as
`ec6e11b2ca781ad6346a1f20f2f1f63d8686c34c270694ea4ee58ac72c868447`.
It also requires the three-process gNB/UE topology validator from commit
`c91c83dd3b69461abdb45a4f8249d9aba08667e8` and source hash
`09fb82c0ba644a44bdf5fb5e5ad92269c1daa31ffbeee1fc0864d5b7ce688c93`.

The locked profile design, accepted prefix, seeds, transition matrices, and
four semantic hashes are verified before any actuation. Each profile creates
one generator, caches samples 0 through 4,199, validates their state/FP64-byte
hash, and advances that same generator once to prove sample 4,200 continues
without wrapping, holding, or reseeding. A production campaign cell still
starts independently at sample zero and continues the same state after 4,199.

## Timing and causal observation rule

The OAI tracer CSV field named `time` is an event time-of-day, but the reusable
collector records a local wall and monotonic timestamp only when each line is
ingested. Phase-14A static anchors deliberately waited 3 seconds after command
acknowledgement and then measured a separate 5-second ingest window. The
successful 100-MHz capture contained 10,981 PUSCH records; the observed
event-to-collector delay was 0.694466 ms at the median, 1.280574 ms at P95,
1.777218 ms at P99, and 11.854475 ms maximum.

Before any dynamic replay, Phase 14B fixes the following rule for every command
and every profile:

- timestamps have `COLLECTOR_INGEST_MONOTONIC_NOT_RF_APPLICATION_TIMESTAMP`
  semantics;
- the window starts 15.0 ms after the validated command ACK, exceeding the
  maximum delay observed in the Phase-14A 100-MHz capture;
- the window ends at the earlier of the absolute 100-ms interval end and the
  next actual command send;
- a usable window requires at least three PUSCH samples and one scheduler
  sample from the single bound RNTI;
- empty and underpopulated windows are separate explicit outcomes and never
  contribute an achieved-SNR value.

The guard and window are configuration-locked. Post-hoc lag/window selection is
forbidden. These are ingest-time associations, so the implementation does not
claim an RF application timestamp or command-to-first-physical-effect latency.

Commands use absolute monotonic deadlines. The scheduler additionally enforces
at least one full 100-ms period between actual sends. An obsolete interval is
recorded as `SKIP_OBSOLETE_NEVER_BURST`; accumulated commands are never emitted
back-to-back.

## Preregistered acceptance

No compatible full-length 100-MHz four-profile contract existed. The prior
40-MHz pilot's 90% coverage and 1.5 dB MAE thresholds inform rationale only.
For each profile independently, Phase 14B requires:

- the exact frozen semantic trace hash and exactly 4,200 ordered targets;
- exact 0.25-dB command quantization through only the monotonic Phase-14A
  mapping, with no extrapolation beyond its measured 5.0-25.5 dB coverage;
- complete ACK-or-obsolete accounting, at least 99% command ACKs, and command
  ACK P95 no greater than 100 ms;
- no burst catch-up, wrapping, final-value hold, or reseeding;
- at least 90% accepted causal observation windows;
- target-versus-achieved MAE no greater than 1.5 dB, absolute bias no greater
  than 1.0 dB, and P95 absolute error no greater than 3.0 dB;
- target and achieved mean, population standard deviation, and Q05/Q25/Q50/
  Q75/Q95;
- one RNTI, unchanged scheduler mode (`mcs_table=0`, `force_ul_mcs=-1`), the
  locked radio identity, the qualified gNB/UE topology, and constant 10-Hz,
  25,000-byte controlled uplink traffic.

For `FADE_RECOVERY`, the frozen prefix must contain exactly 104 Markov-state
transitions and 345 steps of at least 3.0 dB. Each subset is reported and gated
at 80% observation coverage, 2.0 dB MAE, 1.5 dB absolute bias, and 4.0 dB P95
absolute error.

All four profiles must pass. Aggregate-only success is impossible. Final
restoration and read-back of `noise_power_dB=-50` and runner-owned process
cleanup are mandatory.

## Durability and execution boundary

An initial run creates a new leaf under
`experiments/splitfusion_phase14b_four_profile_replay_v1`. It writes an
immutable, hash-bound run manifest before the first replay command. Each
completed profile is one atomic, self-hashed JSON record containing all 4,200
command decisions, ACK timing, causal window status, and selected PUSCH/MCS
observations. A resume reuses only a complete, hash-valid, accepted record. A
crash leaves no completed record for the active profile; its scoped temporary
file is discarded and that profile must restart at sample zero. A rejected
complete profile is never silently replayed.

Tracer, traffic, and process logs live only in a temporary working directory
and are removed after owned processes stop. Finalization occurs only after four
accepted profile records and produces compact qualification JSON, profile CSV,
Markdown, an artifact manifest, and the success terminal. Attached OAI is not
stopped by this attached-state runner; no campaign is authorized by success.

The implementation accepts the live token
`SPLITFUSION_PHASE14B_FOUR_PROFILE_REPLAY`, an explicitly supplied create-only
output leaf, and an explicitly supplied qualified attached-radio state. No live
command was run while preparing this implementation.
