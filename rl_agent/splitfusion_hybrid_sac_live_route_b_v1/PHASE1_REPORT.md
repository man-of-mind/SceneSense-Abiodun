# Phase 1 — audit and frozen-policy runtime

`GENESIS_MATCHED_FROZEN_POLICY_LIVE_PILOT_NOT_SEQUENTIAL_ONLINE_ADAPTATION`

Status: **complete, stopped for Codex review.** No live launch. Nothing in
Phases 2–7 is implemented.

## Scope delivered

A new additive package, `rl_agent/splitfusion_hybrid_sac_live_route_b_v1/`,
containing the frozen-policy runtime and the seams the later phases fill:

| Module | Responsibility |
| --- | --- |
| `pilot_contract.py` | Frozen labels, artifact pins, the Run-3 training-support descriptor, canonical serialization. |
| `checkpoint_loader.py` | Walks the campaign provenance chain and returns read-only seed-17 / update-10,000 actor weights. |
| `frozen_actor.py` | Batch-1, CPU, float32, deterministic inference. No optimizer, no gradient, no RNG consumption. |
| `execution_identity.py` | Exact `(mode_id, q_e4)` identity reconciled against the frozen 72-action catalog; nullable anchor identity. |
| `state_builder.py` | Genesis-matched 31-D observation with registered Run-3 preprocessing and a per-frame training-support audit. |
| `evidence.py` | Create-only session/frame evidence schema and writer. |
| `analyzer.py` | Read-only gate evaluation with an explicit `NOT_EVALUABLE` verdict. |
| `synthetic_fixtures.py` | Test-only, explicitly labelled synthetic observations. |

Every SHA-pinned production module is **imported unchanged**. Nothing under
`splitfusion_hybrid_sac_v1/`, `splitfusion_live_dispatch_v1/`,
`splitfusion_direct_edge_map_v1/`, `splitfusion_quality_feedback_probe_v1/` or
`ue_route_b_split_cell_adapter_v1.py` was modified.

## Verified artifact chain

`load_pilot_actor_weights()` verifies, in order:

```
campaign_complete.json : seed_reports["17"]            = 378bc07b…
RUN3_TRAINING_COMPLETE.json : terminal_sha256, status  = RUN3_FIXED_ENDPOINT_COMPLETE @ 10000
report.json : report_sha256 recomputed                 = 378bc07b…
report.json : artifact_hashes re-hashed                = 27 / 27 files
module pin  : model_010000.pt file digest              = 7ea8e2ca…
snapshot    : snapshot_sha256 via the runner's _hash_state = 313695a2…
full checkpoint : bitwise actor-tensor equality         = 13 / 13 tensors
```

The snapshot is read with `weights_only=True`, so no object pickled by the
training process is reconstructed in the live pilot. The 131 MB full checkpoint
is read only to corroborate the snapshot and is never the weight source.

Resulting identities: `actor_state_sha256 = a4ae74e0…`,
`runner_binding_sha256 = 6e82a29f…`, decision lineage session
`6dc674b2-4fd5-5720-84b4-e59ffe1330a1`. `PILOT_CONTRACT_SHA256 = 3c649c2d…`.

## The finding that matters: 27 of 31 features were constant in training

This was not in the brief's assumptions and it changes how Phase 2 must be
read. Run-3 collected **independent one-step genesis observations**, and the
registered environment enforces three invariants that together freeze most of
the observation:

* `empirical_contextual_environment.reset` raises `D1 genesis state has
  non-zero age` unless **all four measurement ages are exactly zero**, so the
  four normalized freshness features were always `0.0`;
* `empirical_radio_context` samples every row with `bsr_bytes=0` under
  `GENESIS_BSR_JUSTIFICATION`, so `radio_bsr_log1p_scaled` was always `0.0`;
* every state carried `previous=None`, so all 22 previous-outcome features were
  `0.0`.

**Only four features ever varied**: `scene_camera_si_scaled`,
`scene_radar_p40`, `radio_achieved_snr_db_scaled`, `radio_mcs_index_scaled`.

A live frame cannot reproduce that. A real measurement age is not zero and a
real uplink buffer is not always empty. Zeroing them to "stay in distribution"
would be fabricated telemetry, so the pilot does the opposite: it feeds the
causal value and records a per-feature `TrainingSupportAuditV1` on every frame.
The 22 previous-outcome features remain the pilot's own responsibility and are
hard-failed if non-zero.

A concrete consequence, pinned by a test: the registered
`bsr_log1p_scale = 1.0` means `log1p(bytes)` clipped to `[0, 1]` **saturates
above ≈1.72 bytes**. Any non-empty live uplink buffer — 2 bytes or 1 MB — maps
to exactly `1.0`. The registered spec's own provenance already says
`bsr_scale_status: INERT_FOR_EXACT_GENESIS_ZERO_ONLY`; this makes the live
consequence explicit rather than surprising. The BSR feature is effectively a
step function in live operation, not a graded one.

## Actor behaviour on a genesis-matched synthetic grid

375 deterministic decisions over a 5 × 5 × 5 × 3 grid of the four varying
features, with the other 27 held at zero:

* discrete head selects **3 of 12 modes** — mode 11 (279), mode 8 (87),
  mode 9 (9). Not full collapse, but far from uniform.
* continuous head spans `q_e4 ∈ [4385, 8736]`, a span of 4351 — the policy is
  genuinely contextual in `q`.
* every selected `q_e4` lay inside its mode's registered `MODELED_SMOKE_SUPPORT`
  interval.

This is a synthetic-input audit, not a performance claim.

## Tests

Focused: `python3 -m unittest
rl_agent.splitfusion_hybrid_sac_live_route_b_v1.test_phase1_frozen_policy_runtime`
→ **51 tests, OK**, 3.7 s, CPU only.

Covering, as required: deterministic repeated output (bit-identical across
repeats and across a freshly constructed actor); exact feature order (swapping
two features changes the decision); no identifiers in the policy tensor
(mappings, strings, UUIDs and integer timestamps are all refused, and the
registered `FORBIDDEN_POLICY_FEATURE_SUBSTRINGS` guard is re-asserted); no
ambient RNG mutation (Python, NumPy and Torch generator states compared before
and after); no CUDA initialization (asserted at load, at build, before and
after every decision, and in a subprocess).

Also covered: all 72 anchors reproduce their catalog identity bit-for-bit; 70+
representative off-anchor `q_e4` values across all 12 modes preserve exact
keep/drop counts and carry null anchor identity; a forged `q_e4` is rejected
because the contract re-quantizes rather than trusting the actor; create-only
evidence refuses a second write; absent class metrics are null with a status,
never zero; the analyzer abstains rather than passing gates it cannot decide.

Full regression:

| Suite | Result |
| --- | --- |
| `splitfusion_hybrid_sac_v1` (discover) | 852 tests, 752 s — **4 errors**, all in `test_empirical_contextual_run3_controlled_audit` |
| `splitfusion_live_dispatch_v1` | 31 tests, OK |
| `splitfusion_direct_edge_map_v1/tests` | 47 tests, OK |
| `splitfusion_quality_feedback_probe_v1` | 13 tests, OK |

**The four errors are not from this work and were not introduced by it.**
`empirical_contextual_run3_controlled_audit.py` and its test are untracked,
user-owned files that were being patched *while the suite ran*: a `.rej` file
appeared at 11:43:38 mid-run and had disappeared again by the time I looked.
Re-running that module alone a few minutes later produced a *different* failure
set (1 failure + 1 error, `0.09999999999999998 != 0.1`, versus the earlier 4
errors of `payload/datagram relation drift` from a newly added
`datagram_count == ceil(bytes / UDP_PAYLOAD_CAPACITY_BYTES)` check at
`empirical_contextual_run3_controlled_audit.py:233`).

Isolation is verifiable: nothing in this package references
`controlled_audit`, and nothing in the repository imports this package. The
remaining 848 tests in that suite pass. I did not touch, stage or revert any of
those files.

## Blockers for Phase 2 (already measurable, stated now)

**1. The state contract hard-refuses UE-visible live radio evidence.**
`state_reward_transition_contract.py:4554` raises unconditionally for
`RadioEvidencePath.UE_VISIBLE_RUNTIME`: *"the repository has no measured
UE-visible feedback/IPC carrier whose envelope and availability can be opened
and hash-verified."* The only attested factory is
`RadioObservationV1.for_simulator_testbed`, which stamps
`SIMULATOR_TESTBED_PRIVILEGED`. The registered Run-3 normalization spec also
requires `snr_metric == SIMULATOR_EFFECTIVE_UL_SNR_DB`, which only that path
produces.

**2. The live target-SNR runtime measures nothing.**
`rl_agent/ue_target_snr_cell_runtime_v1.py:252` writes `"achieved_snr_db": ""`
on every row — it is an RFsim *actuator*, not a collector. Its `FIELDS`
(line 33) carry no MCS and no BSR at all.

So a causal decision-time SNR/MCS/BSR triple is not obtainable from the current
interface. Phase 2 will measure this precisely and propose the narrowest
sidecar; Phase 1 does not pre-commit to one.

## Explicitly deferred, not silently omitted

`PilotExecutionIdentityV1.execution_bundle_sha256` is `None` behind
`bundle_binding_status = "PHASE1_CATALOG_IDENTITY_ONLY;…"`. Loading
`DynamicExecutionContract` verifies runtime checkpoints, codec sources and
startup artifacts, which is outside Phase 1's boundary. `bind_execution_bundle`
is the seam Phase 3 fills.

## Process confirmation

CARLA, OAI, RFsim, Docker, the map server and CUDA were not launched.
`PYTHONPATH` was not exported. `torch.cuda.is_initialized()` is False
throughout. No user-owned dirty path was reset, staged, edited or deleted. All
work is create-only additive files under `abiodun/`.
