# Run-4 Phase 6 — live integration runner (implementation and offline qualification only)

**Status:** `PHASE6_RUNNER_IMPLEMENTED__OFFLINE_QUALIFIED__NOT_EXECUTED`.
Starting HEAD `2d661703043ad365e829fb3f316f918c689cf41e`. Nothing was launched:
no CARLA, OAI, Docker, CUDA inference or live run. The 300-frame qualification needs a
separate authorization.

## Pipeline

Every step reuses the qualified implementation. New code is versioned adapters only, and no
legacy file is edited; a test proves 11 legacy modules are byte-identical to HEAD.

| Step | Where |
|---|---|
| Sensor preparation, SI/P40 | unchanged Route-B collector (exact-tick quality variant); SI = frozen `camera_spatial_information` on the 768×448 luma; P40 = `profile_current_sweep_p40` on the exact radar window of that frame (`phase6_ue_runtime_v2`) |
| Causal UE telemetry snapshot | Phase-2 provider over the `multi` relay; the parent starts `multi` 2023→2123 plus a durable `record` (the Phase-2C topology) |
| Live 21-D state | `live_state_v2` (training freshness/scaling, raw-support refusal) |
| Frozen actor, batch 1 | `frozen_actor_v2.load_registered_actor` (seed 43 / update 10,000) |
| Continuous action | `DynamicExecutionContract.resolve_q_e4` → the Phase-3 identity seam |
| Hold / fallback | `phase6_decision_engine_v2` over `RewardHoldControllerV2` |
| Front/ranker/AE/codec | the once-preloaded, hash-verified `_preload_ue` modules inside `ContinuousUERuntimeV2` |
| SFD3 + context transport | **SFD4** (`run4_live_wire_v2`): SFD3 plus the exact `FrameContextV1` bytes from the unchanged SFD1-v2 context packer/parser, SHA-256 over the whole frame; unchanged `!IHH` chunking and socket |
| Edge reconstruction + contextual tail | `phase6_edge_runtime_v2`: the qualified `DetachedOptimizedEdgeV3` (hash-verified, loaded once) and the step-for-step `process_compute` sequence; the tail receives the real context (never `tail(c2, None)`) |
| Direct-map branch (immediate) | Run-4 update schema, the legacy chunk/zlib path, container→host on the CN5G bridge; map process = unchanged direct server plus a process-local protocol proxy (`phase6_map_server_v2`) |
| Exact-quality branch (async) | bounded evaluator thread, reward-requested frames only: privileged GT (unchanged `gt_evidence`) → `live_measurement` → authoritative `evaluate_exact_quality` with the `scientific_basis` spec `d5d1e0d2…` |
| R4FB over the UE downlink | compact, SHA-protected `RewardFeedbackV2` encoding; the container routes 10.0.0.0/16 via the UPF; the child verifies `ip route get 10.0.0.2` → `via 192.168.70.134` and that the endpoint differs from the map endpoint |
| Resolution | the engine and controller (170 ms inclusive on `CLOCK_MONOTONIC_RAW`, duplicate, late and unknown orphans) |

## Fixed fallback and session rules (as specified)

- **Open ticket or short hold:** while a ticket is open or has fewer than 2 transmitted
  tensors, the exact action continues. A guard refusal cannot interrupt a hold.
- **Refusal at a decision opportunity:** the actor is not called and no ticket opens. The frame
  carries mode 11 / `q_e4` 9800 / anchor 71 (`split_ae32_uint4_q9800`) with
  `reward_requested=false` and decision/ticket sentinel `2^64−1`.
- **Fallback accounting:** every typed reason is logged. The frame is excluded from policy
  counts and the last policy outcome is untouched. It still passes edge validation and map
  installation, and the next frame retries. There is no TTL.
- **Wire sequence:** a single strictly increasing wire `tensor_seq` spans fallback frames and
  sessions.
- **Infrastructure fault:** any failure after tensor assignment and before a successful send
  faults the engine and stops the route. It is never turned into a timeout.
- **Evaluator-excluded outcomes** (undefined Q_perc, missing GT, evaluator exception) cannot
  be a previous outcome under the Run-4 contract. The next policy decision therefore opens a
  new decision session at genesis; telemetry samples are re-scoped from the provider root
  session. Each such rollover is counted.

## Clocks

`Stamp` makes physical-capture wall time (CARLA identity, map AoI) and `CLOCK_MONOTONIC_RAW`
(commit, action open, feedback receipt, deadline) separate types. Comparing or subtracting
across the two raises an error. The RGB callback is stamped RAW at receipt for the scene
observation; the wall capture time names the frame.

## Prospective plan amendment (before any Phase-6 data)

`live_qualification_300_v2.json`, `PHASE6_PROSPECTIVE_AMENDMENT_1`:

- **P0 coverage:** after 10 decision opportunities, at least 95% must be admitted policy
  decisions, so fallback is at most 5%. Otherwise the result is `INCONCLUSIVE_OR_FAILED`. This
  is the Phase-2C G2 threshold, reused.
- **P7:** feedback over the downlink, reconciled three ways (edge-emitted, on `oaitun_ue1`,
  UE-received) by SHA-256.
- **P8:** no infrastructure fault.
- The fixed fallback and excluded-outcome rules.

The sealed manifest now also pins every Phase-6 module. It was re-sealed as
`c054bcef139e8ba3fa2e1caa32ba65933bdb57b040a73d33be887a0d395d14a3`, and
`phase6_live_runner_v2 --preflight` passes (carrier cell `a71__favorable_stable`, 0 external
processes, no CUDA).

## Tests

- `test_phase6_live_integration_v2`: **24 OK**. Package total: **115 OK**.
- Inherited suites, all OK:

  | Suite | Tests |
  |---|---|
  | direct edge map | 47 |
  | quality probe | 13 |
  | live probe | 10 |
  | dynamic execution contract | 20 |
  | dispatch | 7 |
  | decoded evidence write | 4 |
  | Run-4 contract | 23 |
  | state adapter | 25 |
  | transaction identity | 3 |
  | production capture | 39 |

The Phase-6 tests cover:

- every legal `(mode_id, q_e4)` identity (117,612 pairs) validating against the contract, and
  the 72 anchors unchanged downstream;
- an off-anchor action traversing execution → SFD4 → edge → Run-4 map ingest (proxy) →
  Run-4 ACK → UE ledger reconciliation → evaluator → R4FB → ticket closure, with no
  `action_id` fabricated anywhere (map state label `""`);
- legacy files byte-identical to HEAD, and the proxy forwarding legacy documents byte-exactly;
- the exact `FrameContextV1` round trip into the contextual tail's metadata;
- bit-flip corruption failing closed, and forged bundle/q/anchor refused at the edge (a
  resealed forged ticket can only become an `UNKNOWN_ORPHAN`);
- authoritative Q_perc equal to the stored grid Q_perc on real grid rows, and the live
  measurement consistent with the live scorer;
- exactly 170 ms succeeding and 170 ms + 1 ns timing out, with the late ACK a
  `LATE_ORPHAN`;
- duplicates idempotent and never attaching to a newer ticket;
- k_min = 2 with exactly one reward request per decision;
- fallback: no actor call, no ticket, previous state preserved, retry next frame, edge/map
  still reached; a hold not interrupted by refusal;
- a post-assignment failure producing an infrastructure fault that stops the run and is not a
  timeout;
- evaluator-fault rollover to a genesis session;
- mixed-clock operations refused;
- the coverage gate non-vacuous at both boundaries;
- three-way R4FB reconciliation from a synthetic pcap (host-local or empty evidence fails);
- the feedback path requiring distinct endpoints and a UPF route;
- the edge import closure free of `yaml`/`pandas`;
- parent teardown attempted for every resource when the child fails.

## Remaining risks (live verification required; not blockers to running)

1. **Live semantics of Q_perc.** The authoritative formula and spec are used, but live person
   eligibility is every live GT person. The training grid used an offline AVO ≥ 0.65 table,
   which has no live counterpart, so live rewards are not grid-identical
   (`LIVE_MEASUREMENT_SEMANTICS`). Report reward, do not gate on it.
2. **Genesis rollovers.** About 40% of grid frames have undefined Q_perc. If Route-B frames
   behave similarly live, many decisions will be genesis states, which training rarely saw.
   This follows the contract, but it needs Abiodun–Codex review of whether it is acceptable
   for a qualification.
3. **MCS freshness.** Phase 2C measured a grant age of 50–78 ms at 10 Hz. Smaller off-anchor
   payloads produce shorter UL bursts, which could push MCS past 100 ms. P0 adjudicates this
   and fails the run rather than hiding it.
4. **Not executed:** the CUDA front/codec with off-anchor q, the container edge compute, and
   the relay+record inside the supervisor lifecycle. All are wired from qualified parts but
   only verified offline.
5. The UE-side `STALE_BEFORE_SEND` check is applied only **before** assignment (after
   assignment, lateness is the reward deadline's job). Otherwise a stale frame would become an
   infrastructure fault.

## Proposed command — NOT EXECUTED

```bash
# offline first
env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -m \
  rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_live_runner_v2 --preflight \
  --output-root rl_agent/experiments/splitfusion_hybrid_sac_live_route_b_v2/<run>_phase6_live_qualification
# live (requires separate authorization)
env -u PYTHONPATH python3 -u -m rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_live_runner_v2 \
  --output-root rl_agent/experiments/splitfusion_hybrid_sac_live_route_b_v2/<run>_phase6_live_qualification \
  --transmitted-budget 300 --execute SPLITFUSION_RUN4_PHASE6_LIVE_QUALIFICATION_V2_EXECUTE
```

## Files

- **Created:** `phase6_decision_engine_v2.py`, `run4_live_wire_v2.py`, `run4_map_protocol_v2.py`,
  `run4_ue_ledger_v2.py`, `phase6_edge_runtime_v2.py`, `phase6_map_server_v2.py`,
  `phase6_ue_runtime_v2.py`, `phase6_live_child_v2.py`, `phase6_live_runner_v2.py`,
  `test_phase6_live_integration_v2.py`, `PHASE6_RUNNER_REPORT.md`.
- **Modified (package only):** `live_qualification_300_v2.json` (prospective amendment),
  `readiness_v2.py` (pins the Phase-6 modules), `LIVE_BINDING_MANIFEST_V2.json` (re-sealed).
- **Outside the package:** nothing.
