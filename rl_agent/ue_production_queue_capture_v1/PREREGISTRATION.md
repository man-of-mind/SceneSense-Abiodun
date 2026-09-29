# Production-domain UE queue/transport capture — preregistration

Status: **frozen before collection** (`FROZEN_BEFORE_COLLECTION`).

Execution authority: repository HEAD `8ceb3f5728bb95cec414d0c55f6f2b40dcf72827`.
Contract document digest: see `contract.CONTRACT_SHA256`.

Claim boundary:
`PRODUCTION_DOMAIN_UE_QUEUE_AND_TRANSPORT_CALIBRATION_ONLY__NOT_PERCEPTION_ENDORSEMENT__NOT_POLICY_VALIDATION__NOT_DEPLOYMENT_AUTHORIZATION`.

This package is **additive**. It does not import, modify or supersede
`ue_mcs_backlog_run4_calibration_v1`; that package, its uncommitted working
state and its preserved 1,200-byte calibration design are untouched. Every
previously failed or refused artifact is preserved.

## 0. Why this capture exists

The frozen `ue_mcs_backlog_run4_analysis_v1` preregistration could not be
satisfied from retained evidence: its 12-cell capture was never collected, the
only retained 12-cell capture used different actions/modes/payloads and was
23.3% queue-ceiling censored, and coupling alternative **B** additionally
required a sealed `ByteDomainProofV1` that does not exist. Rather than
manufacture that cross-domain proof, this capture collects **directly in the
production byte domain**, which removes the cross-domain question instead of
answering it.

## 1. Byte-load roles

Three roles, bound to registered SFD1 anchor actions. The payload coordinate
is `total_transmitted_bytes` (SFD1 inner payload plus the 36-byte common
header and the frame-context bytes). Medians are observed in the registered
quality-grid authority, not asserted:

| role  | action | mode | q_e4 | family/quant | median bytes | offered @10 Hz | placement |
|-------|--------|------|------|--------------|--------------|----------------|-----------|
| floor | 71 | 11 | 9800 | AE32/UINT4 | 6,459   | 0.517 Mbps  | lower payload boundary of Run-4 modeled support |
| knee  | 39 | 6  | 7000 | AE64/UINT8 | 374,531 | 29.982 Mbps | inside the retained adverse-capacity uncertainty set |
| guard | 38 | 6  | 5000 | AE64/UINT8 | 619,825 | 49.618 Mbps | above capacity and above the Run-4 execution ceiling |

Action identity is reproduced independently from the registered SFD1
arithmetic `action_id = mode_id * 6 + index(q_e4)` and must agree.

Rationale, all from retained evidence and not re-measured here:

- Run-4 modeled payload support is 6,423–427,605 bytes. Floor and knee are
  inside it; **guard is deliberately outside it** and is a bracket, never an
  operating point. A fitted model refuses rather than extrapolates there.
- The retained adverse-capacity bootstrap uncertainty set is 28.512–31.584
  Mbps (point 30.576). This is explicitly **not** a population confidence
  interval, and the tier selection it originally supported was **REFUSED**.
  It is used only to place byte roles. No new capacity qualification is run
  and no catalog action is dynamically reselected.
- All three actions are `EMERGENCY_ONLY` catalogue entries. This is a
  byte-only queue design; their perception behaviour is **not endorsed**.

## 2. Production packetization, verified not assumed

The deployed uplink binding is verified against the current production sender
and 27 retained live manifests before launch:

- 12,500-byte UDP application datagrams (`udp_chunk_bytes: 12500`);
- 8-byte `!IHH` chunk header;
- at most 12,492 feature bytes per datagram;
- `retransmission: false`.

The 60,000-byte default in `phase2_map_sharing/transport.py` is historical and
is explicitly rejected. No 1,200-byte calibration chunking is used and no
paired-packetization experiment is run.

A full datagram is 12,528 bytes on the wire including UDP and IPv4 headers,
far above the registered 1,500-byte path MTU. **IPv4 fragmentation therefore
happens** (9 fragments per full datagram). It is observed from the kernel
`/proc/net/snmp` counters around every cell, never assumed away.

## 3. Same-domain closure seal

Because collection is already in the production byte domain, the old
cross-domain `ByteDomainProofV1` is replaced by a same-domain closure seal.
All of the following must be present and exact:

1. exact frame/action/payload identity;
2. sender first and last socket-handoff timestamps;
3. UE PDCP ingress;
4. UE RLC ingress and dequeue;
5. causal pre-action RLC backlog;
6. causal previous new-data round-0/table-0 UL MCS;
7. receiver first and last datagram timestamps;
8. receiver complete-reassembly timestamp;
9. observed IP-fragment accounting;
10. zero unexplained sender → PDCP → RLC residual;
11. exactly one terminal outcome for every sent frame.

## 4. Capture design

10-Hz tensor frames, 5-Hz controller. One cycle is

```text
decision frame t (even) -> held frame t+1 (same mode and q) -> successor t+2
```

450 frames per cell (3 blocks x 150), 12 cells, 5,400 raw frames, 224 closed
cycles per cell and 2,688 total. Frame index 448 has no in-cell successor and
is reported `UNCLOSED`, never zero-filled.

12 cells = 2 profiles x 6 counterbalanced tier permutations. Whole cells are
FIT or VALIDATION, and the two partitions use **disjoint permutations and
disjoint scene splits** (`fit` vs `held_scene`), so validation bytes come from
scenes the fit never saw. Validation never chooses bins, back-off, scaling or
a coupling alternative.

Profile identity (`FAVORABLE_STABLE` / `ADVERSE_STABLE`) is **audit-only**. It
is never a feature, a fit key, a back-off key, or an actor input.

Every frame replays the exact `total_transmitted_bytes` of one registered
quality-grid row, selected by SHA-256 rank under a frozen seed. Byte loads are
therefore natural and nonconstant (~410 distinct values per 450 frames) and
every frame is traceable to an authority row digest.

### Frozen sender semantics

The socket is **blocking** with a large send buffer, so every datagram handed
to the socket enters the UE IP/PDCP/RLC path and the byte residual is
closable. `datagrams_dropped_at_socket` must be 0.

The guard role deliberately offers more than the link carries, so once the UE
queue saturates the sender **will** fall behind the nominal 10-Hz grid. That
is expected and is **not** a failure. The measured socket-handoff stamps are
authoritative; the nominal grid is a target, not an assumption. A frozen
per-cell wall-clock guard bounds the overrun.

Per-tier queue-ceiling censoring is **reported, not gated**, because guard
saturation is the intended bracket. Gates 4–8 are evaluated on the
non-censored measured support.

## 5. Latency boundary and the disjoint replacement

Both segments meet at exactly one endpoint, the **last UDP socket handoff**:

```text
UE action path        : seven-channel construction start -> last UDP socket handoff
production transport  : last UDP socket handoff         -> complete receiver reassembly
```

The old Run-4 component `application_feature_uplink_ms` is defined in
`splitfusion_timing_diagnostic_v1` as *edge complete reassembly − UE **first**
send*. It is **REPLACED, never added**.

Disjointness is proven on retained live evidence rather than asserted. Over
the three retained per-frame action files in
`experiments/splitfusion_timing_diagnostic_v1/20260909_live_carla_actions30_15_50_71_retry3/per_frame`,
the identity

```text
application_feature_uplink_ms == ue_send_loop_ms + post_send_to_reassembly_ms
```

holds for every finite row with **zero violations and a maximum absolute
difference of 0.000000000 ms** (n = 183 / 200 / 163). Moving `ue_send_loop_ms`
into the UE action path and replacing the remainder with the new conditional
transport model therefore double-counts nothing. If a future change breaks
this identity, the correct action is to stop, not to add both components.

Sender and receiver run on this one host and stamp the same
`time.monotonic_ns` timeline, which is what makes the interval directly
subtractable. A non-positive transport interval is physically impossible and
means the two stamps raced inside one kernel delivery; such a row is an
instrumentation fault and is **excluded and counted, never clamped to zero**.
Loopback validation showed 9/450 such rows, all single-datagram frames, worst
magnitude 0.19 ms.

## 6. Reward and the anti-bias rule

The frozen reward is unchanged:

```text
success within 170 ms : r = Q_perc - 0.25 * latency_ms / 170
registered timeout    : r = -1
```

No `p_admit`, no SNR, no profile label, no new reward term.

- A sent frame that does not completely reassemble by the 170-ms deadline is a
  **timeout/failure with reward −1**.
- Latency is never fitted or reported on successful survivors only. The
  deadline-outcome gate is evaluated on the **full sent population**.
- Infrastructure, instrumentation or evaluator faults are **excluded**, not
  charged to the policy.

Terminal outcomes, exactly one per sent frame: `COMPLETE_WITHIN_DEADLINE`,
`COMPLETE_AFTER_DEADLINE`, `INCOMPLETE_AT_DEADLINE`, `NEVER_COMPLETED`,
`EXCLUDED_INFRASTRUCTURE_FAULT`.

## 7. Queue conservation

```text
B_next = max(0, B_current + measured_ingress - measured_service)
```

Service is conditioned only on causal previous UE MCS, causal pre-enqueue
backlog and admitted bytes. Future MCS, gNB-only measurements, target SNR and
profile ID are audit-only and never model inputs.

Freshness is one tensor period (100 ms). Missing or stale invokes the
registered external fallback; zero-fill is forbidden.

## 8. Frozen gates

1. `COMPLETE_AND_SEALED_CAPTURE` — 12 cells, 5,400 frames, 2,688 cycles.
2. `SAME_DOMAIN_CLOSURE_SEAL` — all 11 requirements, zero unexplained
   residual, exactly one terminal per sent frame.
3. `CAUSAL_INPUT_COVERAGE` — 100% strictly-prior round-0/table-0 UE MCS and
   pre-enqueue backlog at 100 ms, zero ambiguity.
4. `VALIDATION_NEXT_BACKLOG_ERROR` — NMAE ≤ 0.10 and ≥ 20% better than
   persistence, held-out whole cells.
5. `VALIDATION_TRANSPORT_LATENCY_ERROR` — P50 error ≤ 17 ms, P95 ≤ 34 ms.
6. `VALIDATION_DEADLINE_OUTCOME_CALIBRATION` — false-success ≤ 5%, Brier
   ≤ 0.15, on the full sent population.
7. `MCS_NONHARM_AND_DIRECTION` — Brier degradation ≤ 0.01, correct direction.
8. `MONOTONICITY` — zero violations inside measured support.
9. `BOUNDARY_INTEGRITY` — inversions excluded not clamped, < 1%.
10. `NO_HIDDEN_INPUT` — no profile label, target SNR, gNB-only or future
    measurement reaches the model.

Gates 5 and 6 concern the production transport segment. Clearing transport
inside 170 ms is necessary but not sufficient for a full action to meet 170 ms.

## 9. Execution policy

**Exactly one live attempt is authorized.** The grant is one-use, expiring and
bound to the exact output path, execution commit, config digest, contract
digest and source inventory; it is consumed with `O_EXCL` + `fsync` before any
RAN mutation, so a crash spends it. A failed attempt is preserved and the run
stops on a real structural failure; there is no silent retry and no in-place
cell retry. RF settings are always restored, OAI is always torn down, and a
cold host is verified per cell and at the end.

Analysis outputs are create-only and live outside the immutable raw-evidence
tree. JSON and CSV only; never pickle.
