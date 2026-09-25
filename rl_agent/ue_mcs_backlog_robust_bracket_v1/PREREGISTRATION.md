# Prospective robust capacity bracket, v1

## Decision boundary

This additive amendment does not reopen or relabel the preserved capacity
attempt at
`rl_agent/experiments/ue_mcs_backlog_near_capacity_capacity_v1/20260925_001332`.
That attempt remains `CAPACITY_QUALIFICATION_REFUSED`, `qualified=false`, with
an empty `selected_tiers` list. Its sole audit problem remains exactly
`tier stability: bootstrap capacity interval changes the deterministic tier
triplet`.

The amendment may adopt only the physically verified interval
`[28.512, 31.584] Mbps`, with retained point `30.576 Mbps`, for prospective
downstream design. This is the retained empirical/bootstrap uncertainty set
from the attempt. It is not a population confidence interval.

## Frozen source binding

The builder and verifier require these exact SHA-256 digests:

- result: `0f3b1a3e4d8b04da33876b679e966a0ca11080c20ebed039d5fd18bd9cb17a43`
- manifest: `8a77f0ff913ffb493cc226313bd2fd221394ce7330041770a0b346bda29d0293`
- refused terminal: `6540cbdd003169f46165df564f533d0303999dd51cc845b2ba6b2ca4778defc7`
- final cold state: `d062eb00c0d0f5d1fea88ddcec7904ea3412ea554d371fabf6fae9ffab12bda5`

The verifier checks the preserved manifest closure, result evidence closure,
source/radio/protected-evidence stages, all three retained point records and
their raw derivations, saturation, corroboration, drains, teardown, and cold
state. It separately proves the unmodified
`verify_bound_capacity_result()` still rejects the original result.

## Frozen robust exact-mode selector

Only catalogue actions with `transport_valid=true` and
`agent_action_enabled=true` are candidates. Actions are grouped without
cross-group mixing by family, quantizer, checkpoint SHA-256, decoder identity,
and the full codec/wire identity when present.

Within each group:

- low must be at or below `0.9 * 28.512 Mbps`;
- medium must lie inside `[28.512, 31.584] Mbps`;
- high must be at or above `1.1 * 31.584 Mbps`.

Each tier chooses the action closest to its frozen point target—`0.5 C`, `C`,
or `1.4 C`, where `C=30.576 Mbps`—then the smaller action ID. Feasible groups
are ranked lexicographically by medium distance, low distance, high distance,
medium ID, low ID, high ID, then canonical group-identity digest.

The pinned catalogue must therefore select one AE64/UINT8 identity group:

| tier | action | median payload bytes | offered Mbps |
| --- | ---: | ---: | ---: |
| low | 40 | 126237 | 10.09896 |
| medium | 39 | 374264 | 29.94112 |
| high | 38 | 619563 | 49.56504 |

Any different catalogue, candidate universe, identity grouping, threshold,
ranking rule, or selected triplet is refusal, not an implicit amendment.

## Scientific and operational caveat

All three selected catalogue actions are `EMERGENCY_ONLY`; their perception
service gates are failed or incomplete. This is a byte-only queue-design
bracket. It does not qualify or endorse perception behavior, authorize live
execution, or authorize CUDA or network use.

Artifacts are written into a separate, new directory with create-only writes.
The amendment, manifest, and terminal must all carry the exact status:

`PHYSICAL_CAPACITY_INTERVAL_ADOPTED_FOR_PROSPECTIVE_DOWNSTREAM_DESIGN_ONLY__ORIGINAL_CAPACITY_QUALIFICATION_REFUSED__ORIGINAL_TIER_SELECTION_NOT_QUALIFIED`

