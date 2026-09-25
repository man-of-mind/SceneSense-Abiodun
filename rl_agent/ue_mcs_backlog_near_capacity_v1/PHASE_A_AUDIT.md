# Phase A audit — Run-4 near-capacity sweep (repaired)

Audit and preregistration only. **No live component was launched.** No CARLA, no
CUDA, no model load, no OAI edit or rebuild, no container started, no softmodem
spawned. Repair performed read-only from `fe74997`.

## 1. Protected Run-3 evidence

Hashed before the repair began and again after the full test suite. **All five
byte-identical.**

| File | SHA-256 |
|---|---|
| `manifest.json` | `219db3ed7dacc975ab3bfc56164ad527724989ba12d6a1f0d83f97af382595a1` |
| `INTEGRITY_AMENDMENT_DERIVED_V1_SUPERSEDED.md` | `d68c665d061452fe663d227b6658ed3650244a9af963539e7ed6f891f1b5a00b` |
| `VERIFIER_INPUT_MANIFEST_V2.json` | `c2b796bcf8991891f79d407751c5edd0c8b5774d425513f82bdd528ad4eba640` |
| `analysis_v2.json` | `4c48df847a1be13e29ca325c886412584de6b90692d8063090245867e71be999` |
| `decisions_v2.csv` | `cfddd8b9623b055b80a220f48f71e1619c6e8689ce8caf20c2174da95b7af799` |

Verdict preserved as **`INCONCLUSIVE`**, bound result **4/7 scientific checks and
12/13 structural gates**. Nothing written inside `20260924_131015`.

## 2. What the repair changed

| # | Item | Resolution |
|---|---|---|
| 1 | Radio binding | 106 PRB/7D2U replaced by `OAI_N78_100MHZ_273PRB_4D5U_V1` via the hash-bound launcher and the registered Phase-14a mapping. 23 identities pinned in `radio_binding.py`; drift and the legacy radio are refused. |
| 2 | Capacity | Run-3 106 PRB capacity and actuator anchors removed entirely. New separately-authorized bounded stage measures capacity under this radio; a deterministic rule then selects tiers bracketing the measured boundary. **No action is frozen.** |
| 3 | Identity pinning | Config, starting commit, executed/imported sources, `T_messages` (source and both compiled copies), extractor, radio configs, launcher, mapping and executables verified at `before_preflight`, `before_scientific_cells` and `final_sealing`. |
| 4 | Failure policy | Clamp, skip, RF restore/read-back, sender/receiver, extraction, accounting, teardown notes and non-cold final state are all hard nonzero failures. |
| 5 | UDP probe | Real pre-scientific probe requiring **both** ext-DN receiver arrival and nonzero live `NR_PDCP_TX_SDU`. |
| 6 | Authorization | One-attempt enforcement with supersession, defect statement, prior-attempt listing, committed-revision requirement and `lineage.json`. |
| 7 | Analysis completion | P95 mixture estimator, gate-7 matching/effect rule, exact 200 ms MCS freshness limit, ≤1 µs clock-bridge refusal — all with adversarial tests. |

## 3. Radio identity — reconciled against independent authorities

| Pin | Digest | Independent authority |
|---|---|---|
| launcher | `8e02f091…` | `splitfusion_phase14a_campaign_binding_v1.json:launcher.sha256` |
| launcher runner / config | `09fb82c0…` / `ad541f71…` | same binding, `calibration.runner` / `calibration.config` |
| radio lock | `fd604a37…` | same binding, `capacity_evidence.artifacts.radio_lock` |
| 273 PRB gNB conf | `03fb7ac7…` | `splitfusion_phase14a_100mhz_calibration_v1.json:source_sha256.gnb_source` |
| `ue.conf` / `channelmod_rfsimu.conf` | `b20bf2d9…` / `a47ade41…` | same, `ue_source` / `channel_source` |
| `T_messages.txt`, 5 tracer binaries, extractor | — | `ue_n3_oai_ul_live_stage_v1.json:runtime_seals` |
| mapping manifest | `5f14b43d…` | Phase-14a terminal `manifest_sha256` |

**Recorded, not papered over:** `nr-softmodem` (`ebcd85f4…`) and `nr-uesoftmodem`
(`60ecc9a1…`) **differ** from the `ue_n3_oai_ul_live_stage_v1.json` seal
(`01489dfb…` / `7cdeee94…`). They were rebuilt 2026-08-25, after the 2026-08-03
edit to `gNB_scheduler_ulsch.c` that added the SINR UL-MCS policy gate
(`scenesense_use_sinr_mcs_policy()`, confirmed present in the current source).
The older seal predates that work, so it is marked
`FIRST_PIN_RUN4_REBUILT_AFTER_SINR_MCS_POLICY` and is **not** cited as
corroboration.

**T-tracer byte-compare:** both compiled `T_messages.txt.h` copies are identical
(`e5801830…`, 888,765 B), consistent with the 2026-07-23 source and the
2026-08-25 build. No rebuild is required and none was performed.

**Port correction:** the qualified launcher opens UE `--T_port 2023`. v1 carried
Run-3's **2022**, which would have produced an empty UE trace — no MCS and no
backlog at all — while every other gate passed. Now pinned and asserted against
the launcher text.

## 4. Capacity and tier rule

Run-3's 12.05 / 42.13 Mbps figures are listed in
`FORBIDDEN_LEGACY_CAPACITY_MBPS` purely so a test can prove they are not used.

Stage: three held operating points at the registered `ADVERSE_STABLE` trace's own
percentiles (7.827 / 8.608 / 9.604 dB), a 285.47 Mbps saturating probe, 3 s
settle + 10 s measure each. Gates: all points present, ≥60 samples, ≥80%
continuously backlogged, positive median, capacity non-decreasing in SNR within
10%.

Deterministic rule at ratios 0.50 / 1.00 / 1.40, ties toward the smaller action
id, then refusal unless the tiers bracket the boundary (low strictly below, high
strictly above, medium within ±25%, three distinct actions, strictly increasing
payloads). Verified to bracket correctly at 1, 5, 12.05, 40, 85, 150 and 200
Mbps, and to refuse at 0.4, 0.5, 285.47 and 400 Mbps — the catalogue spans
0.5–285.47 Mbps across 72 eligible actions.

Decoder digests are recorded but sameness is **not** required: no model is
loaded, so a tier is bytes on the wire and decoder identity cannot confound the
measurement. A blank digest is refused.

## 5. Defects found and fixed during the repair

1. **`main()` authorized twice**, the second time after `mkdir`, so a run would
   have counted itself as its own prior attempt and refused. Now authorized once,
   before anything is created.
2. **The shared-checkpoint check was a silent no-op** — `SelectedTier` never
   carried the digest, so the guard compared an empty set. Digest is now carried;
   the rule was also reconsidered and relaxed deliberately, with the reason
   recorded.
3. **The P95 estimator used a different percentile convention** from the observed
   side, so gate 5's two arms would have disagreed about what a percentile means.
   Replaced with a weighted Type-7 form that reduces exactly to the observed-side
   `percentile` under equal weights, asserted at nine quantiles.
4. **Inherited teardown would have leaked the RAN.** The launcher leaves the gNB
   and UE running, detached and root-owned, so they are not in `self.processes`.
   Teardown now stops them explicitly with escalation and checks for stale
   tunnels.

## 6. Host state at repair time

Load average 0.32; no containers; no `nr-softmodem` or `nr-uesoftmodem`; no
`oaitun_ue1`; no CARLA. Cold, with no other user's work running.

## 7. Tests

`test_near_capacity_v1.py` — **106 tests, all passing**, entirely offline.

| Group | n | Covers |
|---|---:|---|
| `CatalogAuthorityTests` | 4 | catalogue digests; digest-mismatch refusal; **no action frozen and the withdrawn 106 PRB symbols are gone**; every eligible action carries a digest |
| `TierRuleTests` | 8 | determinism; bracketing across seven capacities; refusal outside the catalogue span; non-positive capacity; blank-digest and non-increasing-payload adoption refusals; legacy capacity refused; stage freezes nothing |
| `CapacityStageGateTests` | 5 | clean surface qualifies; non-saturating probe, too few samples, capacity falling with SNR, and a missing operating point all refused |
| `RadioBindingTests` | 12 | all pins verify; launcher/config digests match the independent bindings; 273 PRB/100 MHz/4D5U; **UE port 2023 asserted against the launcher text**; compiled `T_messages` consistent; mapping is the Phase-14a one; no Run-3 anchor survives; range covered; qualification not overclaimed; drifted pin detected; forbidden env and legacy launchers refused |
| `DesignBalanceTests` | 16 | 12 cells / 5,400 decisions; six permutations; within-cell load; position and transition balance; FIT/VALIDATION 3+3; Latin squares; **declared transition disjointness**; seeded reproducibility; partitions immune to shuffling |
| `ProtectedEvidenceTests` | 6 | five files unchanged; registered set; verdict preserved; tampering refused; writes inside the protected run refused |
| `CreateOnlyTests` | 5 | `mkdir(exist_ok=False)`; refusal without authorization writing nothing; existing-dir refusal writing nothing; **authorization precedes creation**; protected-run target refused |
| `AnalysisSpecTests` | 12 | bins, zero-bin, monotonicity, eight gate thresholds, degenerate-arm rule, NaN-not-zero improvement, metric values, overflow, wrong-direction detection |
| `AnalysisCompletionTests` | 18 | pinned 200 ms limit; sensitivity does not select it; bridge accepts ≤1 µs, refuses >1 µs and NaN; mixture estimator vs weights, empty bins, no support; **P95 ≠ P50**; **both gate-5 arms share one convention at nine quantiles**; gate-7 gap/support/direction/null/verdict rules; payload levels from selected tiers |
| `AuthorizationTests` | 11 | missing/wrong-stage/wrong-token refusal; unknown fields refused; second attempt refused; supersession must name a real attempt and state a defect; complete supersession accepted; attempts never removed; lineage; dirty tree refused, clean tree accepted |
| `FailurePolicyTests` | 7 | config declares every failure; runner **enforces** clamp/skip/restore/teardown/extraction/cold rather than only recording; UDP probe requires arrival **and** PDCP; three verification points; registered mapping bound, not Run-3 anchors; launcher RAN torn down; config has no 106 PRB block |

## 8. Unresolved assumptions

1. **Adverse capacity under this radio is unmeasured.** 273 PRB is 2.58× the
   legacy bandwidth and 4D5U gives 5/9 uplink slots against 7D2U's 2/10, so it
   could plausibly be several times the legacy 12 Mbps. That is precisely why no
   action is frozen; the number comes from the live stage.
2. **The tier rule can legitimately refuse.** If the measured boundary falls
   outside 0.5–285.47 Mbps, nothing is frozen and the decision returns to
   Abiodun.
3. **The favorable arm is expected to be uninformative for gates 4–7**, handled
   by the registered degenerate-arm rule, not by relaxing a threshold.
4. **Validation extrapolates across transition direction**, by construction.
5. **The Phase-14a mapping is `campaign_mapping_qualified: false`** — that gate
   belonged to the 288-cell campaign and required a four-profile replay. Run 4
   binds the twelve anchor measurements only.
6. **Tier payloads may be large.** At a high measured boundary the rule can
   select multi-megabyte actions (e.g. 1.4 MB at 85 Mbps, 24 chunks/frame). The
   sender and receiver handle chunking, but this has not been exercised live at
   that size by this package.
7. Simulated radio, single UE, one host, one repetition per permutation.
