# Phase A audit — Run-4 near-capacity sweep

Audit and preregistration only. **No live component was launched.** No CARLA,
no CUDA, no model load, no OAI edit or rebuild, no container started.

## 1. Protected Run-3 evidence

Hashed before any file was created, and again after the full test suite ran.
**All five byte-identical.**

| File | SHA-256 |
|---|---|
| `manifest.json` | `219db3ed7dacc975ab3bfc56164ad527724989ba12d6a1f0d83f97af382595a1` |
| `INTEGRITY_AMENDMENT_DERIVED_V1_SUPERSEDED.md` | `d68c665d061452fe663d227b6658ed3650244a9af963539e7ed6f891f1b5a00b` |
| `VERIFIER_INPUT_MANIFEST_V2.json` | `c2b796bcf8991891f79d407751c5edd0c8b5774d425513f82bdd528ad4eba640` |
| `analysis_v2.json` | `4c48df847a1be13e29ca325c886412584de6b90692d8063090245867e71be999` |
| `decisions_v2.csv` | `cfddd8b9623b055b80a220f48f71e1619c6e8689ce8caf20c2174da95b7af799` |

Run-3 verdict preserved as **`INCONCLUSIVE`**, bound result **4/7 scientific
checks and 12/13 structural gates**. The superseded "5/7 / 12/12" claim is not
repeated. Nothing was written inside `20260924_131015`.

## 2. Reuse audit — what is inherited and why it is safe

The Run-3 implementation package is committed and clean at the audited HEAD. It
is **imported, never edited**, so its recorded digests stay valid.

| Component | Decision | Rationale |
|---|---|---|
| `runner.Runner` lifecycle — `preflight`, `assert_cold_ran`, `materialize_configs`, `start_ran`, `wait_attach`, `verify_radio_path`, `start_telemetry`, `open_telnet`, `start_live_pusch`, `calibrate_upper_anchor`, `launch_traffic`, `finish_traffic`, `run_cell`, `extract_ttracer`, `teardown_ran`, `restore_clean`, `final_cold_state` | **inherited unchanged** (15 methods) | Qualified in Run 3: 12/12 cells, radio-path proof, RF restored with read-back, cold-host proof. Semantics unchanged by this design. |
| `tagged_sender` | **reused as a module, unchanged** | Driven entirely by `block_plan.json`; payload size is data, not code. |
| `decision_join` — clock bridge, backlog ticks, round-0 grant extraction, arrivals, `join_cell`, `causal_audit`, `audit_ue_gnb_mcs_provenance` | **reused unchanged** | The causal semantics are exactly what Run 4 must preserve. The provenance auditor already reports per-cell coverage, ambiguity and UE-vs-final mismatch, which is precisely gate 2's input. |
| `contract` invariants — tracer headers, `CHUNK_BYTES`, `FPS`, frames per block/cell, tier order, contrast profiles, transient/steady windows, `AGENT_PATH_BUDGET_MS`, OAI citations, `resolve_profiles` | **imported** | Restating them would let the two runs drift silently. |
| `config_v1.json` `radio` / `actuator` / `telemetry` / `traffic` blocks | **carried over, verified identical** | Same radio, same actuator anchors, same ports, same namespace strategy. |
| `Runner.run` | **overridden** | Run 3 hardcodes its 3×2×2 plan and its plan-audit predicate. |
| `Runner.manifest` | **overridden** | `super()` would stamp the Run-3 contract id onto Run-4 evidence. |
| Cell plan, partitions, gates, capacity anchors | **new** | The design itself is what changed. |

Only `run` and `manifest` are overridden; everything else resolves to the
qualified Run-3 implementation.

**Calibration tier.** The inherited `calibrate_upper_anchor` resolves the Run-3
*calibration* tier (action 71, 6,229 B, 0.50 Mbps) to put enough PUSCH on the
air to measure the noise-power→SNR mapping without flooding the link. It is
retained deliberately so the anchor procedure stays byte-identical to the one
Run 3 qualified (−12.5 dB → 25.0 dB median PUSCH SNR, 412 samples). **It emits
no tagged decision and enters no analysis.**

## 3. Action authority — reconciled

Catalogue JSON `07e0690f…`, CSV `0512cb39…`, shared checkpoint `e2f86775…`: all
three digests match the registered authority. Actions **70 / 69 / 68** resolve
to `split_ae32_uint4_q9000` / `q7000` / `q5000` at **28,109 / 81,087 / 129,707 B**,
**1 / 2 / 3** chunks at 60,000 B, **2.25 / 6.49 / 10.38 Mbps** at 10 fps. All
three are `transport_valid` and `agent_action_enabled` and share one AE32
decoder. The checkpoint digest is an identity assertion only; the file is never
loaded.

## 4. Capacity premise — corrected

Run 3 sited its tiers against "a ~6 Mbps uplink". Re-derived read-only from
Run-3's own `decisions_v2.csv` (`capacity_rederivation.py`, bound by a test):

| Channel | P10 | P50 | P90 | n |
|---|---:|---:|---:|---:|
| `ADVERSE_STABLE` | 10.49 | **12.05** | 22.57 | 1,200 |
| `FAVORABLE_STABLE` | 18.11 | **42.13** | 46.50 | 998 |

Corroborated without the slope estimate: `FAVORABLE_STABLE` sustained 21.08 Mbps
offered with **median backlog 0** and 0.960 complete reassembly — impossible on
a 6 Mbps link.

Consequence, registered in advance: the **`ADVERSE_STABLE` arm spans 0.19→0.86 of
capacity and is the near-capacity arm**, while the **`FAVORABLE_STABLE` arm is
0.05→0.25 of capacity and is expected to drain**. See `PREREGISTRATION.md` §3
and the degenerate-arm rule in §9.

## 5. Gate-2 feasibility

Run-3 per-cell verifier-only gNB coverage was 1.0 in ten cells and 0.99556 in
two — minimum 0.9956. The registered ≥99% floor is therefore calibrated to
measured reality, and the two-cell shortfall that failed Run 3's stricter 100%
structural rule would pass here.

## 6. Host state at registration

Load average 0.10; no containers; no `nr-softmodem` or `nr-uesoftmodem`; no
`oaitun_ue1`; no CARLA. The host is cold and no other user's work is running.

## 7. Tests

`test_near_capacity_v1.py` — **51 tests, all passing**, entirely offline.

| Group | Covers |
|---|---|
| `CatalogIdentityTests` (8) | catalogue digests; exact action/profile/payload/chunk/rate resolution; shared checkpoint; strictly increasing payload; and four refusal paths (profile, payload, chunk count, catalogue digest) |
| `DesignBalanceTests` (16) | 12 cells / 5,400 decisions; all six permutations per channel; within-cell load; position balance; all six transitions ×2; FIT/VALIDATION 3+3; the registered FIT and reversed VALIDATION orders; both partitions Latin; **declared fit/validation transition disjointness**; seeded reproducibility; partitions immune to shuffling; block payload/port binding |
| `ProtectedEvidenceTests` (6) | five files unchanged; registered file set; Run-3 verdict preserved; tampering detected and refused; writes inside the protected run refused; Run-4 output root outside it |
| `CreateOnlyTests` (3) | `mkdir(exist_ok=False)`; `main` refuses an existing output dir **and writes nothing**; `main` refuses a target inside the protected run |
| `AnalysisSpecTests` (13) | payload levels; backlog bin 0 is exactly zero; monotone bins; eight gates with exact thresholds; degenerate arm never passes; improvement over a perfect baseline is NaN not 0; zero scale refused; metric values; false-success undefined when nothing is promised; overflow surfaced; monotonicity detects wrong-direction predictors |
| `CapacityPremiseTests` (5) | anchors recorded for both channels; adverse arm brackets its knee; favorable arm declared sub-capacity; **anchors reproduce from Run-3 evidence**; superseded 6 Mbps figure recorded |

## 8. Unresolved assumptions

1. **The favorable arm is expected to be uninformative for gates 4–7.** Handled
   by the registered degenerate-arm rule, not by relaxing a threshold. If that
   rule is judged unacceptable, the tier set — not the rule — is what should
   change, and that is an Abiodun decision.
2. **Validation extrapolates across transition direction** (FIT and VALIDATION
   transition sets are disjoint by construction). Declared, asserted, and to be
   named explicitly if gates 4–6 fail.
3. **Action 68 at 10.38 Mbps sits just below adverse P10 capacity (10.49 Mbps).**
   Gate 3 censoring is expected to pass comfortably, but if adverse capacity
   drifts below the offered rate the high block will build queue faster than
   planned. This is the intended near-capacity regime, not a defect.
4. **Capacity anchors come from Run-3's RFsim run on this host.** They are a
   planning input, not a claim about a deployed radio.
5. Simulated radio, single UE, one host, one repetition per permutation.
