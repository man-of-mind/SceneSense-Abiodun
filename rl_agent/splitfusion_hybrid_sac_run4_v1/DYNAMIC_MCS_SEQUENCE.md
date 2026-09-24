# Run-4 dynamic UE UL-MCS sequence binding

## Decision

Run 4 may use the retained UE-decoded UL-MCS sequence from
`ue_snr_bridge_qualification_v1/20260923_220340` for offline fitting of
time-varying channel dynamics. It must not infer an MCS sequence from the
288-cell `radio_trace.csv`: those rows record RFsim commands and contain no
measured UE MCS.

This is not a production authorization. The production pin in
`dynamic_mcs_sequence.py` remains `None` until the final composite verifier
binds this sequence together with the state provider, queue model, quality
surface, reward, replay and training evidence.

## What is bound

The loader re-hashes the source manifest, source configuration, UE DCI CSV,
gNB MCS verifier CSV, clock anchors, UE identity, command log and effective
OAI configurations. It accepts only:

- the single retained UE (`IMSI 001010000000001`, `10.0.0.2`, RNTI 33457);
- uplink DCI format 7 with RNTI type 0;
- table-0, round-0, NDI-1, RV-0 grants; and
- the registered `SCENESENSE_MCS_POLICY=sinr` scheduler.

The gNB trace is verifier-only. Of 49,466 eligible UE rows, 49,448 uniquely
join to the gNB final-MCS decision within 5 ms (99.9636%); 18 are unmatched,
and there are zero ambiguous joins and zero MCS mismatches. The policy state
uses the UE-decoded value, never the gNB SNR or profile label.

## Frozen causal split

Each retained dynamic profile has one measured 25-second realization. The
decision grid starts 100 ms after the measured-window boundary so a decision
cannot borrow a warm-up or previous-profile grant.

| Profile | Fit decisions | Internal-validation decisions |
|---|---:|---:|
| MID_VARIABLE | ordinals 1–174 | ordinals 175–249 |
| FADE_RECOVERY | ordinals 1–174 | ordinals 175–249 |

The split is constructed from manifest timestamps before either MCS CSV is
opened. Fit and validation are contiguous but have separate segment
identities and no reused selected grant identity. A transition is emitted
only between adjacent decisions inside one segment. The terminal decision
requires an explicit reset; no transition crosses a split or profile
boundary.

## Causal use

At decision time `t`, selection uses the latest eligible UE DCI with source
timestamp strictly less than `t`. An event stamped exactly at `t` is not
visible. The caller must provide the maximum permitted age. If no prior event
exists, or the latest event is stale, the observation is explicitly
`MISSING` or `STALE` with `mcs_index=None`; zero is never fabricated and no
value is forward-filled.

The sequence supplies only:

```text
current prior_ul_mcs_index -> next prior_ul_mcs_index
```

MCS evolution is exogenous. It is never conditioned on backlog, payload,
mode, q, reward or action. Hidden profile/trace identities remain evidence
metadata and are absent from the policy-feature dictionary.

## Scope limitation

The split is two contiguous portions of one realization per profile. It is
valid for an initial measured-trace training/internal-validation exercise,
not for a claim of run-to-run or unseen-channel generalization. Extra dynamic
repetitions should be captured for final external validation, but are not a
prerequisite for the initial Run-4 fit.
