# SplitFusion post-optimization 288-cell counterfactual

All 288 measured fixed-action cells were replayed offline. Measured feature
capture times, bytes, and per-cell reassembly/admission totals were preserved.
Where frame-level arrival identities were not retained, they were imputed
deterministically. The edge service component was then replaced with the
measured family calibration. The qualified detached two-stage latest-only
scheduler and 25 ms
pre-compute expiry ceiling were simulated.

This is a counterfactual training model, not a new live measurement and not
evidence of 100 ms service readiness.

## Aggregate

| case | installed | useful installs | install AoI <=100 ms | install AoI <=500 ms | install/sent | map AoI cell median |
|---|---:|---:|---:|---:|---:|---:|
| measured source | 334174 | 334174 | 0 | 329722 | 0.3726 | 573.5 ms |
| counterfactual | 429270 | 425022 | 15 | 427287 | 0.4786 | 377.0 ms |

Upstream accounting held fixed: 896,856 sent, 184,124 transport
incomplete, and 110,417 measured pre-queue rejections.
Of 602,315 edge admissions, 344,177 had retained frame-level arrival timestamps and 258,138 were imputed.

## Network-profile behavior

| profile | measured install/sent | candidate install/sent | measured map AoI | candidate map AoI |
|---|---:|---:|---:|---:|
| ADVERSE_STABLE | 0.2702 | 0.3745 | 642.3 ms | 415.9 ms |
| FADE_RECOVERY | 0.3905 | 0.4989 | 587.6 ms | 373.1 ms |
| FAVORABLE_STABLE | 0.4511 | 0.5631 | 528.9 ms | 362.8 ms |
| MID_VARIABLE | 0.3787 | 0.4781 | 546.6 ms | 373.2 ms |

## Interpretation limits

- Optimization is applied to the measured edge service duration, never
  by subtracting a constant from measured AoI.
- The family median service reduction is extrapolated from one live anchor
  per family; quantizer/q-specific optimization variance is not measured.
- Missing old service durations and compact-result install delays are
  deterministically imputed from within-action/profile empirical pools.
- Per-cell reassembly and edge-admission counts are preserved exactly.
  Missing admitted identities and arrival delays are deterministically
  imputed and never presented as measured observations.
- Perception quality is the immutable validation quality of each action;
  failed or superseded frames receive no installation utility.
- The four live scheduling cells selected 25 ms provisionally. They do not
  establish run-to-run variance or universal optimality.
