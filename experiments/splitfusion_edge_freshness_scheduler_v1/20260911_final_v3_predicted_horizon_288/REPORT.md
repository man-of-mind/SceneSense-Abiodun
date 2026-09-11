# Final-edge 288-cell counterfactual

The measured 288-cell arrival/reassembly/admission surface was replayed
through the causal predicted-install, depth-one latest-only scheduler.
AE128/AE64/AE32 use final v3 edge calibration. NoAE conservatively
retains live-valid v2 because action 15 rejected v3 fail-closed.

This is a counterfactual model, not a live remeasurement.

## Aggregate

| Quantity | Measured | Previous simulator | Final-edge simulator |
|---|---:|---:|---:|
| Installed/sent | 0.3726 | 0.5753 | 0.6470 |
| Installed frames | 334,174 | 515,931 | 580,294 |
| Useful installations | 334,174 | 504,199 | 555,484 |
| Median cell install AoI | 364.5 ms | 298.1 ms | 235.8 ms |
| Median cell map AoI | 573.5 ms | 382.1 ms | 307.2 ms |

## Network profiles

| Profile | Source install/sent | Final install/sent | Final install AoI | Final map AoI |
|---|---:|---:|---:|---:|
| ADVERSE_STABLE | 0.2702 | 0.5107 | 272.0 ms | 343.6 ms |
| FADE_RECOVERY | 0.3905 | 0.6685 | 229.2 ms | 304.3 ms |
| FAVORABLE_STABLE | 0.4511 | 0.7580 | 218.0 ms | 293.1 ms |
| MID_VARIABLE | 0.3787 | 0.6510 | 231.2 ms | 302.1 ms |

## Interpretation limits

- Payload, transport completion and edge admission stay fixed per cell.
- Missing admitted identities/timestamps retain deterministic imputations.
- Family calibration is extrapolated from one favorable anchor.
- NoAE does not use v3 and is marked separately in every row.
- This supports simulator construction, not a 100-ms readiness claim.
