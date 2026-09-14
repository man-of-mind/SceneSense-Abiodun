# Corrected 288-cell counterfactual: direct edge-to-map installation

The measured 288-cell arrival/reassembly/admission surface and the final
optimized edge service are held fixed. Only the post-edge installation
path changed: the edge -> UE -> map detour is removed and replaced by the
direct edge-to-map delay measured in the live validation.

**Counterfactual model, not a live remeasurement.**

Pooling rule: `PER_FAMILY` (Kruskal-Wallis H=124.363, p=8.86e-27, pre-registered alpha=0.05).

## Aggregate: old detour versus corrected direct path

| Quantity | Measured campaign | Old edge->UE->map replay | Corrected direct edge->map |
|---|---:|---:|---:|
| Installed/sent | 0.3726 | 0.6470 | 0.6471 |
| Installed frames | 334,174 | 580,294 | 580,355 |
| Useful installations | 334,174 | 555,484 | 558,115 |
| Median cell install AoI | 364.5 ms | 235.8 ms | 224.0 ms |
| Median cell map AoI | 573.5 ms | 307.2 ms | 297.4 ms |

## Extrapolations

- the live direct-map delay was measured only under FAVORABLE_STABLE and is applied to all four network profiles; the justification is physical -- the direct path is a container-to-host datagram on the CN5G bridge that never traverses the radio -- but it is an extrapolation and is not independently confirmed per profile
- the live validation covers actions 15/30/50/71; every other action in a family inherits that family's measured delay
- the edge service model, arrival model and quality are unchanged from the final optimized-edge analysis and are not re-measured

## Limitations

- counterfactual model, not a live remeasurement of 288 cells
- only the post-edge installation path changed; transport, admission and edge service are bit-identical to the baseline replay
- agent-feedback arrival is excluded from physical map AoI by design
- no 100-ms service-readiness claim
