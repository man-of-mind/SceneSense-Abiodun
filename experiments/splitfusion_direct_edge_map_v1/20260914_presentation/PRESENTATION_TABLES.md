# Direct edge-to-map presentation tables

Simulator rows are **counterfactual**: the immutable 288-cell transport,
admission and edge service are replayed with only the post-edge
installation path changed. Rows marked **live** are measurements from the
short four-action validation run.

## 1. Actions meeting each freshness budget (counterfactual)

Primary rule: physical map age within budget for at least 50% of route time.

| Budget | Profile | 25% route | 50% route (primary) | 75% route | Best action | Best fraction |
|---|---|---:|---:|---:|---:|---:|
| 150 ms | Favorable Stable | 0/72 | **0/72** | 0/72 | 71 | 10.2% |
| 150 ms | Mid Variable | 0/72 | **0/72** | 0/72 | 65 | 8.0% |
| 150 ms | Adverse Stable | 0/72 | **0/72** | 0/72 | 65 | 8.6% |
| 150 ms | Fade Recovery | 0/72 | **0/72** | 0/72 | 65 | 8.0% |
| 200 ms | Favorable Stable | 11/72 | **0/72** | 0/72 | 71 | 39.5% |
| 200 ms | Mid Variable | 9/72 | **0/72** | 0/72 | 65 | 35.7% |
| 200 ms | Adverse Stable | 9/72 | **0/72** | 0/72 | 71 | 36.3% |
| 200 ms | Fade Recovery | 10/72 | **0/72** | 0/72 | 65 | 36.5% |
| 250 ms | Favorable Stable | 42/72 | **22/72** | 0/72 | 71 | 70.1% |
| 250 ms | Mid Variable | 33/72 | **15/72** | 0/72 | 65 | 68.1% |
| 250 ms | Adverse Stable | 23/72 | **13/72** | 0/72 | 71 | 69.2% |
| 250 ms | Fade Recovery | 36/72 | **13/72** | 0/72 | 65 | 68.5% |
| 300 ms | Favorable Stable | 53/72 | **43/72** | 23/72 | 53 | 88.0% |
| 300 ms | Mid Variable | 46/72 | **36/72** | 16/72 | 59 | 86.7% |
| 300 ms | Adverse Stable | 35/72 | **24/72** | 13/72 | 71 | 87.6% |
| 300 ms | Fade Recovery | 47/72 | **38/72** | 15/72 | 65 | 86.3% |

## 2. Model quality for the 43 budget-eligible actions

Quality is action-intrinsic: it is a property of the frozen split profile
and does not vary with the network profile or the installation path.

| Action | Family | Veh P | Veh R | Veh F1 | Veh XY MAE | Veh IoU | Per P | Per R | Per F1 | Per XY MAE | Seg mIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | AE64 | 0.929 | 0.866 | 0.896 | 0.492 | 0.895 | 0.777 | 0.571 | 0.658 | 0.833 | 0.708 |
| 32 | AE128 | 0.925 | 0.869 | 0.896 | 0.493 | 0.853 | 0.746 | 0.595 | 0.662 | 0.831 | 0.676 |
| 49 | AE64 | 0.927 | 0.867 | 0.896 | 0.496 | 0.885 | 0.746 | 0.582 | 0.654 | 0.837 | 0.698 |
| 54 | AE32 | 0.931 | 0.864 | 0.896 | 0.511 | 0.889 | 0.728 | 0.556 | 0.630 | 0.850 | 0.685 |
| 60 | AE32 | 0.931 | 0.863 | 0.896 | 0.510 | 0.889 | 0.727 | 0.553 | 0.629 | 0.852 | 0.685 |
| 61 | AE32 | 0.926 | 0.865 | 0.894 | 0.514 | 0.877 | 0.698 | 0.568 | 0.626 | 0.858 | 0.669 |
| 55 | AE32 | 0.926 | 0.864 | 0.894 | 0.512 | 0.877 | 0.699 | 0.569 | 0.627 | 0.851 | 0.669 |
| 66 | AE32 | 0.927 | 0.863 | 0.894 | 0.510 | 0.888 | 0.728 | 0.549 | 0.626 | 0.848 | 0.682 |
| 67 | AE32 | 0.922 | 0.865 | 0.893 | 0.512 | 0.876 | 0.700 | 0.563 | 0.624 | 0.848 | 0.667 |
| 44 | AE64 | 0.915 | 0.870 | 0.892 | 0.498 | 0.852 | 0.743 | 0.588 | 0.656 | 0.826 | 0.676 |
| 50 | AE64 | 0.912 | 0.869 | 0.890 | 0.498 | 0.850 | 0.740 | 0.587 | 0.654 | 0.831 | 0.675 |
| 56 | AE32 | 0.914 | 0.867 | 0.890 | 0.518 | 0.845 | 0.680 | 0.575 | 0.623 | 0.854 | 0.648 |
| 62 | AE32 | 0.914 | 0.867 | 0.890 | 0.518 | 0.845 | 0.678 | 0.572 | 0.621 | 0.848 | 0.648 |
| 68 | AE32 | 0.913 | 0.867 | 0.889 | 0.519 | 0.843 | 0.678 | 0.566 | 0.617 | 0.843 | 0.647 |
| 33 | AE128 | 0.902 | 0.868 | 0.885 | 0.529 | 0.770 | 0.696 | 0.599 | 0.644 | 0.853 | 0.620 |
| 39 | AE64 | 0.894 | 0.870 | 0.882 | 0.535 | 0.767 | 0.693 | 0.595 | 0.641 | 0.850 | 0.616 |
| 45 | AE64 | 0.893 | 0.869 | 0.881 | 0.535 | 0.767 | 0.692 | 0.593 | 0.639 | 0.847 | 0.615 |
| 51 | AE64 | 0.892 | 0.868 | 0.880 | 0.534 | 0.766 | 0.691 | 0.590 | 0.637 | 0.846 | 0.614 |
| 57 | AE32 | 0.886 | 0.865 | 0.875 | 0.551 | 0.763 | 0.627 | 0.580 | 0.602 | 0.870 | 0.596 |
| 63 | AE32 | 0.886 | 0.864 | 0.875 | 0.550 | 0.763 | 0.627 | 0.579 | 0.602 | 0.876 | 0.596 |

(Top 20 of 43 shown; the complete table is in `eligible_action_model_performance.csv`.)

## 5/6. Corrected decomposition and before/after

| Profile | Sensor | UE prep | Uplink+reassembly | Edge queue | Edge proc | Old return+install | Direct publish+install | Saving |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Favorable Stable | 37.2 | 24.9 | 54.9 | 0.0 | 57.7 | 15.4 | 11.8 | 3.7 |
| Mid Variable | 36.5 | 24.5 | 66.1 | 0.0 | 57.4 | 15.4 | 11.7 | 3.6 |
| Adverse Stable | 36.7 | 23.7 | 62.6 | 0.0 | 55.4 | 66.3 | 11.7 | 54.7 |
| Fade Recovery | 38.2 | 24.8 | 60.3 | 0.0 | 56.9 | 15.8 | 11.8 | 4.0 |

Map-feedback arrival at the UE (**live**, median of the four actions): 1.88 ms. This is a controller-observation delay measured separately and is **not** part of physical map-installation AoI.

## Direct-map delay provenance

- Pooling rule: `PER_FAMILY` (Kruskal-Wallis H=124.363, p=8.86e-27, pre-registered alpha=0.05).
- Live family medians (ms): AE128 12.500, AE32 11.722, AE64 10.865, noAE 12.936

### Extrapolations

- the live direct-map delay was measured only under FAVORABLE_STABLE and is applied to all four network profiles; the justification is physical -- the direct path is a container-to-host datagram on the CN5G bridge that never traverses the radio -- but it is an extrapolation and is not independently confirmed per profile
- the live validation covers actions 15/30/50/71; every other action in a family inherits that family's measured delay
- the edge service model, arrival model and quality are unchanged from the final optimized-edge analysis and are not re-measured

### Limitations

- counterfactual model, not a live remeasurement of 288 cells
- only the post-edge installation path changed; transport, admission and edge service are bit-identical to the baseline replay
- agent-feedback arrival is excluded from physical map AoI by design
- no 100-ms service-readiness claim
