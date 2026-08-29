# Route B v3.1 LR-ASPP localizer counterfactual audit

Primary terminal: `LRASPP_INHERITED_LOCALIZER_DOES_NOT_RECOVER`

Secondary attribution: `LRASPP_DEPTH_ERROR_DOMINANT`

## Conclusion

The epoch-40 LR-ASPP localization field improves the corrected visible-anchor detector, but it does not meet the preregistered recovery contract. It is not enough to justify a hybrid qualification run, and the GT-centre-cell diagnostic shows that hard native-cell sampling is not the principal reason it falls short.

The depth/ray counterfactual is decisive. On the 554 score-0.02 fixed-IoU50 pairs that failed the 3 m world criterion, replacing only predicted depth with GT depth recovered 554/554. Replacing only the predicted ray with the GT ray recovered 1/554. The person-private depth/range estimate is therefore the dominant localization failure; the corrected visible anchor and its camera ray are not.

No training is licensed by this audit. Further person-head work under the same frozen LR-ASPP `{low, high}` transport contract should stop. The next person-accuracy effort should use an architecture with a materially stronger depth/range representation, particularly for far and radar-unsupported people.

## Registered execution contract

- Reuse the published visible-anchor epoch-18 detections; do not rerun the candidate.
- Run at most one frozen epoch-40 validation traversal.
- Sample the epoch-40 native localization field at `floor(predicted_full_box_center / 4)`, with no interpolation or clamping.
- Keep fixed score-0.02 and score-0.20 IoU50 associations for localization counterfactuals.
- Evaluate a GT-centre-cell lookup only as a diagnostic oracle.
- Separate the primary composition decision from the secondary ray-versus-depth attribution.
- Do not train, optimize, sweep thresholds/NMS, evaluate v0.25, or touch the locked test set.

The audit followed this contract: candidate traversals 0, epoch-40 traversals 1, segmentation traversals 0, optimizer steps 0, and training runs 0. The one traversal covered all 3,345 validation frames and persisted only 21,261 candidate-cell samples and 3,872 GT-cell samples, not dense maps.

## Reconciliation

The published base and candidate canonical, IoU50, conditional localization, taxonomy, vehicle, and segmentation results reconcile within `1e-12`. Vehicle and segmentation preservation checks are exact.

| Arm | P@.20 | R@.20 | F1@.20 | R@.02 | XY MAE (m) | IoU50 F1@.20 | IoU50 R@.02 | IoU50 <=3 m@.02 | World-wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Epoch-40 base | 0.537513 | 0.518079 | 0.527617 | 0.615186 | 1.341153 | 0.529474 | 0.566374 | 0.814865 | 766 |
| Visible-anchor epoch 18 | 0.543137 | 0.500775 | 0.521096 | 0.595816 | 1.286255 | 0.553834 | 0.596333 | 0.760069 | 861 |
| Epoch-40 field at candidate cell | 0.547672 | 0.510331 | 0.528342 | 0.606663 | 1.350762 | 0.553834 | 0.596333 | 0.770463 | 809 |
| Epoch-40 field at GT cell (diagnostic) | 0.554692 | 0.516012 | 0.534653 | 0.610795 | 1.280324 | 0.553834 | 0.596333 | 0.786921 | 799 |

The deployable lookup passes the F1 and world-wrong-reduction gates, reducing 52 of the candidate's 95 additional world-wrong cases. It fails recall by 0.007748, worsens XY MAE by 0.009609 m, and reaches 0.770463 rather than the registered 0.80 conditional-localization floor.

The GT-cell diagnostic passes F1, XY MAE, and world-wrong reduction, but still misses base recall by 0.002066 and reaches only 0.786921 conditional localization. Under the frozen rule, the result is not sampling-limited.

## Depth/ray attribution

The fixed score-0.02 IoU50 set contains 2,309 pairs.

| Counterfactual | Mean error (m) | Median (m) | P90 (m) | Within 3 m | Canonical F1@.20 |
|---|---:|---:|---:|---:|---:|
| Predicted ray + predicted depth | 2.157309 | 1.641853 | 4.656899 | 0.760069 | 0.521096 |
| Predicted ray + GT depth | 0.073474 | 0.058675 | 0.154323 | 1.000000 | 0.624934 |
| GT ray + predicted depth | 2.156560 | 1.629057 | 4.658111 | 0.757904 | 0.520758 |
| GT ray + GT depth | 0.000004 | 0.000003 | 0.000007 | 1.000000 | 0.624934 |

The deployable value in the GT-depth row is an oracle, not a proposed inference path. Its purpose is attribution: a sufficiently accurate range source would unlock the corrected detector, producing precision 0.640879, recall 0.609762, F1 0.624934, low-score recall 0.723915, and XY MAE 0.244261 m.

The GT/GT world-coordinate round trip has a maximum XY error of `1.979e-05 m`. The offline sanity tolerance was repaired from `1e-6` to `1e-4 m` after the first analysis-only attempt exposed ordinary CSV decimal round-trip error. No inference was repeated and the final tolerance remains two orders of magnitude below the 3 m metric.

## Sampling, transitions, and slices

Candidate and GT centres share the same native cell for 1,258/2,309 pairs (54.48%); 97.62% are within one cell. The inherited field is within 3 m for 0.770463 of pairs at the candidate cell and 0.786921 at the GT cell. The small GT-cell gain is real but insufficient.

At score 0.02, the candidate changes 290 base 2D misses into matches and loses 174 base matches. Canonically, it changes 438 base false negatives into true positives but changes 513 base true positives into false negatives. Another 236 candidate IoU50 matches are lost between score 0.02 and 0.20, confirming a secondary confidence/ranking issue rather than overturning the depth attribution.

| Slice, score 0.02 | Candidate | Epoch-40 at candidate cell | Epoch-40 at GT cell |
|---|---:|---:|---:|
| 0-10 m | 0.9111 | 0.9333 | 0.9185 |
| 10-20 m | 0.8407 | 0.8829 | 0.8935 |
| 20-30 m | 0.6831 | 0.6912 | 0.7095 |
| 30-40 m | 0.6761 | 0.6028 | 0.6423 |
| Radar supported | 0.7681 | 0.7864 | 0.7987 |
| Radar unsupported | 0.6179 | 0.4878 | 0.5772 |
| Clear v0.25 | 0.7748 | 0.7786 | 0.7869 |
| Primary-v0.10-only | 0.5671 | 0.6646 | 0.7866 |

The epoch-40 field helps nearby and radar-supported people but regresses at 30-40 m and without radar support. A follow-up architecture should preserve the visible-centre target correction while adding depth-aware spatial capacity and a stronger radar-conditioned range path; simply reusing this mature LR-ASPP field is not enough.

## Runtime and scope integrity

- Reconciliation wall time: 51.23 s.
- Frozen traversal wall time: 101.46 s.
- Offline analysis wall time: 75.03 s.
- Total measured compute wall time: 227.72 s.
- Traversal peak VRAM: 84.90 MiB allocated and 156.00 MiB reserved.
- Base checkpoint SHA-256: `5c6bb268b43f4dd84bd7a283ff483ec4e87366a50ea51dfacee44979df2bf6e8`.
- Candidate checkpoint SHA-256: `62330263c90ae8d71c2d44b7e0cf164b08dd3f928bee6966f023c619b629b5fc`.

The initial sandboxed traversal launch detected unavailable CUDA and stopped before loading a checkpoint or processing a frame. The single counted traversal then ran with GPU access; this did not consume a second validation traversal. The first sandboxed desktop notification was likewise denied, and the host-context retry completed successfully.

Locked test remained absent and unopened. Existing experiments, production model/decoder/evaluator, CARLA, OAI, q/quant/AE/zstd, live runtime, and the 288 measurements were untouched. The audit wrote no checkpoint and mutated no dataset or published prediction artifact.

Authoritative generated report: `experiments/route_b_v3_1_localizer_counterfactual_audit_v1/20260829_031500/FINAL_REPORT.md`
