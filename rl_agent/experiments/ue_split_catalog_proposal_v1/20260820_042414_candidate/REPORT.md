# UE SPLIT candidate-catalog proposal

**Verdict:** `PASS_CANDIDATE_PROPOSAL_REVIEW_REQUIRED_NO_RUN_AUTHORITY`

This artifact records the approved aggregate quality floor and proposes
candidates using only retained evidence. It is not a final action catalog,
does not authorize a measurement, and cannot launch CARLA/OAI/training.
The point estimates are catalog-development evidence from the frozen offline
set, not live/deployment certification; independent bounded validation remains
required after the final catalog is selected.

## Approved service decision

- Service: `OBJECT_MAP_V1`.
- Required object position: predicted actor-reference world XY (not a mask centroid).
- Segmentation IoU remains secondary and cannot veto object-map eligibility.
- Absolute floor passes 28/72 profiles.
- Absolute plus same-q0 screen yields 26 normal aggregate candidates.
- Rescue candidate: `ae32/u4/q0.9` at 19.8 KiB P95.

## Nonbinding audit-priority shortlist

| Tier | Role | Profile | P95 KiB | Veh R | Ped R | mIoU |
|---|---|---|---:|---:|---:|---:|
| DEGRADED_RESCUE | degraded_rescue | `ae32/u4/q0.9` | 19.8 | 0.921 | 0.841 | 0.424 |
| NORMAL | compact_preferred_pending_high_roi_detail | `ae64/u4/q0.7` | 49.5 | 0.916 | 0.869 | 0.651 |
| NORMAL | compact_evidence_complete_fallback | `ae32/u4/q0.5` | 51.9 | 0.920 | 0.852 | 0.656 |
| NORMAL | mid_payload_localization_balanced | `ae32/u4/q0.3` | 66.1 | 0.922 | 0.860 | 0.715 |
| NORMAL | mid_payload_pedestrian_recall | `ae64/u4/q0.5` | 66.5 | 0.918 | 0.871 | 0.702 |
| NORMAL | high_object_quality_pedestrian_recall | `ae128/u4/q0.3` | 101.3 | 0.924 | 0.886 | 0.684 |
| NORMAL | secondary_segmentation_preserving_reference | `ae64/u4/q0` | 103.8 | 0.919 | 0.862 | 0.824 |

The shortlist is an audit priority only. Exact eight-metric dominance leaves
23/26 normal candidates non-dominated, so reducing to a final N needs an approved
equivalence/catalog-budget rule rather than post-hoc preference.

## Difficult-object evidence

Retained per-object rows reproduce aggregate counts/localization for the
q<=0.5 diagnostic profiles. Horizontal range is auditable; precision by
range is not defined because false positives have no GT range. Small-object
recall remains unresolved because every FN lacks GT box size and the source
dataset is absent.

| Profile | Class | 30m+ recall | 30m+ XY MAE (m) |
|---|---|---:|---:|
| `ae128/u4/q0` | person | 0.868 | 1.007 |
| `ae128/u4/q0` | vehicle | 0.884 | 0.938 |
| `ae128/u4/q0.3` | person | 0.866 | 1.009 |
| `ae128/u4/q0.3` | vehicle | 0.887 | 0.951 |
| `ae32/u4/q0` | person | 0.830 | 0.986 |
| `ae32/u4/q0` | vehicle | 0.887 | 0.976 |
| `ae32/u4/q0.3` | person | 0.831 | 0.970 |
| `ae32/u4/q0.3` | vehicle | 0.885 | 0.976 |
| `ae32/u4/q0.5` | person | 0.820 | 1.090 |
| `ae32/u4/q0.5` | vehicle | 0.883 | 1.067 |
| `ae64/u4/q0` | person | 0.834 | 1.001 |
| `ae64/u4/q0` | vehicle | 0.883 | 0.942 |
| `ae64/u4/q0.5` | person | 0.840 | 1.116 |
| `ae64/u4/q0.5` | vehicle | 0.881 | 1.024 |

## Required review before final N

1. Approve an equivalence/catalog-budget rule; do not equate this proposal with N=27.
2. Decide whether q0.7 compact evidence warrants one bounded offline detail regeneration.
3. Keep the q0.9 rescue provisional and service-debt-labelled.
4. Treat small-object certification as unavailable unless the frozen source GT is restored.
5. Only a later, separately approved freeze may create the N-action catalog or OAI boundary plan.
