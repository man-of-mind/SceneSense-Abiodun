# Training-normalized actor-volume pedestrian visibility — follow-up audit

Reference build `training_reference/20260901_212409` (232 s) ·
audit run `normalized/20260901_213040` (5.2 s) · CPU only.

**Terminal: `TRAIN_NORMALIZED_ACTOR_VOLUME_VISIBILITY_NOT_FEASIBLE_RETAIN_HUMAN_BANDS`**

All 22 qualification checks pass, the training reference is leakage-free, and
every expected-support value is finite and positive — so this is a valid
measurement of the normalised method, not an implementation-invalid result.
Three decision gates fail. Part 3 (the independent human-audit package) was
therefore **not** generated, as specified.

Scope: no model trained or run, no prediction or checkpoint read, no test row,
no CUDA, no CARLA, `torch` never imported, FCOS/LR-ASPP untouched, no full
validation rescore.

---

## 1. Formula

The actor-volume extraction is reused **verbatim** from commit `dc5238d`:
0.05 m containment tolerance, 0.03 m ground rejection, identical
back-projection, identical actor-local containment, identical deterministic
overlapping-actor assignment. Only the score built on top of it changes.

```
support_density               = retained_actor_volume_pixels
                                / clipped_projected_box_area
expected_clear_support_density = 95th percentile (method="higher") of
                                 support_density over a comparable
                                 training-only group
normalized_visibility          = clamp(support_density
                                       / expected_clear_support_density, 0, 1)
truncation                     = 1 - clipped_area / unclipped_area   (separate)
```

Conditioning: `gt_actor_type_id` x folded relative view angle
(`[0,30)`, `[30,60)`, `[60,90]` degrees) x projected full-box height
(`<24`, `[24,48)`, `[48,96)`, `>=96` px).

Folded relative view angle is the angle between the actor's facing axis and the
line of sight in the world XY plane, folded to `[0, 90]` — 0 degrees is head-on
*or* directly away (the silhouette width is the same either way), 90 degrees is
full profile.

Fallback hierarchy, applied in order: actor type + angle + height if n >= 50;
angle + height if n >= 100; height if n >= 100; otherwise the global training
reference.

**Extraction identity was verified, not assumed.** Re-scoring the 100 pilot
samples reproduces the pilot's `retained_actor_point_count`,
`clipped_projected_area_px`, `clipped_bbox_h`, `visible_bbox_*`, `truncation`,
`no_support` and `competing_actor_boxes` with a maximum absolute difference of
exactly `0.0`.

### One thing worth stating precisely

The brief describes this as replacing the denominator, and it does — the cuboid
area is gone. But the *numerator* convention changes too: the pilot scored
`area(B_visible)`, a bounding-box area, while `support_density` counts retained
**pixels**. That is what the specified formula requires, and it is the direct
cause of the result in section 5, so it is recorded here rather than left
implicit.

---

## 2. Training reference — source coverage

Driven strictly off `dataset/manifest.csv`, `split == "train"`. The manifest
contains no `test` split at all (`{"train": 16827, "val": 3345}`).

| | |
|---|---|
| Training episodes | 10 (`canonical_v3_01..04`, `extra_v3_09..14`) |
| Training frames in manifest | 16,827 |
| Person GT rows in those episodes | 130,923 |
| Qualified after the locked filter | **21,972** |
| Unique frames opened | 11,212 |
| Actors attempted / extracted | 21,972 / 21,972 (0 skipped) |
| Zero-support actors (fully occluded) | 2,169 |
| Invalid-support actors | 0 |

Locked filter: training split only, distance <= 40 m, in-frame fraction >= 0.98,
finite geometry. The in-frame fraction is **recomputed** from the recorded
clipped and unclipped projected areas rather than read from a stored field, so
no historical visibility or eligibility flag enters the population.

### Leakage proof

* Split membership enforced by set intersection with the dataset manifest, not
  by trusting episode names.
* Validation sample ids in the population: **0**. In the emitted records: **0**.
* Test rows read: **0** (no test split exists in this dataset).
* Human annotation files read: **0**; the human pilot directory is never opened
  — `build_training_reference.py` does not import `HUMAN_PILOT_DIR` and has no
  path into it. The reference was written and hashed before any annotation was
  loaded, and `run_normalized_audit.py` verifies its sha256 before use.
* `training_reference.json` and `training_support_records.csv` are chmod `444`.

### Group and fallback counts

| Tier | Groups | n >= its threshold |
|---|---|---|
| type + angle + height | 417 | 154 (threshold 50) |
| angle + height | 12 | 12 (threshold 100) |
| height | 4 | 4 (threshold 100) |
| global | 1 | 21,972 |

Height-tier references (95th percentile of support density):
`h_lt24` 0.6229 (n=5066) · `h24_48` 0.5452 (n=12178) ·
`h48_96` 0.5050 (n=3931) · `h_ge96` 0.4682 (n=797). Global 0.5570.

Angle conditioning behaves as designed: head-on/away pedestrians carry more
surface support than oblique ones (e.g. `a00_30|h_lt24` 0.6950 vs
`a30_60|h_lt24` 0.4986).

**No reference group anywhere in any tier is non-positive or non-finite.**

### Fallback usage on the 100 pilot samples

| Tier | All 100 | 77 scoreable |
|---|---|---|
| type + angle + height | 51 | 33 |
| angle + height | 48 | 43 |
| height | 1 | 1 |
| global | 0 | 0 |

Smallest resolved group n = 51. Expected support used spans 0.2051 to 0.8000,
all finite and positive.

---

## 3. Agreement against annotator A

77 scoreable samples; 23 `ambiguous` excluded. Rows = human, columns =
automatic, ordinal order `not-observable / heavy / partial / bare`.

**Normalized actor-volume (new)**

```
              not  heavy  partial  bare
not-observable 11      3        0     0
heavy           2     14        3     0
partial         0     11        5     0
bare            0      3       19     6
```

**Unnormalized actor-volume (pilot)**

```
              not  heavy  partial  bare
not-observable  7      7        0     0
heavy           3     13        2     1
partial         0      5       10     1
bare            0      4       20     4
```

**Depth-interval occupancy (original)**

```
              not  heavy  partial  bare
not-observable 12      2        0     0
heavy           9     10        0     0
partial         1     15        0     0
bare            0     23        5     0
```

| Statistic | Normalized | Unnormalized | Old depth-only |
|---|---|---|---|
| Exact agreement | **0.4675** | 0.4416 | 0.2857 |
| Linear weighted Cohen's kappa | **0.5175** | 0.4581 | 0.2043 |
| Spearman rho | **0.7953** | 0.7169 | 0.8307 |
| >= 0.65 TP / FN / FP / TN | 30 / 14 / 3 / 30 | **35 / 9 / 3 / 30** | 5 / 39 / 0 / 33 |
| Balanced accuracy (>= 0.65) | 0.7955 | **0.8523** | 0.5568 |

### Score distribution by human band

| Human band | n | norm median | norm p25–p75 | norm max | unnorm median | old median |
|---|---|---|---|---|---|---|
| not-observable | 14 | 0.1088 | 0.005–0.189 | 0.609 | 0.2096 | 0.0942 |
| heavy | 19 | 0.3519 | 0.281–0.521 | 0.890 | 0.3659 | 0.2430 |
| partial | 16 | 0.5347 | 0.409–0.673 | 0.801 | 0.7116 | 0.3523 |
| bare | 28 | **0.7878** | 0.732–0.851 | 1.000 | 0.7628 | 0.4998 |

Medians increase strictly (0.1088 -> 0.3519 -> 0.5347 -> 0.7878) and the bands
are better separated than before, so the monotonicity gate passes.

### By distance

| Band | n | norm kappa / bal-acc | unnorm kappa / bal-acc | old kappa / bal-acc |
|---|---|---|---|---|
| 0–10 m | 23 | **0.562 / 0.944** | 0.392 / 0.844 | 0.135 / 0.556 |
| 10–20 m | 24 | 0.359 / 0.667 | 0.346 / **0.900** | 0.127 / 0.533 |
| 20–30 m | 22 | 0.626 / 0.829 | **0.698 / 0.862** | 0.382 / 0.643 |
| 30–40 m | 8 | **0.368 / 0.875** | 0.027 / 0.625 | 0.179 / 0.500 |

The 30–40 m cell has 8 non-ambiguous samples and is not independently
informative.

---

## 4. Twenty largest disagreements

Ranked by ordinal band distance, tie-broken by how far the score falls outside
the human band's own interval. Artifacts: `largest_disagreements.csv` and
`largest_disagreements.png` (20 panels, 4 x 5).

Three are two-band errors, all human-`bare` scored `heavy`: panels 072 (0.369),
056 (0.487) and 057 (0.617). The remaining 17 are one-band. Eight are
human-`partial` pushed down to `heavy` — the single largest error cell and the
one that costs the binary gate. No disagreement in the top twenty carries
non-zero truncation.

---

## 5. Why it fails

The normalisation does what it was designed to do. It fixes the pilot's scale
bias — the `bare` band median rises from 0.7628 toward the 0.90 target while the
`partial` median falls away from it, so the bands separate — and it improves
every ordinal statistic: kappa 0.4581 -> 0.5175, Spearman 0.7169 -> 0.7953,
exact agreement 0.4416 -> 0.4675.

It fails because the statistic it normalises is **pixel count**, and pixel count
conflates external occlusion with silhouette shape. The diagnostic that shows
this is `visible_box_fill_ratio`, how densely retained points fill their *own*
visible box:

| Human band | fill ratio | support density | expected support |
|---|---|---|---|
| not-observable | 0.2872 | 0.0542 | 0.4842 |
| heavy | 0.3944 | 0.1426 | 0.4589 |
| partial | 0.3701 | 0.2662 | 0.4587 |
| bare | 0.4822 | 0.3691 | 0.4589 |

Even a completely unoccluded pedestrian fills only 48 % of its own visible box —
thin limbs, gaps between arms and torso, loose clothing. And `partial` (0.3701)
is *sparser* than `heavy` (0.3944), so fill density does not order the middle of
the range at all. A partially occluded pedestrian is penalised twice: once for
the occlusion, once for a naturally sparse silhouette. Dividing by a group-level
95th percentile corrects the group scale but cannot correct per-instance
silhouette variance.

The consequence is concentrated in the middle band. Among human-`partial`
samples, those scoring >= 0.65 fall from 11/16 to 5/16, while human-`bare`
holds at 24/28 -> 25/28. That single shift is the whole balanced-accuracy loss
(TP 35 -> 30).

The two conventions therefore trade off: the pilot's bounding-box area is robust
to holes in the retained mask but insensitive to partial occlusion; pixel
density is sensitive to occlusion but noisy with respect to pose. Neither, on
its own, clears both bars.

The overlays corroborate this — in panels 056, 057, 072, 075 and 019 the green
retained points sit correctly on the pedestrian but are visibly patchy.

---

## 6. Decision

| Gate | Required | Observed | Result |
|---|---|---|---|
| All geometry qualification checks still pass | yes | 22/22 | PASS |
| Training reference free of validation/test leakage | yes | 0 val, 0 test, 0 human | PASS |
| All expected-support values finite and positive | yes | 0.2051–0.8000, 0 non-positive | PASS |
| Band medians increase monotonically | strictly | 0.1088, 0.3519, 0.5347, 0.7878 | PASS |
| Linear weighted Cohen's kappa | >= 0.60 | **0.5175** | **FAIL** |
| Balanced accuracy at >= 0.65 | >= 0.80 | **0.7955** | **FAIL** |
| Not worse than unnormalized on kappa | >= 0.4581 | 0.5175 | PASS |
| Not worse than unnormalized on balanced accuracy | >= 0.8523 | **0.7955** | **FAIL** |

**`TRAIN_NORMALIZED_ACTOR_VOLUME_VISIBILITY_NOT_FEASIBLE_RETAIN_HUMAN_BANDS`**

No constant was modified and no configuration was re-run after the result was
observed. Part 3 was not executed, because it is conditional on Part 2 passing.
The human bands remain the reference. FCOS and LR-ASPP were not rescored.

---

## 7. Artifacts

Training reference —
`data_collection/experiments/route_b_publication_actor_volume_visibility_v1/training_reference/20260901_212409/`

| Artifact | sha256 |
|---|---|
| `training_reference.json` | `85ab5db71c34a4e0eaee2de49a85858760a56e1de91aafcebb35f5fdada2d689` |
| `training_support_records.csv` (21,972 rows) | `bf428c060291414226f308b0d67c325e6d3ff8489dce04c022faf138c706315f` |

Audit run —
`data_collection/experiments/route_b_publication_actor_volume_visibility_v1/normalized/20260901_213040/`

| Artifact | sha256 |
|---|---|
| `normalized_visibility_scores.csv` | `75d2d9aaa4ab876381abc2cbac0cdb9330a55dd00c2485f5a4795629a8188ec6` |
| `normalized_visibility_with_human_bands.csv` | `a3a23da622915aafb4c25cbff458200eeb7f6e3686cc66c79b2fbcacc572d4d6` |
| `largest_disagreements.csv` | `8e429da0b51b6d4249b2bb6cb9bdde7abbc7322b49b7b18e539b4adf4cbfe6a5` |
| `largest_disagreements.png` (2400x1984) | `356ac4fdb4811a1c6b37fe3687e98374f1e5ff3e43a3b19df47918a61af75283` |
| `RUN_METADATA.json` | `593123c160a4198ec088ba210ef2c4f4f2b0562646de5591fbb7632a4368eb7c` |

Implementation — `data_collection/route_b_publication_actor_volume_visibility_v1/`

| File | sha256 |
|---|---|
| `training_reference.py` | `dbe523fd892a70d3231a2c076ac623c0e092b80cbb7033c9ba5ec2304dfc0e02` |
| `build_training_reference.py` | `f1d419638caa1265176a4974ef11f03e99bec9ae148ee381af1d3cce9f2ae796` |
| `run_normalized_audit.py` | `c0212e12a0e776e3b7616bff9e0a88500539e1586e279d3a51e6e7da725e9898` |
| `contact_sheet.py` | `d6e294592090906d5e7a71cff3d10d8ec0bfc971210423b50fb57c4a27b9c0ef` |
| `core.py` | `02d39fd1d15c31ab8323f0ece09e951e8fc0e42aecf5de06a120918276f1d3e4` |
| `scoring.py` | `3942b8c35c990c27676fc66c13f80ed8011e2a2ddc0768cac5c6e531f85b00dd` |
| `agreement.py` | `a5ebb150cbbdeebaafedb61b99c9130291ee5574d63a1aa9a418dcd559fcce28` |
| `run_audit.py` | `cc6a0afa9997d6d1ccc603ec990046e8c6b0fcbe9ba9b442abaad0589efa7701` |

`core.py`, `scoring.py`, `agreement.py` and `run_audit.py` are unchanged from
commit `dc5238d`; `contact_sheet.py` gained only a reusable tiling helper.

Reproduce:

```
CUDA_VISIBLE_DEVICES="" python3 -m \
  data_collection.route_b_publication_actor_volume_visibility_v1.build_training_reference
CUDA_VISIBLE_DEVICES="" python3 -m \
  data_collection.route_b_publication_actor_volume_visibility_v1.run_normalized_audit \
  --reference-run 20260901_212409
CUDA_VISIBLE_DEVICES="" python3 -m unittest discover \
  -s data_collection/route_b_publication_actor_volume_visibility_v1/tests -t .
```
