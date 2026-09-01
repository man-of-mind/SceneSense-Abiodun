# Box-normalized actor-volume pedestrian visibility — final corrected experiment

Reference build `training_reference/20260901_214026` (202 s) ·
audit run `box_normalized/20260901_214534` (5.1 s) · CPU only · run once.

**Terminal: `BOX_NORMALIZED_ACTOR_VOLUME_VISIBILITY_NOT_FEASIBLE_FINAL_RETAIN_HUMAN_BANDS`**

All 24 qualification, leakage and integrity checks pass, so this is a valid
measurement. Three decision gates fail. Visibility-method development stops
here; the retention decision is in section 7.

Scope: no model inference or training, no CUDA, no CARLA, no predictions, no
test data, no FCOS/LR-ASPP rescoring, no threshold tuning. Source state was
commit `21c3c45`. No prior artifact was modified.

---

## 1. The correction

The previous attempt replaced both the denominator *and* the numerator, swapping
`area(B_visible)` for a retained-pixel count, so pose gaps and rendering holes
read as occlusion. This run applies the intended denominator-only correction:

```
raw_box_visibility   = area(B_visible) / area(B_full_clipped)      (0 if no support)
corrected_visibility = clamp(raw_box_visibility
                             / expected_clear_raw_box_visibility, 0, 1)
```

`expected_clear_raw_box_visibility` is the 95th percentile (`method="higher"`)
of `raw_box_visibility` over the same training-only groups — actor type x folded
view-angle bin x projected-height bin — under the same 50/100/100/global
fallback hierarchy and the same non-truncated training population as before.

Everything else is untouched: actor-volume point extraction, 0.05 m containment
tolerance, 0.03 m ground rejection, overlap assignment, qualification logic and
human-comparison logic.

### Integrity, verified rather than asserted

| Check | Result |
|---|---|
| `raw_box_visibility` equals the originally audited `visibility` | max abs diff **0** |
| Extraction vs pilot (`retained_actor_point_count`, `visible_box_area_px`, `clipped_projected_area_px`, `truncation`) | max abs diff **0** |
| Rebuilding the *pixel* reference from the new training records reproduces the frozen `20260901_212409` tables | **bit-identical** |
| Pixel-normalized score recomputed here vs the previous run's CSV | max abs diff **0** |
| `corrected = clamp(raw / expected)` | max abs diff 2.2e-16 |
| No-support actor-frames | all score exactly 0.0 |

The third row is the important one: because the same records reproduce the
earlier reference exactly, the population filter, binning, percentile and
fallback provably did not change — only which statistic is aggregated.

---

## 2. Corrected training reference

Same 21,972 qualifying training person GT from 11,212 frames across the 10
training episodes; 0 skipped, 0 invalid, 2,169 zero-support (identical to the
previous build). Zero validation rows, zero test rows, zero human-annotation
reads. No non-positive group in any tier.

95th percentile of `raw_box_visibility`, height tier:
`h_lt24` 0.9432 (n=5066) · `h24_48` 0.9225 (n=12178) ·
`h48_96` 0.9228 (n=3931) · `h_ge96` 0.8977 (n=797). Global 0.9259.

Group counts are unchanged: 417 type+angle+height (154 at n >= 50), 12
angle+height, 4 height, 1 global.

### Fallback usage on the 100 pilot samples

| Tier | All 100 | 77 scoreable |
|---|---|---|
| type + angle + height | 51 | 33 |
| angle + height | 48 | 43 |
| height | 1 | 1 |
| global | 0 | 0 |

Smallest resolved group n = 51. Expected-clear values span 0.4694 to 0.9642, all
finite and positive; 3 samples hit the clamp ceiling.

---

## 3. Agreement against annotator A

77 scoreable samples; 23 `ambiguous` excluded. Rows = human, columns =
automatic, ordinal order `not-observable / heavy / partial / bare`.

**Box-normalized (corrected)**

```
              not  heavy  partial  bare
not-observable  6      8        0     0
heavy           2     10        4     3
partial         0      2       12     2
bare            0      3       18     7
```

**Pixel-support-normalized (previous attempt)**

```
              not  heavy  partial  bare
not-observable 11      3        0     0
heavy           2     14        3     0
partial         0     11        5     0
bare            0      3       19     6
```

**Unnormalized actor-volume (original pilot)**

```
              not  heavy  partial  bare
not-observable  7      7        0     0
heavy           3     13        2     1
partial         0      5       10     1
bare            0      4       20     4
```

**Old depth-only occupancy**

```
              not  heavy  partial  bare
not-observable 12      2        0     0
heavy           9     10        0     0
partial         1     15        0     0
bare            0     23        5     0
```

| Statistic | Box-normalized | Pixel-normalized | Unnormalized | Old depth-only |
|---|---|---|---|---|
| Exact agreement | 0.4545 | **0.4675** | 0.4416 | 0.2857 |
| Linear weighted kappa | 0.4533 | **0.5175** | 0.4581 | 0.2043 |
| Spearman rho | 0.7044 | 0.7953 | 0.7169 | **0.8307** |
| >= 0.65 TP / FN / FP / TN | 39 / 5 / 7 / 26 | 30 / 14 / 3 / 30 | 35 / 9 / 3 / 30 | 5 / 39 / 0 / 33 |
| Balanced accuracy | 0.8371 | 0.7955 | **0.8523** | 0.5568 |

### Medians by human band

| Human band | n | corrected | pixel-norm | unnormalized | old |
|---|---|---|---|---|---|
| not-observable | 14 | 0.2318 | 0.1088 | 0.2096 | 0.0942 |
| heavy | 19 | 0.5826 | 0.3519 | 0.3659 | 0.2430 |
| partial | 16 | 0.7987 | 0.5347 | 0.7116 | 0.3523 |
| bare | 28 | **0.8552** | 0.7878 | 0.7628 | 0.4998 |

Corrected medians increase strictly (0.2318 -> 0.5826 -> 0.7987 -> 0.8552), so
the monotonicity gate passes, and the `bare` median moves closest yet to the
0.90 target.

### By distance (weighted kappa / balanced accuracy)

| Band | n | corrected | pixel-norm | unnormalized | old |
|---|---|---|---|---|---|
| 0–10 m | 23 | 0.536 / 0.872 | 0.562 / **0.944** | 0.392 / 0.844 | 0.135 / 0.556 |
| 10–20 m | 24 | 0.370 / **0.933** | 0.359 / 0.667 | 0.346 / 0.900 | 0.127 / 0.533 |
| 20–30 m | 22 | 0.491 / 0.800 | 0.626 / 0.829 | **0.698 / 0.862** | 0.382 / 0.643 |
| 30–40 m | 8 | 0.135 / 0.750 | **0.368 / 0.875** | 0.027 / 0.625 | 0.179 / 0.500 |

The 30–40 m cell has 8 non-ambiguous samples and is not independently
informative.

---

## 4. Twenty largest disagreements

`largest_disagreements.csv` and `largest_disagreements.png` (20 panels, 4 x 5),
ranked by ordinal band distance then by distance outside the human band.

Six are two-band errors, and unlike every previous run they go in **both**
directions: panels 072, 008, 096 are human-`bare` scored `heavy`, while 073, 013
and 014 are human-`heavy` scored `bare`. The remaining 14 are one-band. No
disagreement in the top twenty carries non-zero truncation.

---

## 5. Why it fails

The correction does exactly what it was designed to do about **scale**. The
`bare` median rises 0.7628 -> 0.8552, the closest any variant has come to the
0.90 target, and recall improves in every band that should read as visible:
human-`partial` samples scoring >= 0.65 rise 11/16 -> 14/16 and human-`bare`
24/28 -> 25/28.

But scale was not the binding constraint. Two things show this.

**The correction is close to a uniform rescale.** The expected-clear values
actually used on the pilot cluster tightly around 0.88–0.94 (only one sample
resolved below 0.79). Dividing a score by a near-constant ~0.9 is a monotone
transform, which cannot change ordinal agreement much — and it does not:
weighted kappa moves 0.4581 -> 0.4533 and Spearman 0.7169 -> 0.7044, both
essentially flat, the small losses coming from the per-group divisor adding
variance without adding discrimination.

**Lifting the scale lifts the wrong band too.** The human-`heavy` median rises
0.3659 -> 0.5826, so human-`heavy` samples crossing 0.65 go 3/19 -> 7/19. False
positives rise 3 -> 7 while false negatives fall 9 -> 5, and balanced accuracy
nets out slightly worse: 0.8523 -> 0.8371.

The underlying reason is visible in the overlays. `B_visible` is a *bounding
box* over the retained points, so occlusion that punches holes or removes
interior regions without shrinking the box extent barely moves the score.
Panel 073 is the clearest case: the pedestrian is broken up behind bus-shelter
structure, a human calls it `heavy`, yet the surviving fragments still span
almost the whole projected box, giving `raw_box_visibility` 0.942 and a
corrected 1.000.

So the three variants tried are each limited by a different property of their
statistic, and the limitation is structural rather than one of calibration:

* **bounding-box area** (unnormalized and corrected) — robust to holes, and
  therefore insensitive to exactly the interior occlusion that separates
  `heavy` from `partial`;
* **pixel-fill density** (previous attempt) — sensitive to that occlusion, but
  it also responds to pose and silhouette sparsity, which is noise here.

Correcting the cuboid-denominator bias was a real and necessary fix; it simply
was not what stood between this metric and a 4-band kappa of 0.60.

---

## 6. Decision

| Gate | Required | Observed | Result |
|---|---|---|---|
| All geometry and leakage qualifications pass | yes | 24/24 | PASS |
| Expected-clear values finite and positive | yes | 0.4694–0.9642, 0 non-positive | PASS |
| Band medians increase monotonically | strictly | 0.2318, 0.5826, 0.7987, 0.8552 | PASS |
| Linear weighted Cohen's kappa | >= 0.60 | **0.4533** | **FAIL** |
| Balanced accuracy at >= 0.65 | >= 0.80 | 0.8371 | PASS |
| Not worse than unnormalized on kappa | >= 0.4581 | **0.4533** | **FAIL** |
| Not worse than unnormalized on balanced accuracy | >= 0.8523 | **0.8371** | **FAIL** |

**`BOX_NORMALIZED_ACTOR_VOLUME_VISIBILITY_NOT_FEASIBLE_FINAL_RETAIN_HUMAN_BANDS`**

The experiment was run once. No constant was altered, no threshold was tuned,
and no hybrid or composite score was attempted after seeing the result. The
conditional independent human-audit package was **not** generated.

---

## 7. Retention decision — visibility-method development stops

Per the pre-registered failure path:

* **Human visibility bands** — the reference for publication analysis. Annotator
  A's bands on the 100-panel pilot stand as the visibility ground truth; the
  23 `ambiguous` rows remain excluded from any agreement statistic.
* **Unnormalized actor-volume >= 0.65** — retained as *supporting binary pilot
  evidence* only: balanced accuracy 0.8523, TP 35 / FN 9 / FP 3 / TN 30 on 77
  non-ambiguous samples. It is a binary observability screen, not a graded
  visibility score, and its 4-band kappa of 0.4581 must not be quoted as
  band-level agreement.
* **Old depth-only occupancy** — internal sensitivity analysis only
  (kappa 0.2043, balanced accuracy 0.5568). Not for publication claims.

No further visibility-metric variants are to be developed. FCOS and LR-ASPP
were not rescored and full validation rescoring remains unauthorised.

---

## 8. Artifacts

Corrected training reference —
`.../training_reference/20260901_214026/`

| Artifact | sha256 |
|---|---|
| `training_reference.json` | `a825cffac4a060ee422951bb7d5af0b10d15eb39a347c081af836de35e6c1fff` |
| `training_support_records.csv` (21,972 rows, both statistics) | `8755b1904c821e6942197a3d41abb18806d049131a764ccb9f6100ab80493faf` |

Audit run — `.../box_normalized/20260901_214534/`

| Artifact | sha256 |
|---|---|
| `box_normalized_visibility_scores.csv` | `0e56e91d68b2a98c632fe51970b8685f7cbc638e0afda3f345cd6e2179b310ef` |
| `box_normalized_visibility_with_human_bands.csv` | `b57bfc3f8a8f24b8d772d87c8982f41dd2df185d1123c835df7611da7c05e3a0` |
| `largest_disagreements.csv` | `52357b0b82f9691206d7ee0df5d38d766190e030eeb3ec1eebb8d10cd1cc6aa3` |
| `largest_disagreements.png` (2400x1984) | `9662152970b197becedc93401c7f949994a2b176413da231904eea2531de3981` |
| `RUN_METADATA.json` | `55c39d7699c36a36a7a6349dac54c915d120f12b518e7e2cac30a0875ddae21b` |

Implementation — `data_collection/route_b_publication_actor_volume_visibility_v1/`

| File | sha256 | vs `21c3c45` |
|---|---|---|
| `core.py` | `02d39fd1d15c31ab8323f0ece09e951e8fc0e42aecf5de06a120918276f1d3e4` | unchanged |
| `scoring.py` | `3942b8c35c990c27676fc66c13f80ed8011e2a2ddc0768cac5c6e531f85b00dd` | unchanged |
| `agreement.py` | `a5ebb150cbbdeebaafedb61b99c9130291ee5574d63a1aa9a418dcd559fcce28` | unchanged |
| `run_audit.py` | `cc6a0afa9997d6d1ccc603ec990046e8c6b0fcbe9ba9b442abaad0589efa7701` | unchanged |
| `contact_sheet.py` | `d6e294592090906d5e7a71cff3d10d8ec0bfc971210423b50fb57c4a27b9c0ef` | unchanged |
| `run_normalized_audit.py` | `c0212e12a0e776e3b7616bff9e0a88500539e1586e279d3a51e6e7da725e9898` | unchanged |
| `training_reference.py` | `5206e57893b760bef30045d086ed0dab8047634d2d6574610946c5587e59808e` | statistic parameter added |
| `build_training_reference.py` | `83e82613cff7bfe7b270c3a0a5394d7af513284cf5061db6353f79ad8665917b` | records both statistics, equivalence check |
| `run_box_normalized_audit.py` | `1e8dd717a7850ca9b2383336730bf1273721c1dfe86a9e6c83ee3cae1ec96765` | new |

The two modified files default to the previous behaviour; the bit-identical
rebuild of the frozen pixel reference is the proof.

Reproduce:

```
CUDA_VISIBLE_DEVICES="" python3 -m \
  data_collection.route_b_publication_actor_volume_visibility_v1.build_training_reference \
  --statistic raw_box_visibility --verify-against 20260901_212409
CUDA_VISIBLE_DEVICES="" python3 -m \
  data_collection.route_b_publication_actor_volume_visibility_v1.run_box_normalized_audit \
  --box-reference-run 20260901_214026 --pixel-reference-run 20260901_212409
CUDA_VISIBLE_DEVICES="" python3 -m unittest discover \
  -s data_collection/route_b_publication_actor_volume_visibility_v1/tests -t .
```
