# LR-ASPP to SplitFusion-FCOS: supervisor summary

## Outcome

The supervisor accepted the locked SplitFusion-FCOS checkpoint, which passes 7/9 original numerical gates, as service-ready for progression to compression studies. The two missed person precision/recall targets remain reported unchanged; the acceptance is an explicit service-scope decision rather than a retroactive 9/9 result.

Two clean LR-ASPP approaches were tested:

- **Joint training:** learned useful localization, but detection and segmentation did not pass together.
- **Task-separated training:** achieved strong masks (`0.908` vehicle IoU, `0.573` person IoU), but the frozen object/localization head still produced too many false instances.

This showed that semantic person pixels were not enough to determine how many people existed, separate nearby people, and attach the correct geometry to each individual. LR-ASPP is not proven theoretically incapable; the two tested designs failed the registered service contract.

## Architecture transition

SplitFusion-FCOS preserves the project rather than replacing it:

```text
UE/front
RGB (3) + radar raster (4)
          │
     one 7-channel tensor
          │
  ResNet-50 stem + C2
          │
  one fused split tensor Z
          │
q/ROI, INT8+zstd or AE at this boundary
========== network ==========
          │
 ResNet C3–C5 + FPN P2–P7
          │
   ┌──────┴──────────────┐
   │                     │
semantic masks       FCOS instances
                         │
                  depth + geometry
                         │
                  localized detections
```

`C2` is the early high-resolution ResNet feature map transmitted by the UE. `P2–P7` are multiscale Feature Pyramid maps: P2 retains detail for small pedestrians; higher numbers are progressively coarser. The implemented model includes P7.

FCOS was chosen because it is an anchor-free instance detector, while FPN provides semantically strong multiscale features: [FCOS](https://openaccess.thecvf.com/content_ICCV_2019/html/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.html) and [FPN](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html). The official FCOS centerness target, loss and score equation were retained. Radar can influence centerness through the learned fused seven-channel features, while the custom depth/ray/geometry head handles metric localization explicitly.

## Historical comparison at the v0.50 depth-box-occupancy view

| Model | Person P/R/F1 | Person XY | Segmentation: vehicle/person/foreground IoU | Gates |
|---|---:|---:|---:|---:|
| Joint LR-ASPP | .417/.701/.523 | 1.139 m | .821/.353/.587 | 3/9 |
| Two-stage LR-ASPP | .306/.721/.430 | 1.266 m | .911/.572/.742 | 5/9 |
| SplitFusion-FCOS | **.769/.764/.767** | **.731 m** | **.901/.533/.717** | **7/9** |

## Final FCOS sensitivity

| Depth-box occupancy | Vehicle P/R | Person P/R/F1 | Vehicle/person XY | Gates |
|---|---:|---:|---:|---:|
| ≥10% | .932/.868 | .731/.600/.659 | .479/.844 m | 7/9 |
| ≥25% | .952/.951 | .760/.669/.712 | .443/.834 m | 7/9 |
| ≥50% | .965/.978 | .769/.764/.767 | .403/.731 m | 7/9 |

The registered `visible_fraction` is the fraction of pixels in an actor's projected in-frame rectangle whose synchronized CARLA depth lies in that actor's near/far interval. It is a legacy depth-box-occupancy proxy, not an instance-perfect percentage of visible body: the rectangular denominator contains empty space and same-depth road or other surfaces can be accepted. The canonical contract remains `v0.10`; `v0.25` and `v0.50` remain historical sensitivity views.

## Prediction-blind human visibility audit

A deterministic pilot selected 100 unique target pedestrians without consulting predictions. One human annotator used RGB-only panels to estimate the externally unoccluded fraction of the expected in-frame body: bare `90–100%`, partial `65–90%`, heavy `20–65%`, not observable `0–20%`, or ambiguous. Image-boundary truncation was recorded separately. Ambiguous and severely truncated targets were excluded from the primary view.

| Human view | SplitFusion-FCOS recall / XY | Joint LR-ASPP | Two-stage LR-ASPP |
|---|---:|---:|---:|
| `>=65%` visible, non-severe (N=44) | **.705 / .632 m** | .523 / .709 m | .568 / 1.295 m |
| `>=20%` visible, non-severe (N=63) | **.603 / .737 m** | .429 / .756 m | .460 / 1.298 m |
| Bare `90–100%` (N=28) | **23/28 = .821** | 21/28 = .750 | 20/28 = .714 |

Exact human/AI diagnostic band agreement was 63%, but 73/76 decisive pairs were exact or only one adjacent band apart, and the model ranking did not change under the alternative grouping. The AI result is diagnostic, not a second human annotation. Because only one target per frame was manually banded, this audit measures target recall/localization and does not replace full-validation precision.

At `v0.50`, only person precision and recall miss their original 0.80 targets, by 3.1 and 3.6 percentage points. After reviewing the complete evidence, the supervisor accepted the locked 7/9 checkpoint as service-ready for the next phase.

## Approved next phase

Keep the accepted perception checkpoint and service post-processing frozen. The immediate next action is hybrid-q/ROI training and evaluation at `Z=C2`; clean/zstd, 8-bit quantization and the three registered AE bottlenecks remain the subsequent transport-characterization work.

Full architecture, results, sensitivity tables, evidence limits and paths are in [FULL_TECHNICAL_REPORT.md](./FULL_TECHNICAL_REPORT.md).
