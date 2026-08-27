# Faster R-CNN radar-ROI v1 — final Route B report

Terminal verdict: `FRCNN_RADAR_ROI_NO_GAIN`

## Selected artifact

- Epoch: 12
- Checkpoint: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_fasterrcnn_radar_roi_v1/20260826_224720/checkpoints/fasterrcnn_radar_roi_v1/epoch_012.pt`
- SHA-256: `7d3e1b414a892713fe848cfc81266ae4c321109453f0b60ac93efe30d8a1ef13`

## Architecture and split boundary

COCO-pretrained Faster R-CNN ResNet-50-FPN v2 performs RGB-only RPN, ROI classification and 2D box regression. An independent four-channel radar pyramid is pooled at every positive ROI and concatenated with the visual ROI embedding for camera-local XYZ, dimensions, local yaw, parked state and radar-support regression. A separate visual-FPN decoder produces semantic segmentation.

`encode_front(rgb, radar)` emits five RGB-FPN and five radar-feature tensors; `decode_tail(bundle, image_size)` has no raw modality argument. Boundary payload: 62,853,120 bytes/sample in FP32. Monolithic/split maximum absolute difference: 0.0.

## Pretrained provenance

- FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 from `https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`
- SHA-256 `dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf`; torchvision `0.25.0.dev20251117+cu128`; BSD-3-Clause.
- Route person copies COCO person row 1. Route vehicle is the mean of COCO car 3, motorcycle 4, bus 6 and truck 8, for classifier and class-specific box regressor rows.

## Epoch metrics

| epoch | veh P | veh R | veh F1 | per P | per R | per F1 | veh R@.02 | per R@.02 | veh XY | per XY | veh IoU | person box-mask IoU | mIoU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.3861 | 0.7759 | 0.5156 | 0.2850 | 0.5762 | 0.3814 | 0.8501 | 0.7184 | 1.033 | 1.287 | 0.8598 | 0.3597 | 0.7355 |
| 8 | 0.5834 | 0.8144 | 0.6798 | 0.3528 | 0.6248 | 0.4509 | 0.8510 | 0.7513 | 0.754 | 1.219 | 0.8852 | 0.3525 | 0.7421 |
| 12 | 0.5907 | 0.8290 | 0.6898 | 0.2804 | 0.6448 | 0.3908 | 0.8574 | 0.7591 | 0.692 | 0.970 | 0.8883 | 0.3272 | 0.7347 |

Person segmentation is projected-box-mask IoU, not silhouette IoU.

## Retained-model comparison

| model | veh P | veh R | veh F1 | per P | per R | per F1 | mean F1 | veh XY | per XY | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CenterNet v1 corrected | 0.8144 | 0.5749 | 0.6740 | 0.5404 | 0.4891 | 0.5135 | 0.5937 | 0.914 | 1.095 | 0.6888 |
| CenterNet v2 | 0.7826 | 0.6119 | 0.6868 | 0.5058 | 0.4891 | 0.4973 | 0.5921 | 0.900 | 1.045 | 0.6687 |
| Faster R-CNN selected | 0.5907 | 0.8290 | 0.6898 | 0.2804 | 0.6448 | 0.3908 | 0.5403 | 0.692 | 0.970 | 0.7347 |

## Runtime

Training: 27.7 min, peak allocated/reserved 4771/5180 MiB. Evaluation total: 3.9 min, peak allocated 1378 MiB.

## Service gate

| metric | target | selected | result |
|---|---:|---:|---|
| vehicle_precision | >= 0.80 | 0.5907 | FAIL |
| vehicle_recall | >= 0.85 | 0.8290 | FAIL |
| person_precision | >= 0.80 | 0.2804 | FAIL |
| person_recall | >= 0.80 | 0.6448 | FAIL |
| vehicle_xy_mae_m | <= 1.00 | 0.6922 | PASS |
| person_xy_mae_m | <= 1.20 | 0.9696 | PASS |
| vehicle_iou | >= 0.85 | 0.8883 | PASS |
| person_box_mask_iou | >= 0.50 | 0.3272 | FAIL |
| miou | >= 0.80 | 0.7347 | FAIL |

## Interpretation

Detector, localization and segmentation outcomes are separated in the tables above. The retained manual 32-person panel contains 15 clearly visible, 5 partially visible, 9 heavily occluded and 3 not visible examples. It is stratified rather than random, so those proportions are not extrapolated to the validation corpus; unresolved observability remains a limitation without changing the full GT denominator.

Material-gain rule: vehicle recall >= max(retained v1-corrected, retained v2); person recall >= max(retained v1-corrected, retained v2); and mean class F1 >= strongest retained mean class F1 + 0.05. Result: FAIL.

# FRCNN_RADAR_ROI_NO_GAIN
