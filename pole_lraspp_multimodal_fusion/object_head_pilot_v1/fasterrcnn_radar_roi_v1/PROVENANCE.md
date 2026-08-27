# Faster R-CNN radar-ROI v1 provenance

- Detector: `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`.
- Weights: `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1` from the official
  `download.pytorch.org` URL registered in the JSON config.
- Weight SHA-256: `dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf`.
- License: torchvision BSD-3-Clause; included under `licenses/`.
- Route B class mapping: background copies COCO row 0; person copies COCO row
  1; vehicle is initialized as the arithmetic mean of COCO car (3), motorcycle
  (4), bus (6), and truck (8), for both classification and class-specific box
  regression tensors.
- Native detector postprocess is fixed before validation: class-aware box NMS
  IoU 0.5, score floor 0.02, at most 100 detections per image.
- Split runtime reuse: the complete namespaced RGB/radar FPN bundle delegates
  serialization, quantization, entropy coding, UDP chunking, ImageList
  reconstruction and future worker integration to
  `carla_split_inference_udp_data_collect.py`. Future rank drop and heterogeneous
  per-level AE construction reuse `carla_split_inference_udp_segmentation_demo.py`.
  Clean qualification remains q=0 with no quantization or AE.
- Old AE checkpoint weights are explicitly incompatible and are not loaded:
  the new boundary includes five 256-channel RGB levels and five 64-channel
  radar levels at ResNet50-FPN-v2 spatial shapes.

