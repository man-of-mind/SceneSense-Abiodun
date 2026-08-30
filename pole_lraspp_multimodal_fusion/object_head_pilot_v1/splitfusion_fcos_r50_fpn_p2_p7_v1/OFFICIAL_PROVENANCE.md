# Official FCOS provenance

The model is initialized only from Torchvision's `FCOS_ResNet50_FPN_Weights.COCO_V1` enum. The installed implementation is Torchvision `0.25.0.dev20251117+cu128`, source revision `4efae90d072d0d11e244d6e213208b357f89efe7`, under the BSD-3-Clause source license.

The official URL is `https://download.pytorch.org/models/fcos_resnet50_fpn_coco-99b0c9b7.pth`. The locally hash-verified file is 129,612,099 bytes with SHA-256 `99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7`.

The installed source inspected for this run is `torchvision/models/detection/fcos.py`, `torchvision/models/detection/backbone_utils.py`, and `torchvision/ops/feature_pyramid_network.py`. It establishes the implicit-background sigmoid-focal target, class indices, one-anchor-per-location convention, centre sampling, minimum-area conflict resolution, `sqrt(sigmoid(class)*sigmoid(centerness))` score, classwise NMS, FrozenBatchNorm ResNet, P3-P7 topology, and FPN initialization used here.

The pretrained weights were produced from COCO categories under Torchvision's published detection recipe. COCO dataset rights and terms are separate from Torchvision's BSD-3-Clause source license. Copying the official `car` and `person` classifier rows does not make the two-output project detector output-identical to the complete 91-output detector and does not confer COCO provenance on Route B labels.

