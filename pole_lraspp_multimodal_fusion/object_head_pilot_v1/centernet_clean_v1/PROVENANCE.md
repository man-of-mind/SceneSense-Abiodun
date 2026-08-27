# Clean CenterNet pilot provenance

The RGB encoder uses torchvision's official `ResNet34_Weights.IMAGENET1K_V1`
weights through the installed torchvision API. No CenterFusion or DCNv2 source
code is copied into this pilot.

- Project: torchvision
- Installed version: `0.25.0.dev20251117+cu128`
- Exact installed git revision: `4efae90d072d0d11e244d6e213208b357f89efe7`
- Source revision: `https://github.com/pytorch/vision/tree/4efae90d072d0d11e244d6e213208b357f89efe7`
- Weight enum: `ResNet34_Weights.IMAGENET1K_V1`
- Official weight URL: `https://download.pytorch.org/models/resnet34-b627a593.pth`
- Weight SHA-256: `b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f`
- Licence: BSD-3-Clause; retained in `licenses/torchvision-BSD-3-Clause.txt`

The architecture in `centernet_model_v1.py` is project-local code inspired by
the published CenterNet/CenterFusion design pattern: a stride-four centre
heatmap/regression detector followed by radar-conditioned refinement. It does
not vendor either upstream implementation or use a checkpoint of unclear
provenance.

