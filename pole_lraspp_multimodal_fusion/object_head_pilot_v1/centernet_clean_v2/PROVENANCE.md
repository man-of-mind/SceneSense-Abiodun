# CenterNet v2 provenance

The RGB encoder uses torchvision's official `ResNet34_Weights.IMAGENET1K_V1`
weights through the installed torchvision API, exactly as in v1.  No CenterNet,
CenterFusion or DCNv2 source code is vendored into this package.

- Project: torchvision
- Installed version: `0.25.0.dev20251117+cu128`
- Exact installed git revision: `4efae90d072d0d11e244d6e213208b357f89efe7`
- Weight enum: `ResNet34_Weights.IMAGENET1K_V1`
- Official weight URL: `https://download.pytorch.org/models/resnet34-b627a593.pth`
- Weight SHA-256: `b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f`
- Licence: BSD-3-Clause; retained in `licenses/torchvision-BSD-3-Clause.txt`

`centernet_model_v2.py` is project-local code following the published CenterNet
design pattern (centre heatmap + centre offset + dense regression, decoded by
local-maximum peak picking).  The dual-stride arrangement, the private per-class
offset heads and the stride-2 segmentation skip are specific to this project.

## Relationship to v1 (nothing in v1 is modified)

v2 warm-starts from, and never writes to,
`experiments/route_b_centernet_clean_v1/20260826_175224/checkpoints/resnet34_fpn_centerfusion_v1/epoch_012.pt`
(SHA-256 `59884bb0ed8c291b00dc9dbc40767f1f2f537d6253b7401cf25833f8bea1928e`).
The v1 model file, the v1 experiment directory, the legacy evaluator, the
corrected v1 evaluation artifacts, the dataset, the locked test split and the
production split runtime are all untouched: v2 lives entirely in
`centernet_clean_v2/` plus a new timestamped experiment directory, and reuses v1
code only by import (the RGB/radar/mask input pipeline and the legacy
`summarize` metric function, so v1 and v2 numbers stay comparable).

## Files

| file | role |
|---|---|
| `centernet_model_v2.py` | model, split boundary, warm-start mapping |
| `targets_v2.py` | native-grid Gaussian/offset/regression targets + dataset |
| `losses_v2.py` | per-branch focal + offset + regression losses |
| `decode_v2.py` | the frozen native dual-stride decoder |
| `train_v2.py` | single 24-epoch training run, differential LR |
| `evaluate_v2.py` | validation evaluator at score 0.20 and 0.02 |
| `launch_check_v2.py` | the authorized pre-run checks |
| `select_and_report_v2.py` | selection rule, service-target gate, report |
| `configs/` | v2 yaml config and v2 trial json |
