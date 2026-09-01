# Actor-volume observability model comparison v1

This package performs the one registered CPU-only retrospective comparison of
three frozen Route B validation prediction sets. It imports only the original
unnormalized actor-volume geometry from
`route_b_publication_actor_volume_visibility_v1`, reproduces the registered
100-person pilot exactly, builds one create-only AVO table from raw validation
records, and evaluates all six predeclared AVO eligibility thresholds in a
single process.

It does not load checkpoints, import torch, invoke CUDA or CARLA, run model
inference, modify predictions, choose a threshold, or change any canonical or
service result.

Run once from the repository root with an unused run ID:

```bash
CUDA_VISIBLE_DEVICES="" python3 -m \
  data_collection.route_b_publication_actor_volume_observability_model_comparison_v1.run_comparison \
  --run-id RUN_ID
```
