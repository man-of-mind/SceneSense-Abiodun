# Route B human occlusion pilot v1

This package builds one deterministic, prediction-blind 100-person annotation
pilot from the frozen Route B validation view. It reads only validation manifest,
pedestrian GT geometry, registered depth-count metadata, and RGB frames.

Generate a create-only run:

```bash
python3 data_collection/route_b_publication_human_occlusion_pilot_v1/build_pilot.py \
  --output data_collection/experiments/route_b_publication_human_occlusion_pilot_v1/<run-id>
```

After two people independently complete the generated CSV templates:

```bash
python3 <run>/score_agreement.py \
  --manifest <run>/sample_manifest.csv \
  --annotator-a <run>/annotator_A.csv \
  --annotator-b <run>/annotator_B.csv
```

The builder is create-only. Generated panels and run artifacts are intentionally
kept outside this implementation package and are not committed.
