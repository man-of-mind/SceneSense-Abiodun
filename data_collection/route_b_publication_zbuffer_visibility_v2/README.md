# Publication z-buffer visibility v2

This focused package implements the registered `0.02 m` renderer z-buffer equations and the one controlled qualification. Person support is derived only from isolated actor-only ordinary CARLA depth; pedestrian instance/semantic masks, boxes, ellipses, learned masks and broad near/far intervals are not inputs.

The controlled supervisor starts one fresh Epic CARLA 0.10 server, captures one vehicle and one pedestrian under clear/partial/full static-occluder conditions, writes create-only evidence, and shuts the server down. It does not collect traffic or load a model:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 -m data_collection.route_b_publication_zbuffer_visibility_v2.supervise_controlled --output-root data_collection/experiments/route_b_publication_zbuffer_visibility_v2/<create-only-run>
```

CPU checks:

```bash
python3 -m unittest -v pole_lraspp_multimodal_fusion.object_head_pilot_v1.publication_zbuffer_visibility_evaluation_v2.tests.test_synthetic
```
