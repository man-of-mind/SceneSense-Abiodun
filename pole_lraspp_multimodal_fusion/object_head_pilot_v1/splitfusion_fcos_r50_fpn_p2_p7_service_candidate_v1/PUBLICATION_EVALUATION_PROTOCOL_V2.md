# Publication renderer z-buffer visibility protocol v2

Registered: 2026-09-01, before traffic collection or model inference.

## Why v2 is necessary

The controlled CARLA 0.10 scene showed a visible pedestrian in RGB but zero pedestrian pixels in the synchronized instance-segmentation image. Therefore this build does not provide usable person silhouette or person-instance ground truth. Protocol v1 stopped correctly before traffic collection.

Protocol v2 does not assume that CARLA supplies a pedestrian silhouette. It derives an actor reference support mask from CARLA's ordinary depth renderer with that actor rendered alone.

## Renderer z-buffer definition

For actor `i`, render three lossless depth images using the same resolution, FOV, intrinsics and pixel coordinates:

- `D_empty`: isolated reference camera with no actor;
- `D_actor_i`: the same reference camera with only actor `i`, at the recorded camera-relative transform and pose;
- `D_scene`: the synchronized normal traffic scene.

With both depth tolerances fixed at `0.02 m` before traffic collection:

```text
A_i(p) = D_actor_i(p) + 0.02 < D_empty(p)
V_i(p) = A_i(p) and abs(D_scene(p) - D_actor_i(p)) <= 0.02
visibility_i = pixels(V_i) / pixels(A_i)
```

`A_i` is the renderer-derived in-frame actor support. It is not a CARLA-provided person silhouette annotation. `V_i` is the portion of that support whose exact expected surface depth remains the front-most scene surface. External occluders create a smaller scene depth and therefore remove those pixels from `V_i`.

This is materially different from the historical projected-box metric: it never accepts a pixel merely because its depth lies somewhere inside the actor's broad near/far box interval. Every accepted pixel must agree with the isolated render of that actor's expected surface at that exact pixel.

## Fixed evaluation

The perception baseline, two LR-ASPP comparators, four prospective episode seeds, 40 m range, score views, metrics, and visibility thresholds `0.10/0.25/0.50/0.70/0.85` remain unchanged from v1.

The person segmentation reference is the class-wise union of renderer-derived visible supports `V_i`, not a native CARLA person mask. The instance camera is permitted only as an independent vehicle diagnostic because vehicle instances rendered successfully in the controlled scene.

Before any traffic collection, the controlled scene must prove that actor-only depth renders both classes, reproduces the walker pose, yields clear/partial/full visibility behavior, and agrees with the working vehicle instance mask. Failure stops the protocol; no projected-box fallback is allowed.
