# Route B perception collector v3

v3 reuses the v2 route runner, traffic lifecycle, 20/10/5 Hz cadence, radar
aggregation, renderer proof, intervention policy, persistence layout, and v2
gates. It adds synchronized lossless CARLA depth evidence and a sibling
`object_visibility.csv`, keyed by `sample_id`, `frame_id`, and `gt_actor_id` to
the unchanged `object_boxes.csv` rows.

Each visibility row records clipped and unclipped boxes/areas, in-frame
fraction, actor near/far camera depth, native and 768x432 depth-consistent pixel
counts, visible/closer/farther fractions, `eligible_visible_v010`,
`eligible_clear_v025`, and the three-way visibility tier. All projected v2
object rows are retained, including rows ineligible under either threshold.

Depth is the original CARLA BGRA uint8 encoded-depth buffer stored as a
lossless four-channel PNG in `depth/`. It reproduces metres using
`(R + 256G + 256^2B)/(256^3-1)*1000`. The fixed actor interval tolerance is
0.25 m and the algorithm identifier is
`route_b_depth_visibility_interval_v1`.

Vehicle pixels remain CARLA semantic-mask pixels. Person pixels are visible
depth-consistent regions only, are painted only for `eligible_visible_v010`,
and never replace vehicle pixels. These are depth-derived visible-region
approximations, not guaranteed anatomical silhouettes. There is no filled-box
or ellipse fallback.

