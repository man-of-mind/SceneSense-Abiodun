# Actor-volume observability model comparison — failed-closed final report

## Outcome

The one authorized orchestrated run, `20260901_final_cpu_once`, stopped at the
mandatory exact 100-person pilot-reproduction gate. No frozen prediction file
was opened and no model or AVO-threshold result was scored.

Terminal state: `ACTOR_VOLUME_OBSERVABILITY_MODEL_COMPARISON_FAILED`

The create-only failure artifact is preserved at:

`experiments/actor_volume_observability_model_comparison_v1/20260901_final_cpu_once/FAILURE.json`

## Qualification failure

For the first registered pilot actor-frame
`canonical_v3_05_val_30_30_s601_tm1601_000154_frame865:92`, the candidate
full-table calculation produced:

| Field | Registered pilot | Candidate reproduction |
|---|---:|---:|
| Unclipped projected area | 9362.997141820457 px² | 9362.997141820590 px² |
| Clipped projected area | 9362.997141820457 px² | 9362.997141820590 px² |
| Unnormalized AVO | 0.5527076342772246 | 0.5527076342772167 |

The raw and frozen-dataset calibration strings are identical. The difference
was traced to one-ULP parsing differences in pedestrian half-extents between
Python's standard CSV float conversion used by this orchestrator and the
pandas CSV parser used by the registered pilot. For example, one extent parsed
as `0x1.805dca0000000p-3` instead of the pilot representation
`0x1.805dc9fffffffp-3`. That tiny representation change propagated through the
projected area and AVO arithmetic.

The requirement was exact reproduction, not tolerance-based agreement.
Accordingly, the run failed closed. The candidate table was not published,
the human aggregate was not used in a calculation, and the three frozen
detection files were not opened.

## Scope preserved

- The original unnormalized actor-volume source files were hash-gated to their
  exact `dc5238d` contents before raw data access.
- CUDA was disabled; torch, checkpoints, CARLA/Epic, inference, training, and
  test data were not used.
- No alternate visibility formula or AVO threshold was tried.
- No intermediate or second run was made after observing the failure.
- Vehicle and segmentation metrics were not rescored.
- Canonical v0.10 results, checkpoint selection, service gates, and the
  supervisor-approved SplitFusion-FCOS service decision remain unchanged.
- The dirty `OAI/openairinterface5g` submodule was preserved.

Because the qualification gate failed, there is no valid complete
model×threshold table and the success terminal
`ACTOR_VOLUME_OBSERVABILITY_MODEL_COMPARISON_COMPLETE` is not asserted.
