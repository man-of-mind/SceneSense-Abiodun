# Route B Perception Collection Pilot

Date: 2026-08-22 EDT  
Campaign terminal: `STOPPED_AFTER_LOW_CLEANUP_FAILURE`

The accepted Route B JSON, progress CSV, and qualified density runner passed
their supplied SHA-256 checks before CARLA startup. The low-density episode was
run once. It ended `FAIL` because the perception sensor cleanup postcondition
failed. Per the stop-on-first-failure rule, low was not retried and the medium
and dense episodes were not started.

## Source and fixed settings

- Map: `Carla/Maps/Town10HD_Opt`
- Route: `Town10HD_Opt Route B full-map loop v1`, one loop
- Route JSON SHA-256: `fc4518a8746b9417a64616b8e544f59b16b5a31b7585298a316a59662ecfd6e5`
- Progress CSV SHA-256: `974593859368f24ee2bc4ac31b82118bf2e932d0de1c96858b8771e2dd4d90c0`
- Qualified runner SHA-256: `59592ee83184a227f324ff872d1cc7f5601d5a1efb0300dc08dec7b7f26749a4`
- Scenario/Traffic Manager seeds: `101` / `1101`
- Controller: lane offset `-0.5 m`, walker detection `10 m`, qualified NPC
  hardening and safe-vehicle filtering, interventions disabled
- Weather: unchanged fresh-world qualified default
- Sampling: every tenth 20 Hz tick (target 2 Hz)

## Episode result

| Density | Requested / spawned V/P | Terminal | Sim / wall duration | Saved frames | Route coverage |
|---|---:|---|---:|---:|---|
| low | 5/5 / 5/5 | `FAIL`: perception sensor cleanup failure | 298.80 s / 326.59 s | 597 | 19/19 ordered waypoints; B1, B2, B3 |
| medium | 15/15 / not started | `NOT_RUN` | — | — | — |
| dense | 25/25 / not started | `NOT_RUN` | — | — | — |

Low drove 1,251.69 m of the 1,268.68 m planned route and returned within
0.521 m and 0.002 degrees of loop closure. It had zero collisions, zero
collision incidents, zero interventions, and no watchdog abort. Three replans
and two `OBSERVED_NO_INTERVENTION` blocked-ego events were logged.

The observed saved-frame interval was constant at `0.50000000745 s`; every
successive simulator frame ID differed by exactly 10.

## Perception records

Counts are occurrences across the 597 saved frames, not unique tracked actors.

| Count | Vehicle | Person |
|---|---:|---:|
| Raw / in-view GT | 791 | 332 |
| Training-eligible GT (`area >= 12 px`, `distance <= 40 m`) | 237 | 38 |
| Local GT (`distance <= 50 m`) | 447 | 145 |

| Per-frame density | Vehicle min / mean / max | Person min / mean / max |
|---|---:|---:|
| In-view | 0 / 1.325 / 4 | 0 / 0.556 / 4 |
| Local | 0 / 0.749 / 2 | 0 / 0.243 / 2 |

All 597 RGB JPEGs, segmentation masks, raw semantic-tag images, radar tensors,
and radar point files exist, are non-empty, and passed format decoding/loading.
There were zero missing or corrupt artifact records. The object-GT table has
1,123 rows and zero frame/timestamp mismatches against the manifest.

RGB, semantic, depth, and radar capture shared the same CARLA frame at each
save. Maximum cross-sensor timestamp spread was `0.0 s`; RGB/radar frame-ID
mismatches were zero; RGB/radar timestamp difference was `0.0 s`. Maximum
camera and radar transform displacement from their sampled actor transforms was
`0.0 m`.

## Storage and cleanup

- Low output size: 3,592,778,336 bytes (3.593 GB; 3.346 GiB).
- Historical layout: `manifest.csv`, `object_boxes.csv`, `metadata.json`, RGB,
  masks, semantic tags, radar tensors, and radar points.
- Perception sensor cleanup: **failed**. This is the terminal campaign failure.
- Qualified runner actor cleanup summary: `cleanup_succeeded=true`, although it
  also emitted a warning that one managed vehicle did not acknowledge its
  initial destroy request.
- CARLA was then shut down; process-level verification found no CARLA server.

The low output must remain failure-tagged and must not be admitted to training
until the cleanup failure is reviewed. No AE64 training, checkpoint mutation,
production edit, or final-test access occurred.

## Outputs and reproduction

Created:

`fusion_training_data/route_b_perception_pilot_20260822_231913_EDT/low_5_5_seed101_tm1101`

Not created because the campaign stopped:

`fusion_training_data/route_b_perception_pilot_20260822_231913_EDT/medium_15_15_seed101_tm1101`

`fusion_training_data/route_b_perception_pilot_20260822_231913_EDT/dense_25_25_seed101_tm1101`

Exact server and low-episode commands used:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping
./CarlaUnreal.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -log

cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python \
  data_collection/run_route_b_perception_collection.py \
  --density low \
  --output-dir fusion_training_data/route_b_perception_pilot_20260822_231913_EDT/low_5_5_seed101_tm1101
```

The output path is create-only and now exists; the command will deliberately
refuse to overwrite it. Any rerun or continuation requires explicit review and
a newly authorized create-only campaign directory.
