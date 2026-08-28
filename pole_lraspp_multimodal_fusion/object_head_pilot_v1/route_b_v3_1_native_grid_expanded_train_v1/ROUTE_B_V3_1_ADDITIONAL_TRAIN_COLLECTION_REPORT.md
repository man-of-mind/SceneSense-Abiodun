# Route B v3.1 additional train-only collection and expanded training view

Terminal: `ROUTE_B_V3_1_ADDITIONAL_TRAIN_COLLECTION_COMPLETE`

Six additional independent train-only Route B v3 episodes were collected, their v3.1 GT
contracts materialized, and one expanded symlink training view built. No training,
evaluation, inference, checkpoint load, q/AE, OAI or 288-measurement work occurred.

## Collection

Frozen contract, differing from the canonical v3 collection only in split, episode
identity, scenario seed, Traffic Manager seed and density. Town10HD_Opt, the complete
Route B loop, 25 km/h, a fresh Epic `-RenderOffScreen` CARLA per episode, no hybrid
physics, 600 s budget, 2.0 s replenish, roadblock clearing on, forced overtaking off,
maximum overtakes 0, and the existing sensor/radar/save settings are unchanged. The launch
command is `run_canonical_campaign_v1._run_episode`, reused verbatim; no CLI flag is
invented or duplicated. No retry.

| label | ep | split | density | scenario/TM | wall/sim s | saved/prep/raw | gates | vis/obj rows | person v010/v025 | marg/unobs <=40 m | intv | GiB |
|---|---:|---|---|---:|---:|---|---|---|---|---|---:|---:|
| extra_09 |  9 | train | 30/30 | 801/1801 | 583/308 | 1539/3078/6175 | 37/37 | 21571/21571 | 67.00/51.66% | 217/467 | 2 | 12.02 |
| extra_10 | 10 | train | 50/50 | 802/1802 | 736/358 | 1788/3576/7171 | 37/37 | 38111/38111 | 77.00/63.18% | 415/692 | 4 | 13.95 |
| extra_11 | 11 | train | 30/30 | 803/1803 | 665/358 | 1792/3583/7185 | 37/37 | 21493/21493 | 76.72/62.20% | 232/372 | 2 | 13.95 |
| extra_12 | 12 | train | 50/50 | 804/1804 | 835/418 | 2092/4184/8387 | 37/37 | 42680/42680 | 83.49/74.27% | 341/611 | 8 | 16.35 |
| extra_13 | 13 | train | 30/30 | 805/1805 | 559/298 | 1489/2978/5976 | 37/37 | 18507/18507 | 83.41/76.84% |  69/174 | 4 | 11.64 |
| extra_14 | 14 | train | 50/50 | 806/1806 | 720/358 | 1789/3577/7174 | 37/37 | 39692/39692 | 83.09/66.79% | 490/508 | 3 | 13.96 |

Every episode: route completed and all ordered waypoints reached; no watchdog abort;
RGB/semantic/depth/radar alignment delta exactly `0.0` on the timestamp and both transform
components with zero frame-content failures; visibility rows equal object rows; every
intervention action `DESTROYED`, which is a registered permitted action, with an empty
unexpected-event list; zero forced overtakes; population and controller deficit spans
within the registered bounds; sensor and CARLA cleanup verified.

Total new corpus: 87898824186 bytes (81.86 GiB). Remaining disk: 116946743296 bytes.
Campaign wall time 4521.9 s. All six CARLA servers shut down and verified.

### Gate accounting

The collector emits 21 v2-family gates and 16 v3-family gates. All 37 are required and all
37 pass for every episode; the "21 collection gates" of the authorizing goal are the
v2 family and are a subset of what is enforced here.

### Two deviations worth recording

`camera_frame_parity` is the phase of the first camera frame inside the two-tick prepare
cycle (`camera_frame % 2`), not an alignment delta. A first version of this wrapper's
acceptance list wrongly folded it into the alignment maximum and rejected a conformant
episode 09, whose payload was then reclaimed by the failed-episode policy. The check now
gates only the timestamp and transform deltas, the frame-content failure list and the two
registered alignment gates, and was replayed over all eight canonical episodes - which
include parity 0 and parity 1 - before this run. The rejected attempt is preserved at
`data_collection/experiments/route_b_perception_v3/stopped_attempts/20260828_ep09_rejected_by_defective_parity_check/`.

CARLA is not bit-deterministic across processes at a fixed seed pair: the re-collected
episode 09 has 1539 saved frames and 2 interventions where the discarded run had 1790 and
5. The contract holds the registered seed pair, not byte reproducibility.

## Collection code provenance

- v3 config SHA-256 `084a433e22bac4771cc9889bcb485b42689db5765321747717e3604f8d5e5f97` and
  visibility helper SHA-256 `4a7aa974ea6374eceff35c0fbd8261fba299b2c5de68297591f5ce9756cf980c`
  are unchanged and equal to the retained canonical v3 report.
- The v3 collector required one additive change: the six new episode tuples are registered
  in `ADDITIONAL_TRAIN_EPISODE_KEYS`, since the collector refuses any unregistered
  `(split, density, scenario_seed, tm_seed)`. Its hash is now
  `09a373481d60e1630e69c1ea45403b3bc26104a58050e4159cac3c598304a7df`; the preflight proves
  the change is only that by reverting exactly those three hunks and reproducing the
  canonical-report hash `b17bcc1afa2226372f05fd8f5fe63f08d5fd324d112a108de5ffb6c63d7e0894`.
- Canonical report SHA-256 `27b98ba350d93b40d0dd2901381b481cde2e9f4de3bb2d54f7ca453c0b6fc458`;
  additional-train report SHA-256 `d1f50b97c9873d4efb1121dcfb47627bff8498c5f47b376d832d531e4447a5fc`.
- Town10HD_Opt static-environment catalog SHA-256
  `26ee85ea878204bc93e25e9e3439af47bfc863458d4956e626b3918f859f3c8f`, 123 rows, map
  `Carla/Maps/Town10HD_Opt`, map SHA-256
  `5d883b799f634030af92be1e9d79d107845540ba04338e8c60e095be1aef7be7`. The verified existing
  catalog was reused; nothing was recaptured.
- Route file SHA-256 `fc4518a8746b9417a64616b8e544f59b16b5a31b7585298a316a59662ecfd6e5`.

## v3.1 contract materialization

The retained builders are reused unmodified in behaviour: the frozen aggregate view builder
and `route_b_v3_1_clean_base_v1/build_contract_v1`, then the camera-plane localization
contract. Raw v3 episodes are neither mutated nor rewritten; payloads are directory
symlinks only.

| layer | v010 train pos/ign | v010 val pos/ign | v025 train pos/ign | v025 val pos/ign |
|---|---|---|---|---|
| v3.1 base | 64516/290498 | 13597/57601 | 57465/297549 | 11762/59436 |

v010 train positives by source: 31590 dynamic-actor vehicle, 15339 environment-static
vehicle, 17587 person. Validation is 6599/3126/3872, exactly the retained figures.
Train static-projection rows 39392; validation static rows 8014, unchanged.

Camera-plane localization rule (physical centre with camera-forward depth `<= 0` becomes
localization-ignore, segmentation pixels retained, never converted to background):
v010 train 184 transitions of 64516, v010 validation 34 of 13597; v025 train 25,
v025 validation 1. All nine camera-plane hard gates pass, including the registered
validation expectation of exactly 34 transitions composed of 26 actor and 8
environment-static rows over 11 unique identities and zero person transitions.

Train manifest rows 16827 are derived independently as 16866 reported saved frames minus 39
post-intervention exclusions; validation is required to stay at exactly 3345.

## Expanded training view

`experiments/route_b_v3_1_native_grid_expanded_train_v1/20260828_094151` - symlink only,
zero corpus copies, 40132 symlinks, 1.1 MiB of new payload.

Train, 10 episodes / 16827 frames: canonical 01-04 (seeds 501-504/1501-1504) plus
extra 09-14 (seeds 801-806/1801-1806). Validation, 2 episodes / 3345 frames:
canonical 05-06 (601/1601, 602/1602). Test is absent.

All 17 verification checks pass:

- exactly 10 train and 2 validation episodes; sample IDs globally unique and
  episode-namespaced; train/validation episode IDs and seed pairs disjoint; all ten train
  seed pairs distinct.
- Validation byte-identical at both contract layers. Final camera-plane layer against
  `route_b_v3_1_camera_plane_contract_v1/20260828_060131` and base layer against
  `route_b_v3_1_clean_base_v1/20260828_012309`: all six per-contract validation artifacts
  equal, manifest field sets equal, 3345 validation manifest rows digest-equal, validation
  object-box rows digest-equal. Every validation mask on disk re-hashes to its recorded
  target-manifest hash; 0 mismatches.
- Test absent and unopened: the manifest carries only `train` and `val`; zero symlinks
  resolve outside the ten admitted train and two admitted validation episodes; no text
  artifact contains a locked identifier. No locked directory was listed, resolved or read.
- All 40132 symlink targets resolve; 30 recorded source hashes across both layers reproduce.
- 47 post-intervention-excluded source rows, 0 retained. Collision windows retained with
  provenance exactly as the existing contract does: 579/581 train and 20/20 validation
  present, the two absentees being post-intervention exclusions.

## Paths

- Episodes: `data_collection/experiments/route_b_perception_v3/extra_v3_{09..14}_train_{30_30,50_50}_s80{1..6}_tm180{1..6}`
- Aggregate views: `experiments/route_b_v3_expanded_train_views_v1/20260828_094151`
- v3.1 contract: `experiments/route_b_v3_1_expanded_train_contract_v1/20260828_094151`
- Camera-plane contract: `experiments/route_b_v3_1_expanded_train_camera_plane_v1/20260828_094151`
- Training view: `experiments/route_b_v3_1_native_grid_expanded_train_v1/20260828_094151`
