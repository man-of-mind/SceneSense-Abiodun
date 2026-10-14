# V20: live spatial-map viewer and one supervised full Town10 map GOOD-channel demonstration

Date: 12 October 2026 local time (capture 02:13–02:25 UTC on 13 October). Exactly one attempt; no retry.

## Verdicts (route/integration separate from UI)

| Layer | Result |
|---|---|
| Launch | numeric host exit status **0** (`capture_01/launch.rc`), 705 s lifecycle |
| Route | ROUTE_COMPLETED, ordered coverage 1.0, 1,251.68 m driven, PERMITTED_INTERVENTION (as in V19) |
| Retained runner | `QUALIFICATION_RESULT` CAPTURE_VALID; cell status **INCONCLUSIVE_OR_FAILED** |
| Registered assessment (`registered_assessment_01`) | **STOP**: 30 of 31 checks pass. The only failure is `RETAINED_actor_driven_integration`, because Phase-6 gate **P0_POLICY_COVERAGE** is false |
| Retention | strict verification PASS, 100/100 RETAINED_COMPLETE, GT references 100/100 resolved |
| Live viewer (display only) | 3,576/3,576 installed frames shown; 3,576 pose-bound; 3,576 exact heading joins; 0 refused, 0 dropped, 0 feed errors |

P0 measured 1,629 admitted out of 1,932 scored opportunities (**0.843** against the unchanged 0.95 minimum); V19 had 0.969 and V11 0.975. Of the 306 fallbacks, 249 were `UE_PRIOR_NEW_DATA_UL_MCS_INDEX: MISSING` (no strictly-prior new-data UL grant in the UE telemetry at decision time) and 57 were stale `camera_si` (>100 ms). V19 had 46 and 4, V11 41 and 6. The gate was not lowered and the run was not repeated.

## Did watching affect the pipeline? Yes, very probably

During the drive the host was much busier than in the two comparable runs. The data below come from the unchanged 10 Hz profiler sampler, restricted to samples while the UE process existed.

| | V20 (this run) | V19 | V11 (same 23.5 dB schedule) |
|---|---:|---:|---:|
| host load1 P50 / P95 | **17.2 / 20.4** | 10.8 / 14.4 | 11.1 / 14.1 |
| running entities P50 / P95 / P99 | **14 / 36 / 46** | 10 / 30 / 39 | 10 / 30 / 39 |
| Feedback received ≤ 170 ms | **1,031 / 1,636 (0.630)** | 1,359 / 1,518 (0.895) | 0.873 |
| Prediction completion P50 / P95 / P99 ms | 137.6 / 162.7 / 191.8 | 130.5 / 149.6 / 162.3 | |
| GT published P50 / P99 ms | 126.9 / 229.2 | 119.5 / 159.7 | |
| Feedback receipt P50 / P95 / P99 ms | 165.3 / 192.5 / 269.0 | 155.4 / 176.5 / 203.5 | |
| Payload P50 kB | 108.9 | 111.9 | |

Viewing load in this run:

* Viewer server plus headless recorder: about 1.5 CPUs (sampler P50 152%, max 170%). They were pinned to CPUs 22–23 at nice 10/19. Those E-cores belong to the edge partition.
* **The watching browser was Firefox on this host** (PID 929095, unpinned). Wherever NVML reported it, it used a median **54%** SM share of the shared inference GPU (P95 63%).

Every stage slowed, mostly in the tails, with similar payloads, and the UE-side input-freshness fallbacks rose about six-fold. Both point to host CPU/GPU contention, not to the channel or the taps:

* The map tap wrote 3,576/3,576 with 0 queue drops, and its install-to-write time was 0.34 ms P50.
* The UE tap wrote 10,747/10,747 with 0 drops.
* Their synchronous cost is one `put_nowait` per item.

Session and scene also differ from V19/V11, so the size of the viewing effect is **not attributed exactly**. The direction is clear, though: this capture is not a clean performance measurement. For M4, watch the live view from another machine through the SSH tunnel, and keep the recorder off this host or measure it as a separate condition.

## What ran (smallest prospective revision)

* **V20** = the V19 package stack (actual qualified V17/R1 runtime: GT geometry containment, storage accounting V3.v2, GC, P3, bounded end-of-route drain, registration-bound assessment) with two changes:
  1. Schedule: the **sealed V11 GOOD_ALL** constant 23.5 dB → RFsim −12.0. V11's controller module and channel registration are reused unchanged; the registration's `channel` is byte-identical to the actual V11 registration. V19's position field is not used.
  2. Display-only taps (`rl_agent/splitfusion_live_view_tap_v1`):
     * a map-process post-install copy, through a new map trampoline that installs the unchanged v3 class conversion first;
     * a UE-child copy of the existing `OWNED_SOURCE` and `UE_WIRE` pose evidence.
* Frozen deployed actor, route, traffic_50_50, radio resources, the 170-ms deadline, GT, scorer and map rules are unchanged. The registered map contents and ACKs are unchanged: 3,576 installs, 605 of them late-feedback frames that were still installed and shown.
* Qualification (all on the final sources):
  * binding regression 16/16, including a real map-server install through the class conversion and tap chain, and the pose tap composed with the C5 capacity substitution;
  * fresh-process startup verification 6/6 roles, with 29/29 negative controls rejected (new: altered live-view source, V17/R1 14-dB channel);
  * offline assessment binding 13/13;
  * host verify with the startup proof.
* Unchanged V17/R1, containment, accounting, v8/v7 and V11 qualifications were reused, not rerun.
* Measured PUSCH and detailed channel fidelity were not separately analysed for this demonstration. Schedule checks passed: 5,257 rows all ACK_ON_TIME, and every admitted actor value was 23.5 dB.

Quality (descriptive, different session): vehicle within-tolerance recall 0.719, person 0.538 (V19: 0.747 / 0.450).

## Live UI behaviour

* **Frame and pose joins.** Every installed frame was bound to its own UE frame-context pose by the replay's `bind_pose`. It checks run/cell/stream/frame/capture-ns identity, sensor hash, calibration hashes, ego rotation and mount. Headings come only from the producer world-yaw pair through an exact index/XYZ/size/score/class join; the legacy installed `yaw_deg` of 90 is never used.
* **Pipeline-side ages** (capture → …; the first two rows are both measured from capture, they do not add):

  | Interval | P50 | P95 | P99 | max |
  |---|---:|---:|---:|---:|
  | capture → map install | 199.3 | 239.2 | 308.7 | — |
  | capture → viewer emit | 208.6 | 249.0 | 319.6 | 502.6 |
  | install → tap file write | 0.34 | 1.53 | 3.56 | — |

* **Capture → display**, rolling P50 as shown in the browser: about 288–301 ms. The browser figure includes the wait between successive map updates (~6.9 updates/s), so it saw-tooths between about 200 and 330 ms. The 170-ms scientific deadline is a separate quantity.
* **Render**: your Firefox frame interval was 17.0 ms P50 (60 fps); its 2-s window P95 was 17.2 ms typical and 33 ms at the 95th percentile of windows; the worst single frame was 299 ms. Draw P95 was 4 ms typical. The headless recorder ran at 16.7 ms P50.
* **States**: LIVE from the first install; STALE after the route ended ("no map update N s", tracks faded and expired, ego HELD). Your browser also showed one brief mid-run STALE, around the single capture gap of 587 ms (frames 6710 → 6714). There were no disconnects, no reconnects, no queue drops and no flags.
* **Disconnect, restart and stale-message behaviour** (pre-run browser check on a re-timed V19 feed, `spatial_map_coop/replay_ui_v1/artifacts_live_v1/LIVE_BROWSER_CHECK.json`, PASS):
  * pipeline pause → STALE, then LIVE again;
  * viewer killed → FEED DISCONNECTED after 0.11 s, all objects gone 0.53 s later;
  * viewer restarted → new epoch, tracking reset, LIVE after 0.85 s, without replaying the backlog;
  * injected duplicate, delayed, old-session and old-epoch messages refused with no track change;
  * opening and abruptly closing a second tab did not disturb the stream.
* **Display limitations observed** (no collision-avoidance capability is claimed):
  * Pedestrians: v3.1 limits as accepted; position noise, many `unresolved` ellipses, motion ticks only after statistically supported motion, short identities.
  * Vehicles: occasional body axes diagonal to the lane, and amber conflicting-pair outlines near ego at 30 m (detector heading/range noise).
  * Far glyphs are often unsupported (the known 35–45 m limit).
  * HUD: when STALE, the "unresolved" count and the "map updates /s" figure keep their last values. The count does the same in replay.
  * Building clipping is a 2D approximation only.

## Files

* Viewer (main checkout, `spatial_map_coop/`):
  * new: `replay_ui_v1/live_feed_v1.py`, `viewer_assets/live_v1.html`, `viewer_assets/live_app_v1.js`, `viewer_assets/live_v1.css`, `replay_ui_v1/README_LIVE.md`, `tests/test_live_feed_v1.py`, `replay_ui_v1/live_smoke_v1.py`, `replay_ui_v1/live_browser_check_v1.cjs`, `replay_ui_v1/live_record_v1.cjs`, `replay_ui_v1/artifacts_live_v1/`;
  * edited: `replay_ui_v1/server.py` (adds `--live-root` and `/live` routes; replay routes unchanged), `viewer_assets/replay_tracks_v3.js` and `viewer_assets/replay_policy_v1.js` (also accept the `LIVE_SAME_HOST_WALL` label; replay behaviour unchanged, all JS suites pass).
* Runtime (main checkout, deployed to the worktree): `rl_agent/splitfusion_live_view_tap_v1/` (tap, map trampoline, tests), `rl_agent/splitfusion_m3_route_b_v20/`, `rl_agent/splitfusion_m3_route_b_v20_offline_v1/`. Sealed packages are untouched.
* Evidence:
  * capture: worktree `rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261012_m3_route_b_v20/good_all_live_view_01` (tap files in `capture/cells/*/live_view/`, 196 MB);
  * this directory: `capture_01/`, `registered_assessment_01/`, `retention_01/`, `analysis_01/`;
  * viewer evidence in `live_view_01/`:
    * `recording/95bcbb3eeae6abc6ba6eb66d26d03cc8.webm`: 733 s normal-speed video, 70 MB; the drive starts about 83 s into the video;
    * `VIEWER_EVENTS.jsonl`: per-frame timeline, connections, browser render reports;
    * `recording/STATE_1HZ.jsonl`, `VIEWER_CPU.jsonl`, `LIVE_SUMMARY.json`, `stills/`.
  * Preparation: worktree `experiments/splitfusion_m3_route_b_v20/` (`binding_regression_01`, `startup_verification_01`, `offline_binding_02`) and `deploy_01` here.

## Cleanup and storage

* Host restoration passed and both pre-stop restores returned 0. No pipeline processes or edge containers remain, and the UE tunnel was removed.
* The live viewer and recorder were stopped. Your replay server on 127.0.0.1:8765 was never touched.
* Free space went from 22.06 GB before launch to **13.50 GB (12.57 GiB) now. This is below the 16 GiB launch gate.** Any further capture needs about 3.7 GB freed first. Nothing was deleted.

## Is the live UI a suitable baseline for M4?

Yes for the **adapter and display path**:

* exact frame, pose and heading binding, with no GT;
* correct LIVE / STALE / DISCONNECTED behaviour;
* bounded, non-blocking taps with zero loss;
* replay preserved.

There is one condition, about how it is watched. The viewer process and an on-host browser measurably competed with the pipeline in this run (P0 and timeliness). For M4, view it remotely over the tunnel and keep any recorder off the experiment host, or measure it as a separate condition. Treat this run's timeliness as viewing-contaminated. The viewer's own fan-out could also be made cheaper (it re-serialises about 50 kB per frame per client), but this task did not change it.
