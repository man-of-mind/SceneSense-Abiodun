# Live view v1: the v3.1 ego traffic view on a running capture

The live view uses the same display engine and drawing code as the replay (v3.1: confidence tiers, display tracks, bounded
prediction, expiry, building clipping, pedestrian motion states, raw-candidate Inspect drawer). Only the data source,
the clock and the header/status strip differ. The camera panel is **not** shown in live mode: no camera stream or image
is sent anywhere for the viewer. Replay mode, including its retained-camera panel, is unchanged.

## Start

**Future live demonstrations must run viewing and recording off the experiment host.** Run the viewer/browser and
video recorder on a separate viewing machine, using a read-only copy or transport of the append-only tap inputs.
The commands below describe the file interface; they do not provision that transport. Do not launch the browser or
recorder alongside CARLA/inference/OAI on the experiment host. V20 displayed all 3,576 installed frames but retained
its registered **STOP** assessment; its report identifies likely on-host viewing/recording contention. No new
resource study or live experiment was performed for the display refinement. See [the offline V20 review](README_V20_HEADING.md).

Replay only (unchanged):

```bash
python3 -m spatial_map_coop.replay_ui_v1.server --open            # http://127.0.0.1:8765
```

Live view of ONE running capture (from the repository root; a separate process that only reads files):

```bash
python3 -m spatial_map_coop.replay_ui_v1.server --port 8766 \
  --live-root <prospective root or cell directory> \
  --live-log <new path>/VIEWER_EVENTS.jsonl                        # http://127.0.0.1:8766/live
```

The server binds 127.0.0.1 only. If the browser and viewer server are on separate viewing machines, use
`ssh -N -L 8766:127.0.0.1:8766 <user>@<viewer-host>` from the browser machine.
`/` on the same server is still the recorded replay.

## Data path

```
edge -> direct map install (map process) --[tap: after a SUCCESSFUL install, put_nowait]--> live_view/map_installs.jsonl
UE child pose archive (OWNED_SOURCE, UE_WIRE) --[tap: after the retained add, put_nowait]--> live_view/ue_pose.jsonl
                     viewer process: tail both files -> bind -> bounded per-browser queues -> SSE -> browser
```

* Taps: `rl_agent/splitfusion_live_view_tap_v1/tap.py` (registered in the V20 launcher). Each is one bounded queue
  (256) plus one writer thread; a full queue drops and counts. No socket or request goes from the pipeline to the viewer.
* Detections are the **installed** map record (exactly what the map holds). Heading comes only from the producer's
  p025 world-yaw pair, joined by original record index plus exact XYZ, dimensions, score and class. The installed legacy
  `yaw_deg` (always 90) is never used. If any record disagrees, the frame's headings stay unqualified.
* Pose: the frame-bound UE pose evidence, bound with the replay's own `recording.bind_pose`. It checks the full
  run/cell/stream/frame/capture-ns identity, sensor-source hash, registered calibration hashes, ego rotation and
  camera mount. An install whose pose has not arrived within 400 ms is shown with `pose_error` and ego-follow disabled.
  It is never bound to a nearby or inferred pose.
* GT is never read by the viewer.

## Time, freshness and failure behaviour

* Capture and install stamps are host unix-wall ns. The browser maps its clock to the viewer host's from every message
  (minimum-delay estimate). **Capture age = now − capture time**, so it includes everything before the browser
  (installation, tap, viewer, transport). Display timing is unrelated to the 170-ms scientific deadline. Late
  feedback does not stop a map-eligible prediction from being installed or shown.
* States: `LIVE`; `STALE` when there has been no new map update for ≥ 750 ms (tracks then fade and expire under the
  unchanged policy); `FEED DISCONNECTED` when nothing at all arrives for 1.5 s.
* One feed serves one session (run/cell). Lines from another session, duplicates, out-of-order frames and
  installations read more than 2 s after install are refused before any browser. The browser refuses another epoch,
  another session, and duplicate or earlier sequence numbers before the display engine, which refuses
  duplicate/out-of-order frames again. Expired objects therefore cannot be revived.
* Every (re)connection and every new viewer epoch resets display tracking. No backlog is sent on connect. A restarted
  viewer starts at the end of existing tap files.
* `F` / "Flag moment" writes a timestamped flag (frame, displayed objects) into the event log.

## Checks

```bash
python3 -m unittest rl_agent.splitfusion_live_view_tap_v1.test_tap spatial_map_coop.tests.test_live_feed_v1
node spatial_map_coop/tests/js/replay_tracks_v3.js      # and the other existing JS suites (unchanged results)
# Browser check against a re-timed recorded feed (pause, viewer kill/restart, injected stale messages):
PLAYWRIGHT_MODULE=… CHROMIUM=… node spatial_map_coop/replay_ui_v1/live_browser_check_v1.cjs <V19 cell> <new dir> 8775
```

`live_smoke_v1.py` re-times one sealed capture's journal and pose evidence into the tap format, for testing only.
`live_record_v1.cjs` records a normal-speed video of `/live` and samples its state at 1 Hz.

## Limits

* Building clipping is an approximate 2D static-building aid, not proof of visibility or occlusion.
* Display tracking is presentation only. It is separate from the scientific map and from any future multi-UE association.
* The pedestrian display keeps the accepted v3.1 limits: position noise and delayed motion recognition.
