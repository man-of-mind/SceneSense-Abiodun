# Archived demo clients

This directory preserves superseded demo entry points for source-history and
comparison purposes. They are not the supported launch targets.

- `v9/manual_control_ar_v9.py` was replaced by the v11 client and is now
  superseded by `../manual_control_ar_v12.py`.
- `v9/pedestrian_head_camera_client_v9.py` was replaced by
  the v11 client and is now superseded by
  `../pedestrian_head_camera_client_v13.py`.
- `v11/manual_control_ar_v11.py` was replaced by
  `../manual_control_ar_v12.py`.
- `v11/pedestrian_head_camera_client_v11.py` was replaced by
  `../pedestrian_head_camera_client_v13.py`.
- `v5/spawn_blocker_v5.py` preserves the earlier v5 snapshot that physically
  spawned RGB/radar spatial-map sensors. The supported root
  `../spawn_blocker_v5.py` retains the v5 filename but uses virtual map sensors
  and does not deploy those RGB/radar actors.

The archived scripts originally expected to reside directly beneath CARLA's
`PythonAPI/` directory. Run the current root clients instead of invoking these
archived copies in place.
