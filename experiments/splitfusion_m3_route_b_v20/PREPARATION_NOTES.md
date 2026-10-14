# V20 preparation records

The only attempt is `good_all_live_view_01`; its registration, runtime sources and numeric exit status were not changed after launch.

- Worktree `experiments/splitfusion_m3_route_b_v20/offline_binding_01` is an **empty** directory. The offline binding regression was first started from the deployment worktree. That worktree lacks `rl_agent/splitfusion_m3_route_b_v3_analysis`, which exists only in the main checkout. The run created the directory, then failed on that import before writing anything. It was rerun from the main checkout into `offline_binding_02` (PASS 13/13). The empty directory was kept, not deleted.
- Before the live run, the viewer smoke and browser checks were developed against a re-timed V19 feed in a scratch area. Two harness defects were fixed before the final PASS: the writer pause did not move stamps, and expected network errors during the deliberate server kill were counted as page errors. The final PASS evidence is preserved in `spatial_map_coop/replay_ui_v1/artifacts_live_v1/`.
- An existing replay server (PID 988344, 127.0.0.1:8765, started before this task) was left running. The live view used port 8766.
