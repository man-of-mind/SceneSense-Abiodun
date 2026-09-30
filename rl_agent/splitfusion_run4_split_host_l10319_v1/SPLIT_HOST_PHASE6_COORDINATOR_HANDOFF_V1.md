# Split-host Phase-6 coordinator handoff v1

Status: **offline additive seam and release validators implemented; no live run authorized or launched**.

This is deliberately not a complete split-host live parent. It adds one
process-local adapter around the unchanged Phase-6 child and pure validators
for facts collected by the two existing lifecycle owners. Three operational
pieces remain external:

1. execution of `local_ran_lifecycle_v1` (local gNB, UE, tracers, tunnel and
   RF restore);
2. remote edge/CN startup and project-scoped teardown on L10319; and
3. remote evidence transfer plus manifest verification.

`RemoteEvidenceRetrievalPlanV1` is explicitly **plan-only**. The coordinator
does not execute SSH, SCP, Docker, the RAN, CARLA, CUDA or any network service.

## What the additive child seam does

- Calls the existing `phase6_live_child_nobuild_v2` installer, preserving the
  state, actor, reward, codec, map, ticket and collector logic.
- Temporarily replaces only the direct-map endpoint resolver, so the local map
  binds `10.21.16.222:39320` without trying to inspect a nonexistent local CN
  Docker bridge.
- Replaces the frozen child's local-edge start/stop hooks with an attempt-local
  proxy. Starting the proxy creates a local GT staging directory and connects
  `HighWorkerGtSenderV1` to the already-running listener at
  `192.168.70.140:51015`; stopping it closes and snapshots that sender. It
  never starts or stops a local or remote container.
- Wraps the two GT writers only after `GtWriteRecorderV2` has wrapped them.
- Consumes a prevalidated L10319 `docker exec ... ip route get 10.0.0.2`
  observation instead of invoking Docker on W10275.
- Restores the child, pinned adapter, direct adapter (including its subprocess
  shim and `_ENDPOINT`), quality writers and feedback globals in `finally`.

The front destination remains the frozen `192.168.70.140:51002`. Compact
feedback remains `10.0.0.2:51014` over the OAI radio/UPF. Only GT uses the LAN
sideband `10.21.16.222 -> 192.168.70.140:51015`.

## Mandatory release sequence

1. The L10319 owner starts the existing CN and the attempt-owned remote edge,
   then validates its container, READY record, GT READY record and the
   container's feedback route via UPF `192.168.70.134`.
2. The W10275 local-RAN owner snapshots policy rules/table, starts only the
   local gNB/UE/tracers, and proves attachment.
3. After `oaitun_ue1` exists, collect exactly:

   ```text
   ip -j route get 192.168.70.140 from 10.0.0.2
   ```

   It must resolve `dev oaitun_ue1 table 9999`, source `10.0.0.2`, and must
   not use `wlp130s0f0` or gateway `10.21.16.162`.
4. Send one bounded tensor-path probe and reconcile its identity, payload hash,
   byte count and datagram count between an `oaitun_ue1` capture and the remote
   edge receipt. No cross-host latency is computed.
5. Run the unchanged Phase-6 child under `SplitHostPhase6ChildContextV1`.
6. The child's local proxy stop closes the GT sender and creates
   `run4_phase6/split_host_gt_sender_final.json`. Only then does
   `remote_teardown_release()` succeed.
7. The external L10319 lifecycle owner may then stop its GT listener/edge,
   collect its final records, and execute the plan-only evidence retrieval.
   The local coordinator never stops the remote CN.
8. The local-RAN owner restores RFsim to `-50 dB`, stops only its recorded
   PGIDs/tunnel, and removes only policy entries proven absent before and added
   by this attempt. It must never flush table 9999 or remove pre-existing
   rules; the current host has stale historical `10.0.0.2` rules.

## Files and tests

- `split_host_phase6_coordinator_v1.py`
- `test_split_host_phase6_coordinator_v1.py`

Focused tests cover exact source-route admission and LAN-route refusal,
attempt-scoped cleanup, tunnel capture/remote receipt reconciliation, remote
READY/feedback validation, map resolver bypass, recorder-before-sender order,
local proxy-only ownership, GT-close-before-remote-teardown release, restoration
of all modified globals and plan-only evidence retrieval.

No frozen Phase-6 file, `contract.py`, remote lifecycle, GT transport, or
remote entry point was edited by this task.
