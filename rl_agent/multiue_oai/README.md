# DG-A OAI contention decision gate

This package implements only the approved `D0 -> DG-A -> DG-A.1` stage. It cannot launch DG-B, the
identification campaign, a coordination ladder, or RL.

The runner uses the existing two-UE OAI setup, strong AWGN, SINR-based MCS, 400 KiB production-shaped UDP
messages, the production `!IHH` chunk header, per-UE RLC/BSR/grant traces, and the frozen hard-C1 comparison.
It writes one timestamped experiment directory with raw logs, extracted traces, manifests, progress JSONL,
the provisional N=50/100 sensitivity table, and a human-review decision summary.

## Preflight only

```bash
sudo -v
rl_agent/multiue_oai/launch_dg_a_detached.sh --dry-run
```

The dry run does not execute subprocess checks. Before the real launch, run the stronger preflight:

```bash
sudo -v
rl_agent/multiue_oai/launch_dg_a_detached.sh --preflight-only
```

Preflight-only verifies paths, binaries, privilege availability, the authorization boundary, OAI patch
provenance, all 18 exact sender command contracts, the OAI trace-multiplexer binary, and the local
production-shaped sender/receiver contract. The receiver check uses the same `sudo nsenter` wrapper and
explicit stop-file/final-artifact handshake as a live trial, so a cross-namespace shutdown bug fails before
Docker or either softmodem starts.

## Detached DG-A launch

```bash
sudo -v
rl_agent/multiue_oai/launch_dg_a_detached.sh
```

For each RAN block, the full path attaches both UEs with the explicit channel objects at the clean `-50 dB`
setting, changes both uplinks to the unchanged strong `-4 dB` setting, and runs a short real-traffic gate before
D0 or calibration. That gate requires both fixed tunnels to remain stable, sender traffic to use each tunnel's
dynamically discovered IP, and each UE's PUSCH telemetry to land on the registered two-UE `6.0 dB / MCS 8`
operating point within the unchanged tolerances.

UE identity is not inferred from a two-address lease list. Preflight fixes the expected IMSI -> internal UE ->
tunnel chain, cross-checks both `uiccN` profiles and the core's `oai` DNN, and registers `10.0.0.0/24` as the
address boundary. Every RAN start then requires the same two IMSIs in the softmodem log, the fixed
`oaitun_ue1`/`oaitun_ue2` identities, two unique usable addresses inside that subnet, ext-DN reachability, and
the existing sender bind/tunnel-byte proof. This deliberately accepts lease rollover such as `.2/.3` to
`.4/.5` after a RAN-only restart while still rejecting a wrong subscriber, tunnel, subnet, or traffic path.

OAI's UE `local_tracer` accepts only one upstream client. The runner therefore starts one persistent OAI
`multi` relay per RAN block on port 2033; the raw recorder, live queue probe, and controlled-traffic grant
observer all connect through that relay, while the relay alone connects to UE port 2023. Immediately after
the real-traffic strong-rung gate, an 8-second controlled-path gate runs the recorder and grant observer
concurrently and requires service events for both frozen RNTIs. This validates the A6--A9 telemetry path before
D0 or the longer open-loop trials. Build the shipped relay once, if absent, with:

```bash
make -C OAI/openairinterface5g/common/utils/T/tracer multi
```

The live consumers use OAI `csv` syntax directly (`EVENT FIELD...`); `-OFF/-on` are deliberately restricted to
the separate `record` CLI. A real concurrent `record + csv` relay test guards this external-tool boundary.

Each traffic receiver is stopped through an explicit file request visible on both sides of the network-namespace
boundary. The runner waits for a zero exit and requires `receiver_chunks.csv`, `receiver_frames.csv`, and
`receiver_summary.json` before trace extraction or metrics. Process-group TERM/KILL is cleanup-only after a
timed-out handshake; a timeout fails the trial rather than analyzing partial artifacts.

The tunnel network sampler follows the same rule: the runner requests shutdown through a file, waits for a zero
exit, then validates the time series, two-UE summary, and manifest. It never sends a normal-path signal and then
mistakes that signal's return code for a sampler failure. Preflight exercises both handshakes concurrently with
the production-shaped local transport test.

The launcher immediately prints the exact run directory and exits. Do not poll the process. Re-engage only
after one of these files exists:

- `COMPLETED.json`: DG-A and DG-A.1 finished; inspect `results_summary.json` and `DG_A_DECISION.md`.
- `FAILED.json`: fail-fast HOLD with the error; `results_summary.json` carries the same failure verdict.

`COMPLETED.json` always says `next_stage_launched=false`. Even a candidate GO requires human review and a
separate authorization before DG-B.

## Attach-only reliability smoke

After an attachment-path repair, validate it without spending the DG-A runtime:

```bash
sudo -v
rl_agent/multiue_oai/launch_dg_a_detached.sh --attach-smoke-repeats 3
```

This mode performs three cold gNB/two-UE RAN starts on strong AWGN while keeping the existing core deployment.
Each repetition requires both explicit per-UE uplink channel models, both fixed UE tunnel names, dynamic
discovery of the CN-assigned IP on each tunnel, three stability samples with successful ext-DN pings, clean
softmodem teardown, and no RFsim model fallback. Tunnel names are the stable UE identities; IP assignment may
swap with PDU-session completion order. Trial senders bind to the discovered per-UE IP, while receiver logs
validate the embedded UE/message identity after the OAI UPF source-NAT boundary. Per-UE sender routing is
proved before NAT from the actual socket bind plus fixed-tunnel TX byte counters; the ext-DN correctly sees
the registered UPF N6 address for both UEs. It writes
the same progress, summary, and terminal sentinels as DG-A, but cannot launch D0, A1-A9, DG-A.1, or any later
stage. A passing smoke still requires human review before a separate full DG-A launch.

For the clean-channel diagnostic only, append `--attach-channel-mode clean`. Clean mode is restricted to
attach-only smoke runs and rejects any RFsim channel-model activation.

## Runtime clean-to-strong switch smoke

Before DG-A, validate the live channel-control and sender-routing repairs with one bounded diagnostic:

```bash
sudo -v
rl_agent/multiue_oai/launch_dg_a_detached.sh --runtime-switch-smoke
```

This starts both explicit per-UE RFsim channel objects at an effectively clean `-50 dB` noise setting,
attaches both UEs, records a short clean traffic baseline, and uses the gNB telnet control interface to
modify **both** `rfsimu_channel_ue0` and `rfsimu_channel_ue1` to the measured strong setting (`-4 dB`). It
then sends asymmetric production-shaped traffic from each dynamically discovered UE tunnel IP for the
30-second calibration window. The smoke passes only if both sockets bind to their discovered UE IPs, both
fixed tunnels carry the corresponding application bytes, post-NAT message identity remains valid,
both per-UE gNB PUSCH streams move to the re-registered two-UE `6.0 dB / MCS 8` point within the frozen
tolerances, both model objects
report the target value, and both tunnels remain stable before and after the switch. A partial/no-op
modify fails closed. This mode writes terminal sentinels and cannot launch D0, DG-A, or any later stage.

The registration changed on 2026-08-13 without changing the physical channel or widening either tolerance.
The inherited `8.2 dB / MCS 9` label came from the one-UE sweep; the first valid two-UE, both-uplink `-4 dB`
runtime measurement gave `6.0 dB / MCS 8` for both UEs using
`GNB_MAC_PUSCH_POWER_CONTROL.snrx10 / 10`. This audit-sensitive correction is flagged for advisor review.

The OAI build must contain `cmake_targets/ran_build/build/libtelnetsrv.so`. Build it once, if needed, with
`cmake_targets/build_oai --ninja --build-lib telnetsrv`.

## Frozen comparison

- Estimator window: 1.0 s.
- EWMA alpha: 0.20.
- Observation lag: one 50 ms tick.
- C1 pessimism factor: 0.70.
- Obsolete-frame rule: newest arrival replaces an unsent older frame.
- Every registered arrival stays in deadline denominators; replace/SKIP/timeout is not delivered.
- Paired demand traces must have identical SHA-256 hashes.

The authoritative resolved values live in `configs/dg_a_v1.yaml` and are copied into every run directory.
