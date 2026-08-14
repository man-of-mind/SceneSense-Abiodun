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
provenance, and the local production-shaped sender/receiver contract. It does not start Docker or either
softmodem.

## Detached DG-A launch

```bash
sudo -v
rl_agent/multiue_oai/launch_dg_a_detached.sh
```

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
both per-UE gNB PUSCH streams move to `8.2 dB / MCS 9` within the frozen tolerances, both model objects
report the target value, and both tunnels remain stable before and after the switch. A partial/no-op
modify fails closed. This mode writes terminal sentinels and cannot launch D0, DG-A, or any later stage.

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
