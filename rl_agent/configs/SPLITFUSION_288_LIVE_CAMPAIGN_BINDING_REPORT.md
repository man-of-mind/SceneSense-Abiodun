# SplitFusion 288-cell live campaign binding

Status: `READY_FOR_EXPLICIT_AUTHORIZATION`

This update removes the obsolete 40-MHz/106-PRB runtime fields from the full
campaign contract and binds the 72 actions × four network profiles to the
qualified n78 100-MHz/273-PRB/4D5U path.

The launch gate verifies, before any live process starts:

- the exact 288-cell Cartesian product and its registered mapping digest;
- the qualified Phase-14A RFsim mapping and continuous, non-wrapping traces;
- the current SFD1-v2 dispatcher, adapter, target-SNR runtime, OAI launcher,
  map-install runtime, action catalog and checkpoints;
- the completed 16-cell structural integration pilot;
- the four-cell result-path/freshness recovery qualification;
- the two-cell preparation-path qualification, including preparation coverage
  of at least 0.95 and cold teardown;
- create-only/resumable campaign state and an immutable run manifest.

The binding deliberately does not self-authorize execution. A launch still
requires the exact execution token and an explicit `--authorize-full-sweep`
flag. Missed service deadlines and low delivery are scientific measurements;
corruption, identity/accounting failure, process failure and dirty teardown
remain fatal.

The 100-ms service target and 500-ms ACK horizon remain distinct. The evidence
does not claim that any current split profile is 100-ms service-ready.

Offline reconciliation command:

```bash
/usr/bin/python3 -m rl_agent.splitfusion_288_live_campaign_v1 validate
```

No OAI, RFsim, CARLA, CUDA, model inference or campaign cell is started by that
command.
