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

## Frozen comparison

- Estimator window: 1.0 s.
- EWMA alpha: 0.20.
- Observation lag: one 50 ms tick.
- C1 pessimism factor: 0.70.
- Obsolete-frame rule: newest arrival replaces an unsent older frame.
- Every registered arrival stays in deadline denominators; replace/SKIP/timeout is not delivered.
- Paired demand traces must have identical SHA-256 hashes.

The authoritative resolved values live in `configs/dg_a_v1.yaml` and are copied into every run directory.
