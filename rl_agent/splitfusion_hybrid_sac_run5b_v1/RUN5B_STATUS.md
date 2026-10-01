# Run-5B status

Decision `USE_JOINT_SNR_MCS_CHANNEL` (2026-09-30). Run 5B is Run 4B on the shared joint SNR/MCS
channel, with the causal UL-SNR proxy exposed as feature 21.

## Architecture

- **Shared authority:** `rl_agent/splitfusion_joint_channel_v1/joint_channel.py`.
  - It is the Run-4B environment. Its `step` is inherited unchanged and calls the shared
    operational-latency provider (`056f0fd`, binding `3cc1e6e3…6c29`).
  - Only the MCS source differs: it is the frozen Run-5 joint SNR/MCS channel, with the same
    four profiles, the same balanced blocks and the seed rule `derive(seed, 'train-channel')`.
  - Run-4B-Joint, the paired comparator, must import this module unchanged and bind the same
    digest. The SNR-free Run-4B campaign is a pilot, not the paired ablation comparator.
- **State:** `run5b_state_contract.py`.
  - Positions 0–19 come from the unchanged Run-4B `build_features`.
  - Position 20 is the lease-admitted causal SNR, `(dB − 5.5)/19`.
- **Learner and runner:** `run5b_learner.py` and `run5b_runner.py` are generated from Run-4B's
  `learner.py` and `runner.py` by `derive_from_run4b.py`. Every difference is one listed
  substitution, and a test regenerates both files and requires byte equality.
- **Run-5B-only code:** `run5b_checks.py`, which holds the SNR, reward and Q_perc gates, the disk
  check and the fresh-process actor verification.
- **Registration:** `run5b_registration.py` produces `RUN5B_REGISTRATION.json`.

The earlier scaffold modules, which were built on the Run-5 collector, are superseded and
removed. They remain in history at `5a77b16`.
