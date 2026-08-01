# Diagnostic-only: 273PRB AWGN policy checks

These artifacts came from earlier AWGN policy checks using the 273PRB RFsim setup.

They are useful for one narrow question:

- does the policy code path respond to retransmission/BLER evidence at all?

They are not valid as the official Track 2 bad-channel comparison because the clear-channel baseline uses 106PRB. The 273PRB setup changes grant size, BLER-update sample counts, MCS trajectory, and RLC drain behavior.

For official Track 2 policy conclusions, use the 106PRB clear-channel runs and the pending 106PRB AWGN runs from `run_awgn_106prb_policies.sh`.

