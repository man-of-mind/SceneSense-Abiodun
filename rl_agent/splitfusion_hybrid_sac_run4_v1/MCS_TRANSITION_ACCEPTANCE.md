# Run-4 MCS transition acceptance

The duration-two UE UL-MCS generator is accepted for the bounded offline
Run-4 smoke. It is fitted only on the registered FIT transitions and evaluated
on 176 untouched transitions. Its held-out top-1 successor accuracy is 0.4148,
versus 0.2784 for retaining the previous MCS; Brier and negative log likelihood
are finite.

This is an internal dynamics check, not a claim about deployment or unseen
channels. The actor receives only the UE-decoded prior round-0/table-0 UL MCS.
No profile, gNB SNR, action, backlog, payload, or reward enters this generator.
The exact report, source-evidence digest, fitted-model digest, and source hashes
are sealed under `sealed_mcs_transition_v1/`.
