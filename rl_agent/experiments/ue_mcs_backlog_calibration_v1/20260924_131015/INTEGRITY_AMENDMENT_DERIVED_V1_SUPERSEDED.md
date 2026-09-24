# Integrity amendment: legacy v1 derivatives superseded

Recorded after the live campaign completed on 2026-09-24.

The sealed `manifest.json` was written at 2026-09-24 13:29:35 -0700. A
concurrent post-run process subsequently regenerated the legacy v1 analysis
in place at approximately 13:32--13:33. This changed 15 derived files whose
earlier hashes had already been recorded by the manifest:

- `analysis_v1.json`
- `decisions.csv`
- `decisions_build.json`
- `figures/fig01_mcs_by_channel_and_load.{pdf,png}`
- `figures/fig02_backlog_by_channel_and_load.{pdf,png}`
- `figures/fig03_transient_backlog_after_transition.{pdf,png}`
- `figures/fig04_mcs_age_and_coverage_by_cell.{pdf,png}`
- `figures/fig05_features_vs_next_frame_latency.{pdf,png}`
- `figures/fig06_prediction_comparison_abc.{pdf,png}`

The process identity responsible for the rewrite was not recoverable. The
rewrite was detected by independently recomputing every manifest-listed size
and SHA-256 after campaign exit.

No raw live evidence was changed. The other 652 files listed by the sealed
manifest remained present and matched both their recorded sizes and SHA-256
digests. Those include all 12 cell records, UE/gNB tracer evidence,
sender/receiver records, radio-path proofs, channel commands/read-backs,
terminal accounting, teardown records, and the final cold-state proof.

Consequences:

1. The 15 legacy v1 derivatives above are rejected as manifest-bound evidence.
2. They will not be restored, deleted, or silently treated as the sealed
   versions.
3. Corrected analysis will be generated only under new `v2` filenames and a
   new post-run verifier manifest that binds the 652 unchanged live-evidence
   files, this amendment, and the corrected v2 derivatives.
4. The original `manifest.json` remains untouched as the audit record of what
   the campaign attempted to seal.

