# DG-A + DG-A.1 Decision Summary

**Decision:** `CANDIDATE_GO_DG_B_HUMAN_REVIEW_REQUIRED`

DG-B was **not launched**. This result requires human review.

## Raw N=2 gate

- Scheduler redistribution: `True`
- A6 vs A7: meaningful_gap=`False`, deadline lifts={'0.25': 0.0, '0.50': 0.0}, latency reduction=-692.314 ms, starvation reduction=-479.051 ms.
- A8 vs A9: meaningful_gap=`False`, deadline lifts={'0.25': 0.0, '0.50': 0.0}, latency reduction=-163.085 ms, starvation reduction=1567.585 ms.

## Provisional scale screen

- MODEL-BASED provisional N=50/100 screen fitted on N=1 historical + N=2 DG-A; not measured
- Robust N=50 candidate gap: `True`
- Robust scenarios: 6

All skipped/replaced/timeout demand remains in deadline denominators.
