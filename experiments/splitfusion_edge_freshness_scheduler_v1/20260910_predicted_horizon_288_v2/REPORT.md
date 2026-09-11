# Latest-only scheduler: fixed expiry versus predicted usefulness

This is a counterfactual model over the same 288 reconstructed cells,
not a live remeasurement. The predicted policy never uses the current
frame's realized service time. It uses frozen action-family and network-
profile medians known before the frame starts.

| Metric | Fixed 25 ms | Predicted 500 ms install horizon |
|---|---:|---:|
| Installed/sent | 0.4786 | 0.5753 |
| Install AoI, median of cell medians | 278.8 ms | 298.1 ms |
| Time-weighted map AoI, cell median | 377.0 ms | 382.1 ms |
| Useful installations | 425,022 | 504,199 |
| Fixed queue expiries | 138,894 | 0 |
| Predicted-obsolete drops | 0 | 1,971 |

The candidate processes the only pending frame even after 25 ms when its
predicted map-install age still fits 500 ms. It drops that frame only
when the causally predicted completion would already exceed the horizon.
The 100 ms value remains a reporting target, not this admission horizon.

## Limits

- Service estimates are fixed family medians from prior live measurements.
- Install-delay estimates are fixed network-profile medians from the completed campaign.
- A deployment should update those estimates causally (for example, an EWMA) and retain the frozen fallback.
- This comparison does not include the new v2 post-processing saving, which is not yet live-qualified.
