# Phase-15 retry12 offline reclassification

Status: `SPLITFUSION_PHASE15_LIVE_DEPLOYMENT_QUALIFIED`

This is an explicitly post-observation reclassification of immutable retry12; it is not a new live measurement and does not rewrite the original `FAILED.json`.

Retry12 completed Route B, sent five frames through each representative UE branch, recorded all four edge branches, returned one finite frame-context-valid result, installed it through real OAI, preserved exact terminal accounting, and ended with cold CARLA/OAI/application teardown. Its historical failures were limited to outcomes superseded by protocol-amendment commit `4f122ea`.

Measured preparation: 20/41 (0.487805); installed: 1/20 sent (0.050000), or 1/41 eligible (0.024390). These remain performance outcomes.

This artifact authorizes only the 16-cell pilot, not the 288-cell campaign.
