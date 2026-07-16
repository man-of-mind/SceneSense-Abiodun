# OAI config sweep — overnight run status (launched 2026-07-16 ~01:15)

Autonomous sweep of the OAI network config, **model fixed at no-AE u8** (1141 KB) so the *config* is the only
variable. Phases in the order you asked: **TDD → 5QI → bandwidth**.

## Where to look in the morning
- **Results table:** `oai_config_sweep/oai_config_results.tsv` (one row per config: RTT mean/p95, payload KB,
  frag/frame, delivery %, ping RTT). View: `column -t -s $'\t' oai_config_sweep/oai_config_results.tsv`.
- **Master log:** `oai_config_sweep/logs/sweep.log` (per-config progress; look for `OAI_SWEEP_COMPLETE` at the end).
- **Per-config logs:** `oai_config_sweep/logs/{gnb,ue,front,sampler}_<label>.log`.

## What it runs (each = fixed 300-frame pole front over OAI)
1. **TDD DL:UL** (biggest uplink lever): `tdd_7dl_2ul` (baseline), `tdd_4dl_5ul`, `tdd_2dl_7ul`.
   Expectation: more UL slots → lower RTT + higher delivery (uplink-heavy feature traffic).
2. **5QI** (QoS): `5qi_9`(≈baseline), `5qi_5`, `5qi_1` — on an uplink-favored TDD (2:7) so QoS is visible.
   May **skip** a value if OAI rejects the profile (PDU session won't come up) — logged, sweep continues.
3. **Bandwidth/PRB** (riskiest, last): `prb_162`, `prb_217`, `prb_273` (RIV recomputed correctly for L>138).
   Wide-PRB may **skip** if the UE won't attach (coreset0/SSB/freq not re-derived) — logged, not fatal.

## Validation done before launch (so the night isn't wasted)
- Full cold-start of the stack (CN→gNB→UE→back-half→front) works.
- Smoke test reproduced the known A/B baseline: **delivery 75%, payload 1117 KB, RTT 183 ms, 19.8 frag/frame**.
- Found + fixed a bug: stale PDU sessions made the SMF hand out incrementing UE IPs (10.0.0.4), but the
  back-half returns results to 10.0.0.2 → would zero delivery. Fix: each config restarts amf/smf/upf to reset
  the IP pool, and the tunnel is health-checked at 10.0.0.2 (skip+log if not).
- First sweep config (`tdd_7dl_2ul`) completed live: **delivery 78%, RTT 174 ms, 1115 KB** ✓ (matches A/B).

## Stack state / cleanup
- CN core + the no-AE u8 back-half container are left **up**. The RAN (gNB/UE) is **stopped** at the end.
- Original `config.yaml` (5QI=9) is **restored** on exit; variant gNB confs (`gnb_sweep_*.conf`) are removed.
- To resume/inspect: `scripts/gnb_start.sh` + `scripts/ue_start.sh` bring the RAN back at baseline.
- If the sweep hung: `cat oai_config_sweep/logs/sweep.log`; kill `nr-softmodem`/`nr-uesoftmodem` by pidof; the
  EXIT trap restores configs. To fully tear down the core: `scripts/cn_stop.sh`.

## Caveats / honesty
- Numbers are single 300-frame runs per config (like the A/B), no channel impairment (rfsim), single UE.
- 5QI: OAI's SMF `local_subscription_infos` is the QoS source (config.yaml, not the DB); GBR profiles (5qi 1)
  may need extra params — if a value skipped, that's why.
- Bandwidth: only PRB carrier + RIV are auto-edited; SSB/PointA/coreset0 are left at the 106-PRB values, so
  wide PRB may not bring up (documented skip). A clean wide-PRB run needs those re-derived — a follow-up.
- Next step after reviewing: pick the winning TDD (+ any 5QI/PRB gain), then confirm it *composes* with AE-128
  compression (config × compression), and feed the "network action menu" to the RL controller.
