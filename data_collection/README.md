# Policy corpus collection

This directory owns the L10319 collection loop described in
`rl_agent/policy/DATA_COLLECTION_PLAN.md`.

- `carla_fusion_policy_corpus_collector.py` delegates to the validated shared
  fusion collector and adds pedestrian ground-truth rows. It does not modify or
  fork the real-time perception implementation.
- `run_policy_corpus.py` performs static/live preflight, validates any declared
  collection contract, runs the registered smoke trials or full batch, and stops at the first basic
  pipeline/timing or actor-cleanup failure.
- `verify_policy_corpus.py` applies the full §5 gates and writes a timestamped
  `CORPUS_VERIFICATION.md`, detailed CSVs, an explicit replay split manifest,
  and a hashed verification manifest.
- `rescore_policy_corpus_freshness.py` performs the phase-1, controller-independent
  freshness re-score over an existing immutable batch. It emits GT-seeded and
  detection-seeded views, speed/dwell distributions, right-censored breach times,
  liveness bands, detection coverage, and per-run/split concentration tables.
- `reconcile_detection_coverage.py` applies the identical live direct-coverage
  score to old and new traces, reports range/denominator sensitivity, and audits
  detector configuration plus timeout/empty-result accounting. It is table-only.
- `configs/detection_ab_gate_v1.yaml` defines the six-run matched three-arm gate;
  `analyze_detection_ab_gate.py` computes the hard vehicle/pedestrian gates,
  paired moving-block confidence intervals, fast-in-view dwell, and decoder
  top-80 saturation diagnostics.
- `configs/policy_corpus_v1.yaml` is the single pre-registered experiment
  definition: scenario arguments, 24 seeds, splits, provenance, and gates.
- `configs/policy_corpus_vehicle_v2.yaml` is the independently versioned Track A
  vehicle corpus: corrected 200k/FAST/NMS-2/top-120 recipe and 32 whole-trajectory
  slow/typical/dense/exact-fast runs with 4/2/2 splits per regime.
- `configs/freshness_rescore_v1.yaml` locks the table-driven re-score constants
  (epsilon, range, clock, reference localization profile, QC, and regime bands).
- `configs/freshness_rescore_vehicle_v2.yaml` applies the same locked phase-1
  equation to a verified vehicle-v2 corpus without implying pedestrian evidence.

Use the CARLA 0.10.0 virtual environment; do not export `PYTHONPATH` and do not
start OAI. Start the packaged server with `Town10HD_Opt` already loaded, then:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/run_policy_corpus.py --mode smoke
```

Inspect both smoke records in `batch_manifest.json`. Lock the healthy device
placement in the YAML before the full batch; do not simply keep the faster
placement if it violates `camera_frame_wait`.

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/run_policy_corpus.py --mode full

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/verify_policy_corpus.py \
  data_collection/experiments/policy_corpus_v1/<batch_timestamp>_full

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/rescore_policy_corpus_freshness.py \
  data_collection/experiments/policy_corpus_v1/<batch_timestamp>_full

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/reconcile_detection_coverage.py
```

Never rewrite a `FAIL_QUARANTINED` verification. A later, explicitly versioned
analysis may supersede only its use/disposition under a corrected goal; keep the
batch and both reports immutable. Do not add the batch to replay roots until the
documented human salvage/top-up decision is made.

The historical v1 corpus is additionally held by
`rl_agent/policy/DETECTION_RECONCILIATION.md`: do not make the pedestrian-scope
call or use that quarantined batch for controller training.

The current authorized reconciliation command is:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/run_policy_corpus.py \
  --config data_collection/configs/detection_ab_gate_v1.yaml --mode smoke

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m data_collection.analyze_detection_ab_gate \
  data_collection/experiments/detection_ab_gate_v1/<batch_timestamp>_smoke
```

The analyzer exits non-zero at any failed gate. Its unchanged two-class pass is
still required for Track B and any later pedestrian-inclusive collection; the
accepted vehicle-only Track A branch below is independently authorized.

## Track A corrected vehicle corpus (v2)

The 2026-08-11 joint review accepted the 200k PPS vehicle result and exact
fast-in-view realization independently of the invalid pedestrian arm. Track A
therefore uses its own config/output root and must not rewrite v1. First validate
the full resolved command matrix without contacting CARLA:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/run_policy_corpus.py \
  --config data_collection/configs/policy_corpus_vehicle_v2.yaml --validate-config
```

The live sequence is:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/run_policy_corpus.py \
  --config data_collection/configs/policy_corpus_vehicle_v2.yaml --mode smoke

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/run_policy_corpus.py \
  --config data_collection/configs/policy_corpus_vehicle_v2.yaml --mode full

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/verify_policy_corpus.py \
  data_collection/experiments/policy_corpus_vehicle_v2/<batch_timestamp>_full

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/rescore_policy_corpus_freshness.py \
  data_collection/experiments/policy_corpus_vehicle_v2/<batch_timestamp>_full \
  --config data_collection/configs/freshness_rescore_vehicle_v2.yaml
```

Inspect the smoke manifest and verifier rather than treating requested traffic
or speed as achieved. In particular, each exact-fast run must show its tagged
vehicle at >=10 m/s, in-frustum and <=25 m, for >=5 continuous seconds. The
verifier also checks runtime decoder telemetry and reports pre-top-k saturation.
Only a `PASS` full batch may become a replay root. Track B pedestrian artifacts
and gates remain separate until that experiment passes.

## Advisor-rich Town10HD_Opt corpus (v4, `FAIL_QUARANTINED`)

`run_advisor_policy_corpus.py` composes the read-only advisor traffic/blocker
logic with the fusion collector. The collector is the sole 20 Hz world ticker,
all traffic shares TM port 8010, and the collector uses observe-existing mode.
Version 4 requests the model-training arguments: 10 Hz detection, 1280x720,
camera/radar FOV 120 degrees, 200k radar pps, legacy training rasterizer radius
4, temporal window 2, NMS-2, and top-120. Policy/vehicle control remains 20 Hz.
A later observed-density audit found that this is **not** an exact realized
sensor contract: CARLA budgets radar returns from the 20 Hz physics delta, so
each 10 Hz v4 tensor contains about half the retained 10 Hz training reference.
The safe UI-authored loop is stored as
`routes/town10hd_opt_advisor_safe_perimeter_loop_v3.json`; its companion CSV is
the deterministic ego/NPC controller input. The derived wrappers preserve the
advisor sources as read-only, use one-shot reactive crossings only in the
pedestrian family, and never take a second world-ticker role.

The historical v4 validation/smoke commands were:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m data_collection.run_advisor_policy_corpus \
  --config data_collection/configs/policy_corpus_advisor_rich_v4.yaml \
  --validate-config

/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m data_collection.run_advisor_policy_corpus \
  --config data_collection/configs/policy_corpus_advisor_rich_v4.yaml \
  --mode smoke
```

Do **not** launch those v4 commands again unchanged. The next collection must be
versioned separately and pass the observed radar-density smoke gate documented
in `EVALUATION_CONTRACT_DECISION.md`.

The final v4 smoke is
`experiments/policy_corpus_advisor_rich_v4/20260813_012506_smoke` and passed all perception, fast-dwell, clock, traffic,
and cleanup gates. The resulting 24-run batch is `20260813_014501_full`, but
verification `20260813_023541` is `FAIL_QUARANTINED`: vehicle replay coverage
is 26.14% versus 45.18%, pedestrian replay coverage is 41.41% versus 50%, and
three run-level pedestrian validity/match gates failed. Desk analysis then
confirmed a global input drift: median projected radar density is 9,721/frame,
only 52.29% of the retained 18,591.5/frame reference. Do not run freshness, the
controller ladder, or RL from this batch. See `EVALUATION_CONTRACT_DECISION.md`
for the PR/range analysis, replacement acceptance contract, and re-collect
verdict; `collab/REVIEW_NOTES.md` retains the chronological record.

## Advisor-rich native-10-Hz corpus (v5, accepted)

The replacement v5 collection is complete; **do not collect again**. Its
immutable batch is
`experiments/policy_corpus_advisor_rich_v5/20260813_045142_full`. Native 10 Hz
world/sensor sampling restores on-contract radar density, and 24/24 runs pass
the online collection, traffic, and cleanup checks.

The authoritative structural acceptance is `verification/20260813_061952`
with status `PASS`. It excludes only impact run `pcarv5_mixed_va01`, freezes
validation-selected class thresholds, and admits the remaining 23 trajectories.
Near-field recall and trajectory-grouped CIs are report-only diagnostics for
this controller-training corpus; see `EVALUATION_CONTRACT_DECISION_V5.md`.

The accepted inventory has already been freshness-rescored at
`freshness_rescore/20260813_062203` and evaluated by the reward-v5 controller
ladder. Do not rerun CARLA or weaken/restore the old perception-QA gates. The
next decision is the documented RL go/no-go, not another collection.
