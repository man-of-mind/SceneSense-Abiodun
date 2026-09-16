# Bounded live quality-feedback probe

`live_probe.py` measures privileged per-frame tail quality and the compact
quality-feedback path without running a complete Route-B loop or another
288-cell campaign.

The initial registered matrix is action 50 under `FAVORABLE_STABLE` and
`ADVERSE_STABLE`. Each cell uses a fresh qualified CN5G/gNB/UE lifecycle and a
fresh Epic CARLA lifecycle, then stops after exactly 300 successfully
transmitted feature frames. The child keeps semantic GT, object GT, exact
record retrieval and quality evaluation enabled. It drains those owners before
copying edge quality evidence and before Docker compose-down.

## Offline preflight

This starts no live process:

```bash
python3 -m rl_agent.splitfusion_quality_feedback_probe_v1.live_probe \
  --preflight \
  --output-root experiments/splitfusion_quality_feedback_probe_v1/20260916_action50_favorable_adverse
```

The preflight refuses an action not selected and pinned by the supplied
configuration. Thus `--actions 52,70` is intentionally rejected with the
current four-action direct-validation config; a reviewed config must pin those
profiles before they can be run.

## Authorized live command

Run only after checking that the host is otherwise idle and explicitly
authorizing the two fresh live cells:

```bash
python3 -u -m rl_agent.splitfusion_quality_feedback_probe_v1.live_probe \
  --execute SPLITFUSION_QUALITY_FEEDBACK_LIVE_PROBE_V1_EXECUTE \
  --output-root experiments/splitfusion_quality_feedback_probe_v1/20260916_action50_favorable_adverse
```

There is no `--resume`: an output root is create-only, and a failed cell ends
the matrix. A fresh retry requires a fresh output root and fresh authorization.

## Lifecycle and evidence boundary

- The parent owns the radio and CARLA server so a CARLA-client abort cannot
  bypass their teardown.
- The child owns the direct map, quality-instrumented direct edge, target-SNR
  actuator and CARLA client.
- The collector upper-bounds transmission at 300 and does not replace any
  evaluation queue with a discarding queue.
- The qualified collector `finish()` drains evaluation and feedback before the
  child copies quality evidence and stops the edge container.
- A bounded `tcpdump` on the host-owned UE-softmodem tunnel `oaitun_ue1`
  proves that every compact quality ACK
  traversed the OAI downlink. Its canonical message-digest multiset must equal
  both the edge sender report and UE quality ledger, with zero kernel drops.
- `QUALITY_FEEDBACK_ANALYSIS.json`, `quality_feedback_timing_join.csv`, and
  `QUALITY_FEEDBACK_REPORT.md` keep sent, final-prediction, evaluated and ACK
  denominators separate. They report p50/p95/p99, the 140-ms budget, and
  before-next-decision rates at 8/9/10 FPS from every registered timing
  boundary. The analyzer consumes `quality_policy_timing.csv`; it does not
  install another timestamp proxy or reconstruct a missing time.
- The parent unconditionally stops surviving application resources, CARLA and
  radio state, then requires the same cold postflight used by the live campaign.
- A completed Route-B loop is never claimed. Route non-completion caused by the
  registered 300-frame sentinel is expected.

## Offline checks

```bash
python3 -m py_compile \
  rl_agent/splitfusion_quality_feedback_probe_v1/live_probe.py \
  rl_agent/splitfusion_quality_feedback_probe_v1/live_cell_child.py \
  rl_agent/splitfusion_quality_feedback_probe_v1/packet_evidence.py \
  rl_agent/splitfusion_quality_feedback_probe_v1/analyze_live_probe.py

python3 -m unittest \
  rl_agent.splitfusion_quality_feedback_probe_v1.test_quality_feedback_probe_v1 \
  rl_agent.splitfusion_quality_feedback_probe_v1.test_live_probe -v

git diff --check -- rl_agent/splitfusion_quality_feedback_probe_v1
```

These checks exercise exact budget enforcement, preservation of the real
evaluation queues, the two-cell default selection, fail-closed handling of
unregistered actions, pcap decoding and denominator-explicit percentile
summaries. They do not launch CARLA, Docker, OAI, CUDA or inference.
