"""Focused pre-launch checks for the action-0 baseline (CPU only; no CARLA, OAI, GPU or network). Create-only output.

1. identity    : the registered action-0 document == the dynamic execution contract's resolution of (0, 0); the
                 production profile bypasses the ranker (q 0) and keeps 21504/21504 locations.
2. gt_binding  : the unchanged early-GT v2 anchor-provenance mapping binds catalogue anchor 0 exactly as it binds the
                 seven screen anchors (neutral capture, neutral action identity, validation against the target identity,
                 join identity ANCHOR_PROVENANCE_V1), and refuses a mismatched anchor / off-anchor q for mode 0.
3. codec_scorer: the retained quality-reference replay (99 admitted live frames) already ran action 0 through the
                 production codec encode -> inspect -> require_inner_agreement -> decode -> V3 tail -> retained scorer.
                 Its model-source binding is re-verified against the current checkout, and the per-frame action-0
                 identity / payload is re-read (sealed). The SFD4 application frame adds the measured 455-byte envelope.
Usage: python -m rl_agent.splitfusion_noae_baseline_live_v1_analysis.preflight_checks --out <new dir>
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MAIN = Path(__file__).resolve().parents[2]
WT = MAIN.parent / "abiodun_run4_single_gpu_profiler_v1"
REPLAY = MAIN / "experiments/splitfusion_run5_quality_reference_v1/replay_01/REPLAY.json"
REPLAY_EVIDENCE = MAIN / "experiments/splitfusion_run5_quality_reference_v1/EVIDENCE_SHA256.json"
ENVELOPE_BYTES = 455        # measured SFD4 envelope at FAVORABLE_STABLE (PAYLOAD_COMPARISON_SUMMARY / plot_payload_comparison)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity_check():
    from rl_agent.splitfusion_noae_baseline_live_v1 import contract as C
    from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as dec
    profile = dec.load_dynamic_execution_contract().resolve_q_e4(0, 0)
    checks = dict(document_equals_resolution=C.resolved_forced() == C.FORCED_ACTION,
                  ranker_bypassed=profile.ranker_bypassed is True,
                  keep_all=(int(profile.keep_count), int(profile.drop_count)) == (21504, 0),
                  family_quantizer=(profile.family, profile.quantizer) == ("noAE", "UINT8"),
                  catalogue_anchor=(profile.action_id, profile.profile_id) == (0, "split_noae_uint8_q0000"))
    return dict(checks=checks, passed=all(checks.values()), forced_action=C.FORCED_ACTION)


def gt_binding_check():
    from rl_agent.splitfusion_exact_gt_service_proto_v2 import test_anchor as TA
    from rl_agent.splitfusion_exact_gt_service_proto_v2.deferred import (DeferralRefused, anchor_provenance,
                                                                         neutral_action, validate_anchor_provenance)
    from rl_agent.splitfusion_exact_gt_service_proto_v2.scorer_core import join_identity
    from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as NC
    action = dict(TA.action_for(67), mode_id=0, q_e4=0, anchor_action_id=0)
    capture = dict(TA.TEMPLATE, action_id=0, profile_id="split_noae_uint8_q0000")
    plan = NC.canonical(dict(scope="NOAE_BASELINE_PREFLIGHT", capture=capture, action=action, frame_context=TA.CONTEXT))
    doc = anchor_provenance(TA.TEMPLATE, plan)
    target = dict(capture_identity=doc["neutral_capture"], action_identity=doc["neutral_action_identity"],
                  frame_context=doc["frame_context"], plan_sha256=doc["plan_sha256"])
    validated = validate_anchor_provenance(doc, target_identity=target)
    received = dict(gt_identity=doc["selected_capture"], action_identity=doc["action_identity"],
                    capture_timestamp_ns=TA.TEMPLATE["capture_timestamp_ns"], prediction_digest="f" * 64)
    joined, mode = join_identity(received, doc, TA.TEMPLATE)
    refusals = {}
    for name, changes in (("off_anchor_q", dict(q_e4=1)), ("other_anchor_id", dict(anchor_action_id=1)),
                          ("other_mode", dict(mode_id=1))):
        bad = NC.canonical(dict(scope="NOAE_BASELINE_PREFLIGHT", capture=capture, action=dict(action, **changes),
                                frame_context=TA.CONTEXT))
        try:
            anchor_provenance(TA.TEMPLATE, bad)
            refusals[name] = None
        except DeferralRefused as exc:
            refusals[name] = exc.reason
    checks = dict(catalogue_anchor_0=(doc["catalogue"]["action_id"], doc["catalogue"]["profile_id"], doc["catalogue"]["mode_id"],
                                      doc["catalogue"]["q_e4"]) == (0, "split_noae_uint8_q0000", 0, 0),
                  neutral_capture=doc["neutral_capture"] == TA.TEMPLATE,
                  neutral_action=doc["neutral_action_identity"] == neutral_action(doc["action_identity"])
                  and doc["neutral_action_identity"]["anchor_action_id"] is None,
                  validated=validated.get("version") == 1,
                  join_anchor_provenance=mode == "ANCHOR_PROVENANCE_V1" and joined["gt_identity"] == TA.TEMPLATE,
                  mismatches_refused=all(v == "CATALOGUE_ANCHOR_MISMATCH" for v in refusals.values()))
    return dict(checks=checks, passed=all(checks.values()), refusals=refusals, catalogue=doc["catalogue"])


def codec_scorer_check():
    from rl_agent.splitfusion_paired_quality_replay_v2.model_adapter import source_binding_hashes
    evidence = json.loads(REPLAY_EVIDENCE.read_bytes())
    replay_key = next(k for k in evidence if k.endswith("replay_01/REPLAY.json"))
    replay_sha = sha(REPLAY)
    replay = json.loads(REPLAY.read_bytes())
    rows = [r for r in replay["rows"] if r["admission"] == "ADMITTED"]
    baseline = [r["baseline"] for r in rows]
    inner = np.array([b["payload_bytes"] for b in baseline], dtype=float)
    current = {name: source_binding_hashes(root) for name, root in (("main", MAIN), ("live_worktree", WT))}
    checks = dict(replay_sealed=evidence[replay_key] == replay_sha,
                  model_sources_unchanged_main=replay["model_binding_hashes"] == current["main"],
                  model_sources_unchanged_live_worktree=replay["model_binding_hashes"] == current["live_worktree"],
                  all_admitted_action0=all((b["mode_id"], b["q_e4"], b["anchor_action_id"], b["profile_id"], b["keep_count"],
                                            b["drop_count"], b["ranker_bypassed"], b["execution_bundle_sha256"])
                                           == (0, 0, 0, "split_noae_uint8_q0000", 21504, 0, True,
                                               "583cbab6b68449495e779259507fdf7dcdfba56b5d31894ab9f8955da4deb10a")
                                           for b in baseline),
                  all_scored=all(r["noae_uint8_q0"].get("measurement") is not None or r["noae_uint8_q0"].get("quality_defined") is False
                                 for r in rows),
                  admitted_99=len(rows) == 99)
    return dict(checks=checks, passed=all(checks.values()), replay=str(REPLAY.relative_to(MAIN)), replay_sha256=replay_sha,
                admitted=len(rows), inner_payload_bytes=dict(p50=float(np.percentile(inner, 50)), p99=float(np.percentile(inner, 99)),
                                                            min=int(inner.min()), max=int(inner.max())),
                application_frame_bytes=dict(p50=float(np.percentile(inner, 50)) + ENVELOPE_BYTES,
                                             max=int(inner.max()) + ENVELOPE_BYTES, envelope_bytes=ENVELOPE_BYTES))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    result = dict(schema="splitfusion.noae_baseline_live_v1.preflight_checks.v1", identity=identity_check(),
                  gt_binding=gt_binding_check(), codec_scorer=codec_scorer_check(), live_services_started=False,
                  code_sha256=sha(__file__))
    result["passed"] = all(result[k]["passed"] for k in ("identity", "gt_binding", "codec_scorer"))
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({"RESULT.json": sha(out / "RESULT.json")}, indent=1))
    print(json.dumps(dict(passed=result["passed"], **{k: result[k]["checks"] for k in ("identity", "gt_binding", "codec_scorer")}),
                     indent=1))
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
