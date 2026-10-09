"""Per-arm integration checks for the v3 frontier measurement.

The v2 checks I1-I11 (``splitfusion_fixed_action_screen_analysis_v2.analyze.integration``, unchanged) are run
first. Two checks are then made strict for the off-anchor arm, where v2 was vacuous or anchor-only:

  I11_anchor_provenance  anchor arms: scorer- and producer-side anchor_provenance.json archived for every SCORED
                         decision (v2 rule); off-anchor arm: BOTH files ABSENT for every SCORED decision (template
                         path), instead of the v2 vacuous pass;
  I12_executed_identity  every POLICY_DECISION/POLICY_HOLD executed identity equals the registered dynamic-contract
                         resolution exactly (action_id incl. None, profile_id, keep_count, execution bundle digest);
                         every FALLBACK executed the retained a71 (mode 11, q 9800, split_ae32_uint4_q9800).
"""
from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import Arm, integration as integration_v2, load_jsonl

IDENTITY_KEYS = ("mode_id", "q_e4", "action_id", "profile_id", "keep_count", "execution_bundle_sha256")


def integration(arm, launch_returncode=None, do_rescore=True):
    d = integration_v2(arm, launch_returncode, do_rescore)
    if arm.cell is None:
        return d
    forced = arm.forced
    frames = [f for f in load_jsonl(arm.cell / "numeric_gt/service/FRAMES.jsonl") if f.get("decision")]
    scored = [f for f in frames if f.get("outcome") == "SCORED"]
    archive = arm.cell / "numeric_gt/service/archive"

    def present(fid):
        return [(archive / f"frame_{fid}/{side}/anchor_provenance.json").exists() for side in ("scorer", "producer")]
    if forced.get("action_id") is None:
        i11 = bool(scored) and all(present(f["frame_id"]) == [False, False] for f in scored)
    else:
        i11 = bool(scored) and all(present(f["frame_id"]) == [True, True] for f in scored)
    plans = load_jsonl(arm.cell / "forced_action/FORCED_ACTIONS.jsonl")
    expected = tuple(forced[k] for k in IDENTITY_KEYS)
    policy = [r for r in plans if r["kind"] in ("POLICY_DECISION", "POLICY_HOLD")]
    fallback = [r for r in plans if r["kind"] == "FALLBACK"]
    i12 = bool(policy) and all(tuple(r["executed"].get(k) for k in IDENTITY_KEYS) == expected for r in policy) and all(
        (r["executed"]["mode_id"], r["executed"]["q_e4"], r["executed"]["action_id"], r["executed"]["profile_id"])
        == (11, 9800, 71, "split_ae32_uint4_q9800") for r in fallback)
    checks = dict(d["checks"], I11_anchor_provenance=i11, I12_executed_identity=i12)
    return dict(d, checks=checks, passed=all(checks.values()), v2_i11=d["checks"].get("I11_anchor_provenance"),
                identity=dict(expected=dict(zip(IDENTITY_KEYS, expected)), policy_frames=len(policy), fallback_frames=len(fallback),
                              scored=len(scored)))


__all__ = ["Arm", "integration"]
