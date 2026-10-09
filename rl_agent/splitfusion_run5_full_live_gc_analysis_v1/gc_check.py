"""GC-trace completeness/integrity for one arm (pure over durable evidence). Separate from the retained integration
verdict; a safety stop is reported as such, not as an integration failure."""
import json
from pathlib import Path

from rl_agent.splitfusion_run5_full_live_gc_v1 import contract as C


def _j(path):
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, ValueError):
        return None


def check(arm_dir):
    arm_dir = Path(arm_dir)
    reg = _j(arm_dir / "registration.json") or {}
    arm = (reg.get("gt_gc") or {}).get("arm")
    d = arm_dir / C.TRACE_DIR
    prep, score = _j(d / "prep_gc_trace.json"), _j(d / "score_gc_trace.json")
    ue, sv = _j(d / "ue_substitution.json"), _j(d / "service_substitution.json")
    checks = {}
    checks["traces_present"] = all(x is not None for x in (prep, score, ue, sv))
    if not checks["traces_present"]:
        return dict(arm=arm, passed=False, checks=checks, safety_stop=None)
    checks["ue_substitution_once_restored"] = len(ue["substitutions"]) == 1 and ue["restored"] is True and ue["error"] is None
    checks["service_substitution_roles"] = sorted((s["role"], s["gc_policy"]) for s in sv["substitutions"]) == sorted(
        [("prep", arm), ("score", "control")]) and sv["error"] is None and sv["returncode"] == 0
    checks["roles_as_registered"] = (prep["role"], prep["role_seen"], score["role"], score["role_seen"]) == (
        "prep", "prep", "score", "score") and not prep["refusal"] and not score["refusal"]
    checks["policy_only_in_candidate_prep"] = prep["policy_installed"] is (arm == "candidate") and score["policy_installed"] is False
    checks["gc_state_restored"] = prep["restored_equals_initial"] is True and score["restored_equals_initial"] is True
    checks["clean_worker_exit"] = prep["exit"]["exception"] is None and score["exit"]["exception"] is None
    checks["no_trace_overflow"] = prep["dropped"] == {"ops": 0, "gen2": 0} and score["dropped"] == {"ops": 0, "gen2": 0} \
        and prep["policy_events_dropped"] == 0
    return dict(arm=arm, passed=all(checks.values()), checks=checks, safety_stop=prep.get("safety_stop"))
