"""Affected real-service proof: high-coordinate fixtures through prep BIND and score TARGET.

The sealed v3 rehearsal mechanics are reused.  Two of its scored sources are
replaced before staging with coherent fixture-owned transforms; its existing
send-failure source is additionally given a catalogue-inconsistent action
identity so a genuine identity corruption is refused without killing the
service.  The nested v3 rehearsal output is left sealed and a v4 gate is sealed
around it.
"""
import argparse
import json
from pathlib import Path

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as CC
from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_m3_route_b_v3 import gc_service_rehearsal as R3
from rl_agent.splitfusion_m3_route_b_v3 import gc_seams as GS

from . import contract as C
from .fixture_cases import FixtureCorpus


class FixtureRig(R3.Rig):
    def admit(self, index, *, send=True, anchor=None, tamper=False):
        # Index 9 is already a deliberately unscored edge-terminal scenario,
        # but its source is sent. Make its selected action catalogue-
        # inconsistent so the real prep worker must reject it at BIND.
        corrupt = int(index) == 9
        return super().admit(index, send=send, anchor=(17 if corrupt else anchor),
                             tamper=(bool(tamper) or corrupt))


def _trace(root, arm, role):
    return json.loads((root / arm / C.TRACE_DIR / f"{role}_gc_trace.json").read_bytes())


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    nested = out / "real_service"
    old_corpus, old_rig, old_c, old_gs_c = P.Corpus, R3.Rig, R3.C, GS.C
    reference_pose = None
    result, error = None, None
    try:
        P.Corpus, R3.Rig, R3.C, GS.C = FixtureCorpus, FixtureRig, C, C
        # The in-process reference producer performs the same ready admission
        # as the workers and therefore needs the same registered P3 comparison.
        from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as READY
        from rl_agent.splitfusion_m3_route_b_v3 import capacity as CAP, pose_tolerance as PT
        READY._M3_AGREE_RECONSTRUCTED = PT.agree_reconstructed
        reference_pose = CAP.install(CAP.POSE_GT_WORKER)
        try:
            retained = R3.run(nested)
        finally:
            reference_pose.uninstall()
            del READY._M3_AGREE_RECONSTRUCTED
        corpus = FixtureCorpus.LAST
        fixtures = list(corpus.fixture_evidence)
        frames = [str(row["source_case_frame_id"]) for row in fixtures]
        arms = {}
        for arm, arm_result in retained["arms"].items():
            traces = {role: _trace(nested, arm, role) for role in ("prep", "score")}
            pose = {role: ((trace.get("m3_pose") or {}).get("checks") or {}).get(
                "live context/source camera", {}) for role, trace in traces.items()}
            scored = {frame: arm_result["frames"][frame] for frame in frames}
            corrupt_frame = str(corpus.cases[9].frame_id)
            corrupt_outcome = arm_result["frames"][corrupt_frame]["outcome"]
            checks = dict(
                both_workers_crossed_retained_bound=all(pose[role].get("over_retained_1e5", 0) >= 2
                                                        for role in ("prep", "score")),
                both_workers_accepted=all(pose[role].get("rejected", 1) == 0 for role in ("prep", "score")),
                fixture_processing_scored=all(row["outcome"] == "SCORED" for row in scored.values()),
                fixture_own_input_score_exact=all(row.get("exact") is True for row in scored.values()),
                fixture_archive_rescore_exact=all(row.get("archive_rescore_exact") is True
                                                  for row in scored.values()),
                identity_corruption_rejected=(corrupt_outcome != "SCORED" and
                                              ("REFUSED" in corrupt_outcome or "UNAVAILABLE" in corrupt_outcome)),
                substitutions_restored=arm_result["checks"]["m3_prep_capacity"]
                    and arm_result["checks"]["m3_score_pose"]
                    and arm_result["checks"]["ue_seam_once_restored"],
                process_cleanup=arm_result["checks"]["all_retired"]
                    and arm_result["checks"]["service_exit_zero"])
            arms[arm] = dict(status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                worker_pose=pose, fixture_frames=scored, identity_corruption_frame=int(corrupt_frame),
                identity_corruption_outcome=corrupt_outcome)
        fixture_checks = dict(
            exact_archived_delta=(fixtures[0]["max_abs_matrix_delta_m"] == 1.0009130875232586e-05),
            planned_coordinate_around_141=(140.0 <= fixtures[1]["max_abs_translation_m"] <= 142.0),
            both_cross_retained_bound=all(row["max_abs_matrix_delta_m"] > 1e-5 for row in fixtures),
            transforms_have_exact_inverses=all(row["camera_inverse_max_identity_error"] <= 1e-12 for row in fixtures),
            independently_hashed_sources=(len({row["staged_packet_sha256"] for row in fixtures}) == 2))
        passed = retained["status"] == "PASS" and all(fixture_checks.values()) \
            and all(value["status"] == "PASS" for value in arms.values())
        result = dict(schema="splitfusion.m3_route_b_v4.real_service_high_coordinate_proof.v1",
            status="PASS" if passed else "FAIL", fixtures=fixtures, fixture_checks=fixture_checks,
            arms=arms, retained_rehearsal=str(nested), retained_rehearsal_status=retained["status"],
            registered_pose_rule=C.POSE, sources=C.exact_sources(C.ROOT if hasattr(C, "ROOT") else Path(__file__).resolve().parents[2]),
            live_services_started=False, performance_claim=False)
        (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
        if not passed:
            raise RuntimeError("M3 v4 real-service high-coordinate gate failed")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if result is None:
            result = dict(schema="splitfusion.m3_route_b_v4.real_service_high_coordinate_proof.v1",
                          status="FAIL", error=error, live_services_started=False)
            (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        raise
    finally:
        if reference_pose is not None:
            reference_pose.uninstall()
        P.Corpus, R3.Rig, R3.C, GS.C = old_corpus, old_rig, old_c, old_gs_c
        manifest = {str(path.relative_to(out)): CC.sha256(path.read_bytes())
                    for path in sorted(out.rglob("*")) if path.is_file()
                    and path.name != "OUTPUT_SHA256.json" and "archive" not in path.parts}
        (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "fixture_checks": result.get("fixture_checks"),
                      "arms": {k: v["checks"] for k, v in result.get("arms", {}).items()}}, indent=1))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    run(parser.parse_args().out)
