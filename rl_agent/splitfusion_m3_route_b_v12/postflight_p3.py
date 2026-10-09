"""M3.2 v12 registered offline postflight: the P3 pose rule applied exactly as the live GT workers apply it.

The live GT prep (BIND) and score (TARGET) workers run under the registered P3 substitution
(``splitfusion_m3_route_b_v3.worker_entry.py:138-144``: ``ready._M3_AGREE_RECONSTRUCTED = pose_tolerance
.agree_reconstructed`` plus ``capacity.install(POSE_GT_WORKER)``), i.e. the magnitude-aware float32 translation
tolerance ``max(1e-5 m, 2 x ULP32(max|t|))`` valid to 512 m.  The offline tools did NOT: the verdict tool's archive
rescore and the retention GT-reference adapter both re-admit targets through the same
``numeric_gt_integration_v2.ready._frame_geometry`` under the retained fixed 1e-5 m bound.

In V8 that asymmetry surfaced for the first time (the route finally reached beyond ~128 m): the verdict tool crashed
with ``PoseParityError ... 1.2594e-05`` and the GT-reference adapter refused one of 100 retained frames, while the
live workers had accepted all 1,786 BIND and 1,786 TARGET admissions, 0 rejected, max 1.417e-5 m at 140.66 m
(1.29 ULP).  V8 recovered with an unregistered wrapper after the fact; v12 registers the rule prospectively.

``install_p3()`` is that same install, in this process only, and it covers every ``ready`` entry point
(``_admit``, ``admit_payload``, ``decode_target``) because ``_frame_geometry`` is resolved through the ``ready``
module globals.  It changes no tolerance the live workers do not already use, and it never relaxes the rotation,
homogeneous-row, rigidity, range, identity, timestamp or calibration-hash checks, nor any strict
copy-of-the-same-source comparison.

    python3 -B -m rl_agent.splitfusion_m3_route_b_v12.postflight_p3 verdict   <capture root> <out> <rc> <launch log>
    python3 -B -m rl_agent.splitfusion_m3_route_b_v12.postflight_p3 retention <capture root> <out>

Both modes run from the MAIN checkout (the analysis packages live only there) and write, beside their own output,
``P3_POSTFLIGHT_<mode>.json``: the registered rule, the installed substitution record and the per-check P3
statistics, so the offline admission is auditable against the live workers' ``m3_pose`` trace.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import sys

from rl_agent.splitfusion_m3_route_b_v3 import capacity as CAP
from rl_agent.splitfusion_m3_route_b_v3 import pose_tolerance as PT
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as READY

from . import contract as C

SCHEMA = "splitfusion.m3_route_b_v12.postflight_p3.v1"
MODES = ("verdict", "retention")


@contextmanager
def install_p3():
    """The live GT workers' P3 install, in this process (both roles use exactly these two steps)."""
    if hasattr(READY, "_M3_AGREE_RECONSTRUCTED"):
        raise RuntimeError("REFUSED: GT reconstructed pose comparison already installed")
    READY._M3_AGREE_RECONSTRUCTED = PT.agree_reconstructed
    installed = None
    try:
        installed = CAP.install(CAP.POSE_GT_WORKER)
        yield installed
    finally:
        if installed is not None:
            installed.uninstall()
        del READY._M3_AGREE_RECONSTRUCTED


def document(mode, installed, returncode):
    return dict(schema=SCHEMA, mode=mode, rule=dict(C.POSE), substitutions=list(CAP.POSE_GT_WORKER),
                installed=None if installed is None else installed.document(), pose_checks=PT.snapshot(),
                returncode=returncode, live_equivalence="identical to splitfusion_m3_route_b_v3.worker_entry "
                "(GT prep BIND and GT score TARGET); no other tolerance, gate or strict comparison is changed")


def verdict(root, out, launch_returncode, launch_log):
    """The registered M3 integration verdict tool (archive rescore included) under P3."""
    from rl_agent.splitfusion_m3_route_b_v3_analysis.integration import evaluate
    out, installed, value = Path(out), None, None
    with install_p3() as installed:
        value = evaluate(str(root), launch_returncode=int(launch_returncode), launch_log=str(launch_log),
                         root=C.repo_root())
        returncode = 0 if value.get("verdict") == "PASS_M3_FULL_ROUTE_INTEGRATION" else 3
        (out / "VERDICT_registered_p3.json").write_text(json.dumps(value, indent=1, sort_keys=True, default=str) + "\n")
    (out / "P3_POSTFLIGHT_verdict.json").write_text(
        json.dumps(document("verdict", installed, returncode), indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(dict(verdict=value.get("verdict"), returncode=returncode), default=str))
    return returncode


def retention(root, out):
    """Retention archive verification and GT-reference resolution (the registered postflight) under P3."""
    from rl_agent.splitfusion_m3_route_b_v3_analysis.campaign import POSTFLIGHT
    root, out, installed = Path(root), Path(out), None
    out.mkdir(parents=True, exist_ok=False)
    before = sys.argv
    sys.argv = [__name__, str(root), str(out)]
    try:
        with install_p3() as installed:
            exec(compile(POSTFLIGHT, "<m3-route-b-v12-registered-retention-postflight>", "exec"),
                 dict(__name__="__postflight__"))
    finally:
        sys.argv = before
    (out / "P3_POSTFLIGHT_retention.json").write_text(
        json.dumps(document("retention", installed, 0), indent=1, sort_keys=True, default=str) + "\n")
    return 0


def postflight_registration():
    return dict(schema=SCHEMA, module=C.MODULE + ".postflight_p3", modes=list(MODES),
                rule=dict(C.POSE), substitutions=list(CAP.POSE_GT_WORKER),
                target="rl_agent.splitfusion_numeric_gt_integration_v2.ready:_frame_geometry",
                covers=["the registered M3 integration verdict tool's archive rescore "
                        "(splitfusion_m3_route_b_v3_analysis.integration.evaluate -> "
                        "splitfusion_fixed_action_screen_analysis_v2.analyze.rescore)",
                        "the retention GT-reference adapter "
                        "(splitfusion_paired_retention_v2.gt_adapter.resolve_gt_references)"],
                equivalence="the same two steps the live GT prep and score workers run "
                            "(splitfusion_m3_route_b_v3.worker_entry)",
                unchanged=["rotation / homogeneous-row / rigidity / finiteness bounds", "512 m coordinate range",
                           "identity, frame, timestamp and calibration hashes",
                           "every strict copy-of-the-same-source comparison", "the archive bytes and the scores"],
                v8_evidence="unregistered wrapper experiments/splitfusion_m3_route_b_v8/postflight_registered_p3.py; "
                            "live P3 on that capture: 1,786 calls per GT worker, 0 rejected, max 1.417e-5 m at "
                            "140.66 m, 15 calls above the retained 1e-5 bound")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in MODES:
        raise SystemExit("usage: postflight_p3 verdict <root> <out> <rc> <log> | retention <root> <out>")
    if args[0] == "verdict":
        if len(args) != 5:
            raise SystemExit("verdict requires <root> <out> <launch returncode> <launch log>")
        return verdict(args[1], args[2], args[3], args[4])
    if len(args) != 3:
        raise SystemExit("retention requires <root> <out>")
    return retention(args[1], args[2])


if __name__ == "__main__":
    raise SystemExit(main())
