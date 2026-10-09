"""V4-bound facade over the unchanged v3 prospective publisher.

The broad v3 prerequisites are reused under their sealed v3 source identity.
Only the two changed seams are added: the final high-coordinate real-service
proof and the deployment-path affected-test/startup proof.
"""
import json
from pathlib import Path

from . import contract as C
from rl_agent.splitfusion_m3_route_b_v3 import host as _H
from rl_agent.splitfusion_m3_route_b_v3 import contract as V3

_V3_CHECK = _H.check_prerequisites
_V3_VERIFY_PREREQUISITE = _H.verify_prerequisite


def _bind():
    _H.C = C
    return _H


def check_prerequisites(root, proofs):
    root = Path(root).resolve(strict=True)
    v3_prerequisite = json.loads((root / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
        "20261007_m3_route_b_v3/good_all/PREREQUISITE.json")).read_bytes())
    retained_proofs = dict(proofs)
    for key in ("full_startup", "gc_service_rehearsal"):
        retained_proofs[key] = v3_prerequisite[key]["path"]
    saved = _H.C
    _H.C = V3
    try:
        retained = _V3_CHECK(root, retained_proofs)
    finally:
        _H.C = saved
    worker_path = proofs["gc_service_rehearsal"]
    preflight_path = proofs["full_startup"]
    worker_desc, worker = _H._proof(worker_path, dict(status="PASS", live_services_started=False))
    if (worker.get("sources") != C.exact_sources(root)
            or not all(worker.get("fixture_checks", {}).values())
            or set(worker.get("arms") or {}) != {"candidate_v2", "candidate_v1_rollback"}
            or any(value.get("status") != "PASS" for value in worker["arms"].values())):
        raise ValueError("v4 high-coordinate worker proof differs from the registered sources/gates")
    preflight_desc, preflight = _H._proof(preflight_path, dict(status="PASS", live_services_started=False,
                                                                cuda_initialized=False))
    if preflight.get("sources") != C.source_inventory(root) or preflight.get("test_returncode") != 0:
        raise ValueError("v4 deployment preflight differs from the runtime source inventory")
    return dict(retained, schema="splitfusion.m3_route_b_v4.prerequisite.v1",
                retained_v3_schema=retained["schema"], full_startup=preflight_desc,
                gc_service_rehearsal=worker_desc, v4_worker_rehearsal=worker_desc,
                v4_deployment_preflight=preflight_desc)


def verify_prerequisite(output, record):
    saved_check, saved_c = _H.check_prerequisites, _H.C
    _H.check_prerequisites, _H.C = check_prerequisites, C
    try:
        return _V3_VERIFY_PREREQUISITE(output, record)
    finally:
        _H.check_prerequisites, _H.C = saved_check, saved_c


def verify_prospective(*args, **kwargs): return _bind().verify_prospective(*args, **kwargs)
def prepare(*args, **kwargs):
    saved_check, saved_c = _H.check_prerequisites, _H.C
    _H.check_prerequisites, _H.C = check_prerequisites, C
    try:
        return _H.prepare(*args, **kwargs)
    finally:
        _H.check_prerequisites, _H.C = saved_check, saved_c


def launch(output, *, execute_token):
    saved_verify, saved_c = _H.verify_prerequisite, _H.C
    _H.verify_prerequisite, _H.C = verify_prerequisite, C
    try:
        return _H.launch(output, execute_token=execute_token)
    finally:
        _H.verify_prerequisite, _H.C = saved_verify, saved_c


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "launch"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retained-code", type=Path)
    for name in _H.PROOFS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--arm", choices=C.ARMS)
    parser.add_argument("--producer", choices=C.PRODUCERS)
    parser.add_argument("--execute", default="")
    args = parser.parse_args()
    if args.operation == "prepare":
        proofs = {key: getattr(args, key) for key in _H.PROOFS}
        print(json.dumps(prepare(args.output, arm=args.arm, producer=args.producer,
            retained_code=args.retained_code, proofs=proofs, acceptance=args.acceptance, budget=args.budget)))
        return 0
    return 0 if launch(args.output, execute_token=args.execute) else 1


if __name__ == "__main__":
    main()
