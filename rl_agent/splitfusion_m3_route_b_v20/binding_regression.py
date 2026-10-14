"""Focused v20 binding regression: GOOD_ALL schedule identity, scope diff vs v19, and the live-view taps in the real
map-server install chain and under the C5 pose-archive substitution.  Qualified GT/containment/accounting proofs are
reused unchanged (prerequisite).  No CARLA, OAI, GT service or live inference is started.
"""
import argparse
import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys
import textwrap

from . import contract as C, host as H, entries as E, runner as R


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=1, sort_keys=True, allow_nan=False)


def fresh(code, root, timeout=300):
    """Run ``code`` in a fresh interpreter (no PYTHONPATH, CPU only); return (rc, last JSON line or None, tail)."""
    import os
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'LD_PRELOAD')}
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=root, env=env, capture_output=True, text=True,
                            timeout=timeout)
    lines = [x for x in result.stdout.splitlines() if x.startswith('{')]
    return result.returncode, (json.loads(lines[-1]) if lines else None), (result.stdout + result.stderr)[-3000:]


MAP_CHAIN = textwrap.dedent('''
    import json, sys, tempfile, time
    from pathlib import Path
    sys.argv = ["map_check"]
    from rl_agent.splitfusion_m3_route_b_v3.map_class import install_map_class_conversion
    from rl_agent.splitfusion_live_view_tap_v1 import tap as TAP
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_map_server_v2 as M6
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
    from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as dec
    install_map_class_conversion()
    state = TAP.install_map_tap()
    server = M6.install(dec.load_dynamic_execution_contract())
    baseline = server.baseline
    baseline.CONFIG = baseline.parse_args()
    out = Path(tempfile.mkdtemp()) / "cell"
    state["tap"] = TAP.BoundedJsonlTap(out / "live_view" / "map_installs.jsonl", role="map", build=TAP.map_line).start()
    chain = [c.__name__ for c in server.DirectMapIngestService.__mro__[:3]]
    rec = dict(class_name="vehicle", internal_class=1, score=0.5, world_x=10.0, world_y=-3.0, world_z=0.2,
               size_x=4.5, size_y=1.9, size_z=1.5, yaw_sin=0.6, yaw_cos=0.8, prediction_index=0, candidate_identity="c0")
    doc = dict(schema=MP.RUN4_UPDATE_SCHEMA, run4_identity=dict(anchor_action_id=None), run_id="r", cell_id="c",
               stream_id="s", frame_id=11, capture_timestamp_ns=1_700_000_000_000_000_000, action_id=71,
               records=[rec, dict(rec, class_name="person", score=0.3, world_x=5.0, prediction_index=1)])
    result = server.install_under_state_lock(doc, time.time())
    try:
        server.install_under_state_lock(dict(doc, frame_id=None), time.time())
        failed_raised = False
    except Exception:
        failed_raised = True
    status = state["tap"].close()
    rows = [json.loads(x) for x in (out / "live_view" / "map_installs.jsonl").read_text().splitlines()]
    installed = baseline.installed_frame_history[("s", 11)]
    print(json.dumps(dict(chain=chain, result_keys=sorted(result), rows=len(rows), failed_raised=failed_raised,
        record_equal=rows[0]["record"] == json.loads(json.dumps(installed)),
        types=[o["type"] for o in rows[0]["record"]["objects"]],
        legacy_yaw=[o["yaw_deg"] for o in rows[0]["record"]["objects"]],
        producer_yaw=[[p["yaw_sin"], p["yaw_cos"]] for p in rows[0]["producer_records"]],
        identity=rows[0]["identity"], status=status["counts"], first_error=status["first_error"])))
''')

POSE_CHAIN = textwrap.dedent('''
    import json, tempfile
    from pathlib import Path
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose as POSE
    from rl_agent.splitfusion_m3_route_b_v3 import capacity as CAP
    from rl_agent.splitfusion_live_view_tap_v1.tap import install_ue_pose_tap
    installed = CAP.install(("C5_UE_POSE_ARCHIVE",))
    original_add = POSE.PoseArchive.add
    d = Path(tempfile.mkdtemp())
    try:
        with install_ue_pose_tap(d) as tap:
            archive = POSE.PoseArchive(d / "pose_archive", side="ue", max_frames=3900)
            for frame in range(1200):
                archive.add("OWNED_SOURCE", source={"frame_id": frame}, source_sha256=str(frame))
                archive.add("UE_WIRE", context={"frame_id": frame}, source_sha256=str(frame))
            events = len(archive._events)
    finally:
        restored = POSE.PoseArchive.add is original_add
        installed.uninstall()
    rows = (d / "live_view" / "ue_pose.jsonl").read_text().splitlines()
    print(json.dumps(dict(events=events, rows=len(rows), restored=restored, written=tap.status["counts"]["written"],
                          dropped=tap.status["counts"]["dropped_queue_full"], errors=tap.status["first_error"])))
''')


def scopes(fn):
    return [[ast.unparse(i) for i in w.items] for w in ast.walk(ast.parse(inspect.getsource(fn)))
            if isinstance(w, ast.With)]


def run(name):
    root = C.repo_root()
    out = root / 'experiments/splitfusion_m3_route_b_v20' / name
    out.mkdir(parents=True, exist_ok=False)
    checks, info = {}, {}
    checks['native_runner_host_contract_identities'] = R.C is C and H.C is C and E.C is C
    checks['registered_role_constants'] = all(C.role_constant_audit().values())
    checks['unchanged_containment_accounting_workers'] = (C.SERVICE_ENTRY == C.V17.SERVICE_ENTRY
        and C.WORKER_ENTRY == C.V17.WORKER_ENTRY and C.accounting_registration() == C.V17.accounting_registration())
    v11 = json.loads((root / H.V11_ROOT_RELATIVE / 'registration.json').read_bytes())
    channel = C.channel_registration(root)
    checks['channel_is_the_actual_v11_good_all_registration'] = channel == v11['channel']
    checks['schedule_is_constant_23p5_rfsim_minus_12'] = (channel['schedule']['target_snr_db'] == 23.5
        and channel['schedule']['expected_rfsim_command_db'] == -12.0 and abs(C.ACTOR_SNR_FEATURE - 18 / 19) < 1e-15)
    from rl_agent import ue_target_snr_cell_runtime_v1 as base
    replay = base.load_replay_module()
    mapping = base.load_mapping(root / channel['mapping_csv'], replay)
    checks['existing_calibration_maps_23p5_to_minus_12'] = replay.inverse_interpolate(23.5, mapping, .25) == -12
    checks['controller_is_the_sealed_v11_module'] = (C.CONTROLLER_MODULE == C.V11.CONTROLLER_MODULE
        == 'rl_agent.splitfusion_m3_route_b_v11.good_all_controller')
    from rl_agent.splitfusion_m3_route_b_v19 import entries as V19E
    before, after = scopes(V19E._child), scopes(E._child)
    expected = [list(x) for x in before]
    expected[0][0] = 'install_good_all_controller(attempt)'
    expected[0].remove('install_spatial_pose(attempt)')
    expected[0].insert(expected[0].index("POSE.install_ue_pose(attempt_dir=attempt, max_frames=int(reg['transmitted_budget']))"),
                       'install_ue_pose_tap(attempt)')
    checks['child_scopes_only_registered_changes_vs_v19'] = after == expected
    info['child_scopes'] = after
    checks['edge_scopes_unchanged_vs_v19'] = scopes(E.edge) == scopes(V19E.edge)
    checks['map_trampoline_is_v20_and_retained_path_unchanged'] = (
        C.MAP_SERVER_TRAMPOLINE.endswith('map_profiled_v1.py') and C.RETAINED_MAP_SERVER == C.V3.RETAINED_MAP_SERVER
        and (root / C.MAP_SERVER_TRAMPOLINE).read_bytes() == (root / C.LIVE_VIEW_TRAMPOLINE_SOURCE).read_bytes())
    registration = C.live_view_registration(root)
    checks['live_view_registration_binds_tap_sources'] = registration['sources'] == {
        str(p.relative_to(root)): C.sha(p) for p in sorted((root / C.LIVE_VIEW_PACKAGE).glob('*.py'))}
    rc, doc, tail = fresh(f"import runpy; runpy.run_path({str(root / C.MAP_SERVER_TRAMPOLINE)!r}, run_name='import_check'); "
                          "print('{\"imported\": true}')", root)
    checks['map_trampoline_imports_in_fresh_process'] = rc == 0 and bool(doc and doc.get('imported'))
    rc, doc, tail = fresh(MAP_CHAIN, root)
    info['map_chain'] = doc if doc else tail
    checks['real_map_install_chain_taps_after_success_only'] = bool(
        rc == 0 and doc and doc['rows'] == 1 and doc['failed_raised'] and doc['record_equal']
        and doc['types'] == ['Vehicle', 'Pedestrian'] and doc['legacy_yaw'] == [90.0, 90.0]
        and doc['producer_yaw'] == [[0.6, 0.8], [0.6, 0.8]] and doc['chain'][0] == 'LiveViewMapIngestService'
        and doc['chain'][1] == 'ClassCountingService' and doc['status']['written'] == 1 and doc['first_error'] is None
        and 'install_timestamp' in doc['result_keys'])
    rc, doc, tail = fresh(POSE_CHAIN, root)
    info['pose_chain'] = doc if doc else tail
    checks['pose_tap_composes_with_c5_capacity_and_restores'] = bool(
        rc == 0 and doc and doc['events'] == 2400 and doc['restored'] and doc['errors'] is None
        and doc['rows'] + doc['dropped'] == 2400 and doc['rows'] == doc['written'])
    rc, _, tail = fresh("import unittest,sys; r=unittest.main(module='rl_agent.splitfusion_live_view_tap_v1.test_tap',"
                        "exit=False,argv=['t']).result; print('{\"ok\": %s}' % str(r.wasSuccessful()).lower())", root)
    checks['tap_unit_tests'] = rc == 0 and '"ok": true' in tail
    info['tap_unit_tests_tail'] = tail[-600:]
    def imports(path):
        tree = ast.parse(path.read_text())
        return {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {
            n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    runtime = [p for package in (C.PACKAGE, C.LIVE_VIEW_PACKAGE) for p in (root / package).glob('*.py')]
    checks['runtime_never_imports_the_viewer'] = not any(m.startswith('spatial_map_coop') for p in runtime
                                                         for m in imports(p))
    result = dict(schema=H.REGRESSION_SCHEMA, status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                  info=info, sources=C.exact_sources(root), launcher_binding=C.launcher_binding(),
                  live_services_started=False, carla_started=False, oai_started=False, live_inference_started=False)
    save(out / 'RESULT.json', result)
    save(out / 'OUTPUT_SHA256.json', {str(p.relative_to(out)): C.sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(dict(status=result['status'], checks=checks), sort_keys=True))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    raise SystemExit(run(p.parse_args().name))
