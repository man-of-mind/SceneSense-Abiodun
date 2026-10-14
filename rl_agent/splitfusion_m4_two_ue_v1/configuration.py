"""Compact prospective M4 preparation and strict offline source validation."""
import hashlib
import ast
import json
import os
from pathlib import Path

from .identity import require
from .radio import EXPECTED

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = 'rl_agent/splitfusion_m4_two_ue_v1'
EVIDENCE = 'experiments/splitfusion_m4_two_ue_v1'
BASE = ROOT.parent / 'abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261012_m3_route_b_v20/good_all_live_view_01/registration.json'
BASE_SHA = 'd28f6c5e269fceb8f5554da0caad1bc4e2c3838160cdaa8c8d19267af365c76d'
PROOFS = {
    'offline_checks_04/RESULT.json': '358eb75ccee53ec27eca86e18b0c6c5e5b596b102a321eccc55a00b04c9a73fe',
    'correctness_04/SUMMARY.json': 'c1796eaf5e0a9bd99f1b67525456c45e05daabd5a604845520dc5501f279dadd',
    'capacity_01/SUMMARY.json': '4ab60715b80f5de6c22999db05411e6786f3c80045fd7c3edf1db4f64734b9e2'}
SCHEMA = 'splitfusion.m4.production_preparation.v1'
DISK_GATE = 16 * 1024**3
EDGE_IMAGE = 'sha256:2be62d533b8077ceecab5455d5377f2f952b6a50d43ff8c04dc89ce18027d6ba'
CPU = dict(radio=[0,1,2,3,8,9,10,11,12,13,14,15,16,17,18,19],
           edge_pool=[4,5,6,7,20,21,22,23], host_world=[4,5,6,7,20,21,22,23], gt=[dict(prep=4, score=5, coordinator=20),
           dict(prep=6, score=7, coordinator=21)], front=[22,23], shared_inference=[20,21], map=[22,23])


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'missing/nonregular binding: ' + str(path))
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def source_set(root=ROOT, base_path=BASE):
    require(sha(base_path) == BASE_SHA, 'retained V20 registration drift')
    base = json.loads(Path(base_path).read_bytes())
    result = dict(base['exact_gt_sources'])
    # Read the retained literal model manifest without importing replay/GPU
    # execution into the slim verifier. Bind the actual assets, not only code.
    model_manifest=root/'pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_ae_v1/ae_phase11b_gpu_qualification.py'
    manifests=[n.value for n in ast.parse(model_manifest.read_text()).body if isinstance(n,ast.AnnAssign)
               and isinstance(n.target,ast.Name) and n.target.id=='FROZEN_INPUTS']
    require(len(manifests)==1,'retained model manifest shape drift')
    for record in ast.literal_eval(manifests[0]).values():
        result[record['path']]=record['sha256']
        if 'selection_path' in record: result[record['selection_path']]=record['selection_sha256']
    from rl_agent.ue_mcs_backlog_near_capacity_v1 import radio_binding as RB
    for key,pin in RB.PINS.items():
        digest=sha(root/pin['path']) if key in RB.RESOLVED_AT_VERIFY else pin['sha256']
        require(pin['path'] not in result or result[pin['path']]==digest,'retained radio inventory conflict: '+key)
        result[pin['path']] = digest
    result.update({str(p.relative_to(root)): sha(p) for p in sorted((root / PACKAGE).glob('*.py'))})
    # Run4MapJournal imports this package, whose __init__ imports its public
    # input-adapter modules. Bind and deploy those existing pure dependencies.
    result.update({str(p.relative_to(root)): sha(p) for p in
                   sorted((root / 'spatial_map_coop/coop_input_v1').glob('*.py'))})
    for name in ('data_collection/run_route_b_density_loop.py',
                 'data_collection/run_route_b_perception_collection_v2.py',
                 'rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json',
                 'data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.json',
                 'data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.progress.csv',
                 'rl_agent/multiue_oai/runner.py',
                 'rl_agent/splitfusion_fixed_action_screen_analysis_v2/analyze.py',
                 'rl_agent/splitfusion_m3_route_b_v3_analysis/route_geometry.py',
                 'OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.multi2.conf',
                 'rl_agent/configs/ue_n2_oai_ul_calibration_smoke_v1.json'):
        result.setdefault(name, sha(root / name))
    campaign=json.loads((root/'rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json').read_bytes())
    weights=campaign['deployment']['fcos_constructor_weights']
    require(sha(root/weights['path'])==weights['sha256'],'constructor cache drift')
    result[weights['path']]=weights['sha256']
    priors=campaign['deployment']['perception_train_only_priors']
    result[priors['path']]=priors['sha256']
    result.setdefault(campaign['network']['mapping_csv'],sha(root/campaign['network']['mapping_csv']))
    return dict(sorted(result.items()))


def model_cache():
    campaign=json.loads((ROOT/'rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json').read_bytes())
    item=campaign['deployment']['fcos_constructor_weights']
    require(sha(ROOT/item['path'])==item['sha256'],'frozen constructor cache drift')
    return (ROOT/item['path']).parent.parent


def edge_checkpoint_mount():
    """Expose only the registered FCOS file at the frozen model's cache alias.

    torch.hub uses model_cache(); the historical model independently verifies
    this absolute alias. Do not mount the user's home/cache or download data.
    """
    campaign=json.loads((ROOT/'rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json').read_bytes())
    item=campaign['deployment']['fcos_constructor_weights']
    relative=Path(item['path'])
    require(not relative.is_absolute() and '..' not in relative.parts,'unsafe constructor checkpoint path')
    source=ROOT/relative
    require(sha(source)==item['sha256'],'frozen constructor cache drift')
    target=Path('/home/shr_aisvcs/.cache/torch/hub/checkpoints')/source.name
    require(',' not in str(source),'unsupported comma in bind source')
    return ['--mount',f'type=bind,src={source},dst={target},readonly']


def verify_sources(sources, root=ROOT):
    require(bool(sources), 'empty source inventory')
    for name, digest in sources.items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'unsafe source path')
        require(sha(root / path) == digest, 'source drift: ' + name)


def reused_proofs(root=ROOT):
    result = {}
    for name, digest in PROOFS.items():
        path = root / EVIDENCE / name
        require(sha(path) == digest, 'retained proof drift: ' + name)
        value = json.loads(path.read_bytes())
        # Added composition modules were not covered by these tests. Every
        # previously tested source must remain byte-identical for proof reuse.
        verify_sources(value['sources'], root)
        result[name] = dict(sha256=digest, scope='UNCHANGED_COMPONENTS_ONLY')
    return result


def verify_edge_bodies(root=ROOT):
    names = {'FeedbackRouter', 'SourceEvaluator', 'SharedEdgeProcessor'}
    def selected(path):
        tree = ast.parse(path.read_text())
        return {n.name:ast.dump(n, include_attributes=False) for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name in names}
    original = selected(root / PACKAGE / 'gt.py')
    edge = selected(root / PACKAGE / 'edge_components.py')
    require(set(original) == names and original == edge, 'edge-only GT class bodies differ from qualified components')
    return {name:hashlib.sha256(body.encode()).hexdigest() for name,body in original.items()}


def specification(repository=ROOT):
    return dict(schema=SCHEMA, repository=str(repository), run_id='M4_TWO_UE_GOOD_ALL', cell_id='town10',
        sources=list(EXPECTED), route_json='data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.json',
        carla_host='127.0.0.1', carla_port=2000, cpu=CPU, live_disk_minimum_bytes=DISK_GATE,
        edge_image=EDGE_IMAGE, edge_address='192.168.70.140', edge_network='oai-cn5g-public-net',
        map_ports=[52100,52101], map_api_port=52110,
        radio=dict(band=78, bandwidth_mhz=100, prb=273, numerology=1, dl_slots=4, ul_slots=5,
                   dl_symbols=6, ul_symbols=4, commanded_snr_db=23.5, rfsim_noise_db=-12.,
                   ue_count=2, measured_two_ue_calibration=False),
        actor_weights_sha256='bb9b0d256de79b4498cd622e1daa59edcf43ff3072dc9f675b265aac579ed3ce',
        state_features=22, modes=list(range(12)), frame_ceiling_per_source=3900,
        deadline_ms=170, processing_horizon_ms=500, history_semantics='UNCHANGED',
        traffic=dict(vehicles=50, pedestrians=50, scenario_seed=31, traffic_manager_seed=31),
        no_retry=True, headless=True, fusion=False, deployment_performed=False,
        ownership=dict(world='one density.run / TickOwner', front='one collector/controller per UE, shared read-only weights',
            edge='one detached GPU inference owner, latest pending slot per stream',
            gt='two isolated host services, each prep/score/coordinator/archive',
            radio='one gNB + two logical UEs in one softmodem, one UE tracer relay',
            map='source-bound ingest services, one observation journal; no fusion'),
        cpu_layout_scope='PROSPECTIVE, NOT LIVE QUALIFIED; host/world inherit edge pool; inference shares coordinators, fronts share map',
        offline_space_budget_bytes=100_000_000, projected_capture_bytes=22_950_000_000,
        live_authorization=False)


def prepare(output):
    output = Path(output)
    require(not output.exists(), 'preparation output already exists')
    sources = source_set()
    verify_sources(sources)
    proofs = reused_proofs()
    config = specification()
    value = dict(config=config, sources=sources, reused_proofs=proofs, base_registration_sha256=BASE_SHA)
    value['edge_component_bodies'] = verify_edge_bodies()
    output.mkdir(parents=True)
    from .radio_runtime import prepare_radio
    from rl_agent.splitfusion_m3_route_b_v3.contract import retention_plan
    from rl_agent.splitfusion_paired_retention_v2.plan import write_plan
    value['radio_artifacts'] = prepare_radio(output)
    value['retention_plan_sha256'] = write_plan(output/'RETENTION_PLAN.json',retention_plan(3900))
    value['sha256'] = canonical(value)
    with (output / 'BASE_REGISTRATION.json').open('xb') as stream:
        stream.write(BASE.read_bytes())
    write_new(output / 'PREPARATION.json', value)
    return value


def validate(path, *, root=ROOT):
    value = json.loads(Path(path).read_bytes())
    require(value.get('sha256') == canonical({k:v for k,v in value.items() if k != 'sha256'}), 'preparation digest drift')
    repository = Path(value['config']['repository'])
    require(repository.is_absolute() and '..' not in repository.parts, 'invalid host repository binding')
    require(value['config'] == specification(repository), 'M4 configuration drift')
    expected = source_set(root, Path(path).parent / 'BASE_REGISTRATION.json')
    require(set(value['sources']) == set(expected), 'source inventory incomplete/foreign')
    verify_sources(value['sources'], root)
    require(value['reused_proofs'] == reused_proofs(root), 'proof binding drift')
    require(value['base_registration_sha256'] == BASE_SHA, 'base registration drift')
    require(value['edge_component_bodies'] == verify_edge_bodies(root), 'edge component equivalence binding differs')
    for name in ('gnb.conf','ue.conf'):
        require(sha(Path(path).parent/name)==value['radio_artifacts'][name], 'prepared radio drift: '+name)
    from rl_agent.splitfusion_paired_retention_v2.plan import load_plan, canonical as plan_bytes
    from rl_agent.splitfusion_m3_route_b_v3.contract import retention_plan
    plan = load_plan(Path(path).parent/'RETENTION_PLAN.json',value['retention_plan_sha256'])
    require(plan_bytes(plan)==plan_bytes(retention_plan(3900)), 'retention semantics drift')
    return value
