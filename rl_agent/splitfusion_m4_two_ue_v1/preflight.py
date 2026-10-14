"""Production binding validation and read-only live-host prerequisites."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from . import configuration as C
from .identity import require


def validate_offline(path):
    value=C.validate(path)
    require(Path(value['config']['repository'])==C.ROOT,
            'host preparation belongs to another worktree; prepare and verify in the intended worktree')
    from .edge_entry import import_check
    from .frontend import runtime_class,collector_class
    from .radio_runtime import argv, validate_ue_pair
    from rl_agent.splitfusion_run5_full_live_v1.actor import load_registered_actor
    from rl_agent.splitfusion_numeric_gt_producer_v1.originals import bound
    actor=load_registered_actor()
    from rl_agent.ue_mcs_backlog_near_capacity_v1 import radio_binding as RB
    RB.verify('M4_OFFLINE_PREPARATION',C.ROOT)
    from rl_agent import ue_target_snr_cell_runtime_v1 as target
    campaign=json.loads((C.ROOT/'rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json').read_bytes())
    replay=target.load_replay_module()
    mapping=target.load_mapping(C.ROOT/campaign['network']['mapping_csv'],replay)
    require(replay.inverse_interpolate(23.5,mapping,.25)==-12.,'GOOD_ALL calibration interpolation drift')
    import torch
    require(not torch.cuda.is_initialized(),'offline validation touched CUDA')
    base=json.loads((Path(path).parent/'BASE_REGISTRATION.json').read_bytes())
    require(bound().binding_sha256==base['numeric_producer_binding'],'numeric producer binding changed')
    import_check()
    commands=argv(Path(path).parent)
    validate_ue_pair((Path(path).parent/'ue.conf').read_text())
    require(commands['ue'][commands['ue'].index('--num-ues')+1]=='2' and
            commands['ue'][commands['ue'].index('-r')+1]=='273','two-UE/273-PRB startup drift')
    # Imports resolve every entry dependency without constructing a live object.
    for name in ('host','bootstrap','map_service','telemetry','analysis'):
        importlib.import_module('rl_agent.splitfusion_m4_two_ue_v1.'+name)
    from rl_agent.splitfusion_live_parity_repair_v1.gt_bridge import bind_legacy_namespace
    bind_legacy_namespace()  # the unchanged qualified collector import prerequisite
    for name in ('carla','data_collection.run_route_b_density_loop','agents.navigation.basic_agent',
                 'carla_collect_parked_ego_fusion_training_data'):
        importlib.import_module(name)  # no Client/world or process is constructed
    return value


def host_requirements(registration, *, require_disk):
    """No startup, routing changes, process signals or GPU allocations."""
    failures=[]; checks={}
    free=shutil.disk_usage(C.ROOT).free
    checks['disk']=dict(free_bytes=free,minimum_bytes=C.DISK_GATE,pass_=free>=C.DISK_GATE)
    if require_disk and free<C.DISK_GATE: failures.append('disk below unchanged 16-GiB live gate')
    checks['cpus']=sorted(os.sched_getaffinity(0))
    if not set(range(24))<=set(checks['cpus']): failures.append('required 24 logical CPUs unavailable')
    from . import host_baseline
    baseline=host_baseline.inspect()
    checks['approved_host_baseline']=baseline
    live=[]
    for path in Path('/proc').glob('[0-9]*/comm'):
        try:
            comm=path.read_text().strip()
            if comm.startswith(('CarlaUnreal','nr-softmodem','nr-uesoftmodem','nvidia-cuda-mps')):
                live.append(dict(pid=int(path.parent.name),comm=comm))
        except OSError: pass
    checks['live_processes']=live
    if any(row['pid'] not in baseline['allowed_process_pids'] for row in live):
        failures.append('host has an unapproved existing CARLA/OAI/MPS owner')
    reserved={2000,2001,2002,8010,9090,2021,2023,2123,2031,52100,52101,52110,52010,52011}
    busy=[]
    for name in ('tcp','tcp6','udp','udp6'):
        for row in Path('/proc/net/'+name).read_text().splitlines()[1:]:
            fields=row.split(); port=int(fields[1].split(':')[1],16)
            if port in reserved and (name.startswith('udp') or fields[3]=='0A'):
                busy.append(dict(table=name,port=port))
    checks['reserved_ports_in_use']=busy
    if busy: failures.append('a registered host endpoint is already in use')
    def read(name,args):
        result=subprocess.run(args,capture_output=True,text=True,timeout=20)
        checks[name]=dict(returncode=result.returncode,stdout=result.stdout[:12000],stderr=result.stderr[:1000])
        require(result.returncode==0,'host prerequisite failed: '+name)
        return result.stdout
    gpu=read('gpu_clients',['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'])
    host_baseline.validate_gpu_rows(gpu,set(baseline['allowed_gpu_pids']))
    image=read('edge_image',['sudo','-n','docker','image','inspect',C.EDGE_IMAGE,'--format','{{.Id}}']).strip()
    if image!=C.EDGE_IMAGE: failures.append('edge image differs')
    network=json.loads(read('network',['sudo','-n','docker','network','inspect','oai-cn5g-public-net']))[0]
    if any(v.get('IPv4Address','').split('/')[0]==registration['config']['edge_address']
           for v in network.get('Containers',{}).values()): failures.append('edge address already owned')
    core=json.loads(read('core',['sudo','-n','docker','inspect','oai-amf','oai-smf','oai-upf','mysql']))
    if not all(v['State']['Running'] and v['State'].get('Health',{}).get('Status','healthy')=='healthy' for v in core):
        failures.append('existing CN5G must be healthy; M4 does not recreate it')
    # Registration checks actual subscriber configuration; actual UICC/lease/RNTI
    # and both per-transmission PUSCH streams are separate post-attach gates.
    checks['unverified_live']=['actual attach and subscribers','per-UE achieved PUSCH','two-source shared GPU load',
                                '15-m lane/spawn','full route/density/collision','end-to-end source identity']
    require(not failures,'; '.join(failures))
    return checks


def report(path, *, host=False):
    value=validate_offline(path)
    result=dict(status='OFFLINE_BINDING_PASS',preparation_sha256=value['sha256'],cuda_initialized=False,
        entry='rl_agent.splitfusion_m4_two_ue_v1.cli',contract=C.__name__,
        frozen_actor_weights=value['config']['actor_weights_sha256'],source_count=len(value['sources']),
        live_qualification=False,external_services_started=False,
        free_bytes=shutil.disk_usage(C.ROOT).free,live_disk_minimum_bytes=C.DISK_GATE)
    if host: result['host']=host_requirements(value,require_disk=True)
    return result
