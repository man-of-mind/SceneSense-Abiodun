"""M4 live composition. No function is executed merely by importing it."""
import contextlib
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from . import configuration as C
from .identity import require


def wait_for_carla_rpc(client_factory, host, port, output, *, timeout_s=120.,
                       clock=time.monotonic, sleep=time.sleep):
    """Use a fresh client per probe, as the retained supervisor does.

    CARLA/rpclib clients constructed before the listener exists can remain in
    connection-refused state. Retrying a method on that instance is not a
    reconnection. This changes startup only, never an experiment deadline.
    """
    started=clock(); deadline=started+timeout_s; attempt=0; last_error=None
    with Path(output).open('x') as evidence:
        while clock()<deadline:
            attempt+=1
            row=dict(attempt=attempt,elapsed_s=clock()-started,fresh_client=True)
            try:
                client=client_factory(host,port)
                client.set_timeout(max(.001,min(2.,deadline-clock())))
                version=client.get_server_version()
                row.update(status='READY',server_version=version,completed_elapsed_s=clock()-started)
                evidence.write(json.dumps(row,sort_keys=True)+'\n'); evidence.flush()
                return row
            except RuntimeError as exc:
                last_error=repr(exc)
                row.update(status='RETRY_NEW_CLIENT',error=last_error,completed_elapsed_s=clock()-started)
                evidence.write(json.dumps(row,sort_keys=True)+'\n'); evidence.flush()
            except BaseException as exc:
                row.update(status='FATAL_PROBE_ERROR',error=repr(exc))
                evidence.write(json.dumps(row,sort_keys=True)+'\n'); evidence.flush()
                raise
            sleep(max(0.,min(.2,deadline-clock())))
    raise ValueError('CARLA RPC startup timeout; last error: '+str(last_error))


def campaign_for(source, config, target_start):
    from rl_agent import ue_route_b_split_cell_adapter_v1 as P
    campaign=json.loads((C.ROOT/'rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json').read_bytes())
    campaign['campaign_id']=source.run_id
    campaign['cell']['service_deadline_ms']=170
    campaign['cell']['ack_timeout_ms']=500
    campaign['_target_start_file']=str(target_start)
    campaign['runtime'].update(ue_bind_host=source.ip,edge_remote_host=config['edge_address'],
        edge_receive_port=config['sources'][source.ue_id]['ingress_port'],camera_result_port=source.feedback_port,
        ue_control_port=source.feedback_port,map_ingest_port=config['map_ports'][source.ue_id])
    row=P.action_row(campaign,71)
    cell=dict(row,cell_id=source.cell_id,action_id=71,network_profile_id='FAVORABLE_STABLE',
              trace_id='GM_V2_FAVORABLE_STABLE_SEED_2026082101',seed=2026082101)
    return campaign,cell,row


def run(preparation, output, *, smoke=False):
    """Exactly one future invocation; create-only output, no retry/deployment."""
    from .preflight import validate_offline, host_requirements
    registration=validate_offline(preparation)
    host_requirements(registration,require_disk=True)
    config=registration['config']; output=Path(output).absolute()
    require(not output.exists(),'capture output already exists')
    output.mkdir(parents=True)
    C.write_new(output/'PREPARATION_REFERENCE.json',dict(path=str(Path(preparation).absolute()),sha256=registration['sha256']))
    from .resources import OwnedProcesses,process_birth
    from .radio_runtime import AttachedRadio, GoodAll
    from .telemetry import NativeTelemetry
    from .composition import independent_policy_contexts
    from .gt import SourceHandoff
    from .frontend import runtime_class,collector_class,shared_model_view
    from .bootstrap import run_world
    from .map_service import HeadlessMaps,ExactRecordAPI
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U, continuous_execution_v2 as X
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_ue_ledger_v2 as UL, phase6_prewarm_v2 as PW
    from rl_agent.splitfusion_run5_live_avo_v1.snr_live import LiveSnrProvider
    from rl_agent.splitfusion_m3_route_b_v3.entries import install_ue_capacity
    from rl_agent.splitfusion_m3_route_b_v3.retention import AttemptRetentionObserver,finalize_after_child
    from rl_agent.splitfusion_paired_retention_v2.plan import RetentionConfigV2
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
    from rl_agent.splitfusion_ue_input_prep_v2.candidate import prepare_live_input_fused
    from rl_agent.splitfusion_direct_edge_map_v1.endpoint import resolve_direct_map_endpoint
    from .binding import bind
    import torch
    owner=OwnedProcesses(output)
    C.write_new(output/'HOST_OWNER.json',dict(pid=os.getpid(),birth=process_birth(os.getpid())))
    radio=AttachedRadio(preparation=Path(preparation).parent,output=output,owner=owner,config=config)
    providers={}; handoffs={}; archives={}; observers={}; collectors={}; ledgers={}
    telemetry=maps=api=edge=None
    cleanup={}; error=None; rc=1; started=time.monotonic()
    original_signals={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    def interrupted(sig,*_): raise RuntimeError('M4 interrupted by signal '+str(sig))
    for sig in original_signals: signal.signal(sig,interrupted)
    # The lock is held by this process until all owned cleanup completes.
    import fcntl
    lock=open('/tmp/splitfusion-live-host.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with contextlib.ExitStack() as scopes:
        try:
            previous_affinity=os.sched_getaffinity(0)
            os.sched_setaffinity(0,set(C.CPU['host_world']))
            scopes.callback(os.sched_setaffinity,0,previous_affinity)
            sources=radio.start()
            owner.adopt('pusch_relay',radio.telemetry.relay)
            for stream in radio.telemetry.streams: owner.adopt('radio_'+stream.name,stream.process)
            contexts=independent_policy_contexts(sources,registration['sha256'])
            telemetry=NativeTelemetry(contexts,output)
            for reader in telemetry.readers: owner.adopt('native_'+reader.event,reader.process)
            # Short unique sockets; only providers remove their own inodes.
            socket_root=Path(tempfile.mkdtemp(prefix='m4-snr-'))
            C.write_new(output/'IPC_LOCATION.json',dict(directory=str(socket_root)))
            for source in sources.sources:
                providers[source.stream_id]=LiveSnrProvider(socket_root/(str(source.ue_id)+'.sock'),
                    str(uuid.uuid4()),contexts[source.stream_id]['telemetry'].ue_label)
            radio.snr=GoodAll(sources,providers,output)
            scopes.enter_context(install_ue_capacity(output))
            from rl_agent.splitfusion_m3_route_b_v4.abort_evidence import install_abort_evidence
            scopes.enter_context(install_abort_evidence())
            endpoint=resolve_direct_map_endpoint(port=config['map_ports'][0])
            pose_index={}; pose_lock=threading.Lock()
            def pose_lookup(source,frame):
                with pose_lock:
                    require((source.stream_id,frame) in pose_index,'map has no frame-bound UE wire pose')
                    return copy.deepcopy(pose_index[source.stream_id,frame])
            map_output=output/'map'; map_output.mkdir()
            maps=HeadlessMaps(sources=sources,contract=contexts[sources.sources[0].stream_id]['engine']._contract,
                output=map_output,bind_host=endpoint.host,
                ports={s.stream_id:config['map_ports'][s.ue_id] for s in sources.sources},
                pose_lookup=pose_lookup,history_size=4096,cpus=','.join(map(str,C.CPU['map'])))
            maps.start(); api=ExactRecordAPI(maps.store,config['map_api_port']); api.start()
            require(torch.cuda.is_available(),'host front GPU unavailable')
            torch.hub.set_dir(str(C.model_cache()))
            shared=B._preload_ue(torch.device('cuda:0'))
            contract=contexts[sources.sources[0].stream_id]['engine']._contract
            continuous=X.ContinuousUERuntimeV2(contract,front=U._SqueezedFront(shared[0]._front,torch.device('cuda:0')),
                ranker=shared[0]._ranker,ae_encoders=shared[0]._ae_encoders,codec=shared[0]._codec)
            inputs=bind(PW.synthetic_sensor_inputs,RADAR_SHAPE=(4,432,768))
            warm=bind(PW.warm_ue,synthetic_sensor_inputs=inputs)(continuous,contract,
                prepare_input=lambda f,r:prepare_live_input_fused(f,r,torch.device('cuda:0')))
            C.write_new(output/'PREWARM_FRONT.json',warm)
            require(warm['completed'] and warm['modes_warmed']==list(range(12)),'front prewarm incomplete')
            base=json.loads((Path(preparation).parent/'BASE_REGISTRATION.json').read_bytes())
            for source in sources.sources:
                directory=output/source.stream_id; directory.mkdir()
                evidence=directory/'edge_evidence'; evidence.mkdir()
                placement=C.CPU['gt'][source.ue_id]
                reg=dict(base,exact_gt_sources=registration['sources'],numeric_run_binding=registration['sha256'],
                    exact_gt_limits=dict(base['exact_gt_limits'],prep_cpu=placement['prep'],score_cpu=placement['score'],
                                         service_cpu=placement['coordinator']))
                handoffs[source.stream_id]=SourceHandoff(source=source,attempt_dir=directory,registration=reg,service_config={})
                archive=PO.PoseArchive(directory/'owned_pose',side='ue',max_frames=3900)
                archives[source.stream_id]=archive
                def observed_add(kind,_source=source,_add=archive.add,**values):
                    if kind=='UE_WIRE':
                        gt=values['gt_identity']
                        with pose_lock:
                            pose_index[_source.stream_id,gt['frame_id']]=dict(source=_source.tag,frame_id=gt['frame_id'],
                                capture_timestamp_ns=gt['capture_timestamp_ns'],provenance='OWNED_SOURCE -> UE_WIRE',
                                context=values['context'],source_sha256=values['source_sha256'])
                    return _add(kind,**values)
                archive.add=observed_add
                observer=AttemptRetentionObserver(RetentionConfigV2.from_plan_file(
                    Path(preparation).parent/'RETENTION_PLAN.json',registration['retention_plan_sha256']))
                observer.start(); observers[source.stream_id]=observer
                contexts[source.stream_id]['prewarm_report']=warm
                ledger=UL.Run4CompatLedgerV2(UL.Run4TerminalLedgerV2(output_csv=directory/'map_feedback.csv',
                    experiment_id=source.run_id,cell_id=source.cell_id)); ledgers[source.stream_id]=ledger
            C.write_new(output/'TARGET_STARTED.json',dict(commanded_snr_db=23.5))
            tracer=C.ROOT/'OAI/openairinterface5g/common/utils/T/tracer'
            def make_collector(source,world,ego):
                directory=output/source.stream_id; context=contexts[source.stream_id]
                bindings=U.Run4LiveBindingsV2(registration['sha256'],tracer,tracer.parent/'T_messages.txt',2123,
                                             directory/'run4_phase6')
                cls=runtime_class(bindings,source=source,context=context,snr_provider=providers[source.stream_id],
                    shared_models=shared_model_view(shared,torch.device('cuda:0')),
                    handoff=handoffs[source.stream_id],retention=observers[source.stream_id])
                def factory(**kw):
                    kw.pop('map_host'); kw.pop('map_port')
                    return cls(**kw,ledger=ledgers[source.stream_id],ue_control_port=source.feedback_port)
                collect=collector_class(source=source,runtime_factory=factory,ledger=ledgers[source.stream_id],
                    handoff=handoffs[source.stream_id],pose_archive=archives[source.stream_id])
                campaign,cell,row=campaign_for(source,config,output/'TARGET_STARTED.json')
                collector=collect(world=world,ego=ego,cell=cell,campaign=campaign,row=row,binding={},
                    attempt_dir=directory,map_api_port=config['map_api_port'],feedback_port=source.feedback_port,
                    edge_evidence_dir=directory/'edge_evidence')
                collectors[source.stream_id]=collector
                owner.adopt('gt_'+source.stream_id,handoffs[source.stream_id].process)
                return collector
            runtime=dict(preparation_sha256=registration['sha256'],sources=[s.tag for s in sources.sources],
                output=str(output),map_host=endpoint.host,chunk_bytes=12500,socket_buffer_request_bytes=8388608,
                endpoints={s.stream_id:dict(ingress_port=config['sources'][s.ue_id]['ingress_port'],
                    map_port=config['map_ports'][s.ue_id],edge_evidence=str(output/s.stream_id/'edge_evidence'),
                    gt_socket=str(output/s.stream_id/'edge_evidence'/base['exact_gt_socket_name']),
                    gt_capability=str(output/s.stream_id/'edge_evidence'/base['exact_gt_capability_name'])) for s in sources.sources})
            runtime['sha256']=C.canonical(runtime); C.write_new(output/'RUNTIME.json',runtime)
            def start_edge():
                nonlocal edge
                name='m4-edge-'+uuid.uuid4().hex[:12]
                C.write_new(output/'EDGE_CONTAINER.json',dict(name=name,image=C.EDGE_IMAGE))
                args=['sudo','-n','docker','run','--rm','--name',name,'--label','splitfusion.m4.owner='+runtime['sha256'],
                    '--gpus','all','--network',config['edge_network'],'--ip',config['edge_address'],'--cap-add','NET_ADMIN',
                    '--cpuset-cpus',','.join(map(str,C.CPU['shared_inference'])),'-e','PYTHONDONTWRITEBYTECODE=1',
                    '-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-v',str(C.ROOT)+':'+str(C.ROOT)+':ro',
                    '-v',str(output)+':'+str(output)+':rw',*C.edge_checkpoint_mount(),
                    '-w',str(C.ROOT),'--entrypoint','sh',C.EDGE_IMAGE,'-ec',
                    'ip route add 10.0.0.0/24 via 192.168.70.134; exec env -u PYTHONPATH python3 -B -m rl_agent.splitfusion_m4_two_ue_v1.edge_entry --preparation "$1" --runtime "$2"',
                    'm4-edge',str(Path(preparation).absolute()),str(output/'RUNTIME.json')]
                edge=owner.start('edge',args,cwd=C.ROOT)
                deadline=time.monotonic()+180
                while not (output/'edge/READY.json').is_file():
                    require(edge.poll() is None and time.monotonic()<deadline,'shared edge failed before READY')
                    time.sleep(.1)
                from .radio import check_routes
                C.write_new(output/'ROUTES.json',check_routes(sources,edge_container=name,edge_ip=config['edge_address']))
            def stop_edge():
                if edge is not None and edge.poll() is None:
                    if not (output/'STOP_EDGE').exists(): C.write_new(output/'STOP_EDGE',dict(reason='both-source drain or original failure'))
                    edge.wait(timeout=180)
                if edge is not None:
                    require(edge.returncode==0,'shared edge failed at drain: '+str(edge.returncode))
                    closed=json.loads((output/'edge/CLOSED.json').read_bytes())
                    require(not closed['errors'] and len(closed['evaluators'])==2 and
                        all(not (v.get('error') or v.get('failure') or v.get('worker_alive'))
                            for v in closed['evaluators'].values()),'shared edge/GT drain incomplete')
            # The supervised helper owns only its own CARLA process group and
            # accounts for a wrapper that exits before the actual server.
            from data_collection import run_route_b_collection_supervised_v1 as supervisor
            carla_process,carla_pgid=supervisor.start_carla(config['carla_port'],output/'carla.log')
            C.write_new(output/'CARLA_OWNER.json',dict(pid=carla_process.pid,pgid=carla_pgid))
            cleanup['carla_owner']=dict(process=carla_process,pgid=carla_pgid)
            import carla
            rpc_startup=wait_for_carla_rpc(carla.Client,config['carla_host'],config['carla_port'],
                                         output/'CARLA_RPC_STARTUP.jsonl')
            C.write_new(output/'CARLA_STARTED.json',dict(pgid=carla_pgid,rpc_port=config['carla_port'],
                rpc_startup=rpc_startup,
                members=[dict(row,birth=process_birth(row['pid'])) for row in supervisor.process_group_members(carla_pgid)]))
            world_config=dict(config,smoke=smoke)
            def health():
                require(radio.snr.error is None,'channel controller failed: '+str(radio.snr.error))
                require(edge is not None and edge.poll() is None,'shared edge exited during capture')
                require(all(r.process.poll() is None for r in telemetry.readers),'native telemetry reader exited')
                for handoff in handoffs.values(): handoff._check()
            rc=run_world(config=world_config,sources=sources,output=output,
                catalogues={k:h.catalogue for k,h in handoffs.items()},make_collector=make_collector,
                start_edge=start_edge,stop_edge=stop_edge,health=health)
            if smoke:
                from .analysis import smoke_assessment
                smoke_result=smoke_assessment(output)
                C.write_new(output/'SMOKE.json',smoke_result)
                require(smoke_result['pass_'],'bounded smoke lacks both-source clean data path')
                rc=0  # A passed 20-s smoke is not a completed-route verdict.
        except BaseException as exc:
            error=repr(exc)
            C.write_new(output/'ORIGINAL_FAILURE.json',dict(error=error))
        finally:
            # Preserve the original error; each cleanup failure is additional evidence.
            def clean(name,fn):
                try: cleanup[name]=fn()
                except BaseException as exc: cleanup[name]=dict(error=repr(exc))
            for stream,col in collectors.items():
                if col.worker.is_alive(): clean('collector_'+stream,col.finish)
            if edge is not None and edge.poll() is None:
                clean('edge_drain',stop_edge)
            for stream,h in handoffs.items():
                if h.process is not None: clean('gt_'+stream,h.close)
            if api is not None: clean('map_api',api.close)
            if maps is not None: clean('maps',maps.close)
            for stream,a in archives.items(): clean('pose_'+stream,a.flush)
            for stream,o in observers.items():
                clean('retention_'+stream,lambda s=stream,v=o:finalize_after_child(v,output/s,child_returncode=rc,child_error=error))
            if telemetry is not None: clean('telemetry',telemetry.close)
            clean('radio_restore',radio.stop)
            for stream,provider in providers.items(): clean('snr_'+stream,provider.close)
            if 'carla_owner' in cleanup:
                owned=cleanup.pop('carla_owner')
                clean('carla',lambda:supervisor.stop_carla(owned['process'],owned['pgid'],config['carla_port']))
            clean('processes',owner.close)
            for sig,handler in original_signals.items(): signal.signal(sig,handler)
            fcntl.flock(lock,fcntl.LOCK_UN); lock.close()
            C.write_new(output/'CLEANUP.json',cleanup)
            C.write_new(output/'RUN_RESULT.json',dict(preparation_sha256=registration['sha256'],
                status='ABORTED' if error else ('SMOKE_ONLY' if smoke else 'CAPTURE_ENDED'),error=error,
                returncode=rc,elapsed_s=time.monotonic()-started))
    return 2 if error else int(rc or 0)
