"""One shared detached edge stack; two source-bound ingress/GT/map endpoints."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import socket
import threading
import time

from . import configuration as C
from .identity import Source, Sources, require


def import_check():
    from .receiver_timing import Receiver, ObservedIngress
    from .edge_components import SourceEvaluator, SharedEdgeProcessor
    from .datapath import DataPath
    from rl_agent.splitfusion_edge_optimization_v1.detached_edge_preload_v3 import preload_detached_optimized_edge_v3
    from rl_agent.splitfusion_m3_route_b_v3.edge_guard import build_guarded_evaluator
    from rl_agent.splitfusion_direct_edge_map_v1.direct_v3_edge import _DeadlineGuardedComputeTail
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose
    import torch
    require(not torch.cuda.is_initialized(), 'offline edge check initialized CUDA')
    return dict(verifier=C.__name__, dispatcher=DataPath.__module__, evaluator=SourceEvaluator.__module__,
                receiver=Receiver.__module__, ingress=ObservedIngress.__module__,
                class_body_equivalence=C.verify_edge_bodies(), cuda_initialized=False)


def validate_runtime(registration, runtime, attached):
    """Bind discovered leases/epochs to the one expected prospective roster."""
    require(runtime['preparation_sha256']==registration['sha256'],'edge runtime registration differs')
    require(runtime['sha256']==C.canonical({k:v for k,v in runtime.items() if k!='sha256'}),'runtime digest drift')
    require(runtime['sources']==attached['sources'],'runtime differs from actual attach evidence')
    sources=Sources([Source(**s) for s in runtime['sources']])
    config=registration['config']
    require(runtime['chunk_bytes']==12500 and runtime['socket_buffer_request_bytes']==8388608,'wire limits drift')
    require(set(runtime['endpoints'])==set(sources.by_stream),'endpoint coverage drift')
    for source,expected in zip(sources.sources,config['sources']):
        require(all(getattr(source,k)==v for k,v in expected.items() if k!='ingress_port')
                and source.run_id==config['run_id'] and source.cell_id==config['cell_id'], 'source roster drift')
        entry=runtime['endpoints'][source.stream_id]
        require(entry['ingress_port']==expected['ingress_port']
                and entry['map_port']==config['map_ports'][source.ue_id], 'source endpoint drift')
        directory=Path(runtime['output'])/source.stream_id/'edge_evidence'
        require(Path(entry['edge_evidence'])==directory and Path(entry['gt_socket']).parent==directory
                and Path(entry['gt_capability']).parent==directory,'cross-source GT path')
    return sources


def prewarm_model(output, run_id, cell_id):
    """Exact production model/prewarm path, without receivers or GT services."""
    import torch
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
    from rl_agent.splitfusion_direct_edge_map_v1.direct_v3_edge import _DeadlineGuardedComputeTail
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
    from rl_agent.splitfusion_live_dispatch_v1.dynamic_execution_contract import load_dynamic_execution_contract
    from rl_agent.splitfusion_edge_optimization_v1.detached_edge_preload_v3 import preload_detached_optimized_edge_v3
    from rl_agent.splitfusion_edge_optimization_v1.detached_tail_v3 import DetachedOptimizedTailAdapterV3
    require(torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 5090', 'wrong edge GPU')
    device = torch.device('cuda:0')
    torch.hub.set_dir(str(C.model_cache()))  # pinned local cache, no download
    os.sched_setaffinity(0, set(C.CPU['shared_inference']))
    contract = load_dynamic_execution_contract()
    stack = preload_detached_optimized_edge_v3(device)
    rt = stack.runtime
    rt._detached_tail = _DeadlineGuardedComputeTail(stack.tail, lambda stage, stamp: B.check_deadline(stage,stamp,.5))
    def compute(**kw):
        value = E.run4_compute_on_detached_runtime(rt, **kw)
        published = rt.publish_cpu(value)
        snapshot = DetachedOptimizedTailAdapterV3.snapshot(published.serialized)
        return list(snapshot.records or ()), snapshot.semantic_labels.detach().to(device='cpu',dtype=torch.uint8).contiguous().numpy()
    warm_processor = E.Run4EdgeProcessorV2(contract=contract,run_id=run_id,
        cell_id=cell_id,service_deadline_s=.17,ack_timeout_s=.5,compute=compute)
    warm = PW.warm_edge(warm_processor, rt, contract, device=device,
                       encoders=stack.autoencoders, codec=rt._codec, unguarded_tail=stack.tail)
    C.write_new(output/'PREWARM.json', warm)
    require(warm['completed'] and warm['modes_warmed'] == list(range(12)), 'incomplete edge prewarm')
    return contract, compute


def run(preparation, runtime_path):
    """LIVE ONLY. Entry validation precedes any model/socket/service startup."""
    registration = C.validate(preparation)
    runtime = json.loads(Path(runtime_path).read_bytes())
    sources=validate_runtime(registration,runtime,json.loads((Path(runtime['output'])/'ATTACHED.json').read_bytes()))
    output = Path(runtime['output']) / 'edge'
    output.mkdir()
    from .edge_components import SourceEvaluator, FeedbackRouter
    from .datapath import DataPath, EventLog
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E, run4_live_wire_v2 as W
    from rl_agent.splitfusion_direct_edge_map_v1 import protocol as DP
    from rl_agent.splitfusion_m3_route_b_v3.edge_guard import build_guarded_evaluator
    contract, compute = prewarm_model(output, sources.sources[0].run_id, sources.sources[0].cell_id)
    stop, errors, receivers, readers, evaluators, publishers = threading.Event(), [], [], [], {}, {}
    control = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    log = EventLog(output/'events.jsonl')
    from .receiver_timing import Audit, Receiver, observed_verify
    audit = Audit(output)
    path = None
    router = FeedbackRouter(sources, control.sendto)
    try:
        for source in sources.sources:
            entry = runtime['endpoints'][source.stream_id]
            ev = build_guarded_evaluator(SourceEvaluator)(source=source,router=router,
                socket_path=entry['gt_socket'], capability_path=entry['gt_capability'],
                spec=W.load_run4_quality_spec(C.ROOT), read_ground_truth=None,
                match_distance_m=3.,gt_timeout_s=2.,evidence_dir=Path(entry['edge_evidence']))
            evaluators[source.stream_id] = ev
            ev.start()
            publishers[source.stream_id] = E.Run4MapPublisherV2(map_host=runtime['map_host'],
                map_port=entry['map_port'],chunk_bytes=runtime['chunk_bytes'],
                socket_buffer_request_bytes=runtime['socket_buffer_request_bytes'])
        path = DataPath(sources=sources,contract=contract,evaluators=evaluators,compute=compute,
            publish=lambda s,d: publishers[s.stream_id].publish(d),
            terminal=lambda s,d: control.sendto(DP.encode(d),s.destination),log=log,audit=audit)
        # Verify native context-camera geometry at the production wire boundary.
        from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
        verify = observed_verify(path.processor.verify,sources,audit)
        def owned_verify(frame):
            value = verify(frame)
            envelope, context, profile, identity = value
            camera = PO.context_camera(context)
            source = sources.resolve(context.stream_id)
            log(source,'edge_wire',frame_id=context.frame_id,context=PO.context_document(context),
                capture_timestamp_ns=context.capture_timestamp_ns,action_identity=identity,camera_matrix=camera.tolist())
            return value
        path.processor.verify = owned_verify
        def receive(source, receiver):
            try:
                while not stop.is_set():
                    try: data, measurement = receiver.receive()
                    except socket.timeout: continue
                    path.receive(source,data,measurement=measurement)  # registered endpoint, never NAT IP
            except BaseException as error:
                errors.append(repr(error)); stop.set()
        for source in sources.sources:
            sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,runtime['socket_buffer_request_bytes'])
            sock.bind(('0.0.0.0',runtime['endpoints'][source.stream_id]['ingress_port'])); sock.settimeout(.1)
            receivers.append(sock)
            receiver=Receiver(source,sock,audit)
            thread=threading.Thread(target=receive,args=(source,receiver),name='m4-ingress-'+source.stream_id,daemon=True)
            readers.append(thread); thread.start()
        signal.signal(signal.SIGINT,lambda *_:stop.set())
        signal.signal(signal.SIGTERM,lambda *_:stop.set())
        C.write_new(output/'READY.json',dict(preparation_sha256=registration['sha256'],runtime_sha256=runtime['sha256'],
            sources=[s.tag for s in sources.sources],prewarm=True,shared_stacks=1))
        while not stop.is_set() and not (Path(runtime['output'])/'STOP_EDGE').exists():
            path.process_once()
            require(all(ev._failure is None for ev in evaluators.values()), 'source GT channel failed')
        stop.set()
        for thread in readers: thread.join(timeout=2)
        # Final source slots are independent and each is evaluated against its original horizon.
        while path.process_once(timeout=0): pass
        require(not errors, 'edge ingress failed: '+repr(errors))
    except BaseException as error:
        errors.append(repr(error)); raise
    finally:
        stop.set()
        for thread in readers: thread.join(timeout=2)
        for sock in receivers: sock.close()
        closed={}
        for stream,ev in evaluators.items():
            try:
                closed[stream]=ev.close()
                C.write_new(output/(stream+'_GT_FEEDBACK.json'),ev.records)
            except BaseException as error: closed[stream]=dict(error=repr(error))
        for pub in publishers.values(): pub.close()
        control.close(); log.close()
        audit.close(path.ingress if path is not None else None)
        C.write_new(output/'CLOSED.json',dict(errors=errors,evaluators=closed))


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--runtime',type=Path)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--prewarm-only',action='store_true',help='GPU model validation only; no network/GT receivers')
    p.add_argument('--output',type=Path)
    args=p.parse_args(argv)
    C.validate(args.preparation)
    require(not (args.preflight and args.prewarm_only),'choose one edge validation mode')
    if args.preflight:
        print(json.dumps(dict(status='OFFLINE_EDGE_BINDING_PASS',**import_check()),sort_keys=True)); return 0
    if args.prewarm_only:
        require(args.runtime is None and args.output is not None,'prewarm-only requires output and no runtime')
        args.output.mkdir()
        prewarm_model(args.output,'M4_MODEL_PREFLIGHT','model-only')
        print(json.dumps(dict(status='MODEL_PREWARM_PASS',modes=list(range(12)),live=False))); return 0
    require(args.runtime is not None,'live edge requires registered runtime binding')
    run(args.preparation,args.runtime); return 0

if __name__ == '__main__': raise SystemExit(main())
