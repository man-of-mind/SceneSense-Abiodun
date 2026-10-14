"""Focused CPU real-socket observational-path differential checks; no services."""
import argparse, hashlib, json, socket, sys, time, uuid
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path.cwd()  # run from the production worktree, including inside the slim image
sys.path.insert(0,str(ROOT))
from rl_agent.splitfusion_m4_two_ue_v1.identity import Source,Sources
from rl_agent.splitfusion_m4_two_ue_v1.receiver_timing import Audit,Receiver,SO_TIMESTAMPNS_NEW,parse_timestamp,TIMESPEC,observed_verify
from rl_agent.splitfusion_m4_two_ue_v1.datapath import DataPath
from rl_agent.splitfusion_m4_two_ue_v1.transport import SourceIngress
from phase2_map_sharing.transport import chunk_payload
from rl_agent.splitfusion_live_dispatch_v1.dynamic_execution_contract import load_dynamic_execution_contract
from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X,run4_live_wire_v2 as W
contract=load_dynamic_execution_contract()
roster=Sources([Source(ego_id=f'ego{i}',ue_id=i,imsi=f'00101000000000{i+1}',tunnel=f'oaitun_ue{i+1}',
 ip=f'10.0.0.{20+i}',rnti=1001+i,run_id='M4_RX_FIXTURE',cell_id='cell',stream_id=f'm4-ue{i}',
 session_id=f'attach-{i}',feedback_port=52010+i) for i in range(2)])
def wire(source,frame):
    profile=contract.resolve_q_e4(11,4321);capture=time.time_ns()
    context=build_frame_context_v1(stream_id=source.stream_id,frame_id=frame,sequence_id=frame,
        capture_timestamp_ns=capture,ego_world_x=15.*source.ue_id,ego_world_y=0.,ego_world_z=1.,
        ego_world_pitch=0.,ego_world_yaw=0.,ego_world_roll=0.)
    payload=b'CPU-fixture'
    envelope=X.ExecutionEnvelopeV3(mode_id=11,q_e4=4321,keep_count=profile.keep_count,
        anchor_action_id=None,reward_requested=True,session_uuid=str(uuid.UUID(int=100+source.ue_id)),
        controller_lineage_sha256='1'*64,decision_seq=frame,ticket_seq=frame,frame_id=frame,tensor_seq=frame,
        capture_timestamp_ns=capture,execution_bundle_sha256=profile.execution_bundle_sha256,
        inner_payload_sha256=hashlib.sha256(payload).hexdigest(),inner_payload=payload)
    return W.pack_sfd4(envelope,context),capture
def main():
    out=Path(sys.argv[1]);out.mkdir()
    audit=Audit(out);terminals=[];events=[]
    path=DataPath(sources=roster,contract=contract,
        evaluators={s.stream_id:NS(notify_terminal=lambda *a:None) for s in roster.sources},
        compute=lambda **kw:(_ for _ in ()).throw(AssertionError('inference forbidden')),
        publish=lambda *a:None,terminal=lambda s,d:terminals.append((s.stream_id,d)),
        log=lambda s,k,**v:events.append((s.stream_id,k,v)),audit=audit)
    path.processor.verify=observed_verify(path.processor.verify,roster,audit)
    original=SourceIngress(roster)
    socks=[];receivers=[]
    for s in roster.sources:
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sock.bind(('127.0.0.1',0));sock.settimeout(1.)
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,8388608)
        socks.append(sock);receivers.append(Receiver(s,sock,audit))
    tx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    def get(i,data,ancillary=None,delay=0.):
        tx.sendto(data,socks[i].getsockname())
        if delay:time.sleep(delay)
        actual,meta=receivers[i].receive(ancillary_bytes=ancillary)
        assert actual==data
        return actual,meta
    # Identical IDs interleaved on two endpoints; reordering and duplicate index 0.
    wires=[wire(s,4172) for s in roster.sources]
    chunks=[chunk_payload(w,message_id=4172,chunk_bytes=128) for w,c in wires]
    order=[len(chunks[0])-1,0,0]+list(range(1,len(chunks[0])-1))
    assert all(len(c)==len(chunks[0]) for c in chunks)
    measured=[]
    for j in order:
        for i,s in enumerate(roster.sources):
            data,m=get(i,chunks[i][j]);measured.append(m);assert m['timestamp_status']=='OK'
            wall=wires[i][1]/1e9+.01
            a=path.receive(s,data,monotonic_s=1.,wall_s=wall,measurement=m)
            b=original.ingest(s,data,received_at_s=1.,received_wall_s=wall)
            assert a==b
    for _ in range(2):
        a=path.ingress.pending.take(0);b=original.pending.take(0);assert a==b
        stream,item=a;assert item['payload']==wires[roster.resolve(stream).ue_id][0]
        path.processor.verify(item['payload'])
    # Completed expired frame still traverses the real DataPath terminal route.
    w,capture=wire(roster.sources[0],4173)
    for data in chunk_payload(w,message_id=4173,chunk_bytes=128):
        data,m=get(0,data);path.receive(roster.sources[0],data,monotonic_s=1.1,wall_s=capture/1e9+.6,measurement=m)
    assert len(terminals)==1 and terminals[0][0]=='m4-ue0'
    # Registered endpoint wins, never the peer address; foreign envelope rejected.
    w,capture=wire(roster.sources[0],4174)
    refused=False
    try:
        for data in chunk_payload(w,message_id=4174,chunk_bytes=128):
            data,m=get(1,data);path.receive(roster.sources[1],data,monotonic_s=1.2,wall_s=capture/1e9+.01,measurement=m)
    except ValueError as e:
        assert 'wrong source/frame' in str(e);refused=True
    assert refused
    # Incomplete same-ID messages retained explicitly for BOTH sources.
    for i,s in enumerate(roster.sources):
        w,c=wire(s,4175);chunks2=chunk_payload(w,message_id=4175,chunk_bytes=128)
        data,m=get(i,chunks2[i]);path.receive(s,data,monotonic_s=1.3,wall_s=c/1e9+.01,measurement=m)
    # Actual socket missing/truncated ancillary data; payload unchanged.
    sample=chunk_payload(w,message_id=4176,chunk_bytes=128)[0]
    data,m=get(0,sample,ancillary=0);assert m['timestamp_status']=='ANCILLARY_TRUNCATED' and m['kernel_wall_ns'] is None
    socks[0].setsockopt(socket.SOL_SOCKET,SO_TIMESTAMPNS_NEW,0)
    data,m=get(0,sample);assert m['timestamp_status']=='MISSING' and m['kernel_wall_ns'] is None
    socks[0].setsockopt(socket.SOL_SOCKET,SO_TIMESTAMPNS_NEW,1)
    data,m=get(0,sample,delay=.025)
    service_ns=m['pair']['wall_ns']-m['kernel_wall_ns'];assert service_ns>=20_000_000
    assert parse_timestamp([(socket.SOL_SOCKET,SO_TIMESTAMPNS_NEW,TIMESPEC.pack(1,1_000_000_000))],0)[1]=='INVALID_TIMESPEC'
    for m in measured:
        p=m['pair'];assert p['raw_before_ns']<=p['raw_after_ns']
        assert p['wall_ns']>=m['kernel_wall_ns']
    for sock in socks:sock.close()
    tx.close()
    status=audit.close(path.ingress)
    assert not status['errors'] and not status['writer_alive']
    assert status['counters']['attempted']==status['counters']['enqueued']==status['counters']['written']
    rows=[json.loads(l) for l in (out/'RECEIVER_TIMING.jsonl').read_text().splitlines()]
    complete=[r for r in rows if r['kind']=='assembly' and r['completed']]
    assert len(complete)==4
    assert sum(r['kind']=='deadline_check' and r['result']=='EXPIRED' for r in rows)==1
    assert sum(r['kind']=='incomplete_at_close' and len(r['pending'])==1 for r in rows)==2
    assert sorted(r['duplicates'] for r in complete)==[0,0,1,1]
    bounded=out/'bounded';bounded.mkdir();a=Audit(bounded,row_limit=2)
    for _ in range(5):a.emit(None,'bounded_test')
    bound=a.close();assert bound['counters']['dropped_limit']==5 and bound['counters']['written']==2
    import torch
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS',checks=['two sources same frame/chunk IDs','production DataPath byte/routing parity',
        'reordering/duplicates','incomplete both sources','expired terminal timing retained',
        'foreign source refused','real missing/truncated ancillary','delayed socket service',
        'bounded logging drops visible','CPU only'],delayed_receive_service_ms=service_ns/1e6,
        complete_frames=len(complete),timing_status=status,bounded_status=bound)
    with (out/'RESULT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

