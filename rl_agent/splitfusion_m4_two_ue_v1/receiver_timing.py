"""Bounded observational RX timing; no wire, scheduling or deadline changes.

Linux SO_TIMESTAMPNS_NEW (asm-generic/socket.h, 64) reports a software skb
receive timestamp in CLOCK_REALTIME. This is not a radio/over-the-air stamp.
Only the registered local receiving endpoint supplies source identity.
"""
import collections
import json
import os
from pathlib import Path
import queue
import socket
import struct
import threading
import time

from .identity import require
from phase2_map_sharing.transport import CHUNK_HEADER

SO_TIMESTAMPNS_NEW = 64
TIMESPEC = struct.Struct('=qq')


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def clock_pair():
    first = raw()
    wall = time.time_ns()
    last = raw()
    return dict(raw_before_ns=first, wall_ns=wall, raw_after_ns=last)


def parse_timestamp(ancillary, flags):
    if flags & socket.MSG_CTRUNC:
        return None, 'ANCILLARY_TRUNCATED'
    stamps = [data for level,kind,data in ancillary if level==socket.SOL_SOCKET and kind==SO_TIMESTAMPNS_NEW]
    if not stamps:
        return None, 'MISSING'
    if len(stamps)!=1 or len(stamps[0])!=TIMESPEC.size:
        return None, 'INVALID_LENGTH_OR_MULTIPLE'
    sec,ns=TIMESPEC.unpack(stamps[0])
    if sec<0 or not 0<=ns<1_000_000_000:
        return None, 'INVALID_TIMESPEC'
    return sec*1_000_000_000+ns, 'OK'


class Audit:
    """Nonblocking producers, bounded queue/row count, buffered writer.

    Hot path never serializes JSON or flushes disk. Drops/errors are explicit.
    Existing per-frame production evidence stays unchanged.
    """
    def __init__(self, directory, *, capacity=32768, row_limit=131072):
        self.directory=Path(directory)
        self.q=queue.Queue(maxsize=capacity)
        self.limit=row_limit
        self.counts=collections.Counter()
        self.counter_lock=threading.Lock()
        self.cost_ns=0; self.cost_max_ns=0
        self.errors=[]; self.queue_peak=0
        self.thread=threading.Thread(target=self._write,name='m4-rx-audit',daemon=True)
        self.thread.start()
        self.emit(None,'clock_start',pair=clock_pair(),kernel=os.uname().release,
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  time_namespace=Path('/proc/self/timens_offsets').read_text() if Path('/proc/self/timens_offsets').exists() else None,
                  facility='SO_TIMESTAMPNS_NEW',queue_capacity=capacity,row_limit=row_limit)

    def emit(self, source, kind, **values):
        start=raw()
        with self.counter_lock:
            self.counts['attempted']+=1
            over_limit=self.counts['attempted']>self.limit
        if over_limit:
            self.metric('dropped_limit',1)
        else:
            row=dict(kind=kind,stream_id=source.stream_id if source else None,
                     session_id=source.session_id if source else None,**values)
            try:
                self.q.put_nowait(row)
                self.metric('enqueued',1)
                with self.counter_lock:self.queue_peak=max(self.queue_peak,self.q.qsize())
            except queue.Full:
                self.metric('dropped_queue',1)
        elapsed=raw()-start
        with self.counter_lock:
            self.cost_ns+=elapsed;self.cost_max_ns=max(self.cost_max_ns,elapsed)

    def metric(self,key,value):
        with self.counter_lock:self.counts[key]+=value

    def _write(self):
        try:
            with (self.directory/'RECEIVER_TIMING.jsonl').open('x',buffering=262144) as h:
                while True:
                    try:row=self.q.get(timeout=.25)
                    except queue.Empty:
                        h.flush();continue
                    if row is None:break
                    h.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                    self.metric('written',1)
        except BaseException as e:
            self.errors.append(repr(e))

    def close(self, ingress=None):
        if ingress is not None:
            for stream,reasm in ingress.reassembly.items():
                source=ingress.sources.resolve(stream)
                self.emit(source,'incomplete_at_close',expired_messages=reasm.expired_messages,
                    pending=[dict(message_id=key[1],chunk_count=v['total'],indices=sorted(v['chunks']),
                                  first_chunk_monotonic_s=v['first_chunk_at_s'],duplicates=v['duplicates'])
                             for key,v in reasm.pending.items()])
        self.emit(None,'clock_end',pair=clock_pair())
        if self.thread.is_alive():
            try:self.q.put(None,timeout=5)
            except queue.Full:self.errors.append('writer queue did not drain')
            self.thread.join(timeout=10)
        result=dict(counters=dict(self.counts),errors=self.errors,writer_alive=self.thread.is_alive(),
                    queue_peak=self.queue_peak,producer_wall_ns=self.cost_ns,producer_max_wall_ns=self.cost_max_ns,
                    clock_pair='RAW before / REALTIME / RAW after on each recvmsg; no substitution for missing kernel stamps')
        with (self.directory/'RECEIVER_TIMING_STATUS.json').open('x') as h:json.dump(result,h,indent=2)
        return result


class Receiver:
    def __init__(self, source, sock, audit):
        self.source,self.sock,self.audit=source,sock,audit
        self.seq=0
        before=sock.getsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF)
        sock.setsockopt(socket.SOL_SOCKET,SO_TIMESTAMPNS_NEW,1)
        after=sock.getsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF)
        require(before==after,'timestamp facility changed socket receive buffer')
        require(sock.getsockopt(socket.SOL_SOCKET,SO_TIMESTAMPNS_NEW)==1,'kernel RX timestamp facility unavailable')
        self.local=sock.getsockname()
        audit.emit(source,'socket',registered_source=source.tag,local_endpoint=self.local,rcvbuf_bytes=after,
                   sndbuf_bytes=sock.getsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF),timestamp_option=SO_TIMESTAMPNS_NEW)

    def receive(self, *, ancillary_bytes=None):
        data, ancillary, flags, peer=self.sock.recvmsg(65535,socket.CMSG_SPACE(TIMESPEC.size) if ancillary_bytes is None else ancillary_bytes)
        pair=clock_pair()  # first RAW read immediately after syscall return
        start=pair['raw_before_ns']
        kernel,status=parse_timestamp(ancillary,flags)
        self.seq+=1
        frame=index=count=None
        if len(data)>=CHUNK_HEADER.size:frame,index,count=CHUNK_HEADER.unpack_from(data)
        record=dict(recv_seq=self.seq,frame_id=frame,chunk_index=index,chunk_count=count,datagram_bytes=len(data),
                    kernel_wall_ns=kernel,timestamp_status=status,pair=pair,recv_return_raw_ns=start,
                    msg_flags=flags,payload_truncated=bool(flags&socket.MSG_TRUNC),
                    local_endpoint=self.local,peer_endpoint_diagnostic_only=peer,
                    ancillary=[dict(level=l,type=t,length=len(d)) for l,t,d in ancillary])
        self.audit.metric('timestamp_'+status,1)
        self.audit.emit(self.source,'datagram',**record)
        self.audit.metric('receiver_instrumentation_ns',raw()-start)
        return data,record


def observed_verify(verify, sources, audit):
    def measured(frame):
        start=raw()
        try:value=verify(frame)
        except BaseException as error:
            audit.emit(None,'verification_error',start_raw_ns=start,end_raw_ns=raw(),error=repr(error));raise
        end=raw()
        envelope,context,profile,identity=value
        audit.emit(sources.resolve(context.stream_id),'verified',frame_id=context.frame_id,
                   start_raw_ns=start,end_raw_ns=end,capture_wall_ns=context.capture_timestamp_ns)
        return value
    return measured


# The original qualified SourceIngress remains byte-identical for GT/component
# proof reuse. This observational subclass retains its exact branch/order/data
# flow, checked against the original with real sockets in focused qualification.
from .transport import SourceIngress
from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as base
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W


class ObservedIngress(SourceIngress):
    def __init__(self, sources, audit):
        super().__init__(sources)
        self.audit=audit

    def ingest(self, source, datagram, *, received_at_s, received_wall_s, measurement=None):
        require(self.sources.resolve(source.stream_id)==source,'unbound ingress channel')
        start=raw()
        complete=self.reassembly[source.stream_id].ingest(source.session_id,datagram,received_at_s=received_at_s)
        end=raw()  # before envelope parsing or deadline/admission checks
        header=CHUNK_HEADER.unpack_from(datagram) if len(datagram)>=CHUNK_HEADER.size else (None,None,None)
        self.audit.emit(source,'assembly',frame_id=header[0],recv_seq=(measurement or {}).get('recv_seq'),
            start_raw_ns=start,end_raw_ns=end,receive_monotonic_s=received_at_s,
            completed=complete is not None,payload_bytes=len(complete.payload) if complete else None,
            chunk_count=complete.chunk_count if complete else header[2],
            duplicates=complete.duplicate_chunks if complete else None,
            expired_messages=self.reassembly[source.stream_id].expired_messages)
        if complete is None:return None
        start=raw()
        try:
            envelope,context=W.unpack_sfd4(complete.payload)
            require(context.stream_id==source.stream_id and context.frame_id==int(complete.message_id),
                    'wrong source/frame on bound ingress')
        except BaseException as error:
            self.audit.emit(source,'binding_rejected',frame_id=complete.message_id,start_raw_ns=start,end_raw_ns=raw(),error=repr(error));raise
        self.audit.emit(source,'binding_checked',frame_id=context.frame_id,start_raw_ns=start,end_raw_ns=raw(),
                        capture_wall_ns=context.capture_timestamp_ns,verified_stream=context.stream_id)
        start=raw()
        try:base.check_deadline(base.EDGE_STAGE_AFTER_REASSEMBLY,context.capture_timestamp_ns,self.ack_timeout_s,now_s=received_wall_s)
        except base.DeadlineExpired:
            self.audit.emit(source,'deadline_check',frame_id=context.frame_id,start_raw_ns=start,end_raw_ns=raw(),
                            received_wall_s=received_wall_s,result='EXPIRED');raise
        self.audit.emit(source,'deadline_check',frame_id=context.frame_id,start_raw_ns=start,end_raw_ns=raw(),
                        received_wall_s=received_wall_s,result='PASS')
        item=dict(payload=complete.payload,stream_id=context.stream_id,capture_ns=context.capture_timestamp_ns,
                  received_wall_s=received_wall_s,frame_id=context.frame_id,source=source.tag)
        start=raw()
        admitted,displaced=self.pending.offer(context.stream_id,item,sequence=context.sequence_id)
        self.audit.emit(source,'pending_admission',frame_id=context.frame_id,start_raw_ns=start,end_raw_ns=raw(),admitted=admitted)
        return dict(admitted=admitted,displaced=displaced,offered=item)
