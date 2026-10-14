"""Source-bound reassembly around the retained latest-per-stream pending slot.

Call with the Source assigned to the receiving channel, never a NAT-derived
identity or a caller-supplied UE field. Scheduling and deadline checks are the
retained implementations; no model/precision/rate changes.
"""
from phase2_map_sharing.transport import ChunkReassembler
from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as base
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from .identity import require


class SourceIngress:
    def __init__(self, sources, *, ack_timeout_s=.5):
        self.sources = sources
        self.ack_timeout_s = float(ack_timeout_s)
        self.pending = base.LatestFramePendingSlot()
        self.reassembly = {s.stream_id: ChunkReassembler(timeout_s=2., max_chunks=4096) for s in sources.sources}

    def ingest(self, source, datagram, *, received_at_s, received_wall_s):
        require(self.sources.resolve(source.stream_id) == source, 'unbound ingress channel')
        complete = self.reassembly[source.stream_id].ingest(source.session_id, datagram, received_at_s=received_at_s)
        if complete is None:
            return None
        envelope, context = W.unpack_sfd4(complete.payload)
        require(context.stream_id == source.stream_id and context.frame_id == int(complete.message_id),
                'wrong source/frame on bound ingress')
        base.check_deadline(base.EDGE_STAGE_AFTER_REASSEMBLY, context.capture_timestamp_ns,
                            self.ack_timeout_s, now_s=received_wall_s)
        item = dict(payload=complete.payload, stream_id=context.stream_id, capture_ns=context.capture_timestamp_ns,
                    received_wall_s=received_wall_s, frame_id=context.frame_id, source=source.tag)
        admitted, displaced = self.pending.offer(context.stream_id, item, sequence=context.sequence_id)
        return dict(admitted=admitted, displaced=displaced, offered=item)
