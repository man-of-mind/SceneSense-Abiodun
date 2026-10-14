"""Versioned Run4 observation envelope, NOT fusion or a new map-age policy."""
import copy
import json
import threading
from pathlib import Path

from spatial_map_coop.coop_input_v1.identity_config import CoopIdentityConfig
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
from .identity import require


class Run4MapJournal:
    def __init__(self, sources, contract, path):
        self.sources, self.contract = sources, contract
        self.identity = CoopIdentityConfig.from_documents([dict(run_id=s.run_id, cell_id=s.cell_id,
            stream_id=s.stream_id, ue_id=str(s.ue_id), session_id=s.session_id) for s in sources.sources])
        self.file = Path(path).open('x')
        self.lock = threading.Lock()
        self.latest, self.seen = {}, set()

    def record(self, source, document, *, arrival_ns, feedback, pose):
        # Authoritative validator FIRST, without old-schema retagging or a
        # fabricated catalog action. Original dicts are deep-copied verbatim.
        MP.validate_run4_map_update(document, contract=self.contract)
        MP.validate_run4_map_feedback(feedback)
        require(self.sources.resolve(document['stream_id']) == source, 'wrong map source/connection')
        source.capture(document)
        owner = self.identity.resolve(run_id=document['run_id'], cell_id=document['cell_id'],
                                      stream_id=document['stream_id'])
        for field in ('run_id', 'cell_id', 'stream_id', 'frame_id', 'capture_timestamp_ns', 'run4_identity'):
            require(document[field] == feedback[field], 'wrong-source/action map feedback: ' + field)
        require(type(arrival_ns) is int and arrival_ns > 0, 'actual arrival wall ns required')
        require(isinstance(pose, dict) and pose.get('source') == source.tag
                and pose.get('frame_id') == document['frame_id']
                and pose.get('capture_timestamp_ns') == document['capture_timestamp_ns']
                and bool(pose.get('provenance')), 'frame-bound pose provenance required')
        key = (source.session_id, source.stream_id, document['frame_id'])
        capture = (document['capture_timestamp_ns'], document['frame_id'])
        row = dict(schema='splitfusion.m4.run4_observation.v1', ue_id=owner.ue_id,
            session_id=owner.session_id, source=source.tag, arrival_wall_ns=arrival_ns,
            install_wall_s=feedback['install_timestamp'], wire_document=copy.deepcopy(document),
            map_feedback=copy.deepcopy(feedback), pose=copy.deepcopy(pose),
            count_semantics='source observations, not unique physical objects')
        with self.lock:
            require(key not in self.seen, 'duplicate source map terminal')
            if feedback['accepted']:
                previous = self.latest.get(source.stream_id)
                require(previous is None or capture > previous, 'installed map ordering violation within source')
                self.latest[source.stream_id] = capture
            self.file.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            self.file.flush()
            self.seen.add(key)
        return row

    def close(self):
        self.file.close()
