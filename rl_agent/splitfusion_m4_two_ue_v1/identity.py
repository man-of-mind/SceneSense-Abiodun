"""Explicit M4 source identity. No address, arrival-order or frame-ID inference."""
from dataclasses import asdict, dataclass
import ipaddress


def require(value, message):
    if not value:
        raise ValueError(message)


@dataclass(frozen=True)
class Source:
    ego_id: str
    ue_id: int
    imsi: str
    tunnel: str
    ip: str
    rnti: int
    run_id: str
    cell_id: str
    stream_id: str
    session_id: str  # source/attach epoch; NOT the actor's rolling decision UUID
    feedback_port: int

    def __post_init__(self):
        for name in ('ego_id', 'imsi', 'tunnel', 'run_id', 'cell_id', 'stream_id', 'session_id'):
            require(type(getattr(self, name)) is str and bool(getattr(self, name)), 'empty ' + name)
        require(type(self.ue_id) is int and self.ue_id >= 0, 'invalid UE ID')
        require(type(self.rnti) is int and 0 < self.rnti < 65536, 'invalid RNTI')
        require(type(self.feedback_port) is int and 0 < self.feedback_port < 65536, 'invalid feedback port')
        address = ipaddress.ip_address(self.ip)
        require(address.version == 4 and not address.is_unspecified and not address.is_multicast,
                'explicit usable IPv4 lease required')

    @property
    def tag(self):
        return asdict(self)

    @property
    def destination(self):
        return self.ip, self.feedback_port

    def capture(self, value):
        require(all(value.get(k) == getattr(self, k) for k in ('run_id', 'cell_id', 'stream_id')),
                'foreign capture source')

    def unwrap(self, message):
        require(message.get('_m4_source') == self.tag, 'foreign source/session/connection')
        return {k: v for k, v in message.items() if k != '_m4_source'}

    def wrap(self, message):
        require('_m4_source' not in message, 'already enveloped')
        return dict(message, _m4_source=self.tag)


class Sources:
    def __init__(self, sources):
        self.sources = tuple(sources)
        require(len(self.sources) == 2, 'exactly two explicit sources required')
        for name in ('ego_id', 'ue_id', 'imsi', 'tunnel', 'ip', 'rnti', 'stream_id', 'session_id'):
            require(len({getattr(s, name) for s in self.sources}) == 2, 'ambiguous ' + name)
        self.by_stream = {s.stream_id: s for s in self.sources}

    def resolve(self, stream):
        require(stream in self.by_stream, 'unregistered stream')
        return self.by_stream[stream]
