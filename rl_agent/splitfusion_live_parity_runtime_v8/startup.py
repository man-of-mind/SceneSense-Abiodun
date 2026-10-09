"""Retain stdlib startup diagnostics with the registered v8 relative entry."""
from types import FunctionType
from rl_agent.splitfusion_live_parity_runtime_v2 import startup as _v2

preserve_startup = _v2.preserve_startup
write_x = _v2.write_x
edge_entry = FunctionType(_v2.edge_entry.__code__,
    dict(_v2.edge_entry.__globals__, __package__=__package__, __spec__=__spec__),
    _v2.edge_entry.__name__, _v2.edge_entry.__defaults__, _v2.edge_entry.__closure__)
