"""The same stdlib edge startup guard, dispatching this registered package."""
from types import FunctionType
from rl_agent.splitfusion_live_parity_runtime_v8 import startup as V8

write_x, preserve_startup = V8.write_x, V8.preserve_startup
edge_entry = FunctionType(V8.edge_entry.__code__,
    dict(V8.edge_entry.__globals__, __package__=__package__, __spec__=__spec__),
    V8.edge_entry.__name__, V8.edge_entry.__defaults__, V8.edge_entry.__closure__)
