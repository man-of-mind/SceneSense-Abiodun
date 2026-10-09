"""Retained pinned-image CPU startup proof, bound to the new additive entry."""
from rl_agent.splitfusion_live_parity_runtime_v8 import container_check as V8
from .bindings import namespace, specialize

_bound = namespace(V8, __file__=__file__)
specialize(V8.docker_command, _bound, [
    ('"rl_agent.splitfusion_live_parity_runtime_v8.container_check"',
     '"rl_agent.splitfusion_live_parity_runtime_v9.container_check"')])
globals().update({name: value for name, value in _bound.items() if not name.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(main())
