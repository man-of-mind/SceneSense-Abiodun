"""Offline-only exact JSON normalization in private handoff namespaces.

No validator, hash, canonical JSON option, packet field, array operation,
publication boundary, archive operation, GC policy or scheduling rule changes.
Only the already-qualified v1 normalizer replaces the original recursive
typing.Mapping-first conversion. No frame values or validation results cache.
"""
from functools import wraps
from types import MethodType, SimpleNamespace


COUNTERS = ("publish", "serialize", "json_normalize", "transport_canonical",
            "transport_input_metadata", "packet_encode", "target_files",
            "archive_canonical", "archive_sha")


def private_bindings():
    """Fresh function globals and scalar counters; never mutate original modules."""
    from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
    from rl_agent.splitfusion_live_gt_optimization_v1.source_geometry import json_value
    from rl_agent.splitfusion_live_gt_optimization_v2.candidate import clone_functions
    from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T, gt_bridge as G
    from rl_agent.splitfusion_live_gt_handoff_v1 import engine as H, scoring_packet as P, deferred_files as D

    counters = dict.fromkeys(COUNTERS, 0)

    def counted(name, function):
        @wraps(function)
        def call(*args, **kwargs):
            counters[name] += 1
            return function(*args, **kwargs)
        return call

    # Canonical serialization itself stays A.canonical, including sort order,
    # separators, Unicode behavior, nonfinite rejection and unsupported types.
    adapter = SimpleNamespace(**dict(vars(A), _json_value=counted("json_normalize", json_value)))
    transport = clone_functions(T, targets=adapter)
    for name, counter in (("_canonical", "transport_canonical"),
                          ("_input_metadata", "transport_input_metadata")):
        transport[name] = counted(counter, transport[name])
    private_transport = SimpleNamespace(**transport)
    bridge = clone_functions(G, A=adapter)
    bridge["canonical"] = counted("archive_canonical", bridge["canonical"])
    bridge["sha"] = counted("archive_sha", bridge["sha"])
    private_bridge = SimpleNamespace(**bridge)
    packet = clone_functions(P, T=private_transport)
    packet["encode_packet"] = counted("packet_encode", packet["encode_packet"])
    deferred = clone_functions(D, T=private_transport)
    deferred["target_files"] = counted("target_files", deferred["target_files"])

    # These are from-import globals in the original engine, so rebinding only
    # the packet/deferred module attributes would leave the slow calls active.
    namespace = dict(H.Engine._publish.__globals__, T=private_transport, G=private_bridge,
                     encode_packet=packet["encode_packet"], target_files=deferred["target_files"])
    from types import FunctionType
    methods = {}
    for name, counter in (("_publish", "publish"), ("_serialize", "serialize")):
        original = getattr(H.Engine, name)
        private = FunctionType(original.__code__, namespace, original.__name__,
                               original.__defaults__, original.__closure__)
        private.__kwdefaults__ = original.__kwdefaults__
        methods[name] = counted(counter, private)
    return dict(methods=methods, counters=counters, transport=private_transport,
                bridge=private_bridge, packet=SimpleNamespace(**packet),
                deferred=SimpleNamespace(**deferred))


def make_engine_class(base_engine):
    """Build a subclass whose private methods exist before native timer binding.

    The base must inherit the sealed handoff publisher/serializer unchanged.
    Its startup, GC, placement, validation, request loop and teardown remain
    authoritative. Per-instance diagnostics are ``metadata_counters``.
    """
    from rl_agent.splitfusion_live_gt_handoff_v1.engine import Engine as HandoffEngine
    if (not isinstance(base_engine, type) or not issubclass(base_engine, HandoffEngine)
            or base_engine._publish is not HandoffEngine._publish
            or base_engine._serialize is not HandoffEngine._serialize):
        raise ValueError("unchanged sealed handoff engine methods required")

    class ExactMetadataEngine(base_engine):
        def __init__(self, *args, **kwargs):
            private = private_bindings()
            self.metadata_counters = private["counters"]
            for name, function in private["methods"].items():
                setattr(self, name, MethodType(function, self))
            # HandoffEngine.__init__ now wraps these instance methods. Its
            # existing V2-fast self.validate_inputs is deliberately untouched.
            super().__init__(*args, **kwargs)

    return ExactMetadataEngine
