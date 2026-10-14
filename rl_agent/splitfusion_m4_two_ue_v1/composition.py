"""Immediate M4 ownership primitives. No CARLA/OAI launcher or global seams.

The owner accepts two actual collector/controller pairs; integrating their
construction with the retained live route harness remains a separate explicit
step. This module deliberately offers no invented live CLI.
"""
import concurrent.futures
from .identity import require


class BothStreamLifecycle:
    def __init__(self, sources):
        self.sources = sources
        self.completed = {s.stream_id: False for s in sources.sources}
        self.drained = dict(self.completed)

    def complete(self, source):
        require(self.sources.resolve(source.stream_id) == source, 'wrong route owner')
        self.completed[source.stream_id] = True

    def drain(self, callbacks):
        require(set(callbacks) == set(self.completed), 'both source drains required')
        errors = {}
        # Independent drain waits, not an inference scheduling change. One
        # failed/finished source never cancels the other source's cleanup.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {name: pool.submit(callback) for name, callback in callbacks.items()}
            for name, future in futures.items():
                try:
                    future.result()
                    self.drained[name] = True
                except BaseException as error:
                    errors[name] = repr(error)
        require(not errors, 'source drain failures: ' + repr(errors))

    @property
    def done(self):
        return all(self.completed.values()) and all(self.drained.values())


def independent_policy_contexts(sources, lineage):
    """Separate history/controllers/telemetry; share read-only CPU actor weights.

    No PolicySeams mutation. Callers must use the retained Run5 22-feature
    builder and ACKed commanded-SNR provider; no alternative state semantics.
    """
    from rl_agent.splitfusion_run5_full_live_v1.actor import load_registered_actor
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_decision_engine_v2 import Run4DecisionEngineV2
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.ue_telemetry_provider_v2 import UeTelemetryProviderV2, CausalClockBridgeV2
    from rl_agent.splitfusion_live_dispatch_v1.dynamic_execution_contract import load_dynamic_execution_contract
    actor, contract = load_registered_actor(), load_dynamic_execution_contract()
    result = {}
    for source in sources.sources:
        telemetry = UeTelemetryProviderV2(bridge=CausalClockBridgeV2())
        telemetry.bind_ue(rnti=source.rnti, oai_ue_id=source.ue_id)
        result[source.stream_id] = dict(source=source, telemetry=telemetry,
            engine=Run4DecisionEngineV2(contract_=contract, actor=actor, ue_id=telemetry.ue_label,
                                       controller_lineage_sha256=lineage))
    return result
