"""Native Run5 state construction per source, with no shared policy seams."""
import types
import dataclasses

from .binding import bind
from .identity import require
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import live_state_v2 as LS
from rl_agent.splitfusion_run5_live_avo_v1.actor_binding import LiveStateBuilder


class NativePipeline(U.Run4FramePipelineV2):
    def __init__(self, *, source, snr_provider, **kwargs):
        require(kwargs['provider'].ue_label == kwargs['engine'].ue_id == snr_provider.ue_id,
                'telemetry/SNR/controller source identity mismatch')
        super().__init__(stream_id=source.stream_id, run_id=source.run_id, cell_id=source.cell_id, **kwargs)
        self.source = source
        self.state_builder = LiveStateBuilder(snr_provider)
        self.snr_rows = []

        def build(**state):
            require(state['identity'].ue_id == snr_provider.ue_id, 'foreign state UE')
            result = self.state_builder.build(**state)
            self.snr_rows.append(dict(frame_id=state['camera_si'].metadata.identity.sample_seq,
                ue_id=state['identity'].ue_id, session=state['identity'].session_uuid,
                observation=dataclasses.asdict(self.state_builder.last_observation),
                features=list(result[1].as_tuple())))
            return result

        # Retained plan/order/hold/fallback behavior and bytecode, with only the
        # native state builder bound locally. No 21-feature fallback path.
        local_state = types.SimpleNamespace(scene_observations=LS.scene_observations, build_live_state=build)
        self.plan = types.MethodType(bind(U.Run4FramePipelineV2.plan, LS=local_state), self)


class TelemetryFanout:
    """Parse once per reader, dispatch explicitly; post-NAT IP is never used."""
    def __init__(self, contexts):
        self.contexts = contexts
        self.rejected = []

    def dispatch(self, event, row, clock_pair):
        if event == 'NRUE_MAC_DCI_GRANT':
            owners = [v for v in self.contexts.values() if v['source'].rnti == int(row['rnti'])]
            method = 'on_dci'
        elif event == 'NRUE_MAC_RLC_BUFFER_STATUS':
            owners = [v for v in self.contexts.values() if v['source'].rnti == int(row['rnti'])
                      and v['source'].ue_id == int(row['ue_id'])]
            method = 'on_rlc'
        elif event == 'NR_PDCP_TX_SDU':
            # Clock anchoring is host-wide; the provider itself validates it.
            owners, method = list(self.contexts.values()), 'on_pdcp'
        else:
            raise ValueError('unknown telemetry event')
        if not owners:
            self.rejected.append(dict(event=event, row=dict(row)))
            return False
        for owner in owners:
            getattr(owner['telemetry'], method)(row, clock_pair)
        return True
