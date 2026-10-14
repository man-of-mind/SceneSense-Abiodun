"""Source-owned frontend construction around the retained Run4 methods.

Constructing the factory is offline; constructing a runtime is LIVE ONLY.
The two runtime instances share one immutable preloaded front/ranker/AE set,
not sockets, histories, plans, controllers, ledgers or preparation slots.
"""
import collections
import threading
import types
from pathlib import Path

from .binding import bind, construct
from .identity import require
from .policy import NativePipeline


def shared_model_view(shared, device):
    """Separate call ledger/runtime counters around the same immutable weights."""
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
    ue, _, models = shared
    ledger = B._Ledger()
    runtime = B.PreloadedSplitUERuntime(B.SplitActionRegistry.from_runtime_binding(),
        front=B._Front(ue._front._model,ledger),ranker=B._Ranker(ue._ranker._model,ledger),
        ae_encoders={name:B._AE(name,wrapped._model,ledger) for name,wrapped in ue._ae_encoders.items()},
        device=device,codec=B.ProductionSplitCodec(),prepare_modules=False,
        startup_model_load_operations=0,startup_model_construction_operations=0)
    return runtime,ledger,models


def runtime_class(bindings, *, source, context, snr_provider, shared_models, handoff, retention=None):
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_object_gt_v2 as OG
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
    from rl_agent.splitfusion_direct_edge_map_v1 import live_pilot_runtime_direct_v1 as DR
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
    from rl_agent.splitfusion_ue_input_prep_v2.candidate import prepare_live_input_fused
    from phase2_map_sharing.transport import chunk_payload
    inherited = U.build_run4_runtime_class(bindings)
    base_init = bind(B.LivePilotCellRuntime.__init__, _preload_ue=lambda device: shared_models)
    packing = prepare_live_input_fused
    if retention is not None:
        from rl_agent.splitfusion_paired_retention_v2.observer import retained_fused_packing
        packing = retained_fused_packing(packing, retention.observe_input)
    submit_base = types.SimpleNamespace(**{**vars(B), '_prepare_live_input': packing})
    retained_submit = bind(inherited.submit, base=submit_base)

    class SourceRuntime(inherited):
        def __init__(self, *, ledger, campaign, cell, attempt_dir, evidence_dir, ue_control_port):
            require(ue_control_port == source.feedback_port, 'wrong UE feedback endpoint')
            require(campaign['runtime']['ue_bind_host'] == source.ip, 'front not bound to actual lease')
            self._run4_ready = threading.Event()
            self._engine_lock = threading.Lock()
            self._gt_identity, self._run4_identity = {}, {}
            self._feedback_rows, self._terminal_rows = [], []
            self.scene_hooks = self.reward_planned_hook = None
            self.last_datagram = OG.LastDatagramMarksV2()
            self.ledger = ledger
            self.control_counters = B._Counters()
            self._terminal_frames = {}
            campaign = dict(campaign, runtime=dict(campaign['runtime'], camera_result_port=ue_control_port))
            base_init(self, campaign=campaign, cell=cell, attempt_dir=attempt_dir,
                      evidence_dir=evidence_dir, map_host='127.0.0.1', map_port=1)
            from .sender import SourceSender
            self.sender = SourceSender(self.sender,source)
            self.map_socket.close()
            self.map_socket, self.map_remote = DR._ForbiddenSocket(), None
            self.ue_control_port = ue_control_port
            self.contract, self.engine = context['engine']._contract, context['engine']
            self.actor, self.provider = self.engine._actor, context['telemetry']
            self.readers, self.audit = [], None  # the explicit relay owner owns all readers
            continuous = X.ContinuousUERuntimeV2(self.contract,
                front=U._SqueezedFront(self.ue._front, self.device), ranker=self.ue._ranker,
                ae_encoders=dict(self.ue._ae_encoders), codec=self.ue._codec)
            self.pipeline = NativePipeline(source=source, snr_provider=snr_provider, engine=self.engine,
                                            continuous_ue=continuous, provider=self.provider)
            bind_owned_materialize(self.pipeline)
            if retention is not None:
                materialize = self.pipeline.materialize
                def observed(planned, **kw):
                    token = prepared = None
                    try:
                        token = retention.begin(planned)
                        prepared = materialize(planned, **kw)
                        return prepared
                    finally:
                        retention.end(token, prepared)
                self.pipeline.materialize = observed
            install_controller_observer(self.engine, handoff)
            self._chunk_payload = chunk_payload
            self.sensor_first = U.SensorFirstPlannerV2(pipeline=self.pipeline, lock=self._engine_lock)
            self.cycle_budget = U.CycleBudgetV2(frame_budget=3900)
            self.sensor_stages = collections.OrderedDict()
            self.cycle_boundary_reached = False
            # Warm the immutable shared model bundle once, not twice.
            self.prewarm_report = context['prewarm_report']
            require(self.prewarm_report['completed'] and self.prewarm_report['modes_warmed'] == list(range(12)),
                    'shared UE prewarm missing modes')
            self.gt_log = OG.GtTicketLogV3()
            self._run4_ready.set()

        def submit(self, **kwargs):
            require(kwargs['stream_id'] == source.stream_id, 'wrong collector/runtime source')
            binding = getattr(PO._LOCAL, 'binding', None)
            require(binding is not None and binding['owner'] is handoff.collector, 'missing source-owned pose')
            doc = binding['source'].document
            require(doc['frame_id'] == kwargs['frame_id'] and doc['stream_id'] == source.stream_id
                    and float(doc['timestamp']).hex() == float(kwargs['carla_timestamp']).hex(), 'pose/source drift')
            handoff.stage_before_action(self, **{k: kwargs[k] for k in
                ('frame_id', 'carla_timestamp', 'capture_timestamp_ns', 'ego_pose')})
            self._numeric_current_ego_pose = tuple(doc['ego_world'])
            try:
                result = retained_submit(self, **kwargs)
            except BaseException:
                handoff.after_submit(self, kwargs['frame_id'], {'sent': False})
                raise
            finally:
                self._numeric_current_ego_pose = None
            handoff.after_submit(self, kwargs['frame_id'], result)
            return result

        def close(self):
            from .configuration import write_new
            # Before inherited close releases the sender descriptor.
            write_new(Path(self.attempt_dir)/'SENDER_BINDING.json',self.sender.evidence())
            write_new(Path(self.attempt_dir)/'ACKED_SNR_STATE.json',self.pipeline.snr_rows)
            return super().close()
    return SourceRuntime


def collector_class(*, source, runtime_factory, ledger, handoff, pose_archive):
    """Retained sensor/AVO/Run4/drain layers with explicit source dependencies."""
    from rl_agent import ue_route_b_split_cell_adapter_v1 as P
    from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as Q
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
    from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge as GB, perception
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
    from rl_agent.splitfusion_m3_route_b_v20.route_drain import build_drain_layer
    from rl_agent.splitfusion_m3_route_b_v3.route_completion import FinishResult

    scene = GB.make_complete_scene_source(LG.build_timestamped_scene_source(Q.QualitySceneSnapshotSource,
                                                                           Q.QualityFrozenWorld))
    constructor = construct(P.PassiveSplitCollector.__init__, [
        ('from rl_agent.ue_map_install_feedback_v1 import InstallFeedbackLedger',
         'InstallFeedbackLedger = _m4_ledger_factory'),
        ('self.stream_id = f"ue288_{cell[\'cell_id\']}"', 'self.stream_id = _m4_stream')],
        LivePilotCellRuntime=runtime_factory, SceneSnapshotSource=scene,
        _m4_ledger_factory=lambda **kw: ledger, _m4_stream=source.stream_id)

    class SourceBase(P.PassiveSplitCollector):
        __init__ = constructor
        _process_token = bind(P.PassiveSplitCollector._process_token,
            actor_world_matrix=lambda actor: owned_matrix(actor),
            actor_world_inverse_matrix=lambda actor: owned_matrix(actor, inverse=True))
        def _spawn_sensors(self):
            perception.bind_collector_geometry(self)
            super()._spawn_sensors()

    class QualitySource(Q.QualityPassiveSplitCollector, SourceBase):
        pass

    class Owned(LG.build_live_gt_collector(QualitySource)):
        def __init__(self, **kwargs):
            self._pose_lock, self._numeric_radar_lock = threading.Lock(), threading.Lock()
            self._pose_sources, self._numeric_radar = collections.OrderedDict(), {}
            self._pose_fault = None
            super().__init__(**kwargs)
            handoff.catalogue.require_world(self.world)

        def wait_for_sensor_frame(self, frame_id, *, snapshot, timeout_s=2.):
            barrier = super().wait_for_sensor_frame(frame_id, snapshot=snapshot, timeout_s=timeout_s)
            owned = PO.capture_owned_pose(collector=self, frame_id=frame_id, snapshot=snapshot, barrier=barrier)
            with self._pose_lock:
                require(frame_id not in self._pose_sources, 'duplicate owned pose')
                self._pose_sources[frame_id] = owned
                while len(self._pose_sources) > 512:
                    self._pose_sources.popitem(last=False)
            pose_archive.add('OWNED_SOURCE', source=owned.document, source_sha256=owned.sha256)
            return barrier

        def _records_for(self, frame_id, timeout_s=.25):
            result = super()._records_for(frame_id, timeout_s)
            if result is not None:
                binding = getattr(PO._LOCAL, 'binding', None)
                require(binding is not None and binding['owner'] is self, 'foreign pose preparation')
                for image, role in ((result[0], 'camera'), (result[3], 'radar')):
                    require(int(image.frame) == frame_id == binding['source'].document['frame_id']
                            and image.timestamp == binding['source'].document['timestamp'], 'sensor time/frame drift')
                    PO._agree(image.transform.get_matrix(), binding['source'].matrix(role), 'preparation ' + role)
                binding['used'].add('sensor_records')
            return result

        def _process_token(self, token):
            import os
            from .configuration import CPU
            os.sched_setaffinity(0, {CPU['front'][source.ue_id]})
            with self._pose_lock:
                owned = self._pose_sources.pop(int(token['frame_id']), None)
            require(owned is not None, 'missing frame-bound pose')
            try:
                with PO._bind_preparation(self, owned, pose_archive):
                    result = super()._process_token(token)
                pose_archive.add('UE_PREPARATION_DONE', frame_id=token['frame_id'], stream_id=self.stream_id,
                                 sent=token['frame_id'] in self.sent_frames)
                return result
            except BaseException as error:
                self._pose_fault = repr(error)
                pose_archive.error(error, frame_id=token['frame_id'])
                raise

        def on_world_tick(self, frame_id, route_tick):
            require(self._pose_fault is None, 'owned pose failure: ' + str(self._pose_fault))
            require(self.sent < 3900, 'frame safety ceiling reached; not route completion')
            return super().on_world_tick(frame_id, route_tick)

        def _ground_truth(self, **kwargs):
            if kwargs.get('world') is not None:
                frame = int(kwargs['frame_id'])
                with self._quality_scene_lock:
                    self._quality_scenes.pop(frame, None)
                self._run4_gate.close(frame, 'NUMERIC_GT_LEGACY_BYPASS')
                return []
            return super()._ground_truth(**kwargs)

        def finish(self):
            return FinishResult(super().finish())

    class Collector(U.build_run4_collector_class(build_drain_layer(Owned))):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            handoff.attach(self)
            handoff.prestart_from_scene()

        def _run4_timed_build(self, real_build, **kwargs):
            result = super()._run4_timed_build(real_build, **kwargs)
            require(type(result) in (tuple, list) and len(result) == 3 and type(result[1]) is dict,
                    'actual radar preparation required')
            with self._numeric_radar_lock:
                self._numeric_radar[float(kwargs['frame_time_s'])] = result[1]
                require(len(self._numeric_radar) <= 64, 'radar ownership bound exceeded')
            return result
    Collector.__m4_binding__ = constructor.__m4_binding__
    return Collector


def owned_matrix(actor, inverse=False):
    import numpy as np
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
    from rl_agent import ue_route_b_split_cell_adapter_v1 as P
    binding = getattr(PO._LOCAL, 'binding', None)
    if binding is None:
        return (P.actor_world_inverse_matrix if inverse else P.actor_world_matrix)(actor)
    for role in ('camera', 'radar'):
        if actor is getattr(binding['owner'], role):
            value = binding['source'].matrix(role)
            binding['used'].add(role + ('_inverse' if inverse else '_matrix'))
            return np.linalg.inv(value) if inverse else value
    raise ValueError('unknown actor in owned preparation')


def bind_owned_materialize(pipeline):
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_m3_route_b_v3.pose_tolerance import agree_reconstructed
    original = pipeline.materialize
    def materialize(planned, **kwargs):
        binding = getattr(PO._LOCAL, 'binding', None)
        require(binding is not None, 'FrameContext outside owned-pose preparation')
        source, used = binding['source'], binding['used']
        doc = source.document
        require(kwargs['frame_id'] == doc['frame_id'] and pipeline.stream_id == doc['stream_id']
                and kwargs.get('carla_timestamp') == doc['timestamp'], 'materialize source mismatch')
        require({'sensor_records', 'camera_matrix', 'camera_inverse', 'radar_matrix', 'radar_inverse'} <= used,
                'materialize lacks exact projection poses')
        legacy = list(kwargs['ego_pose'])
        result = original(planned, **dict(kwargs, ego_pose=tuple(doc['ego_world'])))
        envelope, context = W.unpack_sfd4(result.wire)
        require(envelope == result.envelope and context.frame_id == doc['frame_id']
                and context.stream_id == doc['stream_id']
                and tuple(context.ego_world.values()) == tuple(doc['ego_world']), 'wire/source identity drift')
        delta = agree_reconstructed(PO.context_camera(context), source.matrix('camera'), 'wire/source camera')
        binding['archive'].add('UE_WIRE', gt_identity=dict(result.gt_identity), action_identity=dict(result.identity),
            context=PO.context_document(context), source_sha256=source.sha256, inherited_ego_world=legacy,
            max_abs_matrix_delta=delta, preparation_pose_reads=sorted(used))
        return result
    pipeline.materialize = materialize


def install_controller_observer(engine, handoff):
    """Attach timeout retirement to each controller, including future rollovers."""
    def observe(controller):
        original = controller.poll
        def poll(now):
            before = controller.current
            resolution = None if before is None else before.resolution
            result = original(now)
            handoff.on_controller_poll(before, resolution, now, controller.current)
            return result
        controller.poll = poll
    for controller in engine.controllers:
        observe(controller)
    original = engine._new_session
    def new_session():
        result = original()
        observe(engine.controller)
        return result
    engine._new_session = new_session
