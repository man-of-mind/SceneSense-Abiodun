"""Deferred action binding over the unchanged numeric producer.

The pinned ``NumericSourceProducer.compute`` takes no plan. The pinned
``ProducerOwner.compute`` nevertheless validates the full plan first, and the
runtime ``PREPARE_STAGED`` binds the action before computing. This module
splits that single step into:

1. ``prepare_unbound``: exact owned-source compute, numerical assembly and
   product admission on the action-neutral capture template (both selected
   capture fields null). History is only *staged*.
2. ``bind``: the full plan's checks from ``ProducerOwner.compute`` plus the
   unchanged ``ready.admit_payload`` / ``set_ready`` admission.
3. Unchanged ``confirm_send`` -> commit, ``publish_ready``, consumers.

Deferral is refused (fail closed) whenever the bound capture identity would
differ from the template, because identity enters source-synchronisation and
provenance evidence computed during step 1. A refused or cancelled frame is
cancelled through the pinned producer, which restores the exact pre-frame
history digest; speculative holds/fallbacks therefore never advance history.

All numerical functions, validators, commit/cancel and admission are imported
unchanged. Only orchestration is new.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import threading

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN, ready as R, composition as COM
from rl_agent.splitfusion_numeric_gt_producer_v1 import source as NS
from rl_agent.splitfusion_numeric_gt_runtime_v1 import producer as PR
from rl_agent.splitfusion_numeric_gt_runtime_v1 import staging as ST


class DeferralRefused(OWN.OwnershipError):
    """The selected plan cannot reuse the action-neutral computation."""


def is_template(capture):
    owned = C._json_owned(capture)
    C._capture(owned)
    return owned == ST.neutral_identity(owned)


class DeferredProducerOwner(OWN.ProducerOwner):
    """``ProducerOwner`` with compute split from plan binding."""

    def compute(self, *args, **kwargs):  # never reachable through the deferred API
        raise OWN.OwnershipError("deferred owner computes only through compute_unbound")

    def compute_unbound(self, sourcekwargs, *, reserved_bytes):
        self._check()
        if self._pending is not None:
            self._fail("another real producer transaction is pending")
        try:
            capture = C._json_owned(sourcekwargs["identity"])
            C._capture(capture)
            if capture != ST.neutral_identity(capture):
                raise OWN.OwnershipError("unbound computation requires the action-neutral capture template")
            frame_id = capture["frame_id"]
            if type(frame_id) is not int or frame_id <= self._last_frame:
                raise OWN.OwnershipError("regressing or reused producer frame")
            if type(reserved_bytes) is not int or reserved_bytes <= 0:
                raise OWN.OwnershipError("explicit frame payload reservation required")
            if len(self._frames) >= self._max_jobs or self._reserved + reserved_bytes > self._max_bytes:
                self._fail("producer retention capacity exceeded; no eviction", OWN.CapacityError)
        except BaseException as error:
            self._failure = self._failure or f"invalid unbound producer source: {error}"[:1024]
            raise
        self._sequence += 1
        token = OWN.ProducerToken(self._sequence, self._issuer)
        row = OWN._Frame(token, None, None, capture, reserved_bytes, context=None)
        self._frames[token.sequence] = row
        self._reserved += reserved_bytes
        self._pending = row
        try:
            # Identical aliasing/copy rules to ProducerOwner.compute.
            owned = dict(sourcekwargs)
            for name in ("camera_matrix", "camera_inverse", "radar_world_xyz"):
                value = np.asarray(owned[name])
                if value.dtype.hasobject:
                    raise OWN.OwnershipError("object source array is forbidden")
                owned[name] = np.frombuffer(value.tobytes(order="C"), value.dtype).reshape(value.shape)
            for name in ("identity", "dynamic_snapshot_evidence"):
                owned[name] = copy.deepcopy(owned[name])
            row.staged = self._producer.compute(**owned)
            if row.staged is not self._producer.staged:
                self._fail("producer returned a foreign staged transaction")
            self._account(row)
            self._last_frame = frame_id
            return token
        except BaseException as error:
            self._failure = self._failure or f"producer compute failed: {type(error).__name__}: {error}"[:1024]
            if row.staged is not None and row.producer_terminal is None:
                self._cancel_real(row)
            self._discard(row)
            raise

    def bind_plan(self, token, plan_identity_bytes):
        """The full-plan checks of ProducerOwner.compute, applied after compute.

        Raises DeferralRefused (without poisoning the owner) when the plan is a
        valid accepted reward plan whose capture differs from the template;
        any other defect is an ordinary owner failure.
        """
        row = self._frame(token)
        if row.plan is not None or row.products is None or row.ready is not None or row.producer_terminal is not None:
            self._fail("plan binding requires one unbound, assembled, non-terminal transaction")
        try:
            plan = C._parse(plan_identity_bytes)
            if C.canonical(plan) != plan_identity_bytes or not isinstance(plan, dict):
                raise OWN.OwnershipError("canonical full plan identity bytes required")
            capture, action = plan["capture"], plan["action"]
            C._capture(capture)
            if type(plan.get("frame_context")) is not dict:
                raise OWN.OwnershipError("full immutable plan must include its frame context")
            from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
            MP.validate_run4_identity(action)
            if action["reward_requested"] is not True or action["frame_id"] != capture["frame_id"]:
                raise OWN.OwnershipError("only accepted reward-requested frame plans are admitted")
            if action.get("frame_kind") != "POLICY_DECISION":
                raise OWN.OwnershipError("only policy decisions are admitted; holds/fallbacks never bind")
            if len(plan_identity_bytes) >= row.reservation:
                raise OWN.OwnershipError("explicit frame payload reservation required")
        except BaseException as error:
            self._failure = self._failure or f"invalid producer plan: {error}"[:1024]
            raise
        if C.canonical(capture) != C.canonical(row.capture):
            # Valid plan, but its selected capture is not the computed template.
            raise DeferralRefused("selected capture differs from the action-neutral computation")
        row.plan, row.action, row.context = plan_identity_bytes, action, plan["frame_context"]
        self._account(row)
        return plan


@dataclass(frozen=True, slots=True)
class UnboundHandle:
    sequence: int
    frame_id: int
    template_sha256: str
    _issuer: object


@dataclass
class _Unbound:
    handle: UnboundHandle
    owner_token: object
    payload: object
    template: dict
    clocks: dict | None
    filenames: dict | None
    prescore_filename: str | None
    rgb_bytes: bytes | None
    rgb_shape: tuple | None
    before_bytes: int
    begin: int
    neutral_ready: int


class DeferredProducerEngine(PR.ProducerEngine):
    """ProducerEngine whose prepare is split into prepare_unbound + bind.

    Every other method (confirm_send, publish_ready, export_live_source,
    notifications, offline archive, cancel, fail, snapshot, close) is the
    inherited, unchanged implementation.
    """

    def __init__(self, *, producer_args, run_binding, producer_binding, prediction_contract,
                 max_frames, max_retained_jobs=2, max_retained_bytes=2 * PR.RESERVATION,
                 reserved_bytes=PR.RESERVATION, offline_replay=False):
        # Identical to ProducerEngine.__init__ except for the owner class.
        PR._require(type(offline_replay) is bool, "explicit offline replay boolean required")
        PR._require(C.is_digest(run_binding) and C.is_digest(producer_binding), "real run/producer bindings required")
        PR._require(type(max_frames) is int and 1 <= max_frames <= 1000, "bounded frame count required")
        PR._require(type(reserved_bytes) is int and 0 < reserved_bytes <= max_retained_bytes,
                    "bounded per-frame storage reservation required")
        contract = C._parse(C.canonical(prediction_contract))
        args = dict(producer_args)
        source_max_frames = args.get("max_frames")
        PR._require(type(source_max_frames) is int and 1 <= source_max_frames <= 1000,
                    "explicit registered producer max_frames in [1,1000] required")
        PR._require(max_frames <= source_max_frames, "runtime frame budget exceeds registered producer capacity")
        producer = NS.NumericSourceProducer(**args)
        PR._require(producer.bound.binding_sha256 == producer_binding, "actual producer source binding differs")
        self._owner = DeferredProducerOwner(producer, max_retained_jobs=max_retained_jobs,
                                            max_retained_bytes=max_retained_bytes)
        self._thread = threading.current_thread()
        self._issuer = object()
        self._run_binding, self._producer_binding = run_binding, producer_binding
        self._prediction_contract = contract
        self._reservation, self._max_frames = reserved_bytes, max_frames
        self._source_max_frames = source_max_frames
        self._offline = offline_replay
        self._frames = {}
        self._completed_events = {}
        self._issued = self._retired = 0
        self._failure = None
        self._closed = False
        self._unbound = None
        self._unbound_sequence = 0
        self._unbound_issuer = object()
        self._deferral = dict(prepared_unbound=0, bound=0, cancelled_unbound=0, refused=0)

    def prepare(self, *args, **kwargs):
        raise PR.ProducerError("deferred engine prepares only through prepare_unbound + bind")

    def prepare_unbound(self, sourcekwargs, *, population, archive_clocks=None, payload_filenames=None,
                        prescore_filename=None, replay_observed_raw_ns=None):
        """Action-independent part of ProducerEngine.prepare, same checks and order."""
        self._check()
        PR._require(self._unbound is None, "one unbound producer transaction at a time")
        owner_token = None
        before_bytes = self._owner.snapshot()["payload_bytes"]
        begin = PR._raw()
        try:
            PR._require(self._issued < self._max_frames, "runtime frame budget exhausted")
            if not self._offline:
                PR._require(archive_clocks is None and replay_observed_raw_ns is None,
                            "recorded legacy clocks require explicit offline replay")
            if archive_clocks is not None:
                PR._require(type(archive_clocks) is dict and set(archive_clocks) == set(PR.AR.AVO_CLOCKS)
                    and all(type(x) is int and x > 0 for x in archive_clocks.values()),
                    "seven actual recorded AVO clocks required")
            clocks = None if archive_clocks is None else dict(archive_clocks)
            names = None if payload_filenames is None else dict(payload_filenames)
            if names is not None:
                PR._require(set(names) == {"depth_bgra", "rawsemantic"}, "exact AVO native filenames required")
                for name in names.values():
                    PR._basename(name)
                PR._basename(prescore_filename)
                PR._require(len(set((*names.values(), prescore_filename))) == 3, "duplicate AVO artifact filenames")
            else:
                PR._require(prescore_filename is None, "AVO filename declarations must be complete")
            template = C._json_owned(sourcekwargs["identity"])
            PR._require(is_template(template), "unbound preparation requires the action-neutral template")
            PR._require(C.canonical(population) == C.canonical(sourcekwargs["dynamic_snapshot_evidence"]),
                        "population differs from owned source kwargs")
            ids, digest = population["registered_population_ids"], population["population_sha256"]
            PR._require(type(ids) in (list, tuple) and all(type(i) is int for i in ids)
                and len(set(ids)) == len(ids) and C.is_digest(digest), "complete registered population required")
            if self._owner.population != frozenset(ids):
                self._owner.register_population(ids, digest, before_frame_id=template["frame_id"])
            rgb = sourcekwargs["rgb"].get("_rgb_image")
            rgb_bytes = rgb_shape = None
            if rgb is not None:
                rgb_shape = (int(rgb.height), int(rgb.width), 4)
                PR._require(rgb_shape == (720, 1280, 4) and type(rgb.frame) is int
                    and rgb.frame == template["frame_id"]
                    and float(rgb.timestamp) == float(sourcekwargs["rgb"]["timestamp"]),
                    "native RGB source identity/geometry differs")
                PR._require(isinstance(rgb.raw_data, (bytes, bytearray, memoryview))
                    and len(rgb.raw_data) == int(np.prod(rgb_shape)), "complete native RGB byte buffer required")
                PR._require(np.array_equal(np.asarray(rgb.transform.get_matrix()),
                    np.asarray(sourcekwargs["camera_matrix"])), "native RGB camera pose differs")
                rgb_bytes = bytes(rgb.raw_data)
            with self._observed_clock(replay_observed_raw_ns):
                owner_token = self._owner.compute_unbound(sourcekwargs, reserved_bytes=self._reservation)
            staged = self._owner._staged_for_assembly(owner_token)
            numeric, avo, payload = COM.assemble(staged, sourcekwargs)
            self._owner.stage_products(owner_token, target=numeric, avo=avo, payload=payload)
            self._unbound_sequence += 1
            handle = UnboundHandle(self._unbound_sequence, template["frame_id"],
                                   C.sha256(C.canonical(template)), self._unbound_issuer)
            self._unbound = _Unbound(handle, owner_token, payload, template, clocks, names, prescore_filename,
                                     rgb_bytes, rgb_shape, before_bytes, begin, PR._raw())
            self._deferral["prepared_unbound"] += 1
            return handle
        except BaseException as error:
            self._failure = self._failure or f"unbound prepare failed: {type(error).__name__}: {error}"[:1024]
            if owner_token is not None:
                try:
                    self._owner.cancel(owner_token, "runtime unbound preparation failed")
                except BaseException as cleanup_error:
                    raise error from cleanup_error
            raise

    def _take_unbound(self, handle):
        self._check(healthy=False)
        state = self._unbound
        if (type(handle) is not UnboundHandle or handle._issuer is not self._unbound_issuer
                or state is None or state.handle is not handle):
            self._fail("foreign, stale or already resolved unbound handle")
        return state

    def bind(self, handle, plan_identity_bytes, *, frame_context):
        """Bind the actual selected plan; same admission as ProducerEngine.prepare."""
        self._check()
        state = self._take_unbound(handle)
        try:
            plan = self._owner.bind_plan(state.owner_token, plan_identity_bytes)
        except DeferralRefused:
            self._owner.cancel(state.owner_token, "deferral refused: selected capture differs from template")
            self._unbound = None
            self._deferral["refused"] += 1
            raise
        except BaseException as error:
            self._failure = self._failure or f"plan binding failed: {error}"[:1024]
            raise
        try:
            target = R.admit_payload(state.payload, expected_capture_identity=state.template,
                expected_action_identity=plan["action"], frame_context=frame_context,
                plan_sha256=C.sha256(plan_identity_bytes), run_binding=self._run_binding,
                producer_binding=self._producer_binding, prediction_contract=self._prediction_contract)
            PR._require(self._owner.set_ready(state.owner_token, target) is None,
                        "prepare cannot commit without send confirmation")
            self._issued += 1
            token = PR.ProducerToken(self._issued, self._issuer)
            observed = PR._raw()
            prepared = PR.Prepared(token, target.identity_bytes, target.semantic_sha256, observed)
            metadata_size = len(PR._canonical(dict(clocks=state.clocks, filenames=state.filenames,
                                                   prescore_filename=state.prescore_filename)))
            row = PR._Frame(token, state.owner_token, plan_identity_bytes, prepared, state.clocks, state.filenames,
                state.prescore_filename, state.rgb_bytes, state.rgb_shape,
                self._owner.snapshot()["payload_bytes"] - state.before_bytes, metadata_size,
                events=dict(prepare_start_raw_ns=state.begin, neutral_ready_raw_ns=state.neutral_ready,
                            numerical_ready_raw_ns=observed))
            self._budget(row)
            self._frames[token.sequence] = row
            self._unbound = None
            self._deferral["bound"] += 1
            return prepared
        except BaseException as error:
            self._failure = self._failure or f"bound admission failed: {type(error).__name__}: {error}"[:1024]
            raise

    def cancel_unbound(self, handle, reason):
        """Hold, fallback, displaced or unsent before binding: exact history rollback."""
        state = self._take_unbound(handle)
        PR._require(type(reason) is str and bool(reason), "explicit cancellation reason required")
        result = self._owner.cancel(state.owner_token, reason)
        self._unbound = None
        self._deferral["cancelled_unbound"] += 1
        return result

    # ------------------------------------------------------------------ prototype retirement (live mode)
    def _receipt_binds(self, row, receipt, relative_prefix, files):
        from pathlib import PurePosixPath
        prefix = PurePosixPath(relative_prefix) if type(relative_prefix) is str else None
        PR._require(prefix is not None and relative_prefix and not prefix.is_absolute()
                    and str(prefix) == relative_prefix and all(p not in (".", "..") for p in prefix.parts),
                    "safe relative archive prefix required")
        capture = C._parse(row.prepared.identity_bytes)["capture_identity"]
        PR._require(type(receipt) is dict and receipt.get("status") == "COMPLETE" and receipt.get("durable") is True
                    and receipt.get("error") is None and C.canonical(receipt.get("identity")) == C.canonical(capture),
                    "actual complete durable producer receipt for this capture required")
        rows = {item["relative"]: item for item in receipt.get("written", []) if type(item) is dict}
        PR._require({n for n in rows if n.startswith(relative_prefix + "/")} == {f"{relative_prefix}/{n}" for n in files},
                    "producer receipt inventory differs")
        for name, raw in files.items():
            item = rows[f"{relative_prefix}/{name}"]
            PR._require(item.get("sha256") == C.sha256(raw) and item.get("bytes") == len(raw),
                        "durable receipt does not bind exact producer file: " + name)

    def complete_scored_durable(self, token, identity_bytes, semantic_sha256, *, files, receipt,
                                relative_prefix, scorer_proof):
        """Retire a scored live-mode frame after its result was handed out for emission and
        both producer evidence and scorer evidence are durable. No UE outcome is claimed."""
        row = self._notification(token, identity_bytes, semantic_sha256)
        PR._require(not self._offline and row.committed and row.scored and not row.feedback
                    and row.archive is None and row.live_unscored_terminal is None,
                    "prototype scored retirement requires a scored, published live-mode frame")
        proof = C._json_owned(scorer_proof)
        PR._require(type(proof) is dict and proof.get("status") == "DURABLE" and proof.get("scored") is True
                    and proof.get("ready_identity_sha256") == C.sha256(identity_bytes)
                    and proof.get("ready_semantic_sha256") == semantic_sha256
                    and C.is_digest(proof.get("archive_manifest_sha256")), "scorer durable proof differs")
        self._receipt_binds(row, receipt, relative_prefix, files)
        self._owner.confirm_feedback(row.owner_token)      # result handed to emission (not a UE receipt)
        row.feedback = True
        self._owner.complete_archive(row.owner_token)
        row.events["proto_scored_durable_raw_ns"] = PR._raw()
        self._retire(row, "PROTO_SCORED_DURABLE")
        return dict(status="RETIRED", outcome="PROTO_SCORED_DURABLE", ready_identity_sha256=C.sha256(identity_bytes))

    def prepare_proto_unscored(self, token, identity_bytes, semantic_sha256, reason):
        """Compact no-score evidence for a committed, published frame (missing prediction,
        scorer refusal or edge terminal). It never fabricates a score or feedback."""
        import re
        row = self._notification(token, identity_bytes, semantic_sha256)
        PR._require(row.committed and not row.scored and not row.feedback and row.archive is None,
                    "unscored terminal requires a published, unscored frame")
        PR._require(type(reason) is str and re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", reason) is not None,
                    "bounded unscored reason required")
        target = self._owner.ready_target(row.owner_token)
        terminal = dict(schema="splitfusion.exact_gt_service_proto.producer_unscored_terminal.v1",
            ready_identity=C._parse(identity_bytes), ready_identity_sha256=C.sha256(identity_bytes),
            ready_semantic_sha256=semantic_sha256, input_content_sha256=target.metadata["input_content_sha256"],
            history_after_sha256=target.metadata["history_after_sha256"], reason=reason, score_exists=False,
            quality_feedback_emitted=False, ue_outcome_observed=False,
            controller_terminal_basis="NO_QUALITY_FEEDBACK_EMITTED_BY_SERVICE")
        files = {"unscored_terminal.json": C.canonical(terminal)}
        row.events["proto_unscored_files"] = C.sha256(files["unscored_terminal.json"])
        self._proto_unscored = getattr(self, "_proto_unscored", {})
        self._proto_unscored[token.sequence] = files
        return files

    def complete_proto_unscored(self, token, identity_bytes, semantic_sha256, *, receipt, relative_prefix):
        row = self._notification(token, identity_bytes, semantic_sha256)
        files = getattr(self, "_proto_unscored", {}).get(token.sequence)
        PR._require(files is not None and row.committed and not row.scored, "prepared unscored terminal required")
        self._receipt_binds(row, receipt, relative_prefix, files)
        # Owner primitive for a committed no-score frame; history is NOT rolled back
        # (the frame was sent, so it belongs to accepted history).
        self._owner.resolve_live_unscored_terminal(row.owner_token, terminal="FEEDBACK_TIMEOUT")
        self._proto_unscored.pop(token.sequence)
        self._retire(row, "PROTO_UNSCORED_DURABLE")
        return dict(status="RETIRED", outcome="PROTO_UNSCORED_DURABLE", ready_identity_sha256=C.sha256(identity_bytes))

    def snapshot(self):
        value = super().snapshot()
        value["deferral"] = dict(self._deferral, unbound_pending=self._unbound is not None)
        return value

    def close(self, *, abandon=False):
        if self._unbound is not None:
            state, self._unbound = self._unbound, None
            if not abandon:
                self._failure = self._failure or "closed with an unbound transaction"
            try:
                self._owner.cancel(state.owner_token, "closed while unbound")
            except BaseException:
                if not abandon:
                    raise
        return super().close(abandon=abandon)
