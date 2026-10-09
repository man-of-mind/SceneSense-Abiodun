"""Prospective, opt-in numerical-GT edge composition for short live trials.

The retained SFD4 compute and map publisher are not changed.  This module
replaces only the legacy evaluator class inside an explicitly scoped edge
entry. TARGET is an already-admitted numerical ReadyTarget packet from the UE;
TERMINAL carries actual owner-stamped events and the producer's source export.
Neither a missing target nor a missing terminal is converted into a score.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import queue
import socket
import stat
import struct
import threading
import time

import numpy as np

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import reward_hold_controller_v2 as H
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
from rl_agent.splitfusion_live_gt_handoff_v1 import broker as B
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
from rl_agent.splitfusion_minimal_quality_scorer_v1 import scorer as MS
from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R

from . import ipc as I, live_receipt as LR, scorer as SC
from .live_clock import ASYNC_HANDOFF_ID, LiveClockLedger, read_boot_id


SCHEMA = "splitfusion.numeric_gt.edge_input.v1"
OBSERVED_EVENTS_SCHEMA = "splitfusion.numeric_gt.edge_observed_events.v1"
ASYNC_OBSERVED_EVENTS_SCHEMA = "splitfusion.numeric_gt.edge_observed_events.v2"
ACK_SCHEMA = "splitfusion.numeric_gt.edge_input_ack.v1"
MAX_TARGETS = 1000
MAX_INBOX_BYTES = 256 * 1024 * 1024


class NumericEdgeError(RuntimeError):
    pass


def _require(value, message):
    if not value:
        raise NumericEdgeError(message)


def _expected(identity_bytes):
    _require(type(identity_bytes) is bytes and len(identity_bytes) <= C.MAX_METADATA_BYTES,
             "canonical full target identity required")
    value = C._parse(identity_bytes)
    _require(C.canonical(value) == identity_bytes and type(value) is dict and set(value) == {
        "capture_identity", "action_identity", "plan_sha256", "run_binding", "producer_binding",
        "frame_context", "prediction_contract"}, "exact full target identity required")
    expected = dict(expected_capture_identity=value["capture_identity"],
        expected_action_identity=value["action_identity"], frame_context=value["frame_context"],
        plan_sha256=value["plan_sha256"], run_binding=value["run_binding"],
        producer_binding=value["producer_binding"], prediction_contract=value["prediction_contract"])
    _require(SC.registration_key(expected) == identity_bytes, "target registration key drift")
    return expected


def _ticket_matches(identity_bytes, ticket):
    value = C._parse(identity_bytes)
    return (C.canonical(value["capture_identity"]) == C.canonical(ticket.gt_identity)
        and C.canonical(value["action_identity"]) == C.canonical(MP.run4_identity(ticket.envelope))
        and C.canonical(value["frame_context"]) == C.canonical(PO.context_document(ticket.context)))


def _observed_handoff(observed):
    """Select an explicit owner clock contract, never infer V2 from timing."""
    _require(type(observed) is dict and type(observed.get("events")) is list
             and 1 <= len(observed["events"]) <= 16
             and all(type(item) is dict for item in observed["events"]),
             "bounded owner-stamped UE/collector events required")
    names = [(item.get("role"), item.get("name")) for item in observed["events"]]
    if observed.get("schema") == OBSERVED_EVENTS_SCHEMA:
        _require(set(observed) in ({"schema", "events"},
                  {"schema", "events", "feedback"})
                 and ("ue", "source_owned") not in names,
                 "V1 UE event envelope contains async source claim")
        return None
    _require(observed.get("schema") == ASYNC_OBSERVED_EVENTS_SCHEMA
             and set(observed) in ({"schema", "handoff_id", "events"},
                                   {"schema", "handoff_id", "events", "feedback"})
             and observed.get("handoff_id") == ASYNC_HANDOFF_ID,
             "unknown or crossed observed handoff version")
    required = (("collector", "owned_tick_snapshot"), ("ue", "source_owned"),
                ("ue", "action_open"), ("ue", "last_datagram_sent"),
                ("producer", "source_stage_received"))
    _require(all(names.count(key) == 1 for key in required),
             "V2 owner events lack exact source/action/send/stage stamps")
    selected = {key: observed["events"][names.index(key)] for key in required}
    _require(all(set(item) == {"role", "name", "raw_ns", "pid", "boot_id"}
                 and type(item["raw_ns"]) is int and item["raw_ns"] > 0
                 and type(item["pid"]) is int and item["pid"] > 0
                 and type(item["boot_id"]) is str and bool(item["boot_id"])
                 for item in selected.values()),
             "V2 owner events require actual bounded RAW stamps")
    tick, owned, opened, sent, staged = (selected[key] for key in required)
    _require((owned["boot_id"], owned["pid"]) == (opened["boot_id"], opened["pid"])
             == (sent["boot_id"], sent["pid"]),
             "V2 UE source/action/send process ownership drift")
    _require(tick["boot_id"] == owned["boot_id"] == staged["boot_id"],
             "V2 after-send source claim requires one observed Linux boot")
    _require(tick["raw_ns"] <= owned["raw_ns"], "V2 source predates owned tick")
    _require(owned["raw_ns"] <= opened["raw_ns"] <= sent["raw_ns"],
             "V2 source was not owned before action-open and send")
    _require(sent["raw_ns"] <= staged["raw_ns"],
             "V2 producer stage was not observed after actual send")
    return ASYNC_HANDOFF_ID


def _peer(connection):
    pid, uid, gid = struct.unpack("=iII", connection.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("=iII")))
    _require(pid >= 0, "invalid publisher PID namespace credential")
    return pid, uid, gid


def _socket_path(path):
    path = Path(path).absolute()
    _require(path.name == "numeric_gt.sock" and ".." not in path.parts
             and path.parent.is_dir() and not path.parent.is_symlink(),
             "existing real numeric edge evidence parent required")
    return path


class NumericEdgeInbox:
    """Bounded same-UID local receiver; no scoring or filesystem archive work."""

    def __init__(self, socket_path, *, max_frames, max_bytes=MAX_INBOX_BYTES,
                 publisher_uid=None, publisher_gid=None):
        self.path = _socket_path(socket_path)
        _require(type(max_frames) is int and 0 < max_frames <= MAX_TARGETS
                 and type(max_bytes) is int and 0 < max_bytes <= MAX_INBOX_BYTES,
                 "bounded edge inbox capacity required")
        self.max_frames, self.max_bytes = max_frames, max_bytes
        self.publisher_uid = os.geteuid() if publisher_uid is None else publisher_uid
        self.publisher_gid = os.getegid() if publisher_gid is None else publisher_gid
        _require(type(self.publisher_uid) is int and type(self.publisher_gid) is int
                 and self.publisher_uid >= 0 and self.publisher_gid >= 0,
                 "explicit publisher credentials required")
        self._rows = OrderedDict()
        self._bytes = 0
        self._lock = threading.Condition()
        self._listener = self._thread = None
        self._socket_identity = None
        self._stop = False
        self._failure = None
        self._rejected = 0
        self._admitted = 0
        self._results = {}
        self._unscored_files = {}
        # Edge admission and asynchronous GT production are independent. A
        # reward frame can be superseded before its numerical TARGET exists.
        # Retain that real terminal, not a fabricated score or a timeout guess.
        self._edge_terminals = OrderedDict()
        self._action_targets = {}
        self._early_terminals = 0
        self._joined_early_terminals = 0
        self._processing = {}
        self._processing_expired = set()
        self._admission_contract = None
        self._admission_expiries = {}

    @staticmethod
    def _publish(socket_path, value, timeout_s):
        _require(type(timeout_s) in (int, float) and 0 < timeout_s <= 30,
                 "bounded target publication timeout required")
        path = _socket_path(socket_path)
        deadline = time.monotonic() + timeout_s
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout_s)
            with B._address(path) as address:
                connection.connect(address)
            I.send(connection, value, deadline=deadline)
            reply = I.receive(connection, deadline=deadline)
        _require(type(reply) is dict and reply.get("schema") == ACK_SCHEMA
                 and reply.get("status") == "ADMITTED"
                 and reply.get("identity_sha256") == C.sha256(value["identity_bytes"]),
                 "numeric target publication was not admitted: " + str(reply)[:250])
        return reply

    @staticmethod
    def publish_target(socket_path, *, identity_bytes: bytes, packet: bytes,
                       semantic_sha256: str, timeout_s=3.):
        return NumericEdgeInbox._publish(socket_path, dict(schema=SCHEMA, op="TARGET",
            identity_bytes=identity_bytes, packet=packet,
            semantic_sha256=semantic_sha256), timeout_s)

    @staticmethod
    def publish_terminal(socket_path, *, identity_bytes: bytes, ue_events_bytes: bytes,
                         source_export: dict, timeout_s=3.):
        return NumericEdgeInbox._publish(socket_path, dict(schema=SCHEMA, op="TERMINAL",
            identity_bytes=identity_bytes, ue_events_bytes=ue_events_bytes,
            source_export=source_export), timeout_s)

    @staticmethod
    def publish_unscored_files(socket_path, *, identity_bytes: bytes, files: dict,
                               timeout_s=3.):
        return NumericEdgeInbox._publish(socket_path, dict(schema=SCHEMA, op="UNSCORED_FILES",
            identity_bytes=identity_bytes, files=files), timeout_s)

    @staticmethod
    def await_completion(socket_path, *, identity_bytes: bytes, timeout_s=30.):
        _expected(identity_bytes)
        _require(type(timeout_s) in (int, float) and 0 < timeout_s <= 120,
                 "bounded completion wait required")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            path = _socket_path(socket_path)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(min(3., max(.1, deadline-time.monotonic())))
                with B._address(path) as address:
                    connection.connect(address)
                I.send(connection, dict(schema=SCHEMA, op="GET_RESULT",
                    identity_bytes=identity_bytes), deadline=time.monotonic()+3.)
                reply = I.receive(connection, deadline=time.monotonic()+3.)
            _require(type(reply) is dict and reply.get("schema") == ACK_SCHEMA
                     and reply.get("identity_sha256") == C.sha256(identity_bytes),
                     "foreign/rejected edge completion reply: " + str(reply)[:500])
            if reply.get("status") != "PENDING":
                _require(reply.get("status") in ("SCORED_DURABLE", "UNSCORED_REQUIRED",
                         "UNSCORED_DURABLE", "FAILED"), "unknown edge completion status")
                return reply
            time.sleep(min(.02, max(0., deadline-time.monotonic())))
        raise NumericEdgeError("bounded edge completion wait expired")

    def _admit(self, value):
        _require(type(value) is dict and value.get("schema") == SCHEMA
                 and value.get("op") in ("TARGET", "TERMINAL", "GET_RESULT", "UNSCORED_FILES"),
                 "unknown numerical target message")
        key = value.get("identity_bytes")
        expected = _expected(key)
        with self._lock:
            _require(self._failure is None and not self._stop, "edge target inbox failed or closed")
            if value["op"] == "GET_RESULT":
                _require(set(value) == {"schema", "op", "identity_bytes"}
                         and key in self._rows, "result request lacks admitted target")
                result = self._results.get(key, {})
                if result.get("status") == "UNSCORED_REQUIRED" and key in self._unscored_files:
                    return dict(schema=ACK_SCHEMA, status="PENDING",
                                identity_sha256=C.sha256(key))
                return dict(schema=ACK_SCHEMA, status=result.get("status", "PENDING"),
                    identity_sha256=C.sha256(key), **{name: item for name, item in
                    result.items() if name not in ("status", "identity_sha256")})
            if value["op"] == "UNSCORED_FILES":
                _require(set(value) == {"schema", "op", "identity_bytes", "files"}
                         and self._results.get(key, {}).get("status") == "UNSCORED_REQUIRED"
                         and key not in self._unscored_files,
                         "unscored file handoff precedes observed terminal or repeats")
                files = value["files"]
                _require(type(files) is dict and set(files) == {"live_clock.json", "unscored_terminal.json"}
                         and all(type(raw) is bytes and 0 < len(raw) <= C.MAX_METADATA_BYTES
                                 for raw in files.values())
                         and files["live_clock.json"] == self._results[key]["clock_descriptor_bytes"],
                         "exact bounded producer unscored terminal files required")
                self._unscored_files[key] = dict(files)
                self._lock.notify_all()
            elif value["op"] == "TARGET":
                _require(set(value) == {"schema", "op", "identity_bytes", "packet", "semantic_sha256"},
                         "exact TARGET fields required")
                _require(key not in self._rows and len(self._rows) < self.max_frames,
                         "duplicate target or frame limit")
                packet, semantic = value["packet"], value["semantic_sha256"]
                _require(type(packet) is bytes and C.is_digest(semantic)
                         and self._bytes + len(packet) <= self.max_bytes,
                         "bounded immutable target packet required")
                target = R.decode_target(packet, **SC._expected(expected))
                _require(target.identity_bytes == key and target.semantic_sha256 == semantic,
                         "target wire identity/semantic differs")
                action = C.canonical(expected["expected_action_identity"])
                _require(action not in self._action_targets,
                         "ambiguous full target bindings for one action")
                edge_terminal = self._edge_terminals.get(action)
                if edge_terminal is not None:
                    self._check_terminal_capture(edge_terminal, expected)
                processing = self._processing.get(action)
                if processing is not None:
                    _require(C.canonical(processing) == C.canonical(
                        expected["expected_capture_identity"]),
                        "processing capture differs from numerical target")
                self._rows[key] = dict(packet=packet, semantic_sha256=semantic,
                    expected=expected, terminal=None, target_claimed=False,
                    edge_terminal=edge_terminal)
                self._action_targets[action] = key
                if edge_terminal is not None:
                    self._joined_early_terminals += 1
                self._bytes += len(packet)
            else:
                _require(set(value) == {"schema", "op", "identity_bytes", "ue_events_bytes",
                                         "source_export"}, "exact TERMINAL fields required")
                row = self._rows.get(key)
                _require(row is not None and row["terminal"] is None,
                         "terminal precedes target or repeats one frame")
                events = value["ue_events_bytes"]
                export = value["source_export"]
                _require(type(events) is bytes and 0 < len(events) <= C.MAX_METADATA_BYTES
                         and type(export) is dict and set(export) == {
                             "schema", "identity_bytes", "semantic_sha256", "input_metadata_bytes",
                             "source_synchronization_bytes", "numerical_avo_bytes",
                             "producer_events_bytes", "payload_bytes"},
                         "exact owner event/source export required")
                _require(export["identity_bytes"] == key
                         and export["semantic_sha256"] == row["semantic_sha256"]
                         and export["schema"] == "splitfusion.numeric_gt.live_source_export.v1"
                         and all(type(export[name]) is bytes and len(export[name]) <= C.MAX_METADATA_BYTES
                                 for name in ("input_metadata_bytes", "source_synchronization_bytes",
                                              "numerical_avo_bytes", "producer_events_bytes")),
                         "source export belongs to another target or exceeds bound")
                complete_bytes = len(events) + sum(len(export[name]) for name in (
                    "input_metadata_bytes", "source_synchronization_bytes",
                    "numerical_avo_bytes", "producer_events_bytes"))
                _require(type(export["payload_bytes"]) is int
                         and export["payload_bytes"] == len(key) + complete_bytes - len(events),
                         "producer source byte inventory differs")
                _require(self._bytes + complete_bytes <= self.max_bytes,
                         "edge terminal/source evidence capacity exceeded")
                observed = C._parse(events)
                handoff_id = _observed_handoff(observed)
                if "feedback" in observed:
                    feedback = observed["feedback"]
                    _require(type(feedback) is dict and set(feedback) == {
                        "packet_sha256", "kind", "reason", "q_perc"}
                        and C.is_digest(feedback["packet_sha256"])
                        and type(feedback["kind"]) is str and type(feedback["reason"]) is str,
                        "actual received feedback digest/kind/reason required")
                row["terminal"] = dict(ue_events=observed["events"], handoff_id=handoff_id,
                    feedback=observed.get("feedback"), source_export=export)
                self._bytes += complete_bytes
            self._admitted += 1
            self._lock.notify_all()
        return dict(schema=ACK_SCHEMA, status="ADMITTED", op=value["op"],
                    identity_sha256=C.sha256(key))

    def start(self):
        _require(self._listener is None and not self.path.exists(), "numeric socket must be create-only")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            with B._address(self.path) as address:
                listener.bind(address)
            observed = self.path.lstat()
            _require(stat.S_ISSOCK(observed.st_mode), "new numerical socket is not a socket")
            self._socket_identity = (observed.st_dev, observed.st_ino)
            descriptor = os.open(self.path, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                pinned = os.fstat(descriptor)
                _require(stat.S_ISSOCK(pinned.st_mode)
                         and (pinned.st_dev, pinned.st_ino) == self._socket_identity,
                         "numeric socket inode changed during owner assignment")
                handle = f"/proc/self/fd/{descriptor}"
                if (pinned.st_uid, pinned.st_gid) != (self.publisher_uid, self.publisher_gid):
                    os.chown(handle, self.publisher_uid, self.publisher_gid)
                os.chmod(handle, 0o600)
            finally:
                os.close(descriptor)
            secured = self.path.lstat()
            _require(stat.S_ISSOCK(secured.st_mode)
                     and (secured.st_dev, secured.st_ino) == self._socket_identity
                     and (secured.st_uid, secured.st_gid) ==
                         (self.publisher_uid, self.publisher_gid)
                     and stat.S_IMODE(secured.st_mode) == 0o600,
                     "numeric socket owner/mode/inode differs after setup")
            listener.listen(8)
            listener.settimeout(.1)
        except BaseException:
            listener.close()
            raise
        self._listener = listener
        self._thread = threading.Thread(target=self._serve, name="numeric-edge-inbox", daemon=True)
        self._thread.start()

    def _serve(self):
        while True:
            with self._lock:
                if self._stop:
                    break
            try:
                connection, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                try:
                    _, uid, gid = _peer(connection)
                    _require((uid, gid) == (self.publisher_uid, self.publisher_gid),
                             "foreign numeric publisher credentials")
                    value = I.receive(connection, deadline=time.monotonic() + 3.)
                    answer = self._admit(value)
                    I.send(connection, answer, deadline=time.monotonic() + 3.)
                except Exception as error:
                    with self._lock:
                        self._rejected += 1
                        self._failure = self._failure or f"{type(error).__name__}: {error}"[:512]
                        self._lock.notify_all()
                    try:
                        I.send(connection, dict(schema=ACK_SCHEMA, status="REJECTED",
                            error=str(error)[:300]), deadline=time.monotonic() + .2)
                    except Exception:
                        pass

    def matching_target(self, ticket, *, claim=False):
        with self._lock:
            _require(self._failure is None, self._failure or "numeric inbox failed")
            matches = [(key, row) for key, row in self._rows.items() if _ticket_matches(key, ticket)]
            _require(len(matches) <= 1, "ambiguous ready target for one edge ticket")
            if not matches:
                return None
            key, row = matches[0]
            _require(row["edge_terminal"] is None,
                     "evaluation conflicts with observed no-score edge terminal")
            _require(key not in self._admission_expiries,
                     "evaluation follows closed edge admission window")
            if claim:
                _require(not row["target_claimed"] and key not in self._results,
                         "numerical target already claimed or completed")
                row["target_claimed"] = True
            return key, row

    def wait_terminal(self, key, timeout_s):
        deadline = time.monotonic() + timeout_s
        with self._lock:
            while True:
                _require(self._failure is None, self._failure or "numeric inbox failed")
                row = self._rows.get(key)
                _require(row is not None, "archive requested for absent target")
                if row["terminal"] is not None:
                    return row["terminal"]
                left = deadline - time.monotonic()
                _require(left > 0, "observed UE terminal missing for scored target")
                self._lock.wait(left)

    def terminal_if_ready(self, key):
        with self._lock:
            row = self._rows.get(key)
            return None if row is None else row["terminal"]

    def result(self, key, value):
        with self._lock:
            _require(key in self._rows and key not in self._results,
                     "one final result per admitted target required")
            self._results[key] = dict(value)
            self._lock.notify_all()

    def replace_unscored_result(self, key, value):
        with self._lock:
            _require(self._results.get(key, {}).get("status") == "UNSCORED_REQUIRED",
                     "unscored final result requires prior observed request")
            self._results[key] = dict(value)
            self._lock.notify_all()

    def unscored_files_if_ready(self, key):
        with self._lock:
            return self._unscored_files.get(key)

    def rows_snapshot(self):
        with self._lock:
            return [(key, dict(row), self._results.get(key))
                    for key, row in self._rows.items()]

    @staticmethod
    def _check_terminal_capture(document, expected):
        capture = expected["expected_capture_identity"]
        _require(all(document.get(name) == capture[name] for name in (
            "run_id", "cell_id", "stream_id", "frame_id", "capture_timestamp_ns")),
            "edge terminal capture differs from full admitted target")

    def bind_processing_horizon(self, *, run_id, cell_id, ack_timeout_s):
        """Bind the processor's existing capture deadline, never the RL clock."""
        _require(type(ack_timeout_s) in (float, int) and 0 < ack_timeout_s <= 30
                 and all(type(x) is str and x for x in (run_id, cell_id)),
                 "actual bounded edge processing horizon required")
        binding = dict(run_id=run_id, cell_id=cell_id, ack_timeout_s=float(ack_timeout_s))
        with self._lock:
            _require(self._admission_contract in (None, binding),
                     "processing horizon changed within edge session")
            self._admission_contract = binding

    def reserve_processing(self, capture, action):
        """Atomically own a verified reward frame before compute can start.

        The existing before-decode check occurs outside this lock. Repeating
        the same check here closes the race with no-ticket archival, including
        a receiver/compute thread descheduled between that check and process().
        """
        from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as base
        MP.validate_run4_identity(action)
        identity = C.canonical(action)
        with self._lock:
            binding = self._admission_contract
            _require(binding is not None and capture["run_id"] == binding["run_id"]
                     and capture["cell_id"] == binding["cell_id"],
                     "verified frame belongs to another processing session")
            key = self._action_targets.get(identity)
            now = time.time()
            if key in self._admission_expiries:
                raise base.DeadlineExpired(base.EDGE_STAGE_BEFORE_DECODE,
                    capture_timestamp_ns=capture["capture_timestamp_ns"],
                    age_ms=(now-capture["capture_timestamp_ns"]/1e9)*1e3)
            base.check_deadline(base.EDGE_STAGE_BEFORE_DECODE, capture["capture_timestamp_ns"],
                                binding["ack_timeout_s"], now_s=now)
            if not action["reward_requested"]:
                return
            _require(identity not in self._processing and identity not in self._edge_terminals
                     and len(self._processing) < self.max_frames,
                     "duplicate, terminated or excess processing admission")
            if key is not None:
                _require(C.canonical(capture) == C.canonical(
                    self._rows[key]["expected"]["expected_capture_identity"]),
                    "verified processing capture differs from target")
            self._processing[identity] = C._json_owned(capture)

    def observe_evaluation_submission(self, ticket):
        """Retain queued/in-flight evaluator ownership even before TARGET."""
        identity = C.canonical(MP.run4_identity(ticket.envelope))
        with self._lock:
            key = self._action_targets.get(identity)
            _require(key not in self._admission_expiries and identity not in self._edge_terminals,
                     "evaluation submitted for closed/no-score frame")
            _require(identity not in self._processing_expired,
                     "evaluation submitted after processor reported deadline failure")
            capture = C._json_owned(ticket.gt_identity)
            prior = self._processing.get(identity)
            _require(prior is None or C.canonical(prior) == C.canonical(capture),
                     "evaluation capture differs from processor ownership")
            _require(prior is not None or len(self._processing) < self.max_frames,
                     "bounded evaluation admission capacity exceeded")
            if self._admission_contract is not None:
                _require(prior is not None, "live evaluation lacks verified processor reservation")
            self._processing[identity] = capture

    def processing_expired_before_evaluation(self, capture, action):
        """Called only on a real retained DeadlineExpired from process()."""
        identity = C.canonical(action)
        with self._lock:
            _require(identity in self._processing
                     and C.canonical(self._processing[identity]) == C.canonical(capture),
                     "processing expiry lacks exact reserved capture")
            key = self._action_targets.get(identity)
            _require(key is None or (not self._rows[key]["target_claimed"]
                                    and key not in self._results),
                     "processing expiry conflicts with scoring ownership")
            self._processing_expired.add(identity)

    def fail_processing(self, error):
        # An unexpected decoder/tail failure is not a policy timeout or an
        # unscored expiry. Surface it to the coordinator rather than orphaning
        # a reserved slot and failing two minutes later during CLOSE.
        with self._lock:
            self._failure = self._failure or f"processing: {type(error).__name__}: {error}"[:512]
            self._lock.notify_all()

    def close_unadmitted_after_horizon(self, key):
        """Prove no scoring owner exists after the unchanged admission window.

        This handles partial or wholly lost uplinks without guessing that a
        170-ms UE timeout means no score exists. Already-admitted processing,
        pending evaluation and late scoring are never cancelled by this path.
        """
        with self._lock:
            if key in self._admission_expiries:
                return self._admission_expiries[key]
            row = self._rows[key]
            binding = self._admission_contract
            terminal = row["terminal"]
            if binding is None or terminal is None or row["edge_terminal"] is not None:
                return None
            capture = row["expected"]["expected_capture_identity"]
            _require(capture["run_id"] == binding["run_id"]
                     and capture["cell_id"] == binding["cell_id"],
                     "target does not belong to active edge admission contract")
            identity = C.canonical(row["expected"]["expected_action_identity"])
            if (identity in self._processing or row["target_claimed"] or key in self._results
                    or terminal.get("feedback") is not None
                    or not any(item["role"] == "ue" and item["name"] == "feedback_timeout"
                               for item in terminal["ue_events"])):
                return None
            now = time.time()
            limit = capture["capture_timestamp_ns"]/1e9 + binding["ack_timeout_s"]
            if now <= limit:  # Same inclusive wall-clock admission as the retained edge.
                return None
            proof = dict(schema="splitfusion.numeric_gt.no_processing_after_horizon.v1",
                ready_identity_sha256=C.sha256(key), ready_identity=C._parse(key),
                semantic_sha256=row["semantic_sha256"], capture_deadline_wall_s=limit,
                observed_wall_s=now, observed_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                pid=os.getpid(), boot_id=read_boot_id(), processing_horizon=binding,
                verified_processing_admitted=False, evaluation_submitted=False,
                score_exists=False, future_processing_refused=True,
                reason="NO_VERIFIED_PROCESSING_BEFORE_EXISTING_EDGE_HORIZON")
            self._admission_expiries[key] = proof
            self._lock.notify_all()
            return proof

    def observe_edge_terminal(self, document):
        """Record a real Phase-6 terminal, never infer supersession from time."""
        MP.validate_run4_edge_terminal(document)
        if not document["run4_identity"]["reward_requested"]:
            # Unlike absence from _rows, the transmitted envelope actually
            # proves that a hold/fallback has no numerical reward target.
            return False
        _require(document.get("frame_id") == document["run4_identity"]["frame_id"]
                 and type(document.get("capture_timestamp_ns")) is int
                 and document["capture_timestamp_ns"] > 0
                 and all(type(document.get(name)) is str and bool(document[name])
                         for name in ("run_id", "cell_id", "stream_id")),
                 "edge terminal lacks exact capture routing identity")
        encoded = C.canonical(document)
        _require(len(encoded) <= C.MAX_METADATA_BYTES,
                 "bounded edge terminal document required")
        document = C._parse(encoded)  # Own the producer's mutable document.
        identity = C.canonical(document["run4_identity"])
        with self._lock:
            _require(self._failure is None and not self._stop, "edge target inbox failed or closed")
            _require(identity not in self._edge_terminals,
                     "duplicate or conflicting edge terminal for numeric target")
            _require(identity not in self._processing or identity in self._processing_expired,
                     "edge terminal conflicts with reserved processing/evaluation owner")
            if identity in self._processing:
                self._check_terminal_capture(document,
                    {"expected_capture_identity": self._processing[identity]})
            _require(len(self._edge_terminals) < self.max_frames
                     and self._bytes + len(encoded) <= self.max_bytes,
                     "edge terminal frame/byte capacity exceeded")
            key = self._action_targets.get(identity)
            row = None if key is None else self._rows[key]
            if row is not None:
                self._check_terminal_capture(document, row["expected"])
                _require(not row["target_claimed"] and (key not in self._results
                         or key in self._admission_expiries),
                         "edge terminal conflicts with claimed or completed score")
            self._edge_terminals[identity] = document
            self._bytes += len(encoded)
            if row is None:
                self._early_terminals += 1
            else:
                row["edge_terminal"] = document
            self._lock.notify_all()
            return True

    def close(self):
        with self._lock:
            self._stop = True
            self._lock.notify_all()
        if self._listener is not None:
            self._listener.close()
        if self._thread is not None:
            self._thread.join(timeout=2.)
        try:
            if self.path.exists():
                current = self.path.lstat()
                _require(stat.S_ISSOCK(current.st_mode)
                         and (current.st_dev, current.st_ino) == self._socket_identity,
                         "numeric socket inode changed before teardown")
                self.path.unlink()
        except OSError as error:
            self._failure = self._failure or f"socket unlink: {error}"
        with self._lock:
            return dict(schema=SCHEMA, admitted=self._admitted, rejected=self._rejected,
                targets=len(self._rows), terminals=sum(row["terminal"] is not None
                    for row in self._rows.values()), pending_bytes=self._bytes,
                edge_terminals=len(self._edge_terminals),
                edge_terminals_before_target=self._early_terminals,
                early_edge_terminals_joined=self._joined_early_terminals,
                edge_terminals_without_target=sum(action not in self._action_targets
                    for action in self._edge_terminals),
                processing_admissions=len(self._processing),
                processing_expired_before_evaluation=len(self._processing_expired),
                admission_expiries=list(self._admission_expiries.values()),
                worker_alive=self._thread is not None and self._thread.is_alive(),
                failure=self._failure)


def prediction_for_ticket(ticket):
    """Use the real native output and the retained prediction-digest rule."""
    labels = ticket.predicted_mask
    _require(type(labels) is np.ndarray and labels.dtype == np.uint8
             and labels.shape == (720, 1280) and labels.flags.c_contiguous,
             "real 720x1280 native UINT8 label map required")
    records = C._json_owned(list(ticket.records))
    raw = labels.tobytes(order="C")
    action = MP.run4_identity(ticket.envelope)
    identity = dict(gt_identity=C._json_owned(ticket.gt_identity), action_identity=action,
        capture_timestamp_ns=ticket.context.capture_timestamp_ns,
        prediction_digest=hashlib.sha256(raw + MS.canonical(records)).hexdigest())
    return dict(identity=identity, labels=raw, shape=[720, 1280], records=records)


def _observed_event(identity_bytes, semantic_sha256, item):
    _require(type(item) is dict and set(item) == {"role", "name", "raw_ns", "pid", "boot_id"},
             "exact owner event primitive required")
    return dict(item, clock_domain="CLOCK_MONOTONIC_RAW",
        ready_identity=C._parse(identity_bytes), ready_identity_sha256=C.sha256(identity_bytes),
        semantic_sha256=semantic_sha256)


def archive_clock_events(identity_bytes, semantic_sha256, terminal, emitted_event):
    """Normalize actual owner stamps; never infer or backdate a missing event."""
    export = terminal["source_export"]
    producer = C._parse(export["producer_events_bytes"])
    _require(type(producer) is dict and producer.get("role") == "producer"
             and producer.get("ready_identity_sha256") == C.sha256(identity_bytes)
             and producer.get("semantic_sha256") == semantic_sha256
             and producer.get("clock_domain") == "CLOCK_MONOTONIC_RAW"
             and type(producer.get("events")) is list,
             "actual producer source-event bundle required")
    owned = list(terminal["ue_events"])
    owned.extend(dict(role="producer", name=item["name"], raw_ns=item["raw_ns"],
        pid=producer["pid"], boot_id=producer["boot_id"]) for item in producer["events"])
    if emitted_event is not None:
        owned.append(emitted_event)
    return [_observed_event(identity_bytes, semantic_sha256, item) for item in owned]


class NumericEvaluatorV1(E.Run4EvaluatorV2):
    """Phase-6 evaluator interface backed by an off-compute-thread CPU child."""

    _binding = None

    def __init__(self, *, spec, read_ground_truth, send, match_distance_m, gt_timeout_s,
                 queue_depth=64, probe_timeout_s=.002, poll_interval_s=.005,
                 clock=time.monotonic, evidence_dir=None):
        binding = type(self)._binding
        _require(type(binding) is dict, "numeric edge binding must precede evaluator construction")
        _require(type(queue_depth) is int and 0 < queue_depth <= 1000
                 and type(gt_timeout_s) in (float, int) and 0 < gt_timeout_s <= 30
                 and type(binding["terminal_timeout_s"]) in (float, int)
                 and 0 < binding["terminal_timeout_s"] <= 60,
                 "bounded numeric evaluator queues and timeouts required")
        self.inbox, self.scorer_client, self.archive_client = (binding["inbox"],
            binding["scorer_client"], binding["archive_client"])
        self.archive_root_index = binding["archive_root_index"]
        self.sidecar_directory = Path(binding["sidecar_directory"])
        self._report_sink = binding.get("report_sink")
        self.terminal_timeout_s = float(binding["terminal_timeout_s"])
        self.gt_timeout_s = float(gt_timeout_s)
        self._observed_send = None
        self._sending_ticket = None

        def send_observed(payload):
            send(payload)
            actual_feedback, actual_reason = W.decode_feedback(payload)
            self._sent_packet = payload
            self._sent_feedback = actual_feedback
            self._sent_reason = actual_reason
            self._observed_send = dict(role="coordinator", name="feedback_emitted",
                raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                pid=os.getpid(), boot_id=read_boot_id())

        super().__init__(spec=spec, read_ground_truth=read_ground_truth,
            send=send_observed, match_distance_m=match_distance_m,
            gt_timeout_s=gt_timeout_s, queue_depth=queue_depth,
            probe_timeout_s=probe_timeout_s, poll_interval_s=poll_interval_s,
            clock=clock, evidence_dir=evidence_dir)
        self._stop = False
        self._failure = None
        self.counters.update(scored=0, archived=0, unscored=0)
        self._scored = OrderedDict()
        self._pending = OrderedDict()
        self._archive_thread = None
        self._archive_records = []
        self._client_lock = threading.Lock()
        self._emission_lock = threading.Lock()
        self._emitted = OrderedDict()
        self._scoring_keys = set()
        self._archived_keys = set()
        self._archive_stop = threading.Event()
        self._archive_thread = threading.Thread(target=self._archive_loop,
            name="numeric-edge-archive", daemon=True)

    def start(self):
        super().start()
        self._archive_thread.start()

    def submit(self, ticket):
        self.inbox.observe_evaluation_submission(ticket)
        return super().submit(ticket)

    def _emit(self, ticket, feedback, reason, **kwargs):
        """Retain the original wire/counters; stamp only after the actual send."""
        with self._emission_lock:
            self._observed_send = None
            self._sending_ticket = ticket
            try:
                super()._emit(ticket, feedback, reason, **kwargs)
                event = self._observed_send
            finally:
                self._sending_ticket = None
            if event is not None:
                _require(self._key(ticket) not in self._emitted,
                         "duplicate numerical feedback emission")
                actual = dict(ticket=ticket, event=event, packet_bytes=self._sent_packet,
                    packet_sha256=C.sha256(self._sent_packet),
                    feedback_kind=self._sent_feedback.kind,
                    reason=self._sent_reason.name,
                    q_perc=self._sent_feedback.q_perc)
                with self._lock:
                    self._emitted[self._key(ticket)] = actual
                    if feedback is None:
                        self.counters["unscored"] += 1
            return None

    def _service(self, pending, *, draining):
        """Reuse retained bounded pending loop but wait for numerical READY, not legacy GT."""
        for ticket_key, (ticket, submitted, timing) in list(pending.items()):
            try:
                matched = self.inbox.matching_target(ticket, claim=True)
            except BaseException as error:
                self._failure = f"inbox: {type(error).__name__}: {error}"[:1024]
                raise
            if matched is not None:
                del pending[ticket_key]
                with self._lock:
                    self._scoring_keys.add(ticket_key)
                try:
                    self._evaluate_with(ticket, {"numeric_target": matched}, timing)
                except Exception as error:
                    timing["numeric_error"] = f"{type(error).__name__}: {error}"[:200]
                    self._emit(ticket, None, W.EvaluatorReason.EVALUATOR_EXCEPTION,
                               timing=timing)
                finally:
                    with self._lock:
                        self._scoring_keys.discard(ticket_key)
            elif self._clock() - submitted >= self._gt_timeout:
                del pending[ticket_key]
                with self._lock:
                    self.counters["gt_expired"] += 1
                    if draining:
                        self.counters["drained_at_shutdown"] += 1
                self._emit(ticket, None, W.EvaluatorReason.GROUND_TRUTH_UNAVAILABLE,
                           timing=timing)

    def _evaluate_with(self, ticket, gt, timing):
        """The profiler's edge.qperc_evaluate span covers real CPU-child scoring."""
        key, row = gt["numeric_target"]
        _require(_ticket_matches(key, ticket), "ready target differs from actual edge ticket")
        target = R.decode_target(row["packet"], **SC._expected(row["expected"]))
        prediction = prediction_for_ticket(ticket)
        prediction["prediction_contract"] = C.thaw(target.metadata["prediction_contract"])
        timing["evaluator_start_wall_ns"] = time.time_ns()
        with self._client_lock:
            self.scorer_client.call("REGISTER", dict(key=key, expected=row["expected"],
                reservation_bytes=32 * 1024 * 1024))
            self.scorer_client.call("TARGET", dict(key=key, packet=row["packet"]))
            self.scorer_client.call("PREDICTION", dict(key=key, **prediction))
            score = self.scorer_client.call("SCORE", dict(key=key))
        _require(score is not None and score["ready_semantic_sha256"] == row["semantic_sha256"]
                 and score["identity_sha256"] == C.sha256(key),
                 "foreign/missing CPU numerical score")
        feedback, reason = W.quality_feedback(self._spec, score["measurement"], ticket.envelope)
        _require(feedback.q_perc == score["q_perc"],
                 "numerical quality differs from unchanged feedback formula")
        timing["evaluator_end_wall_ns"] = time.time_ns()
        self._emit(ticket, feedback, reason, measurement=score["measurement"], timing=timing)
        with self._lock:
            emitted = self._emitted[self._key(ticket)]
            self._scored[key] = dict(ticket=ticket, score=score, emitted_event=emitted["event"])
            self.counters["scored"] += 1

    def _archive_one(self, key, record):
        terminal = self.inbox.terminal_if_ready(key)
        _require(terminal is not None, "archive precedes actual UE terminal")
        source = terminal["source_export"]
        row = self.inbox._rows[key]
        emitted = self._emitted[self._key(record["ticket"])]
        feedback = terminal.get("feedback")
        if feedback is not None:
            _require(feedback == dict(packet_sha256=emitted["packet_sha256"],
                kind=emitted["feedback_kind"], reason=emitted["reason"],
                q_perc=emitted["q_perc"]),
                "UE received feedback differs from exact emitted numerical feedback")
        arguments = dict(input_metadata=C._parse(source["input_metadata_bytes"]),
            source_synchronization=C._parse(source["source_synchronization_bytes"]),
            numerical_avo=C._parse(source["numerical_avo_bytes"]),
            clock_events=archive_clock_events(key, row["semantic_sha256"], terminal,
                                               record["emitted_event"]),
            handoff_id=terminal.get("handoff_id"))
        with self._client_lock:
            attached = self.scorer_client.call("ATTACH_LIVE_ARCHIVE", dict(key=key, arguments=arguments))
        archive = LR.archive_from_scorer_reply(key, attached)
        capture = C._parse(key)["capture_identity"]
        prefix = f"numeric_gt/frame_{capture['frame_id']:010d}/"
        files = LR.archive_files_for_submit(self.archive_client, archive,
            relative_prefix=prefix, root_index=self.archive_root_index)
        job = self.archive_client.submit(capture, files)
        submitted = LR.observe_archive_submission(self.archive_client, job, archive,
            relative_prefix=prefix, root_index=self.archive_root_index)
        deadline = time.monotonic() + 30.
        while True:
            self.archive_client.poll()
            matches = [item for item in self.archive_client.completions if item["job_id"] == job]
            if matches:
                _require(len(matches) == 1, "duplicate ArchiveClient completion")
                receipt = matches[0]
                break
            _require(time.monotonic() < deadline, "ArchiveClient completion timeout")
            time.sleep(.01)
        completed = LR.observe_durable_completion(self.archive_client, receipt, archive,
            relative_prefix=prefix, root_index=self.archive_root_index)
        with self._client_lock:
            pending = self.scorer_client.call("LIVE_ARCHIVE_DURABLE", dict(key=key, receipt=receipt,
                relative_prefix=prefix, archive_submitted_event=submitted,
                durable_completion_event=completed))
        sidecar_directory = self.sidecar_directory / f"frame_{capture['frame_id']:010d}"
        sidecar_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        proof = LR.persist_live_completion(self.archive_client, receipt, archive, pending,
            submitted, completed, relative_prefix=prefix, root_index=self.archive_root_index,
            sidecar_directory=sidecar_directory)
        with self._client_lock:
            final = self.scorer_client.call("LIVE_COMPLETION_DURABLE", dict(key=key, proof=proof))
        _require(final["status"] == "DURABLE" and final["ready_identity_sha256"] == C.sha256(key),
                 "scorer did not retire exact archived frame")
        self.counters["archived"] += 1
        return dict(status="SCORED_DURABLE", scorer_durable_proof=final,
            identity_sha256=C.sha256(key), terminal=final["terminal"],
            archive_manifest_sha256=final["archive_manifest_sha256"],
            sidecar_sha256=final["sidecar_sha256"])

    def _unscored_descriptor(self, key, row, terminal, emitted):
        """Give the UE producer an exact observed clock, never a synthetic score."""
        source = terminal["source_export"]
        ledger = LiveClockLedger(key, row["semantic_sha256"],
            handoff_id=terminal.get("handoff_id"))
        for event in archive_clock_events(key, row["semantic_sha256"], terminal,
                                          None if emitted is None else emitted["event"]):
            observed = ledger.record(role=event["role"], name=event["name"],
                raw_ns=event["raw_ns"], pid=event["pid"], boot_id=event["boot_id"],
                ready_identity_bytes=key, semantic_sha256=row["semantic_sha256"])
            _require(C.canonical(observed) == C.canonical(event),
                     "unscored owner event identity/clock differs")
        if emitted is not None:
            feedback = terminal.get("feedback")
            if feedback is not None:
                _require(feedback == dict(packet_sha256=emitted["packet_sha256"],
                    kind=emitted["feedback_kind"], reason=emitted["reason"],
                    q_perc=emitted["q_perc"]),
                    "actual UE fault receipt differs from coordinator packet")
            _require(emitted["feedback_kind"] in ("EVALUATOR_FAULT", "INFRASTRUCTURE_FAULT")
                     and emitted["q_perc"] is None,
                     "unscored coordinator emission must be evaluator fault")
        descriptor = ledger.archive_descriptor(
            avo_sha256=C.sha256(source["numerical_avo_bytes"]),
            source_synchronization_sha256=C.sha256(source["source_synchronization_bytes"]),
            fault_feedback_sha256=None if emitted is None else emitted["packet_sha256"],
            fault_reason=None if emitted is None else emitted["reason"])
        return dict(status="UNSCORED_REQUIRED", clock_descriptor_bytes=descriptor,
            reason="FEEDBACK_TIMEOUT" if emitted is None else emitted["reason"],
            fault_feedback=None if emitted is None else dict(
                packet_bytes=emitted["packet_bytes"],
                packet_sha256=emitted["packet_sha256"],
                kind=emitted["feedback_kind"], reason=emitted["reason"],
                q_perc=emitted["q_perc"]))

    def _archive_unscored_files(self, key, files):
        clock = C._parse(files["live_clock.json"])
        terminal_doc = C._parse(files["unscored_terminal.json"])
        _require(C.canonical(clock) == files["live_clock.json"]
                 and C.canonical(terminal_doc) == files["unscored_terminal.json"]
                 and terminal_doc.get("ready_identity_sha256") == C.sha256(key)
                 and terminal_doc.get("clock_descriptor_sha256") == C.sha256(files["live_clock.json"])
                 and terminal_doc.get("score_exists") is False,
                 "producer unscored sidecar must bind exact clock and no-score identity")
        capture = C._parse(key)["capture_identity"]
        prefix = f"numeric_gt/frame_{capture['frame_id']:010d}/"
        root = Path(self.archive_client.config["roots"][self.archive_root_index]["path"])
        outgoing = {prefix + name: raw for name, raw in files.items()}
        proof = self.inbox._admission_expiries.get(key)
        proof_prefix = "numeric_gt_admission/"
        if proof is not None:
            outgoing[proof_prefix + f"frame_{capture['frame_id']:010d}.json"] = C.canonical(proof)
        job = self.archive_client.submit(capture,
            {root / name: raw for name, raw in outgoing.items()})
        deadline = time.monotonic() + 30.
        while True:
            self.archive_client.poll()
            matches = [item for item in self.archive_client.completions if item["job_id"] == job]
            if matches:
                _require(len(matches) == 1, "duplicate unscored ArchiveClient completion")
                receipt = matches[0]
                break
            _require(time.monotonic() < deadline, "unscored ArchiveClient completion timeout")
            time.sleep(.01)
        _require(receipt.get("status") == "COMPLETE" and receipt.get("durable") is True
                 and receipt.get("error") is None and receipt.get("identity") == capture,
                 "unscored archive not durably completed for this capture")
        selected = {item["relative"]: item for item in receipt["written"]}
        _require(set(selected) == set(outgoing) and len(selected) == len(receipt["written"]),
                 "unscored durable inventory incomplete")
        for name, raw in outgoing.items():
            item = selected[name]
            _require(item["root"] == self.archive_root_index
                     and item["bytes"] == len(raw) and item["sha256"] == C.sha256(raw)
                     and item["ready"] is False,
                     "unscored durable bytes/hash differ")
        LR._readback(root, {name: item for name, item in selected.items()
                           if name.startswith(prefix)}, prefix)
        if proof is not None:
            LR._readback(root, {name: item for name, item in selected.items()
                               if name.startswith(proof_prefix)}, proof_prefix)
        _require(clock["ready_identity_sha256"] == C.sha256(key),
                 "unscored clock differs from identity")
        return dict(status="UNSCORED_DURABLE", receipt=receipt,
            relative_prefix=prefix[:-1], root_index=self.archive_root_index,
            clock_descriptor_sha256=C.sha256(files["live_clock.json"]),
            unscored_terminal_sha256=C.sha256(files["unscored_terminal.json"]))

    def _archive_loop(self):
        """Archive after terminal while the evaluator keeps scoring new tickets."""
        deadline_after_stop = None
        try:
            while True:
                progressed = False
                with self._lock:
                    scored = list(self._scored.items())
                    emitted = list(self._emitted.items())
                    scoring_keys = set(self._scoring_keys)
                for key, record in scored:
                    if key in self._archived_keys or self.inbox.terminal_if_ready(key) is None:
                        continue
                    result = self._archive_one(key, record)
                    self.inbox.result(key, result)
                    self._archive_records.append(result)
                    self._archived_keys.add(key)
                    progressed = True
                for ticket_key, emission in emitted:
                    if ticket_key in scoring_keys or any(
                            self._key(record["ticket"]) == ticket_key for _, record in scored):
                        continue
                    target_rows = [(key, row) for key, row, _ in self.inbox.rows_snapshot()
                                   if key not in self._archived_keys and row["terminal"] is not None
                                   and _ticket_matches(key, emission["ticket"])]
                    for key, row in target_rows:
                        result = self._unscored_descriptor(key, row, row["terminal"], emission)
                        self.inbox.result(key, result)
                        self._archived_keys.add(key)
                        progressed = True
                # An edge control terminal proves this frame was never scored.
                # A mere 170-ms UE timeout does not: a late score may still be
                # in flight and must retain its own scored ownership.
                for key, row, _ in self.inbox.rows_snapshot():
                    if key in self._archived_keys or row["terminal"] is None:
                        continue
                    admission_expiry = self.inbox.close_unadmitted_after_horizon(key)
                    if row["edge_terminal"] is None and admission_expiry is None:
                        continue
                    _require(not any(_ticket_matches(key, record["ticket"])
                                     for _, record in scored),
                             "scored ticket conflicts with edge no-score terminal")
                    result = self._unscored_descriptor(key, row, row["terminal"], None)
                    result["edge_terminal"] = row["edge_terminal"]
                    if admission_expiry is not None:
                        result["admission_expiry"] = admission_expiry
                    self.inbox.result(key, result)
                    self._archived_keys.add(key)
                    progressed = True
                for key, row, result in self.inbox.rows_snapshot():
                    if (result or {}).get("status") != "UNSCORED_REQUIRED":
                        continue
                    files = self.inbox.unscored_files_if_ready(key)
                    if files is None:
                        continue
                    result = self._archive_unscored_files(key, files)
                    self.inbox.replace_unscored_result(key, result)
                    self._archive_records.append(result)
                    progressed = True
                if self._archive_stop.is_set():
                    if deadline_after_stop is None:
                        deadline_after_stop = time.monotonic() + self.terminal_timeout_s
                    pending = [key for key, row, result in self.inbox.rows_snapshot()
                               if (result or {}).get("status") not in
                               ("SCORED_DURABLE", "UNSCORED_DURABLE")]
                    if not pending:
                        return
                    _require(time.monotonic() < deadline_after_stop,
                             "unarchived observed terminals at edge shutdown")
                if not progressed:
                    time.sleep(.01)
        except BaseException as error:
            self._failure = f"archive: {type(error).__name__}: {error}"[:1024]

    def close(self, timeout_s=None):
        summary = super().close(timeout_s=timeout_s)
        self._archive_stop.set()
        self._archive_thread.join(timeout=self.terminal_timeout_s+2.)
        report = dict(summary, archive_worker_alive=self._archive_thread.is_alive(),
            archive_records=list(self._archive_records), failure=self._failure,
            live_qualified=False)
        _require(not report["archive_worker_alive"] and self._failure is None,
                 self._failure or "numerical archive worker did not drain")
        if self._report_sink is not None:
            self._report_sink.append(report)
        return report


class NumericProcessorV1(E.Run4EdgeProcessorV2):
    """Observe actual never-scored edge terminals without changing their bytes."""

    _inbox = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _require(type(self)._inbox is not None, "numeric processor inbox not installed")
        type(self)._inbox.bind_processing_horizon(run_id=self.run_id, cell_id=self.cell_id,
                                                ack_timeout_s=self._ack_s)

    def process(self, frame, *, edge_timing, segmentation=None):
        from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as base
        envelope, context, profile, identity = self.verify(frame)
        capture = MP.gt_identity(run_id=self.run_id, cell_id=self.cell_id,
            stream_id=context.stream_id, frame_id=context.frame_id,
            anchor_action_id=profile.action_id, anchor_profile_id=profile.profile_id,
            capture_timestamp_ns=context.capture_timestamp_ns)
        type(self)._inbox.reserve_processing(capture, identity)
        try:
            return super().process(frame, edge_timing=edge_timing, segmentation=segmentation)
        except base.DeadlineExpired:
            if identity["reward_requested"]:
                type(self)._inbox.processing_expired_before_evaluation(capture, identity)
            raise
        except Exception as error:
            type(self)._inbox.fail_processing(error)
            raise

    def terminal(self, frame, *, outcome, stage, superseded_by_frame_id=None):
        document = super().terminal(frame, outcome=outcome, stage=stage,
                                    superseded_by_frame_id=superseded_by_frame_id)
        inbox = type(self)._inbox
        _require(inbox is not None, "numeric edge terminal observer not installed")
        inbox.observe_edge_terminal(document)
        return document


@contextmanager
def install_numeric_edge(*, socket_path, scorer_client, archive_client,
                         archive_root_index, sidecar_directory, max_frames,
                         max_inbox_bytes=MAX_INBOX_BYTES, terminal_timeout_s=30.,
                         publisher_uid=None, publisher_gid=None):
    """Replace only the evaluator inside the retained edge service entry."""
    inbox = NumericEdgeInbox(socket_path, max_frames=max_frames, max_bytes=max_inbox_bytes,
        publisher_uid=publisher_uid, publisher_gid=publisher_gid)
    original = E.Run4EvaluatorV2
    original_processor = E.Run4EdgeProcessorV2
    _require(NumericEvaluatorV1._binding is None,
             "nested numerical evaluator binding forbidden")
    _require(NumericProcessorV1._inbox is None,
             "nested numerical terminal observer forbidden")
    state = dict(schema="splitfusion.numeric_gt.edge_live_scope.v1", installed=False,
        evaluator_reports=[], inbox=None, errors=[], live_qualified=False,
        evaluator_class="NumericEvaluatorV1")
    NumericEvaluatorV1._binding = dict(inbox=inbox, scorer_client=scorer_client,
        archive_client=archive_client, archive_root_index=archive_root_index,
        sidecar_directory=sidecar_directory, terminal_timeout_s=terminal_timeout_s,
        report_sink=state["evaluator_reports"])
    inbox.start()
    E.Run4EvaluatorV2 = NumericEvaluatorV1
    NumericProcessorV1._inbox = inbox
    E.Run4EdgeProcessorV2 = NumericProcessorV1
    state["installed"] = True
    try:
        yield state
    finally:
        E.Run4EvaluatorV2 = original
        E.Run4EdgeProcessorV2 = original_processor
        NumericEvaluatorV1._binding = None
        NumericProcessorV1._inbox = None
        state["inbox"] = inbox.close()


__all__ = ["NumericEdgeError", "NumericEdgeInbox", "NumericEvaluatorV1", "NumericProcessorV1", "prediction_for_ticket",
           "archive_clock_events", "install_numeric_edge", "SCHEMA", "OBSERVED_EVENTS_SCHEMA"]
