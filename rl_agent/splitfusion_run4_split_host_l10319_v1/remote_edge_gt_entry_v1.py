"""Persistent GT ingress entry seam for the remote Run-4 edge.

This module composes the measured-GPU entry with the exact-byte GT transport.
It changes neither the frozen edge processor nor evaluator source.  Instead,
for this process only, it installs a strict evaluator subclass whose
``submit`` authorizes the already-verified reward ticket before delegating to
the unchanged bounded evaluator queue.

The GT listener binds and listens synchronously before the evaluator
constructor returns.  A guard around the frozen pre-warm publisher rechecks
listener health immediately before edge READY is written.  Importing this
module binds no socket, starts no thread, imports no torch, and reads no file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import threading
from typing import Any, Callable, Mapping, Optional, Sequence

from . import contract as C
from . import gt_transport as GT


SCHEMA = "scenesense.run4.remote_edge_gt_ingress.v1"
REGISTERED_GT_PORT = 51015
MAX_GT_TICKETS = 4096


class RemoteEdgeGtError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RemoteEdgeGtError(message)


def _write_create_only(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(document), sort_keys=True, indent=1) + "\n")


class PersistentGtIngressListenerV1:
    """Caller-owned listener repeatedly applying :func:`GT.serve_one`."""

    def __init__(
        self, *, run_id: str, cell_id: str, evidence_dir: Path,
        bind_host: str, advertised_host: str, port: int,
        socket_timeout_s: float, expectation_timeout_s: float,
        ready_evidence: Path, final_evidence: Path,
        socket_factory: Callable[..., Any] = socket.socket,
        serve_one_fn: Callable[..., Any] = GT.serve_one,
    ) -> None:
        topology = C.default_topology()
        _require(bind_host == topology.edge_ip, "GT bind host must be the edge address")
        _require(advertised_host == topology.edge_ip,
                 "GT advertised host must be the edge address")
        _require(type(port) is int and port == REGISTERED_GT_PORT,
                 "GT port is not the registered port")
        _require(type(socket_timeout_s) in (int, float)
                 and 0 < float(socket_timeout_s) <= GT.MAX_SOCKET_TIMEOUT_S,
                 "GT socket timeout is outside the transport bound")
        _require(type(expectation_timeout_s) in (int, float)
                 and 0 <= float(expectation_timeout_s)
                 <= GT.MAX_EXPECTATION_TIMEOUT_S,
                 "GT expectation timeout is outside the transport bound")
        self.run_id, self.cell_id = str(run_id), str(cell_id)
        self.bind_host, self.advertised_host = bind_host, advertised_host
        self.port = port
        self.socket_timeout_s = float(socket_timeout_s)
        self.expectation_timeout_s = float(expectation_timeout_s)
        self.ready_evidence, self.final_evidence = (Path(ready_evidence),
                                                    Path(final_evidence))
        self.registry = GT.ExpectedTicketRegistryV1(
            run_id=self.run_id, cell_id=self.cell_id, max_tickets=MAX_GT_TICKETS)
        self.ingress = GT.GtIngressStoreV1(Path(evidence_dir), self.registry)
        self._socket_factory, self._serve_one = socket_factory, serve_one_fn
        self._stop = threading.Event()
        self._started = threading.Event()
        self._lock = threading.Lock()
        self._server: Any = None
        self._client: Any = None
        self._thread: Optional[threading.Thread] = None
        self._failure: Optional[str] = None
        self._stopped = False
        self._edge_ready_checked = False
        self._counters = {
            "connections": 0, "stored": 0, "duplicate_identical": 0,
            "rejected": 0, "authorized": 0, "idle_disconnects": 0,
        }

    def start(self) -> None:
        _require(self._thread is None, "GT listener already started")
        server = self._socket_factory(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.bind_host, self.port))
            server.listen(1)
            server.settimeout(0.1)
        except BaseException:
            server.close()
            raise
        self._server = server
        self._thread = threading.Thread(
            target=self._run, name="run4-remote-gt-ingress", daemon=False)
        self._thread.start()
        try:
            _require(self._started.wait(1.0), "GT listener thread did not start")
            self.require_healthy()
            _write_create_only(self.ready_evidence, {
                "schema": SCHEMA,
                "status": "LISTENING",
                "run_id": self.run_id, "cell_id": self.cell_id,
                "bind_host": self.bind_host,
                "advertised_endpoint": f"{self.advertised_host}:{self.port}",
                "max_tickets": MAX_GT_TICKETS,
                "socket_timeout_s": self.socket_timeout_s,
                "expectation_timeout_s": self.expectation_timeout_s,
                "cross_host_clock_subtraction": False,
                "policy_deadline_clock_owner": "W10275",
            })
        except BaseException:
            try:
                self.stop()
            except BaseException:
                pass
            raise

    def _set_failure(self, exc: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = f"{type(exc).__name__}: {exc}"[:500]
        self._stop.set()

    def _run(self) -> None:
        self._started.set()
        while not self._stop.is_set():
            try:
                stream, _peer = self._server.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                if self._stop.is_set():
                    return
                self._set_failure(exc)
                return
            with self._lock:
                self._client = stream
                self._counters["connections"] += 1
            try:
                while not self._stop.is_set():
                    try:
                        ack = self._serve_one(
                            stream, self.ingress,
                            socket_timeout_s=self.socket_timeout_s,
                            expectation_timeout_s=self.expectation_timeout_s)
                    except socket.timeout:
                        # Framing state after a mid-message timeout is unknowable.
                        with self._lock:
                            self._counters["idle_disconnects"] += 1
                        break
                    except GT.ProtocolError as exc:
                        if "stream closed during framed message" in str(exc):
                            with self._lock:
                                self._counters["idle_disconnects"] += 1
                            break
                        self._set_failure(exc)
                        return
                    except (GT.IdentityConflictError, GT.StorageError,
                            GT.IdentityError) as exc:
                        self._set_failure(exc)
                        return
                    except OSError as exc:
                        if self._stop.is_set():
                            return
                        self._set_failure(exc)
                        return
                    except BaseException as exc:  # fail closed on foreign failure
                        self._set_failure(exc)
                        return
                    with self._lock:
                        key = ("stored" if ack.status == "STORED" else
                               "duplicate_identical" if ack.status == "DUPLICATE_IDENTICAL"
                               else "rejected")
                        self._counters[key] += 1
            finally:
                try:
                    stream.close()
                finally:
                    with self._lock:
                        if self._client is stream:
                            self._client = None

    def authorize(self, ticket: Any) -> GT.GtTransportIdentityV1:
        self.require_healthy()
        identity = GT.identity_from_phase6(
            run_id=self.run_id, cell_id=self.cell_id,
            envelope=ticket.envelope, context=ticket.context,
            gt_identity=ticket.gt_identity)
        try:
            self.registry.authorize(identity)
        except BaseException as exc:
            self._set_failure(exc)
            raise
        with self._lock:
            self._counters["authorized"] += 1
        return identity

    def require_healthy(self) -> None:
        with self._lock:
            failure = self._failure
        _require(self._thread is not None and self._thread.is_alive(),
                 "GT listener is not alive")
        _require(failure is None, f"GT listener failed: {failure}")

    def mark_edge_ready_checked(self) -> None:
        self.require_healthy()
        with self._lock:
            self._edge_ready_checked = True

    def stop(self, *, write_final: bool = True) -> Mapping[str, Any]:
        with self._lock:
            already_stopped = self._stopped
            if not already_stopped:
                self._stopped = True
        if already_stopped:
            return self._final_document()
        self._stop.set()
        for stream in (self._client, self._server):
            if stream is None:
                continue
            try:
                stream.shutdown(socket.SHUT_RDWR)
            except (OSError, AttributeError):
                pass
            try:
                stream.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=self.socket_timeout_s + 1.0)
        document = self._final_document()
        if write_final:
            _write_create_only(self.final_evidence, document)
        _require(not document["thread_alive"], "GT listener failed to stop")
        return document

    def _final_document(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                "schema": SCHEMA,
                "status": "FAILED" if self._failure else "STOPPED",
                "run_id": self.run_id, "cell_id": self.cell_id,
                "advertised_endpoint": f"{self.advertised_host}:{self.port}",
                "edge_ready_health_checked": self._edge_ready_checked,
                "failure": self._failure,
                "counters": dict(self._counters),
                "thread_alive": bool(self._thread and self._thread.is_alive()),
                "cross_host_clock_subtraction": False,
                "policy_deadline_clock_owner": "W10275",
            }


class InstalledGtEvaluatorHooksV1:
    """Process-local evaluator/prewarm hooks, fully restored on exit."""

    def __init__(self, *, frozen: Any, prewarm: Any,
                 listener_factory: Callable[[Path], PersistentGtIngressListenerV1]) -> None:
        self.frozen, self.prewarm = frozen, prewarm
        self.listener_factory = listener_factory
        self.original_evaluator = frozen.Run4EvaluatorV2
        self.original_publish_ready = prewarm.publish_ready_after_warmup
        self.latest: Any = None
        self.installed = False

    def install(self) -> None:
        _require(not self.installed, "remote GT hooks already installed")
        owner = self
        original = self.original_evaluator

        class RemoteGtEvaluatorV1(original):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                evidence = kwargs.get("evidence_dir")
                _require(evidence is not None, "remote evaluator lacks evidence directory")
                self._remote_gt_listener = owner.listener_factory(Path(evidence))
                self._remote_gt_listener.start()
                self._remote_base_started = False
                self._remote_close_result: Optional[Mapping[str, Any]] = None
                try:
                    super().__init__(*args, **kwargs)
                except BaseException:
                    self._remote_gt_listener.stop()
                    raise
                owner.latest = self

            def start(self) -> None:
                self._remote_gt_listener.require_healthy()
                super().start()
                self._remote_base_started = True

            def submit(self, ticket: Any) -> None:
                # The frozen processor already verified and constructed ticket.
                self._remote_gt_listener.authorize(ticket)
                super().submit(ticket)

            def close(self, timeout_s: Optional[float] = None) -> Mapping[str, Any]:
                if self._remote_close_result is not None:
                    return self._remote_close_result
                try:
                    if self._remote_base_started:
                        base_result = super().close(timeout_s=timeout_s)
                    else:
                        base_result = {**self.counters, "worker_alive": False,
                                       "records": len(self.records)}
                finally:
                    # Keep ingress available while the base evaluator drains,
                    # but always stop/join it even if that drain fails.
                    listener_result = self._remote_gt_listener.stop()
                self._remote_close_result = {
                    **base_result, "remote_gt_listener": listener_result}
                return self._remote_close_result

        def publish_ready_after_warmup(warm: Callable[[], Any],
                                       write_ready: Callable[[], Any],
                                       *args: Any, **kwargs: Any) -> Any:
            def guarded_write() -> Any:
                _require(owner.latest is not None, "remote GT evaluator was not constructed")
                owner.latest._remote_gt_listener.mark_edge_ready_checked()
                return write_ready()

            return owner.original_publish_ready(warm, guarded_write, *args, **kwargs)

        self.frozen.Run4EvaluatorV2 = RemoteGtEvaluatorV1
        self.prewarm.publish_ready_after_warmup = publish_ready_after_warmup
        self.installed = True

    def close_latest(self) -> None:
        if self.latest is not None:
            self.latest.close()

    def restore(self) -> None:
        if not self.installed:
            return
        self.frozen.Run4EvaluatorV2 = self.original_evaluator
        self.prewarm.publish_ready_after_warmup = self.original_publish_ready
        self.installed = False


def parse_remote_gt_args(
        argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse only the additive GT options; importing torch/services is unnecessary."""
    values = list(argv) if argv is not None else None
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--remote-gt-bind-host", required=True)
    parser.add_argument("--remote-gt-advertised-host", required=True)
    parser.add_argument("--remote-gt-port", type=int, required=True)
    parser.add_argument("--remote-gt-ready-evidence", type=Path, required=True)
    parser.add_argument("--remote-gt-final-evidence", type=Path, required=True)
    parser.add_argument("--remote-gt-socket-timeout-s", type=float, required=True)
    parser.add_argument("--remote-gt-expectation-timeout-s", type=float, required=True)
    return parser.parse_known_args(values)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - live seam
    known, remaining = parse_remote_gt_args(argv)
    identity_parser = argparse.ArgumentParser(add_help=False)
    identity_parser.add_argument("--run-id", required=True)
    identity_parser.add_argument("--cell-id", required=True)
    identity_parser.add_argument("--edge-segmentation-evidence-dir", type=Path,
                                 required=True)
    identity, _ignored = identity_parser.parse_known_args(remaining)

    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2
    from . import remote_edge_entry_v1 as GPU

    def factory(evidence_dir: Path) -> PersistentGtIngressListenerV1:
        _require(evidence_dir == identity.edge_segmentation_evidence_dir,
                 "frozen evaluator and GT ingress evidence directories differ")
        return PersistentGtIngressListenerV1(
            run_id=identity.run_id, cell_id=identity.cell_id,
            evidence_dir=evidence_dir, bind_host=known.remote_gt_bind_host,
            advertised_host=known.remote_gt_advertised_host, port=known.remote_gt_port,
            socket_timeout_s=known.remote_gt_socket_timeout_s,
            expectation_timeout_s=known.remote_gt_expectation_timeout_s,
            ready_evidence=known.remote_gt_ready_evidence,
            final_evidence=known.remote_gt_final_evidence)

    hooks = InstalledGtEvaluatorHooksV1(
        frozen=phase6_edge_runtime_v2, prewarm=phase6_prewarm_v2,
        listener_factory=factory)
    hooks.install()
    try:
        return GPU.main(remaining)
    finally:
        try:
            hooks.close_latest()
        finally:
            hooks.restore()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
