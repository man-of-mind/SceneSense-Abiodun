"""Adapt retained joint-drain lifecycle to truthful deferred archive receipts."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import FunctionType

from rl_agent.splitfusion_live_gt_optimization_v4 import runtime as R
from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
from .client import Client
from . import contract as C
from .deferred_files import resolve_records


def _owned_json(path, root):
    path, root = Path(path), Path(root).resolve(strict=True)
    W.require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
        and not any(p.is_symlink() for p in path.parents if p.is_relative_to(root)),
        "missing or unsafe owned archival drain evidence")
    W.require(path.stat().st_size <= 32 * 1024 * 1024, "archival drain evidence exceeds bound")
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


class Session(R._Session):
    def start(self, owner, catalogue):
        factory = self.client_factory
        # Existing startup places the edge evidence directory before collector
        # construction; no external launch or frame is started by this seam.
        config = dict(directory=str(Path(owner.edge_evidence_dir).resolve(strict=True)),
                      evidence_root=str(self.attempt))
        self.client_factory = lambda **kw: factory(handoff_config=config, **kw)
        self.handoff_binding = C.binding()
        self.state.update(schema="splitfusion.gt_scoring_handoff.runtime_scope.v1",
            new_cpu_processes=2, new_queues=2, new_python_threads=0,
            deferred_evidence=True, scoring_ready_is_not_archive_durable=True)
        super().start(owner, catalogue)
        self.state["new_cpu_processes"] = 2

    def before_preserve(self):
        # Base stops admission, verifies retained GT worker drain, closes the
        # CPU child (which drains archive), joins and copies its genuine log.
        super().before_preserve()
        # A handled RUN failure has a real failure_close receipt. A disk fault
        # during CLOSE can prevent that response entirely, although the child
        # already stopped its writer and wrote its truthful FAILED drain proof.
        # Recover only that owned, source-bound proof after the child is dead.
        timing = archive = None
        failures = []
        try:
            W.require(self.client.process.poll() is not None, "archival salvage while GT child is alive")
            engine_root = self.attempt / "cpu_gt_worker/engine"
            timing, drain_sha = _owned_json(engine_root / "HANDOFF_DRAIN.json", engine_root)
            W.require(timing.get("binding") == self.handoff_binding, "foreign archival drain source binding")
            response = self.client.handoff_close
            if response is not None:
                W.require(isinstance(response, dict) and response.get("timing") == timing,
                    "reported and durable archival drain disagree")
            else:
                failures.append("CLOSE response missing; recovered owned failed drain after child exit")
            descriptor = timing.get("ledger", {})
            ledger_path = engine_root / "HANDOFF_ARCHIVE_LEDGER.json"
            W.require(descriptor.get("path") == str(ledger_path), "foreign archival ledger path")
            ledger, ledger_sha = _owned_json(ledger_path, engine_root)
            W.require(ledger_sha == descriptor.get("sha256") and ledger.get("binding") == self.handoff_binding,
                "archival ledger digest or source binding differs")
            archive = ledger.get("archive")
            W.require(isinstance(archive, dict) and archive.get("binding") == {
                "schema": C.SCHEMA, "source_binding": self.handoff_binding}, "foreign archive process binding")
            W.require(type(archive.get("child_returncode")) is int, "archive writer has not exited")
            compact = {key: value for key, value in archive.items() if key not in ("submissions", "completions")}
            compact["completions"] = [{key: value for key, value in row.items() if key != "written"}
                for row in archive["completions"]]
            W.require(timing.get("archive") == compact, "compact archival drain differs from full ledger")
            W.require(type(archive.get("submitted")) is int and 0 <= archive["submitted"] <= self.max_frames
                and len(archive["submissions"]) == archive["submitted"], "archive admission count differs")
            completions = archive["completions"]
            W.require(len({row["job_id"] for row in completions}) == len(completions), "duplicate archival completion")
            clean = (timing.get("archive_drained") is True and archive.get("failed") is False
                and archive.get("outstanding_jobs") == 0 and archive.get("outstanding_bytes") == 0
                and archive.get("submitted") == archive.get("completed")
                and archive.get("child_returncode") == 0)
            if not clean:
                failures.append("deferred archive failed or did not durably complete every admitted job")
            self.state.update(archive=timing["archive"], handoff_ledger=descriptor,
                archival_drain_sha256=drain_sha, archival_ledger_sha256=ledger_sha,
                close_receipt_recovered_from_owned_evidence=response is None)
        except Exception as exc:
            failures.append(f"drain verification: {type(exc).__name__}: {exc}")
            archive = None
        self._preserve_completed(archive, failures)

    def _verify_job_files(self, archive, completion):
        """Verify actual durable output, not merely a completion flag."""
        W.require(completion.get("status") == "COMPLETE" and completion.get("durable") is True,
            "archive job lacks a successful durable completion")
        submissions = [row for row in archive["submissions"] if row["job_id"] == completion["job_id"]]
        W.require(len(submissions) == 1 and submissions[0]["identity"] == completion["identity"],
            "archival completion has no exact admitted identity")
        admitted = {(row["root"], row["relative"]): row for row in submissions[0]["files"]}
        written = completion.get("written", [])
        W.require(len(written) == len(admitted)
            and {(row["root"], row["relative"]) for row in written} == set(admitted),
            "successful archive job omitted or duplicated files")
        roots = tuple(dict.fromkeys((Path(self.owner.edge_evidence_dir).resolve(strict=True),
            self.attempt.resolve(strict=True), (self.attempt / "cpu_gt_worker/engine").resolve(strict=True))))
        hashes = {}
        for row in written:
            index, relative = row["root"], Path(row["relative"])
            W.require(type(index) is int and 0 <= index < len(roots) and not relative.is_absolute()
                and ".." not in relative.parts, "unsafe archival output path")
            path = roots[index] / relative
            W.require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(roots[index])
                and not any(p.is_symlink() for p in path.parents if p.is_relative_to(roots[index])),
                "missing or symbolic archival output")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            W.require(digest == row["sha256"] and path.stat().st_size == row["bytes"],
                "archival file differs from durable completion")
            source = admitted[(index, str(relative))]
            W.require(row["ready"] == source["ready"], "archival publication kind differs")
            if not row["ready"]:
                W.require(row["sha256"] == source["sha256"] and row["bytes"] == source["bytes"],
                    "archival non-READY content differs from admission")
            hashes[str(path)] = digest
        return hashes

    def _preserve_completed(self, archive, failures):
        originals = {"records": json.loads(W.canonical(self.owner._avo_results)),
            "rejected_records": json.loads(W.canonical(self.owner._avo_rejected_results))}
        kept, unresolved = {name: [] for name in originals}, []
        completions = [] if archive is None else archive["completions"]
        for group, records in originals.items():
            for original in records:
                # Work on an independent record: failed partial resolution may
                # not mutate the receipt retained as failed evidence.
                candidate = json.loads(W.canonical(original))
                try:
                    if group == "records":
                        job = candidate["archive_receipt"]["job_id"]
                        matches = [row for row in completions if row["job_id"] == job]
                    else:
                        matches = [row for row in completions if row.get("identity", {}).get("frame_id")
                            == candidate["frame_id"] and row.get("status") == "COMPLETE"]
                    W.require(len(matches) == 1, "no unique successful archival receipt for retained record")
                    completion = matches[0]
                    hashes = self._verify_job_files(archive, completion)
                    W.require(hashes.get(candidate["path"]) == candidate["sha256"],
                        "AVO sidecar lacks exact durable content")
                    sidecar = json.loads(Path(candidate["path"]).read_bytes())
                    W.require(sidecar.get("identity") == completion["identity"], "AVO/archive identity differs")
                    if group == "records":
                        resolve_records([candidate], [completion])
                    kept[group].append(candidate)
                except Exception as exc:
                    unresolved.append(dict(group=group, record=original,
                        error=f"{type(exc).__name__}: {exc}"))
        if unresolved or failures:
            self.fail(R.RuntimeErrorGT("deferred archive partial preservation required"))
            path = self.attempt / "FAILED_ARCHIVE_PARTIAL_PRESERVATION.json"
            report = dict(schema="splitfusion.gt_handoff.failed_partial_preservation.v1", status="FAILED",
                binding=self.handoff_binding, input_records=originals, unresolved=unresolved,
                failures=list(failures), original_counts={key: len(value) for key, value in originals.items()},
                preserved_counts={key: len(value) for key, value in kept.items()},
                preserved_identities={key: [row.get("complete_identity", {"frame_id": row["frame_id"]})
                    for row in values] for key, values in kept.items()},
                durable_success_for_unresolved=False,
                gt_child_stopped_before_salvage=self.client.process.poll() is not None,
                archive_writer_exit_verified=archive is not None)
            # Write the complete failed descriptors BEFORE changing collector
            # lists; no silent filtering and no fabricated successful evidence.
            R._write(path, report)
            self.state["failed_partial_preservation"] = dict(path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                original_counts=report["original_counts"], preserved_counts=report["preserved_counts"],
                unresolved=len(unresolved))
        self.owner._avo_results[:] = kept["records"]
        self.owner._avo_rejected_results[:] = kept["rejected_records"]
        self.state["pending_receipts_resolved_before_preservation"] = not unresolved


def _installation():
    original = R.install_cpu_gt_worker.__wrapped__
    function = FunctionType(original.__code__, dict(original.__globals__, _Session=Session),
        original.__name__, original.__defaults__, original.__closure__)
    function.__kwdefaults__ = original.__kwdefaults__
    return contextmanager(function)


@contextmanager
def install_cpu_gt_worker(*, attempt_dir, max_frames, expected_sources,
                          expected_bootstrap_sources, timeout_s=30., _client_factory=None):
    if expected_bootstrap_sources != C.sources():
        raise ValueError("handoff worker source binding differs")
    factory = _client_factory or Client

    def bound(**kwargs):
        return factory(expected_bootstrap_sources=expected_bootstrap_sources, **kwargs)

    with _installation()(attempt_dir=attempt_dir, max_frames=max_frames,
            expected_sources=expected_sources, timeout_s=timeout_s, _client_factory=bound) as state:
        state["bootstrap_sources"] = dict(expected_bootstrap_sources)
        yield state
