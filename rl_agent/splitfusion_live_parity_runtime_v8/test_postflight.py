"""Bounded tamper/failure regressions; no services, sensors, CUDA or live runs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import postflight_checks as P
from rl_agent.splitfusion_live_gt_handoff_v1 import contract as H


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))


def broker_fixture():
    policy = dict(publisher_uid=1000, publisher_gid=1001, explicitly_supplied=True,
        creator_uid=0, creator_gid=0, mode="0600", authentication="SO_PEERCRED_BEFORE_APPLICATION_READ")
    row = dict(identity=dict(frame_id=1), binding="a"*64, packet_sha256="b"*64, consumed=True,
        publisher_credentials=dict(pid=0, uid=1000, gid=1001))
    broker = dict(binding="a"*64, started=True, closed=True, worker_alive=False, failure=None,
        rejected_credentials=None, all_consumed=True, pending_jobs=0, pending_bytes=0,
        payloads_retained_for_unconsumed=0, credential_policy=policy,
        socket_proof=dict(uid=1000,gid=1001,mode="0600",device=1,inode=2),
        ledger=[row], admitted=1, consumed=1,
        limits=dict(max_jobs=2,max_bytes=32*1024*1024,max_frames=100), high_water_jobs=1, high_water_bytes=100)
    return dict(schema="splitfusion.gt_packet_edge_scope.v1", binding="a"*64, max_frames=100,
        complete=True, errors=[], transport_semantics=P.TRANSPORT, target_reference_kind=P.REFERENCE_KIND,
        new_broker_threads=1, durability_claim=False, reward_deadline_changed=False,
        quality_formula_changed=False, broker=broker, publisher_credential_policy=policy,
        prediction_state=dict(flushed=True,archive_errors=[]), pose_state=dict(flushed=True,errors=[]),
        evaluator_close=dict(worker_alive=False,parity_archive_complete=True,parity_archive_errors=[]))


class BrokerTests(unittest.TestCase):
    def check(self, state):
        return P._verify_broker(state,binding="a"*64,uid=1000,gid=1001,max_frames=100)

    def test_kernel_authenticated_namespace_pid_zero_is_valid(self):
        self.assertEqual(set(self.check(broker_fixture())), {1})

    def test_foreign_uid_gid_no_auth_or_surviving_packets_refused(self):
        mutations = (
            lambda s:s["broker"]["ledger"][0]["publisher_credentials"].update(uid=0),
            lambda s:s["broker"]["ledger"][0]["publisher_credentials"].update(gid=0),
            lambda s:s["broker"]["credential_policy"].update(authentication="PATH_ONLY"),
            lambda s:s["broker"].update(pending_jobs=1),
            lambda s:s["broker"].update(pending_bytes=1),
            lambda s:s["broker"].update(worker_alive=True),
            lambda s:s["broker"].update(failure="read failed"),
            lambda s:s["broker"].update(rejected_credentials=dict(uid=0)),
            lambda s:s["broker"]["socket_proof"].update(mode="0666"),
            lambda s:s.update(new_broker_threads=0),
            lambda s:s["broker"]["ledger"].append(copy.deepcopy(s["broker"]["ledger"][0])),
        )
        for mutation in mutations:
            state=broker_fixture(); mutation(state)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.check(state)

    def test_scoring_reference_cannot_be_called_disk_ready(self):
        self.assertEqual(P._packet_reference(dict(scoring_handoff=dict(packet_sha256="a"*64,archive_durable=False))), "a"*64)
        with self.assertRaises(ValueError):
            P._packet_reference(dict(scoring_handoff=dict(packet_sha256="a"*64,archive_durable=True)))


class ArchiveTests(unittest.TestCase):
    def fixture(self, root):
        identity=dict(frame_id=1)
        pending=dict(schema="splitfusion.prepared_live_v010_transport.v1",identity=identity,
            ready_raw_ns=0,ready_wall_ns=0,prepared_content_sha256="c"*64)
        ready=dict(pending,ready_raw_ns=1300,ready_wall_ns=9900)
        payload=b"native source bytes"
        (root/"native.bin").write_bytes(payload); write(root/"READY.json",ready)
        before=[dict(root=0,relative="native.bin",sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload),ready=False),
            dict(root=0,relative="READY.json",sha256=hashlib.sha256(encoded(pending)).hexdigest(),bytes=len(encoded(pending)),ready=True)]
        after=copy.deepcopy(before)
        after[1].update(sha256=hashlib.sha256(encoded(ready)).hexdigest(),bytes=len(encoded(ready)))
        submitted=dict(job_id=1,identity=identity,files=before,bytes=sum(r["bytes"] for r in before),
            stamp_ready_paths=[str(root/"READY.json")],ownership="SEALED_KERNEL_TRANSFER_NOT_DURABLE")
        completed=dict(job_id=1,identity=identity,status="COMPLETE",durable=True,error=None,pid=42,
            bytes=submitted["bytes"],written=after,start_ns=500,end_ns=700,
            finalized_ready=[dict(path=str(root/"READY.json"),sha256=after[1]["sha256"],ready_raw_ns=1300,ready_wall_ns=9900)])
        return dict(schema="splitfusion.gt_archive_handoff.v1",binding=dict(schema=H.SCHEMA,source_binding="a"*64),
            failed=False,fault=None,outstanding_jobs=0,outstanding_bytes=0,child_returncode=0,child_stderr="",
            child_pid=42,submissions=[submitted],completions=[completed],submitted=1,completed=1)

    def verify(self, value, root):
        return P._verify_archive(value,binding="a"*64,roots=(root,),max_frames=100)

    def test_real_stamped_ready_and_all_native_bytes_required(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); value=self.fixture(root)
            self.assertEqual(set(self.verify(value,root)[1]),{1})
            (root/"native.bin").write_bytes(b"changed source")
            with self.assertRaisesRegex(ValueError,"actual archived bytes"): self.verify(value,root)

    def test_archive_flags_cannot_replace_actual_complete_inventory(self):
        mutations=(lambda a:a.update(failed=True), lambda a:a.update(outstanding_jobs=1),
            lambda a:a.update(child_returncode=None), lambda a:a["completions"][0].update(durable=False),
            lambda a:a["completions"][0]["written"].pop(),
            lambda a:a["completions"][0].update(finalized_ready=[]),
            lambda a:a["completions"][0]["written"][0].update(relative="../native.bin"),
            lambda a:a["completions"].append(copy.deepcopy(a["completions"][0])))
        for mutation in mutations:
            with tempfile.TemporaryDirectory() as name:
                root=Path(name); value=self.fixture(root); mutation(value)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.verify(value,root)

    def test_ready_may_only_change_its_finalization_clocks(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); value=self.fixture(root)
            ready=json.loads((root/"READY.json").read_bytes()); ready["prepared_content_sha256"]="e"*64
            write(root/"READY.json",ready)
            digest=hashlib.sha256(encoded(ready)).hexdigest()
            value["completions"][0]["written"][1].update(sha256=digest)
            value["completions"][0]["finalized_ready"][0].update(sha256=digest)
            with self.assertRaisesRegex(ValueError,"beyond archival timestamps"): self.verify(value,root)

    def test_teardown_mapping_uses_preserved_exact_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); scratch=root/"scratch"; scratch.mkdir(); value=self.fixture(scratch)
            cell=root/"cell"; destination=cell/"run5_live_avo_inputs"; destination.mkdir(parents=True)
            for path in scratch.iterdir(): path.rename(destination/path.name)
            scratch.rmdir()
            result=P._verify_archive(value,binding="a"*64,roots=(scratch,),max_frames=100,preserved_cell=cell)
            self.assertEqual(set(result[1]),{1})


class TimingTests(unittest.TestCase):
    def fixture(self, directory, mutate=None):
        rows=[dict(frame_id=1,stage=stage,start_raw_ns=100+index*10,end_raw_ns=105+index*10,
            wall_ns=5,thread_cpu_ns=4,completed=True) for index,stage in enumerate(P.STAGES)]
        if mutate: mutate(rows)
        raw=b"".join(encoded(row)+b"\n" for row in rows)
        (directory/"stages.jsonl").write_bytes(raw)
        write(directory/"MANIFEST.json",dict(schema="splitfusion.gt_scoring_handoff.stage_timing.v1",
            clock="CLOCK_MONOTONIC_RAW",cpu_clock="thread_time_ns",flush_after_worker_drain=True,dropped=0,
            scoring_ready_is_not_archive_durable=True,work_and_readiness_unchanged=False,
            sha256={"stages.jsonl":hashlib.sha256(raw).hexdigest()},records=len(rows),bound=800))

    def test_six_real_handoff_stages_replace_old_five_disk_stages(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); self.fixture(root)
            self.assertEqual(P.verify_gt_timing(root,expected_frames={1})["stages"],6)

    def test_missing_duplicated_failed_crossed_or_old_stage_refused(self):
        for mutate in (lambda rows:rows.pop(),lambda rows:rows.append(rows[0]),
                lambda rows:rows[0].update(completed=False),lambda rows:rows[1].update(start_raw_ns=99,end_raw_ns=104),
                lambda rows:rows[3].update(stage="publish_complete_target")):
            with tempfile.TemporaryDirectory() as name:
                root=Path(name); self.fixture(root,mutate)
                with self.assertRaises(ValueError):P.verify_gt_timing(root,expected_frames={1})

    def test_latency_bound_is_inclusive_and_clocks_stay_separate(self):
        from rl_agent.splitfusion_live_parity_runtime_v1 import postflight as retained
        for elapsed, expected in ((170_000_000,True),(170_000_001,False)):
            with tempfile.TemporaryDirectory() as name:
                root=Path(name)
                data=dict(decisions={1:dict(action_open=dict(domain="CLOCK_MONOTONIC_RAW",ns=10))},
                    evaluations={1:dict(timing=dict(evaluator_start_wall_ns=10_000,evaluator_end_wall_ns=10_010),
                        emit_wall_ns=10_020,feedback_sha256="a"*64)},
                    ue=dict(feedback_rows=[dict(frame_id=1,receipt=dict(domain="CLOCK_MONOTONIC_RAW",ns=10+elapsed),sha256="a"*64)]))
                write(root/"CPU_GT_PROCESS_TIMELINE.json",dict(requests=[dict(identity=dict(frame_id=1),worker=dict(start_raw_ns=20,end_raw_ns=30))]))
                write(root/"cpu_gt_worker/engine/HANDOFF_ARCHIVE_LEDGER.json",dict(archive=dict(completions=[dict(identity=dict(frame_id=1),start_ns=4,end_ns=6)])))
                write(root/"profiler/edge/EDGE_HANDOFF_SCOPE.json",dict(broker=dict(ledger=[dict(identity=dict(frame_id=1),admitted_raw_ns=25,admitted_wall_ns=9900)])))
                directory=root/"source_ownership"; directory.mkdir()
                (directory/"events.jsonl").write_bytes(b"\n".join(encoded(row) for row in [dict(kind="SOURCE_ADMITTED",frame_id=1,observed_raw_ns=15),dict(kind="SOURCE_CONSUMED",frame_id=1,observed_raw_ns=19)]))
                with mock.patch.object(retained,"_population",return_value=data):
                    report=P.latency_report(root)
                self.assertEqual(report["timing_passed"],expected)
                self.assertFalse(report["cross_clock_subtraction"])
                self.assertEqual(report["rows"][0]["archive_service_monotonic_ms"],2/1e6)

    def test_failed_population_remains_visible(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            write(root/"run5_live_avo_gt_summary.json",dict(records=[dict(frame_id=1)],rejected_records=[dict(frame_id=2)]))
            write(root/"cpu_gt_worker/engine/HANDOFF_ARCHIVE_LEDGER.json",dict(failed_batches=[dict(identity=dict(frame_id=2))]))
            write(root/"FAILED_ARCHIVE_PARTIAL_PRESERVATION.json",dict(unresolved=[dict(frame_id=3)]))
            report=P.population_accounting(root)
            self.assertFalse(report["clean"])
            self.assertEqual(report["counts"],dict(accepted=1,rejected=1,failed=1,unresolved=1))

    def test_retained_reader_replacement_is_scoped(self):
        from rl_agent.splitfusion_live_parity_runtime_v1 import postflight as retained
        original=retained.verify_cell
        with P.install_postflight_reader():self.assertIsNot(retained.verify_cell,original)
        self.assertIs(retained.verify_cell,original)


if __name__=="__main__": unittest.main()
