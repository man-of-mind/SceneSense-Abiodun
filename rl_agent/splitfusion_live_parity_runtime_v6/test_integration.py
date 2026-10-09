"""CPU-only order, drain and failure tests of the actual new child entry."""
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest import mock

from . import entries as E, contract as C, optimization_scope as O, startup as S
from rl_agent.splitfusion_paired_quality_replay_v1 import capture as A, scope as I


def module(name, **values):
    result=ModuleType(name);result.__dict__.update(values);result.__path__=[]
    return result


class ChildTests(unittest.TestCase):
    @contextmanager
    def fixture(self, *, failure=None, front_alive=False, gt_alive=False, observer_fault=False):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            output=Path(tmp); attempt=output/"capture/cells/cell"; attempt.mkdir(parents=True)
            record={"budget":100,"optimization_sources":{"runtime.py":"a"*64}}
            (output/"PROSPECTIVE.json").write_text(json.dumps(record))
            config={"qualification_id":C.ID}; config["sha256"]=C.canonical(config)
            events=[]
            collectors=[NS(worker=NS(is_alive=lambda:front_alive),evaluation_worker=NS(is_alive=lambda:gt_alive))]
            owned={"collectors":collectors,"manifest":None,"inspect_drain":lambda:{"worker_drained":not gt_alive}}
            optimized={"sources":record["optimization_sources"],"settings_before":{"fixed":True}}
            @contextmanager
            def scope(name,value=None,**kwargs):
                events.append("enter:"+name)
                try: yield value
                finally:
                    events.append("exit:"+name)
                    if name=="ownership": owned["manifest"]={"complete":not gt_alive}
                    if name=="optimization": optimized.update(settings_after={"fixed":True},retained_settings_unchanged=True,source_integrity_error=None)
            class Archive:
                errors=["injected"] if observer_fault else []
                error_count=1 if observer_fault else 0
                policy_opportunities=1
                def __init__(self,**kwargs):
                    if kwargs != {"max_frames":50,"max_bytes":512*1024*1024}: raise AssertionError(kwargs)
                def flush(self,path,*,collectors,source_binding):
                    events.append("flush")
                    if "exit:ownership" not in events or "exit:optimization" not in events or "exit:capture" not in events:
                        raise AssertionError("flush before retained contexts exit")
                    drain=A.verified_drain(collectors)
                    if self.error_count: raise ValueError("observer fault")
                    path.mkdir()
                    (path/"MANIFEST.json").write_text("fixture")
                    return {"frames":[{}],"retained_bytes":1,"drain":drain}
            def run(args,**kwargs):
                events.append("retained")
                if failure: raise failure
                return 0
            fake={
                "rl_agent.splitfusion_live_parity_repair_v1":module("repair",composition=NS(run_ue_child=run)),
                "rl_agent.splitfusion_single_gpu_profiler_v1":module("profiler",child_v1=NS(attempt_dir_from_argv=lambda _:attempt,load_config=lambda _:config)),
                "rl_agent.splitfusion_live_parity_runtime_v1":module("prior",pose=NS(install_ue_pose=lambda **kw:scope("pose"))),
                "rl_agent.splitfusion_live_parity_repair_v2.source_ownership":module("ownership",install_source_ownership=lambda **kw:scope("ownership",owned)),
                "rl_agent.splitfusion_live_parity_repair_v2.gt_timing":module("timing",install_gt_timing=lambda **kw:scope("timing")),
                "rl_agent.splitfusion_live_parity_runtime_v6.host_capture":module("host",verify_prospective=lambda _:record),
            }
            stack.enter_context(mock.patch.dict(sys.modules,fake))
            stack.enter_context(mock.patch.object(A,"InputArchive",Archive))
            stack.enter_context(mock.patch.object(I,"install_capture",lambda _:scope("capture")))
            stack.enter_context(mock.patch.object(O,"install_cpu_gt_optimizations",lambda **kw:scope("optimization",optimized)))
            stack.enter_context(mock.patch.object(E,"evidence_scope",lambda _:scope("evidence")))
            stack.enter_context(mock.patch.object(S,"preserve_startup",lambda _:scope("startup")))
            yield NS(attempt=attempt,events=events,config=config,
                report=lambda:json.loads((attempt/"PAIRED_INPUT_SCOPE.json").read_text()))

    def test_order_and_postdrain_only_flush(self):
        with self.fixture() as f:
            self.assertEqual(E.child(["fixture"]),0)
            self.assertEqual(f.events[:3],["enter:capture","enter:optimization","enter:ownership"])
            self.assertEqual(f.events[-3:],["exit:optimization","exit:capture","flush"])
            self.assertTrue(f.report()["complete"])
            self.assertEqual(f.report()["max_frames"],50)

    def test_front_worker_alive_refuses_flush_even_when_gt_drained(self):
        with self.fixture(front_alive=True) as f:
            with self.assertRaisesRegex(ValueError,"worker has not drained"): E.child([])
            self.assertFalse(f.report()["complete"])

    def test_gt_worker_alive_refuses_before_flush(self):
        with self.fixture(gt_alive=True) as f:
            with self.assertRaisesRegex(RuntimeError,"did not close"): E.child([])
            self.assertNotIn("flush",f.events)

    def test_observer_fault_cannot_be_complete(self):
        with self.fixture(observer_fault=True) as f:
            with self.assertRaisesRegex(ValueError,"observer fault"): E.child([])
            self.assertEqual(f.report()["observer_error_count"],1)
            self.assertFalse(f.report()["complete"])

    def test_runtime_exception_preserved_no_flush(self):
        error=RuntimeError("actual runtime error")
        with self.fixture(failure=error) as f:
            with self.assertRaises(RuntimeError) as caught:E.child([])
            self.assertIs(caught.exception,error)
            self.assertNotIn("flush",f.events)
            self.assertFalse(f.report()["complete"])

    def test_foreign_registration_before_any_scope(self):
        with self.fixture() as f:
            f.config["qualification_id"]="foreign"
            with self.assertRaises(ValueError):E.child([])
            self.assertEqual(f.events,[])


if __name__=="__main__": unittest.main()
