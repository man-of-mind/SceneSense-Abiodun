"""Retained C2 child plus diagnostic launch/flush barriers, no pipeline edits."""
import os,sys,time
from pathlib import Path

def main(argv=None):
    from . import child_v1_threadcap as C
    from . import nsys_control_diagnostic_v2 as CT
    from . import nsys_markers_diagnostic as N
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_launch_v2 as EL
    out=Path(os.environ['SCENESENSE_NSYS_OUTPUT'])
    sys.path.insert(0,str(out))
    from collector_lib import bounded,wait_file,write,inventory
    state=CT.install(out/'route/host');N.install(state/'nvtx_cost.json')
    original=EL.up_command
    def command(variables):
        a=original(variables);i=a.index('up')
        return a[:i]+['-f',str(out/'compose_nsys.yaml')]+a[i:]
    EL.up_command=command
    old=C.install_ordering_route
    def ordering(*args,**kwargs):
        old(*args,**kwargs)
        from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as D
        from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
        D.DIRECT_EDGE_MODULE='-m rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_nsys_diagnostic_v2'
        old_stop=pinned.stop_tail
        def stop():
            edge=out/'route/edge';errors=[]
            # Every wait is bounded. Neither timeout nor normal teardown deletes trace storage.
            if (edge/'supervisor_start.json').exists():
                try:wait_file(edge/'flush_complete.json',90)
                except Exception as exc:errors.append(repr(exc));write(edge/'FLUSH_TIMEOUT_PRESERVED.json',dict(available=inventory(edge)))
                (edge/'app_stop').touch()
                for name in ['application_footer.json','application_launch_exit.json']:
                    try:wait_file(edge/name,30)
                    except Exception as exc:errors.append(repr(exc))
                result=bounded(['docker','exec','-e','TMPDIR=/diag_output/route/edge/tmp','oai-perception-rx',
                    '/diag_nsys/target-linux-x64/nsys','shutdown','--session=codex_v2_route_edge','--kill=none'],edge/'shutdown.log',15)
                if result['timed_out']:errors.append('collector shutdown timeout; persistent evidence retained')
                (edge/'supervisor_exit').touch()
                try:wait_file(edge/'supervisor_footer.json',15)
                except Exception as exc:errors.append(repr(exc))
                write(edge/'teardown_barrier.json',dict(errors=errors,**CT.stamp()))
            return old_stop()
        pinned.stop_tail=stop
    C.install_ordering_route=ordering
    try:return C.main(argv)
    finally:CT.footer(state)

if __name__=='__main__':raise SystemExit(main())
