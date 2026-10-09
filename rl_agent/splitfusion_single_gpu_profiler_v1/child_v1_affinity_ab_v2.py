"""Retained C2 child; persistent affinity evidence and pre-teardown restoration."""
import os,sys,subprocess,time,json
from pathlib import Path

def main():
    from . import child_v1_threadcap as C
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_launch_v2 as EL
    from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
    p=Path(os.environ['SCENESENSE_AFFINITY_OUTPUT'])
    orig=EL.up_command
    def command(v):
        a=orig(v);i=a.index('up');return a[:i]+['-f',str(p/'compose_affinity.yaml')]+a[i:]
    EL.up_command=command;C.ORDERING_EDGE_MODULE='-m rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_affinity_ab_v2'
    previous=C.install_ordering_route
    def ordering(*args,**kwargs):
        previous(*args,**kwargs)
        stop=pinned.stop_tail
        def stop_tail():
            (p/'collection_stop').touch()
            try:
                if (p/'worker_ready.json').exists():
                    q=subprocess.run(['docker','exec','oai-perception-rx','python3','/affinity_output/restore_in_container.py','/affinity_output'],capture_output=True,text=True,timeout=15)
                    (p/'restore_command.json').write_text(json.dumps(dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr,raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)),indent=2)+'\n')
            finally:return stop()
        pinned.stop_tail=stop_tail
    C.install_ordering_route=ordering
    return C.main()
if __name__=='__main__':raise SystemExit(main())
