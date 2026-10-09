"""Container-local collector supervisor with persistent raw/report storage."""
import os,sys

def main():
    if len(sys.argv)<2 or sys.argv[1]!='--nsys-inner':
        env=dict(os.environ,SCENESENSE_NSYS_ARM_FIRST='1')
        args=[sys.executable,'/diag_output/container_supervisor.py','/diag_output/route/edge',
              'codex_v3_route_edge',sys.executable,'-u','-m',__spec__.name,'--nsys-inner',*sys.argv[1:]]
        os.execve(sys.executable,args,env)
    from . import nsys_control_diagnostic_v2 as C
    from . import nsys_markers_diagnostic as N
    from . import edge_profiled_v1_ordering as E
    state=C.install('/diag_output/route/edge')
    N.STAGES={'edge.codec.frozen_tail'}
    N.install(state/'nvtx_cost.json')
    from . import recorder_v1 as R
    import threading
    saved=set();enter=R._Span.__enter__
    def identify(span):
        if span._stage=='edge.compute' and span._rec._phase=='live':
            tid=threading.get_native_id()
            if tid not in saved:
                C.write(state/('compute_tid_'+str(tid)+'.json'),dict(pid=os.getpid(),tid=tid,context=span._rec.context(),**C.stamp()))
                saved.add(tid)
        return enter(span)
    R._Span.__enter__=identify
    try:return E.main(sys.argv[2:])
    finally:C.footer(state)

if __name__=='__main__':raise SystemExit(main())
