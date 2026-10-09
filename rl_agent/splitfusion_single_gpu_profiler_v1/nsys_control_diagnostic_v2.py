"""Diagnostic-only capture barriers; no tensor, queue or scientific changes."""
import json, os, sys, threading, time
from pathlib import Path

def stamp():
    a=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    w=time.time_ns(); b=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    return dict(raw_a=a,wall_ns=w,raw_b=b)

def write(p,body):
    p=Path(p);q=p.with_suffix(p.suffix+'.pending')
    q.write_text(json.dumps(body,indent=2)+'\n');q.replace(p)

def install(state):
    p=Path(state);p.mkdir(parents=True,exist_ok=True)
    write(p/'application_pid.json',dict(pid=os.getpid(),**stamp()))
    def control():
        started=False
        try:
            while True:
                torch=sys.modules.get('torch')
                if not started and (p/'trace_start').exists() and torch is not None and torch.cuda.is_initialized():
                    before=stamp();torch.cuda.profiler.start();started=True
                    write(p/'trace_started.json',dict(before=before,after=stamp()))
                if started and (p/'trace_stop').exists():
                    before=stamp();torch.cuda.profiler.stop()
                    write(p/'trace_stopped.json',dict(before=before,after=stamp()));return
                time.sleep(.05)
        except BaseException as exc:
            write(p/'trace_control_error.json',dict(error=repr(exc),**stamp()))
    threading.Thread(target=control,name='diagnostic-nsys-control',daemon=True).start()
    return p

def footer(state):
    write(Path(state)/'application_footer.json',dict(pid=os.getpid(),main_returned=True,**stamp()))
