import sys,time
from pathlib import Path
from affinity_lib import write,raw
p=Path(sys.argv[1]);write(p/'perf_ready.json',dict(raw_ns=raw()));end=time.monotonic()+600
while not (p/'collection_stop').exists() and time.monotonic()<end:time.sleep(.1)
write(p/'perf_sentinel_footer.json',dict(raw_ns=raw(),stop_marker=(p/'collection_stop').exists()))
