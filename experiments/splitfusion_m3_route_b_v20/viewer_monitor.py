"""Low-rate CPU/RSS sampler of the live-viewer processes only (viewer server, recorder node/chromium). Display tooling.

Every 2 s: per-process CPU seconds delta (utime+stime from /proc/<pid>/stat), RSS, CPU affinity, and /proc/loadavg.
Stops when <stopfile> exists.  Never touches pipeline processes.
"""
import json
import os
from pathlib import Path
import sys
import time

NEEDLES = ("replay_ui_v1.server --port 8766", "live_record_v1.cjs", "headless_shell")
TICK = os.sysconf("SC_CLK_TCK")


def matching():
    out = {}
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if any(n in command for n in NEEDLES) and "viewer_monitor" not in command:
            out[int(path.parent.name)] = command[:160]
    return out


def stat(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        rss = int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
        return (int(fields[11]) + int(fields[12])) / TICK, rss, sorted(os.sched_getaffinity(pid))
    except (OSError, ValueError, IndexError):
        return None


def main(out, stopfile):
    previous = {}
    with open(out, "x", buffering=1) as handle:
        while not Path(stopfile).exists():
            now = time.time()
            rows = {}
            for pid, command in matching().items():
                value = stat(pid)
                if value is None:
                    continue
                cpu, rss, affinity = value
                last = previous.get(pid)
                rows[pid] = dict(cmd=command, cpu_s=cpu, rss_mb=round(rss / 2**20, 1), cpus=affinity,
                                 cpu_pct=None if last is None else round(100 * (cpu - last[0]) / (now - last[1]), 1))
                previous[pid] = (cpu, now)
            handle.write(json.dumps(dict(wall=now, loadavg=Path("/proc/loadavg").read_text().split()[:3],
                                         total_cpu_pct=round(sum(r["cpu_pct"] or 0 for r in rows.values()), 1),
                                         processes=rows)) + "\n")
            time.sleep(2.0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
