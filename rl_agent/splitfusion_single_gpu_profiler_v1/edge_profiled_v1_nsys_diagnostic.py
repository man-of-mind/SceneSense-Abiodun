"""Launch the serial cap-only edge under an in-container Nsight collector."""
import os
import sys

def main():
    if os.environ.get('SCENESENSE_NSYS_EDGE_INJECTED') != '1':
        env = dict(os.environ, SCENESENSE_NSYS_EDGE_INJECTED='1')
        tool = '/diag_nsys/target-linux-x64/nsys'
        args = [tool, 'profile', '--trace=cuda,nvtx,osrt', '--sample=none',
                '--cpuctxsw=none', '--cuda-event-trace=false', '--start-later=true',
                '--session-new=codex_edge_045255', '--kill=none', '--duration=25',
                '--force-overwrite=false', '--output=/diag_output/edge_trace',
                sys.executable, '-u', '-m', __spec__.name, *sys.argv[1:]]
        os.execve(tool, args, env)
    from . import nsys_markers_diagnostic as N
    from . import edge_profiled_v1_ordering as E
    N.install('/diag_output/edge_nvtx_cost.json')
    return E.main()

if __name__ == '__main__':
    raise SystemExit(main())
