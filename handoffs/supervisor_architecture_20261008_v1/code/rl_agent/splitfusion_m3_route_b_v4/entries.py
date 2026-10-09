"""V4 entry points: install the abort-safe exporter outside the retained v3 composition."""
from . import contract as C


def _bind_v3():
    from rl_agent.splitfusion_m3_route_b_v3 import entries as E
    from rl_agent.splitfusion_m3_route_b_v3 import gc_seams as GS
    E.C = C
    GS.C = C
    return E


def registration(path, root):
    return _bind_v3().registration(path, root)


def child(argv=None):
    from .abort_evidence import install_abort_evidence
    with install_abort_evidence() as state:
        result = _bind_v3().child(argv)
    return result


def edge(argv=None):
    return _bind_v3().edge(argv)
