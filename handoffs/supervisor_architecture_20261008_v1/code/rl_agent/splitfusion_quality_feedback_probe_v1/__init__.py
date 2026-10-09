"""Privileged CARLA quality-feedback timing probe.

This package is diagnostic/training infrastructure.  It deliberately consumes
CARLA ground truth and therefore is not a deployable vehicle feedback path.
Production inference, map publication and the terminal map ledger remain owned
by :mod:`rl_agent.splitfusion_direct_edge_map_v1`.
"""

from .protocol import QUALITY_EVALUATED_ACK_SCHEMA

__all__ = ("QUALITY_EVALUATED_ACK_SCHEMA",)
