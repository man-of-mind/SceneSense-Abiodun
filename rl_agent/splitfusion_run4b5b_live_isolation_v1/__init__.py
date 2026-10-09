"""Deployment-safe primitives shared by the Run-4B and Run-5B live paths.

This additive package deliberately contains no launcher and performs no I/O at
import time.  It separates the operational tail-output acknowledgement from
the map and research-evaluation branches.  CARLA ground truth and Q_perc are
therefore not prerequisites for closing a live policy action.
"""

from .operational_ack_v1 import (  # noqa: F401
    ACK_DEADLINE_NS,
    AckClass,
    FrameActionIdentityV1,
    OperationalAckLedgerV1,
    OperationalTerminal,
    TailOutputAckV1,
    decode_ack,
    encode_ack,
)
