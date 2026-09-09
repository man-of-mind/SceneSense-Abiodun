"""SplitFusion timing diagnostic v1.

One short, targeted decomposition of the SplitFusion edge service and of the
application-level feature uplink through the live OAI 5G path. This package
adds instrumentation only: it wraps the qualified Phase-13/Phase-15 runtime
through subclasses and callbacks and never edits a hash-bound production
runtime file, an action definition, a codec, a threshold or a checkpoint.
"""

from __future__ import annotations

DIAGNOSTIC_ID = "splitfusion_timing_diagnostic_v1"
EXECUTE_TOKEN = "SPLITFUSION_TIMING_DIAGNOSTIC_V1_EXECUTE"
