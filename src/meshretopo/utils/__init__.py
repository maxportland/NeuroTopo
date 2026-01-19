"""Utility modules for MeshRetopo."""

from meshretopo.utils.timing import (
    TimeoutError,
    TimingResult,
    TimingLog,
    get_timing_log,
    reset_timing_log,
    timed_operation,
    with_timeout,
    run_with_timeout,
    ProgressTimer,
    configure_timeouts,
    TIMEOUT_CURVATURE,
    TIMEOUT_FEATURES,
    TIMEOUT_REMESH,
    TIMEOUT_EVALUATION,
    TIMEOUT_OPTIMIZATION,
    TIMEOUT_AUTOTUNER_ITERATION,
)

__all__ = [
    "TimeoutError",
    "TimingResult",
    "TimingLog",
    "get_timing_log",
    "reset_timing_log",
    "timed_operation",
    "with_timeout",
    "run_with_timeout",
    "ProgressTimer",
    "configure_timeouts",
    "TIMEOUT_CURVATURE",
    "TIMEOUT_FEATURES",
    "TIMEOUT_REMESH",
    "TIMEOUT_EVALUATION",
    "TIMEOUT_OPTIMIZATION",
    "TIMEOUT_AUTOTUNER_ITERATION",
]
