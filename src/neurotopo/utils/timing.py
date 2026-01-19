"""
Timing and timeout utilities for long-running processes.

Provides:
- Timeout decorators and context managers
- Performance logging
- Progress tracking for iterative operations
"""

from __future__ import annotations

import time
import signal
import functools
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from threading import Thread
import queue

# Configure logging
logger = logging.getLogger("neurotopo.timing")


class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


@dataclass
class TimingResult:
    """Result of a timed operation."""
    operation: str
    elapsed_seconds: float
    success: bool
    timed_out: bool = False
    error: Optional[str] = None
    
    def __str__(self) -> str:
        status = "OK" if self.success else ("TIMEOUT" if self.timed_out else "ERROR")
        return f"{self.operation}: {self.elapsed_seconds:.3f}s [{status}]"


@dataclass
class TimingLog:
    """Accumulated timing information for a pipeline run."""
    entries: list[TimingResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    def add(self, result: TimingResult):
        """Add a timing result."""
        self.entries.append(result)
        logger.info(str(result))
    
    def total_time(self) -> float:
        """Total elapsed time."""
        return time.time() - self.start_time
    
    def summary(self) -> str:
        """Generate summary of all timings."""
        lines = ["Timing Summary:", "-" * 40]
        for entry in self.entries:
            lines.append(f"  {entry}")
        lines.append("-" * 40)
        lines.append(f"  Total: {self.total_time():.3f}s")
        return "\n".join(lines)
    
    def get_slowest(self, n: int = 3) -> list[TimingResult]:
        """Get the n slowest operations."""
        return sorted(self.entries, key=lambda x: x.elapsed_seconds, reverse=True)[:n]


# Global timing log for current pipeline run
_current_timing_log: Optional[TimingLog] = None


def get_timing_log() -> TimingLog:
    """Get or create the current timing log."""
    global _current_timing_log
    if _current_timing_log is None:
        _current_timing_log = TimingLog()
    return _current_timing_log


def reset_timing_log():
    """Reset the timing log for a new run."""
    global _current_timing_log
    _current_timing_log = TimingLog()
    return _current_timing_log


@contextmanager
def timed_operation(name: str, timeout: Optional[float] = None, log: bool = True):
    """
    Context manager for timing an operation with optional timeout.
    
    Args:
        name: Name of the operation for logging
        timeout: Maximum time in seconds (None = no timeout)
        log: Whether to log the timing
        
    Yields:
        TimingResult that will be populated on exit
        
    Raises:
        TimeoutError: If operation exceeds timeout
    """
    result = TimingResult(operation=name, elapsed_seconds=0, success=False)
    start = time.time()
    
    try:
        yield result
        result.success = True
    except TimeoutError:
        result.timed_out = True
        result.error = f"Timeout after {timeout}s"
        raise
    except Exception as e:
        result.error = str(e)
        raise
    finally:
        result.elapsed_seconds = time.time() - start
        if log:
            get_timing_log().add(result)


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")


def with_timeout(timeout: float, operation_name: str = "operation"):
    """
    Decorator to add timeout to a function.
    
    Args:
        timeout: Maximum time in seconds
        operation_name: Name for logging
        
    Note: Uses SIGALRM, only works on Unix systems.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set up signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout)
            
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                logger.debug(f"{operation_name} completed in {elapsed:.3f}s")
                return result
            except TimeoutError:
                elapsed = time.time() - start
                logger.warning(f"{operation_name} timed out after {elapsed:.3f}s (limit: {timeout}s)")
                raise
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


def run_with_timeout(func: Callable, args: tuple = (), kwargs: dict = None,
                     timeout: float = 60.0, operation_name: str = "operation") -> Any:
    """
    Run a function with timeout using a thread.
    
    This is more portable than signal-based timeout.
    
    Args:
        func: Function to run
        args: Positional arguments
        kwargs: Keyword arguments
        timeout: Maximum time in seconds
        operation_name: Name for logging
        
    Returns:
        Function result
        
    Raises:
        TimeoutError: If function exceeds timeout
    """
    kwargs = kwargs or {}
    result_queue = queue.Queue()
    error_queue = queue.Queue()
    
    def target():
        try:
            result = func(*args, **kwargs)
            result_queue.put(result)
        except Exception as e:
            error_queue.put(e)
    
    start = time.time()
    thread = Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    elapsed = time.time() - start
    
    if thread.is_alive():
        logger.warning(f"{operation_name} timed out after {elapsed:.3f}s (limit: {timeout}s)")
        raise TimeoutError(f"{operation_name} timed out after {timeout}s")
    
    if not error_queue.empty():
        raise error_queue.get()
    
    if not result_queue.empty():
        logger.debug(f"{operation_name} completed in {elapsed:.3f}s")
        return result_queue.get()
    
    raise RuntimeError(f"{operation_name} completed but produced no result")


class ProgressTimer:
    """
    Track progress of iterative operations.
    
    Estimates time remaining and logs progress.
    """
    
    def __init__(self, total: int, operation_name: str = "Processing",
                 log_interval: float = 5.0):
        self.total = total
        self.operation_name = operation_name
        self.log_interval = log_interval
        self.current = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
    
    def update(self, n: int = 1):
        """Update progress by n items."""
        self.current += n
        
        now = time.time()
        if now - self.last_log_time >= self.log_interval:
            self._log_progress()
            self.last_log_time = now
    
    def _log_progress(self):
        """Log current progress."""
        elapsed = time.time() - self.start_time
        if self.current > 0:
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            pct = 100 * self.current / self.total
            logger.info(f"{self.operation_name}: {pct:.1f}% ({self.current}/{self.total}) "
                       f"- {elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining")
    
    def finish(self):
        """Mark operation as complete and log final timing."""
        elapsed = time.time() - self.start_time
        rate = self.total / elapsed if elapsed > 0 else 0
        logger.info(f"{self.operation_name}: Complete - {self.total} items in {elapsed:.3f}s "
                   f"({rate:.1f} items/s)")


# Default timeout values (in seconds)
TIMEOUT_CURVATURE = 30.0
TIMEOUT_FEATURES = 30.0
TIMEOUT_REMESH = 120.0
TIMEOUT_EVALUATION = 60.0
TIMEOUT_OPTIMIZATION = 60.0
TIMEOUT_AUTOTUNER_ITERATION = 30.0


def configure_timeouts(
    curvature: float = None,
    features: float = None,
    remesh: float = None,
    evaluation: float = None,
    optimization: float = None,
    autotuner_iteration: float = None,
):
    """Configure default timeout values."""
    global TIMEOUT_CURVATURE, TIMEOUT_FEATURES, TIMEOUT_REMESH
    global TIMEOUT_EVALUATION, TIMEOUT_OPTIMIZATION, TIMEOUT_AUTOTUNER_ITERATION
    
    if curvature is not None:
        TIMEOUT_CURVATURE = curvature
    if features is not None:
        TIMEOUT_FEATURES = features
    if remesh is not None:
        TIMEOUT_REMESH = remesh
    if evaluation is not None:
        TIMEOUT_EVALUATION = evaluation
    if optimization is not None:
        TIMEOUT_OPTIMIZATION = optimization
    if autotuner_iteration is not None:
        TIMEOUT_AUTOTUNER_ITERATION = autotuner_iteration
