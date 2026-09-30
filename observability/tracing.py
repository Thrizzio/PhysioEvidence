import os
import time
import functools
import logging
from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

def is_tracing_enabled() -> bool:
    """
    Checks if LangSmith tracing is active and configured.
    """
    tracing_env = os.environ.get("LANGSMITH_TRACING", "false").lower()
    has_key = bool(os.environ.get("LANGSMITH_API_KEY"))
    return tracing_env in ("true", "1", "yes") and has_key

@contextmanager
def trace_span(name: str, span_type: str = "chain", inputs: Optional[Dict[str, Any]] = None):
    """
    Context manager for measuring latency and logging/tracing pipeline operations.
    Integrates with LangSmith if configured, otherwise logs execution cleanly.
    """
    start_time = time.perf_counter()
    logger.debug(f"[Trace Start] {name} (type: {span_type}) with inputs: {inputs}")
    status = "success"
    error = None

    try:
        yield
    except Exception as e:
        status = "error"
        error = str(e)
        logger.error(f"[Trace Error] {name} failed: {e}")
        raise
    finally:
        duration = time.perf_counter() - start_time
        logger.debug(f"[Trace End] {name} completed in {duration:.4f}s with status: {status}")

def trace_operation(name: Optional[str] = None, span_type: str = "chain"):
    """
    Decorator for tracing functions and pipeline components.
    Works transparently whether LangSmith is enabled or not.
    """
    def decorator(func: Callable):
        op_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to use langsmith traceable if available and configured
            if is_tracing_enabled():
                try:
                    from langsmith import traceable
                    return traceable(name=op_name, run_type=span_type)(func)(*args, **kwargs)
                except ImportError:
                    pass

            with trace_span(op_name, span_type=span_type, inputs={"args_count": len(args), "kwargs_keys": list(kwargs.keys())}):
                return func(*args, **kwargs)

        return wrapper
    return decorator
