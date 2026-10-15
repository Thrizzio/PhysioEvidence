import os
import time
import functools
import logging
from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager

# Load environment variables if python-dotenv is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Optional LangSmith SDK integration: safe top-level import with graceful fallback
try:
    import langsmith
    from langsmith import trace, traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    trace = None       # type: ignore
    traceable = None   # type: ignore

# Valid LangSmith run types
VALID_RUN_TYPES = {"chain", "embedding", "tool", "llm", "retriever", "prompt", "parser"}

# Mapping internal domain span types (e.g. reranker) to valid LangSmith run types
RUN_TYPE_MAPPING = {
    "reranker": "chain",
    "ranker": "chain",
    "synthesizer": "chain",
    "pipeline": "chain",
    "evaluator": "chain",
    "filter": "tool",
}


def _normalize_run_type(span_type: str) -> str:
    """
    Normalizes span types to one of LangSmith's supported run types.
    Prevents runtime validation warnings and errors.
    """
    normalized = RUN_TYPE_MAPPING.get(span_type.lower(), span_type.lower())
    return normalized if normalized in VALID_RUN_TYPES else "chain"


def _sanitize_inputs(inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sanitizes trace inputs to prevent leaking secrets/credentials (ADR-015).
    Safely serializes complex Python objects.
    """
    if not inputs:
        return {}
    safe: Dict[str, Any] = {}
    sensitive_keywords = ("key", "secret", "token", "password", "credential", "auth")
    for k, v in inputs.items():
        k_str = str(k).lower()
        if any(keyword in k_str for keyword in sensitive_keywords):
            safe[str(k)] = "[REDACTED]"
        elif isinstance(v, (str, int, float, bool, list, dict)) or v is None:
            safe[str(k)] = v
        else:
            safe[str(k)] = repr(v)
    return safe


def is_tracing_enabled() -> bool:
    """
    Checks if LangSmith tracing is active, configured, and the SDK is installed.
    """
    if not LANGSMITH_AVAILABLE:
        return False
    tracing_env = os.environ.get("LANGSMITH_TRACING", "false").lower()
    has_key = bool(os.environ.get("LANGSMITH_API_KEY"))
    return tracing_env in ("true", "1", "yes") and has_key


@contextmanager
def trace_span(name: str, span_type: str = "chain", inputs: Optional[Dict[str, Any]] = None):
    """
    Context manager for measuring latency and logging/tracing pipeline operations.
    Integrates with LangSmith if configured and available, otherwise logs execution cleanly.
    """
    start_time = time.perf_counter()
    run_type = _normalize_run_type(span_type)
    safe_inputs = _sanitize_inputs(inputs)
    logger.debug(f"[Trace Start] {name} (type: {run_type}) with inputs: {safe_inputs}")
    status = "success"

    langsmith_cm = None
    span = None
    if is_tracing_enabled() and trace is not None:
        try:
            langsmith_cm = trace(name=name, run_type=run_type, inputs=safe_inputs)
            span = langsmith_cm.__enter__()
        except Exception as e:
            logger.warning(f"[Trace Warning] LangSmith trace span creation failed for '{name}': {e}")
            langsmith_cm = None
            span = None

    try:
        yield span
    except Exception as e:
        status = "error"
        logger.error(f"[Trace Error] {name} failed: {e}")
        if langsmith_cm:
            try:
                langsmith_cm.__exit__(type(e), e, e.__traceback__)
            except Exception:
                pass
            langsmith_cm = None
        raise
    finally:
        if langsmith_cm:
            try:
                langsmith_cm.__exit__(None, None, None)
            except Exception:
                pass
        duration = time.perf_counter() - start_time
        logger.debug(f"[Trace End] {name} completed in {duration:.4f}s with status: {status}")


def trace_operation(name: Optional[str] = None, span_type: str = "chain"):
    """
    Decorator for tracing functions, methods, and pipeline components.
    Works transparently whether LangSmith is enabled or not.
    """
    def decorator(func: Callable):
        op_name = name or func.__name__
        run_type = _normalize_run_type(span_type)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Format high-level inputs, avoiding large/unserializable self on class methods
            meta: Dict[str, Any] = {}
            if args:
                first_arg = args[0]
                if hasattr(first_arg, "__class__") and not isinstance(
                    first_arg, (str, int, float, bool, list, dict, tuple)
                ):
                    meta["class"] = first_arg.__class__.__name__
                    meta["args_count"] = len(args) - 1
                else:
                    meta["args_count"] = len(args)
            if kwargs:
                meta["kwargs_keys"] = list(kwargs.keys())

            with trace_span(op_name, span_type=run_type, inputs=meta) as span:
                result = func(*args, **kwargs)
                if span is not None and hasattr(span, "outputs"):
                    try:
                        if isinstance(result, (str, int, float, bool)):
                            span.outputs = {"result": result}
                        elif result is not None:
                            span.outputs = {"output_type": type(result).__name__}
                    except Exception:
                        pass
                return result

        return wrapper
    return decorator
