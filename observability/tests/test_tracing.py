import pytest
from observability.tracing import (
    is_tracing_enabled,
    trace_span,
    trace_operation,
    _normalize_run_type,
    _sanitize_inputs,
)


def test_is_tracing_enabled_false_by_default(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert is_tracing_enabled() is False


def test_is_tracing_enabled_requires_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert is_tracing_enabled() is False

    monkeypatch.setenv("LANGSMITH_API_KEY", "test_key_123")
    assert is_tracing_enabled() is True


def test_is_tracing_enabled_sdk_not_available(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test_key_123")
    monkeypatch.setattr("observability.tracing.LANGSMITH_AVAILABLE", False)
    assert is_tracing_enabled() is False


def test_normalize_run_type():
    assert _normalize_run_type("reranker") == "chain"
    assert _normalize_run_type("retriever") == "retriever"
    assert _normalize_run_type("tool") == "tool"
    assert _normalize_run_type("unknown_custom_type") == "chain"


def test_sanitize_inputs():
    inputs = {
        "api_key": "secret_123",
        "user_token": "bearer_abc",
        "query": "knee rehab",
        "top_k": 5,
        "is_valid": True,
        "obj": object(),
    }
    sanitized = _sanitize_inputs(inputs)
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["user_token"] == "[REDACTED]"
    assert sanitized["query"] == "knee rehab"
    assert sanitized["top_k"] == 5
    assert sanitized["is_valid"] is True
    assert "object at" in sanitized["obj"]


def test_trace_span_context_manager():
    executed = False
    with trace_span("test_span", span_type="tool", inputs={"query": "patella"}):
        executed = True
    assert executed is True


def test_trace_operation_decorator():
    @trace_operation(name="sample_calculation", span_type="chain")
    def calculate(a: int, b: int) -> int:
        return a + b

    result = calculate(3, 4)
    assert result == 7


def test_trace_operation_on_class_method():
    class DummyService:
        @trace_operation(name="service_work", span_type="reranker")
        def work(self, text: str) -> str:
            return text.upper()

    svc = DummyService()
    assert svc.work("patella") == "PATELLA"


def test_trace_operation_exception_propagation():
    @trace_operation(name="failing_func")
    def failing():
        raise ValueError("simulated pipeline error")

    with pytest.raises(ValueError, match="simulated pipeline error"):
        failing()
