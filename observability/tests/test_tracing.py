import pytest
from observability.tracing import is_tracing_enabled, trace_span, trace_operation

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
