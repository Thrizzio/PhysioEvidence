import os
from storage.db import get_connection_string

def test_get_connection_string_default(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_HOST", "db.example.com")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "test_db")
    monkeypatch.setenv("POSTGRES_USER", "tester")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret123")

    conn_str = get_connection_string()
    assert conn_str == "postgresql://tester:secret123@db.example.com:5433/test_db"

def test_get_connection_string_url_override(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://custom_user:custom_pass@other_host:5432/custom_db")
    conn_str = get_connection_string()
    assert conn_str == "postgresql://custom_user:custom_pass@other_host:5432/custom_db"
