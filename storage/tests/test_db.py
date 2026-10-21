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


def test_get_connection_string_supabase_pooler(monkeypatch):
    from services.config import mask_connection_string

    pooler_url = "postgresql://postgres.sample_project_ref:super_secret_pw@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres?sslmode=require"
    monkeypatch.setenv("DATABASE_URL", pooler_url)
    conn_str = get_connection_string()
    assert conn_str == pooler_url

    masked = mask_connection_string(conn_str)
    assert "super_secret_pw" not in masked
    assert "postgres.sample_project_ref" in masked
    assert "aws-0-ap-southeast-1.pooler.supabase.com" in masked
    assert "****" in masked

