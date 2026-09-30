import os
import logging
from pathlib import Path
from typing import Optional
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def get_connection_string() -> str:
    """
    Constructs the PostgreSQL connection string from environment variables.
    Prefers DATABASE_URL if explicitly set.
    """
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return database_url

    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db_name = os.environ.get("POSTGRES_DB", "physioevidence")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "")

    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    return f"postgresql://{user}@{host}:{port}/{db_name}"

def get_connection(conn_str: Optional[str] = None) -> psycopg.Connection:
    """
    Opens a psycopg connection and registers the pgvector type adapter.
    """
    target_conn_str = conn_str or get_connection_string()
    try:
        conn = psycopg.connect(target_conn_str, autocommit=True)
        register_vector(conn)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL at '{target_conn_str}': {e}")
        raise

def init_db(conn: Optional[psycopg.Connection] = None, schema_path: Optional[Path] = None) -> None:
    """
    Initializes the database schema using storage/schema.sql.
    """
    if schema_path is None:
        schema_path = Path(__file__).parent / "schema.sql"

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at: {schema_path}")

    sql = schema_path.read_text(encoding="utf-8")
    
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        logger.info("Database schema initialized successfully.")
    finally:
        if close_after:
            conn.close()
