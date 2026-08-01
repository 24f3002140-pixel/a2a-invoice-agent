import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = os.getenv(
    "DATABASE_PATH",
    str(BASE_DIR / "invoice_agent.db"),
)


def get_connection() -> sqlite3.Connection:
    """
    Create a SQLite connection configured for concurrent API requests.
    """
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
        check_same_thread=False,
    )

    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=FULL;")
    connection.execute("PRAGMA foreign_keys=ON;")
    connection.execute("PRAGMA busy_timeout=30000;")

    return connection


@contextmanager
def database_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that commits successful operations
    and rolls back failed operations.
    """
    connection = get_connection()

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    """
    Create all required tables and indexes.
    """

    with database_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                principal_hash TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                original_message_id TEXT NOT NULL,
                original_message_hash TEXT NOT NULL,

                task_state TEXT NOT NULL,
                task_json TEXT NOT NULL,
                proposals_json TEXT,
                receipts_json TEXT,

                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                terminal_at TEXT,

                UNIQUE(principal_hash, original_message_id)
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_principal
            ON tasks(principal_hash);

            CREATE INDEX IF NOT EXISTS idx_tasks_principal_created
            ON tasks(principal_hash, created_at);

            CREATE INDEX IF NOT EXISTS idx_tasks_context
            ON tasks(context_id);

            CREATE INDEX IF NOT EXISTS idx_tasks_batch
            ON tasks(batch_id);


            CREATE TABLE IF NOT EXISTS message_idempotency (
                principal_hash TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_hash TEXT NOT NULL,
                task_id TEXT NOT NULL,
                created_at TEXT NOT NULL,

                PRIMARY KEY(principal_hash, message_id),

                FOREIGN KEY(task_id)
                    REFERENCES tasks(task_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_idempotency_task
            ON message_idempotency(task_id);


            CREATE TABLE IF NOT EXISTS package_decision_cache (
                package_hash TEXT PRIMARY KEY,
                package_json TEXT NOT NULL,
                decision_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );


            CREATE TABLE IF NOT EXISTS processed_continuations (
                principal_hash TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_hash TEXT NOT NULL,
                task_id TEXT NOT NULL,
                response_task_json TEXT NOT NULL,
                created_at TEXT NOT NULL,

                PRIMARY KEY(principal_hash, message_id),

                FOREIGN KEY(task_id)
                    REFERENCES tasks(task_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_continuations_task
            ON processed_continuations(task_id);


            CREATE TABLE IF NOT EXISTS task_locks (
                task_id TEXT PRIMARY KEY,
                lock_owner TEXT NOT NULL,
                acquired_at TEXT NOT NULL
            );
            """
        )


def database_healthcheck() -> bool:
    """
    Verify that SQLite is reachable.
    """
    try:
        with database_connection() as connection:
            row = connection.execute("SELECT 1 AS healthy").fetchone()
            return bool(row and row["healthy"] == 1)
    except Exception:
        return False