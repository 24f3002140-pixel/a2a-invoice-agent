import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from database import database_connection


TERMINAL_STATES = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_CANCELED",
}


class StorageError(Exception):
    pass


class TaskNotFoundError(StorageError):
    pass


class TaskAccessError(StorageError):
    pass


class IdempotencyConflictError(StorageError):
    pass


class TaskConflictError(StorageError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    """
    Recursively key-sort JSON and remove unnecessary whitespace.
    Array order is preserved because array order can be semantically important.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def hash_json(value: Any) -> str:
    encoded = canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_principal(bearer_token: str) -> str:
    """
    Store only a hash of the Bearer token.
    Never persist the raw authentication token.
    """
    return hashlib.sha256(
        bearer_token.encode("utf-8")
    ).hexdigest()


def parse_json(value: Optional[str], default: Any = None) -> Any:
    if value is None:
        return default

    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise StorageError("Stored JSON is invalid.") from exc


def task_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return parse_json(row["task_json"], {})


def get_task_record(
    task_id: str,
    principal_hash: str,
) -> Optional[sqlite3.Row]:
    with database_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM tasks
            WHERE task_id = ?
              AND principal_hash = ?
            """,
            (task_id, principal_hash),
        ).fetchone()

        return row


def get_task(
    task_id: str,
    principal_hash: str,
) -> Dict[str, Any]:
    """
    Read a task only when it belongs to the authenticated principal.
    """
    row = get_task_record(task_id, principal_hash)

    if row is None:
        raise TaskNotFoundError("Task not found.")

    return task_row_to_dict(row)


def list_tasks(
    principal_hash: str,
) -> List[Dict[str, Any]]:
    """
    Return only tasks owned by this principal.
    """
    with database_connection() as connection:
        rows = connection.execute(
            """
            SELECT task_json
            FROM tasks
            WHERE principal_hash = ?
            ORDER BY created_at ASC
            """,
            (principal_hash,),
        ).fetchall()

    return [
        parse_json(row["task_json"], {})
        for row in rows
    ]


def find_initial_idempotency(
    principal_hash: str,
    message_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Return the stored idempotency record for an initial request.
    """
    with database_connection() as connection:
        row = connection.execute(
            """
            SELECT
                i.message_hash,
                i.task_id,
                t.task_json
            FROM message_idempotency AS i
            JOIN tasks AS t
              ON t.task_id = i.task_id
            WHERE i.principal_hash = ?
              AND i.message_id = ?
            """,
            (principal_hash, message_id),
        ).fetchone()

    if row is None:
        return None

    return {
        "message_hash": row["message_hash"],
        "task_id": row["task_id"],
        "task": parse_json(row["task_json"], {}),
    }


def resolve_initial_replay(
    principal_hash: str,
    message_id: str,
    message_hash: str,
) -> Optional[Dict[str, Any]]:
    """
    Return the previous task for a semantically identical retry.

    Raise a conflict when the same message ID is reused with different
    semantic message content.
    """
    record = find_initial_idempotency(
        principal_hash=principal_hash,
        message_id=message_id,
    )

    if record is None:
        return None

    if record["message_hash"] != message_hash:
        raise IdempotencyConflictError(
            "The message ID was reused with different content."
        )

    return record["task"]


def create_initial_task(
    *,
    task_id: str,
    context_id: str,
    principal_hash: str,
    batch_id: str,
    message_id: str,
    message_hash: str,
    task: Dict[str, Any],
    proposals: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    """
    Atomically store an initial INPUT_REQUIRED task.

    Returns:
        (task, created)

    If another equivalent request won the race, this returns the already
    stored task. If the message ID was reused with different content,
    an IdempotencyConflictError is raised.
    """
    created_at = utc_now()
    task_json = canonical_json(task)
    proposals_json = canonical_json(proposals)

    with database_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")

        existing = connection.execute(
            """
            SELECT
                i.message_hash,
                t.task_json
            FROM message_idempotency AS i
            JOIN tasks AS t
              ON t.task_id = i.task_id
            WHERE i.principal_hash = ?
              AND i.message_id = ?
            """,
            (principal_hash, message_id),
        ).fetchone()

        if existing is not None:
            if existing["message_hash"] != message_hash:
                raise IdempotencyConflictError(
                    "The message ID was reused with different content."
                )

            return parse_json(existing["task_json"], {}), False

        connection.execute(
            """
            INSERT INTO tasks (
                task_id,
                context_id,
                principal_hash,
                batch_id,
                original_message_id,
                original_message_hash,
                task_state,
                task_json,
                proposals_json,
                receipts_json,
                created_at,
                updated_at,
                terminal_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL)
            """,
            (
                task_id,
                context_id,
                principal_hash,
                batch_id,
                message_id,
                message_hash,
                task.get("state", "TASK_STATE_INPUT_REQUIRED"),
                task_json,
                proposals_json,
                created_at,
                created_at,
            ),
        )

        connection.execute(
            """
            INSERT INTO message_idempotency (
                principal_hash,
                message_id,
                message_hash,
                task_id,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                principal_hash,
                message_id,
                message_hash,
                task_id,
                created_at,
            ),
        )

    return task, True


def find_continuation_idempotency(
    principal_hash: str,
    message_id: str,
) -> Optional[Dict[str, Any]]:
    with database_connection() as connection:
        row = connection.execute(
            """
            SELECT
                message_hash,
                task_id,
                response_task_json
            FROM processed_continuations
            WHERE principal_hash = ?
              AND message_id = ?
            """,
            (principal_hash, message_id),
        ).fetchone()

    if row is None:
        return None

    return {
        "message_hash": row["message_hash"],
        "task_id": row["task_id"],
        "task": parse_json(row["response_task_json"], {}),
    }


def resolve_continuation_replay(
    principal_hash: str,
    message_id: str,
    message_hash: str,
) -> Optional[Dict[str, Any]]:
    record = find_continuation_idempotency(
        principal_hash=principal_hash,
        message_id=message_id,
    )

    if record is None:
        return None

    if record["message_hash"] != message_hash:
        raise IdempotencyConflictError(
            "The continuation message ID was reused with different content."
        )

    return record["task"]


def get_task_with_proposals(
    task_id: str,
    principal_hash: str,
) -> Dict[str, Any]:
    """
    Read task metadata and its stored proposal artifact.
    """
    with database_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM tasks
            WHERE task_id = ?
              AND principal_hash = ?
            """,
            (task_id, principal_hash),
        ).fetchone()

    if row is None:
        raise TaskNotFoundError("Task not found.")

    return {
        "task_id": row["task_id"],
        "context_id": row["context_id"],
        "batch_id": row["batch_id"],
        "state": row["task_state"],
        "task": parse_json(row["task_json"], {}),
        "proposals": parse_json(row["proposals_json"], {}),
        "receipts": parse_json(row["receipts_json"], None),
    }


def complete_task(
    *,
    task_id: str,
    principal_hash: str,
    continuation_message_id: str,
    continuation_message_hash: str,
    completed_task: Dict[str, Any],
    receipts: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    """
    Atomically change INPUT_REQUIRED to COMPLETED and store the continuation.

    Exactly one of completion or cancellation can win because both use
    BEGIN IMMEDIATE and verify the stored nonterminal state.

    Returns:
        (task, changed)

    changed=False means an identical continuation had already completed.
    """
    now = utc_now()
    completed_task_json = canonical_json(completed_task)
    receipts_json = canonical_json(receipts)

    with database_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")

        replay = connection.execute(
            """
            SELECT
                message_hash,
                response_task_json
            FROM processed_continuations
            WHERE principal_hash = ?
              AND message_id = ?
            """,
            (
                principal_hash,
                continuation_message_id,
            ),
        ).fetchone()

        if replay is not None:
            if replay["message_hash"] != continuation_message_hash:
                raise IdempotencyConflictError(
                    "The continuation message ID was reused with different content."
                )

            return parse_json(replay["response_task_json"], {}), False

        row = connection.execute(
            """
            SELECT
                task_state,
                task_json
            FROM tasks
            WHERE task_id = ?
              AND principal_hash = ?
            """,
            (
                task_id,
                principal_hash,
            ),
        ).fetchone()

        if row is None:
            raise TaskNotFoundError("Task not found.")

        current_state = row["task_state"]

        if current_state in TERMINAL_STATES:
            raise TaskConflictError(
                "The task is already terminal."
            )

        if current_state != "TASK_STATE_INPUT_REQUIRED":
            raise TaskConflictError(
                "The task is not waiting for results."
            )

        connection.execute(
            """
            UPDATE tasks
            SET task_state = ?,
                task_json = ?,
                receipts_json = ?,
                updated_at = ?,
                terminal_at = ?
            WHERE task_id = ?
              AND principal_hash = ?
              AND task_state = ?
            """,
            (
                "TASK_STATE_COMPLETED",
                completed_task_json,
                receipts_json,
                now,
                now,
                task_id,
                principal_hash,
                "TASK_STATE_INPUT_REQUIRED",
            ),
        )

        if connection.total_changes < 1:
            raise TaskConflictError(
                "The task changed concurrently."
            )

        connection.execute(
            """
            INSERT INTO processed_continuations (
                principal_hash,
                message_id,
                message_hash,
                task_id,
                response_task_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                principal_hash,
                continuation_message_id,
                continuation_message_hash,
                task_id,
                completed_task_json,
                now,
            ),
        )

    return completed_task, True


def cancel_task(
    *,
    task_id: str,
    principal_hash: str,
    canceled_task: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Atomically cancel a nonterminal task.

    A terminal task never changes. This ensures completion and cancellation
    cannot both succeed.
    """
    now = utc_now()
    canceled_task_json = canonical_json(canceled_task)

    with database_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")

        row = connection.execute(
            """
            SELECT
                task_state,
                task_json
            FROM tasks
            WHERE task_id = ?
              AND principal_hash = ?
            """,
            (
                task_id,
                principal_hash,
            ),
        ).fetchone()

        if row is None:
            raise TaskNotFoundError("Task not found.")

        current_state = row["task_state"]

        if current_state in TERMINAL_STATES:
            raise TaskConflictError(
                "The task is already terminal."
            )

        cursor = connection.execute(
            """
            UPDATE tasks
            SET task_state = ?,
                task_json = ?,
                receipts_json = NULL,
                updated_at = ?,
                terminal_at = ?
            WHERE task_id = ?
              AND principal_hash = ?
              AND task_state NOT IN (?, ?)
            """,
            (
                "TASK_STATE_CANCELED",
                canceled_task_json,
                now,
                now,
                task_id,
                principal_hash,
                "TASK_STATE_COMPLETED",
                "TASK_STATE_CANCELED",
            ),
        )

        if cursor.rowcount != 1:
            raise TaskConflictError(
                "The task changed concurrently."
            )

    return canceled_task


def get_cached_decision(
    package_hash: str,
) -> Optional[Dict[str, Any]]:
    with database_connection() as connection:
        row = connection.execute(
            """
            SELECT decision_json
            FROM package_decision_cache
            WHERE package_hash = ?
            """,
            (package_hash,),
        ).fetchone()

    if row is None:
        return None

    return parse_json(row["decision_json"], {})


def save_cached_decision(
    *,
    package_hash: str,
    package: Dict[str, Any],
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Cache a semantic package decision.

    Concurrent attempts are safe. The first stored decision is retained.
    """
    now = utc_now()
    package_json = canonical_json(package)
    decision_json = canonical_json(decision)

    with database_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")

        existing = connection.execute(
            """
            SELECT decision_json
            FROM package_decision_cache
            WHERE package_hash = ?
            """,
            (package_hash,),
        ).fetchone()

        if existing is not None:
            return parse_json(existing["decision_json"], {})

        connection.execute(
            """
            INSERT INTO package_decision_cache (
                package_hash,
                package_json,
                decision_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                package_hash,
                package_json,
                decision_json,
                now,
                now,
            ),
        )

    return decision


def get_many_cached_decisions(
    packages: List[Dict[str, Any]],
) -> Tuple[
    Dict[str, Dict[str, Any]],
    List[Tuple[str, Dict[str, Any]]],
]:
    """
    Split packages into cached and uncached collections.

    Returns:
        cached:
            package_hash -> cached decision

        uncached:
            list of (package_hash, package)
    """
    cached: Dict[str, Dict[str, Any]] = {}
    uncached: List[Tuple[str, Dict[str, Any]]] = []

    for package in packages:
        package_hash = hash_json(package)
        decision = get_cached_decision(package_hash)

        if decision is None:
            uncached.append((package_hash, package))
        else:
            cached[package_hash] = decision

    return cached, uncached


def save_many_cached_decisions(
    package_decisions: List[
        Tuple[str, Dict[str, Any], Dict[str, Any]]
    ],
) -> None:
    for package_hash, package, decision in package_decisions:
        save_cached_decision(
            package_hash=package_hash,
            package=package,
            decision=decision,
        )


def task_exists_for_other_principal(
    task_id: str,
    principal_hash: str,
) -> bool:
    """
    Internal helper only.

    Do not expose its result in an API error, because errors must not reveal
    whether another user's task exists.
    """
    with database_connection() as connection:
        row = connection.execute(
            """
            SELECT 1
            FROM tasks
            WHERE task_id = ?
              AND principal_hash != ?
            LIMIT 1
            """,
            (
                task_id,
                principal_hash,
            ),
        ).fetchone()

    return row is not None


def count_tasks_for_principal(
    principal_hash: str,
) -> int:
    with database_connection() as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS task_count
            FROM tasks
            WHERE principal_hash = ?
            """,
            (principal_hash,),
        ).fetchone()

    return int(row["task_count"]) if row else 0


def delete_all_data() -> None:
    """
    Development/testing helper.

    Do not expose this through a public HTTP route.
    """
    with database_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM processed_continuations")
        connection.execute("DELETE FROM message_idempotency")
        connection.execute("DELETE FROM package_decision_cache")
        connection.execute("DELETE FROM task_locks")
        connection.execute("DELETE FROM tasks")