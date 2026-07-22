from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

TaskStatus = Literal["queued", "running", "succeeded", "failed"]
TaskKind = Literal["generation", "upscale", "background-removal"]
TASK_KINDS: tuple[TaskKind, ...] = ("generation", "upscale", "background-removal")


class IdempotencyConflict(RuntimeError):
    pass


class QueueFull(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    task_kind: TaskKind
    status: TaskStatus
    request: dict[str, Any]
    error_code: str | None
    image_name: str | None
    output_sha256: str | None
    output_width: int | None
    output_height: int | None
    output_mode: str | None
    created_at: int
    updated_at: int


class TaskStore:
    def __init__(self, path: Path, max_queue_depth: int = 100) -> None:
        self.path = Path(path)
        self.max_queue_depth = max_queue_depth
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS generation_tasks (
                    task_id TEXT PRIMARY KEY,
                    idempotency_hash TEXT NOT NULL UNIQUE,
                    fingerprint TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed')),
                    worker_id TEXT,
                    error_code TEXT,
                    image_name TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    task_kind TEXT NOT NULL DEFAULT 'generation',
                    output_sha256 TEXT,
                    output_width INTEGER,
                    output_height INTEGER,
                    output_mode TEXT
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(generation_tasks)").fetchall()
            }
            migrations = {
                "task_kind": "TEXT NOT NULL DEFAULT 'generation'",
                "output_sha256": "TEXT",
                "output_width": "INTEGER",
                "output_height": "INTEGER",
                "output_mode": "TEXT",
            }
            for name, definition in migrations.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE generation_tasks ADD COLUMN {name} {definition}"
                    )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS generation_tasks_capability_claim "
                "ON generation_tasks(task_kind,status,created_at)"
            )

    @staticmethod
    def _canonical(request: dict[str, Any]) -> str:
        return json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _row(cls, row: sqlite3.Row) -> TaskRecord:
        task_kind = row["task_kind"]
        if task_kind not in TASK_KINDS:
            raise RuntimeError("persisted task has invalid kind")
        return TaskRecord(
            task_id=row["task_id"],
            task_kind=task_kind,
            status=row["status"],
            request=json.loads(row["request_json"]),
            error_code=row["error_code"],
            image_name=row["image_name"],
            output_sha256=row["output_sha256"],
            output_width=row["output_width"],
            output_height=row["output_height"],
            output_mode=row["output_mode"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def admit(
        self,
        idempotency_key: str,
        request: dict[str, Any],
        task_kind: TaskKind = "generation",
    ) -> TaskRecord:
        canonical = self._canonical(request)
        fingerprint = self._hash(self._canonical({"task_kind": task_kind, "request": request}))
        key_hash = self._hash(idempotency_key)
        now = time.time_ns()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM generation_tasks WHERE idempotency_hash = ?", (key_hash,)
            ).fetchone()
            if existing is not None:
                connection.commit()
                if existing["fingerprint"] != fingerprint:
                    # Legacy rows used a request-only fingerprint. Preserve exact legacy replay.
                    legacy = self._hash(canonical)
                    if not (
                        existing["task_kind"] == "generation" and existing["fingerprint"] == legacy
                    ):
                        raise IdempotencyConflict(
                            "idempotency key already identifies another request"
                        )
                return self._row(existing)
            depth = connection.execute(
                "SELECT COUNT(*) FROM generation_tasks WHERE status IN ('queued','running')"
            ).fetchone()[0]
            if depth >= self.max_queue_depth:
                connection.rollback()
                raise QueueFull("task queue is full")
            task_id = uuid.uuid4().hex
            connection.execute(
                """INSERT INTO generation_tasks
                   (task_id,idempotency_hash,fingerprint,request_json,status,task_kind,
                    created_at,updated_at)
                   VALUES (?,?,?,?, 'queued',?,?,?)""",
                (task_id, key_hash, fingerprint, canonical, task_kind, now, now),
            )
            row = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            connection.commit()
            return self._row(row)

    def get(self, task_id: str) -> TaskRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return self._row(row)

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM generation_tasks").fetchone()[0])

    def source_referenced(self, source_name: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT 1 FROM generation_tasks
                   WHERE status IN ('queued','running')
                     AND json_extract(request_json, '$.source_image_name') = ? LIMIT 1""",
                (source_name,),
            ).fetchone()
        return row is not None

    def active_source_names(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT DISTINCT json_extract(request_json, '$.source_image_name') AS source_name
                   FROM generation_tasks
                   WHERE status IN ('queued','running')
                     AND json_type(request_json, '$.source_image_name') = 'text'"""
            ).fetchall()
        return {str(row["source_name"]) for row in rows}

    def claim_next(self, worker_id: str, task_kind: TaskKind = "generation") -> TaskRecord | None:
        now = time.time_ns()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT task_id FROM generation_tasks
                   WHERE status='queued' AND task_kind=? ORDER BY created_at LIMIT 1""",
                (task_kind,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            task_id = row["task_id"]
            changed = connection.execute(
                """UPDATE generation_tasks SET status='running',worker_id=?,updated_at=?
                   WHERE task_id=? AND status='queued' AND task_kind=?""",
                (worker_id, now, task_id, task_kind),
            ).rowcount
            if changed != 1:
                connection.rollback()
                return None
            claimed = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            connection.commit()
            return self._row(claimed)

    def succeed(
        self,
        task_id: str,
        image_name: str,
        *,
        output_sha256: str | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
        output_mode: str | None = None,
    ) -> None:
        self._transition(
            task_id,
            "succeeded",
            image_name=image_name,
            output_sha256=output_sha256,
            output_width=output_width,
            output_height=output_height,
            output_mode=output_mode,
        )

    def fail(self, task_id: str, error_code: str) -> None:
        self._transition(task_id, "failed", error_code=error_code[:64])

    def _transition(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        error_code: str | None = None,
        image_name: str | None = None,
        output_sha256: str | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
        output_mode: str | None = None,
    ) -> None:
        with self._connect() as connection:
            changed = connection.execute(
                """UPDATE generation_tasks
                   SET status=?,error_code=?,image_name=?,output_sha256=?,output_width=?,
                       output_height=?,output_mode=?,worker_id=NULL,updated_at=?
                   WHERE task_id=? AND status='running'""",
                (
                    status,
                    error_code,
                    image_name,
                    output_sha256,
                    output_width,
                    output_height,
                    output_mode,
                    time.time_ns(),
                    task_id,
                ),
            ).rowcount
        if changed != 1:
            raise RuntimeError("invalid task state transition")

    def running(self, task_kind: TaskKind | None = None) -> list[TaskRecord]:
        with self._connect() as connection:
            if task_kind is None:
                rows = connection.execute(
                    "SELECT * FROM generation_tasks WHERE status='running' ORDER BY created_at"
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT * FROM generation_tasks
                       WHERE status='running' AND task_kind=? ORDER BY created_at""",
                    (task_kind,),
                ).fetchall()
        return [self._row(row) for row in rows]

    def reconcile_success(
        self,
        task_id: str,
        image_name: str,
        *,
        output_sha256: str | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
        output_mode: str | None = None,
    ) -> bool:
        """Idempotently bind a validated canonical output to a running task."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT status,image_name,output_sha256,output_width,output_height,output_mode
                   FROM generation_tasks WHERE task_id=?""",
                (task_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return False
            expected_metadata = (
                image_name,
                output_sha256,
                output_width,
                output_height,
                output_mode,
            )
            stored_metadata = (
                row["image_name"],
                row["output_sha256"],
                row["output_width"],
                row["output_height"],
                row["output_mode"],
            )
            if row["status"] == "succeeded" and stored_metadata == expected_metadata:
                connection.commit()
                return True
            if row["status"] != "running":
                connection.rollback()
                return False
            connection.execute(
                """UPDATE generation_tasks
                   SET status='succeeded',error_code=NULL,image_name=?,output_sha256=?,
                       output_width=?,output_height=?,output_mode=?,worker_id=NULL,updated_at=?
                   WHERE task_id=? AND status='running'""",
                (
                    image_name,
                    output_sha256,
                    output_width,
                    output_height,
                    output_mode,
                    time.time_ns(),
                    task_id,
                ),
            )
            connection.commit()
            return True
