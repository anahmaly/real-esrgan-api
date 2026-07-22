from __future__ import annotations

import argparse
import errno
import logging
import os
import secrets
import sqlite3
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import fcntl

logger = logging.getLogger(__name__)
STATE_UID = 10001
STATE_GID = 10001
DEFAULT_MAX_ENTRIES = 100_000
DEFAULT_MAX_DEPTH = 32
SOURCE_LOCK_NAME = ".source-files.lock"
STATE_INIT_LOCK_NAME = ".state-init.lock"
READINESS_FILE_PREFIX = ".write-readiness."


class StateMigrationLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class _StateEntry:
    path: Path
    fd: int
    metadata: os.stat_result


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _open_verified_entry(
    name: str,
    *,
    parent_fd: int,
    metadata: os.stat_result,
) -> int | None:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if stat.S_ISDIR(metadata.st_mode):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}:
            return None
        raise
    opened = os.fstat(fd)
    if not _same_inode(metadata, opened) or stat.S_IFMT(metadata.st_mode) != stat.S_IFMT(
        opened.st_mode
    ):
        os.close(fd)
        return None
    return fd


def _state_entries(
    root_fd: int,
    root: Path,
    max_entries: int,
    max_depth: int,
) -> Iterator[_StateEntry]:
    root_metadata = os.fstat(root_fd)
    root_device = root_metadata.st_dev
    root_entry = _StateEntry(root, os.dup(root_fd), root_metadata)
    stack: list[tuple[_StateEntry, Any, int]] = []
    unstacked_fd: int | None = root_entry.fd
    seen = 1
    try:
        yield root_entry
        if stat.S_ISDIR(root_entry.metadata.st_mode):
            stack.append((root_entry, os.scandir(root_entry.fd), 0))
            unstacked_fd = None
        else:
            os.close(root_entry.fd)
            unstacked_fd = None
        while stack:
            parent, directory_entries, parent_depth = stack[-1]
            try:
                directory_entry = next(directory_entries)
            except StopIteration:
                directory_entries.close()
                os.close(parent.fd)
                stack.pop()
                continue
            seen += 1
            if seen > max_entries:
                raise StateMigrationLimit("state migration entry limit exceeded")
            try:
                metadata = os.stat(
                    directory_entry.name,
                    dir_fd=parent.fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            if metadata.st_dev != root_device or stat.S_ISLNK(metadata.st_mode):
                continue
            if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
                continue
            child_depth = parent_depth + 1
            if stat.S_ISDIR(metadata.st_mode) and child_depth > max_depth:
                raise StateMigrationLimit("state migration depth limit exceeded")
            child_fd = _open_verified_entry(
                directory_entry.name,
                parent_fd=parent.fd,
                metadata=metadata,
            )
            if child_fd is None:
                continue
            child = _StateEntry(parent.path / directory_entry.name, child_fd, metadata)
            unstacked_fd = child_fd
            yield child
            if stat.S_ISDIR(metadata.st_mode):
                stack.append((child, os.scandir(child_fd), child_depth))
                unstacked_fd = None
            else:
                os.close(child_fd)
                unstacked_fd = None
    finally:
        if unstacked_fd is not None:
            os.close(unstacked_fd)
        for entry, directory_entries, _depth in reversed(stack):
            directory_entries.close()
            os.close(entry.fd)


def _open_directory(path: Path) -> int:
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


def initialize_state(
    state_dir: Path,
    *,
    uid: int = STATE_UID,
    gid: int = STATE_GID,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> None:
    """Preflight and migrate a quiescent state volume within explicit resource bounds."""
    if max_entries < 1:
        raise ValueError("state migration entry limit must be positive")
    if max_depth < 1:
        raise ValueError("state migration depth limit must be positive")
    state_dir.mkdir(parents=True, exist_ok=True)
    root_fd = _open_directory(state_dir)
    lock_fd = -1
    try:
        lock_fd = os.open(
            STATE_INIT_LOCK_NAME,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("state migration is already running") from exc
        for _entry in _state_entries(root_fd, state_dir, max_entries, max_depth):
            pass
        for entry in _state_entries(root_fd, state_dir, max_entries, max_depth):
            os.fchown(entry.fd, uid, gid)
            if stat.S_ISDIR(entry.metadata.st_mode):
                os.fchmod(entry.fd, entry.metadata.st_mode | stat.S_IRWXU)
            else:
                os.fchmod(entry.fd, entry.metadata.st_mode | stat.S_IRUSR | stat.S_IWUSR)
    finally:
        if lock_fd >= 0:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(root_fd)


def _unlink_created_probe(source_fd: int, name: str, created: os.stat_result) -> None:
    try:
        current = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not _same_inode(current, created):
        raise OSError("readiness probe entry changed before cleanup")
    os.unlink(name, dir_fd=source_fd)


def _probe_source_write(source_dir: Path) -> None:
    source_fd = _open_directory(source_dir)
    lock_fd = -1
    try:
        lock_fd = os.open(
            SOURCE_LOCK_NAME,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
            dir_fd=source_fd,
        )
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        probe_name = f"{READINESS_FILE_PREFIX}{secrets.token_hex(16)}.tmp"
        probe_fd = os.open(
            probe_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=source_fd,
        )
        created: os.stat_result | None = None
        try:
            created = os.fstat(probe_fd)
            view = memoryview(b"ready")
            while view:
                written = os.write(probe_fd, view)
                if written < 1:
                    raise OSError("readiness probe write made no progress")
                view = view[written:]
            os.fsync(probe_fd)
        finally:
            os.close(probe_fd)
            if created is not None:
                _unlink_created_probe(source_fd, probe_name, created)
        os.fsync(source_fd)
    finally:
        if lock_fd >= 0:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(source_fd)


def _probe_sqlite_write(database_path: Path) -> None:
    with sqlite3.connect(database_path, timeout=5, isolation_level=None) as connection:
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("BEGIN IMMEDIATE")
        try:
            current = int(connection.execute("PRAGMA user_version").fetchone()[0])
            probe_value = 0 if current == 2**31 - 1 else current + 1
            connection.execute(f"PRAGMA user_version = {probe_value}")
        finally:
            connection.rollback()


def state_write_ready(state_dir: Path, database_path: Path, source_dir: Path) -> bool:
    """Exercise source publication and a rollback-only SQLite write without persistent data."""
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        source_dir.mkdir(parents=True, exist_ok=True)
        _probe_source_write(source_dir)
        _probe_sqlite_write(database_path)
        return True
    except (OSError, sqlite3.Error) as exc:
        logger.error(
            "shared state write readiness failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "check"))
    args = parser.parse_args()
    state = Path(os.getenv("IMAGE_API_STATE_DIR", "/state"))
    if args.command == "init":
        initialize_state(
            state,
            uid=int(os.getenv("IMAGE_API_STATE_UID", str(STATE_UID))),
            gid=int(os.getenv("IMAGE_API_STATE_GID", str(STATE_GID))),
            max_entries=int(
                os.getenv("IMAGE_API_STATE_INIT_MAX_ENTRIES", str(DEFAULT_MAX_ENTRIES))
            ),
            max_depth=int(os.getenv("IMAGE_API_STATE_INIT_MAX_DEPTH", str(DEFAULT_MAX_DEPTH))),
        )
        return
    if not state_write_ready(state, state / "tasks.sqlite3", state / "sources"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
