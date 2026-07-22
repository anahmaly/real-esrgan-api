from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import pytest

from image_api import state as state_module
from image_api.state import StateMigrationLimit, initialize_state, state_write_ready

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ("image-api", "upscale-worker", "background-worker", "generation-worker")


def test_legacy_root_owned_state_is_migrated_to_shared_numeric_identity(
    tmp_path: Path, monkeypatch
) -> None:
    state = tmp_path / "state"
    sources = state / "sources"
    sources.mkdir(parents=True)
    lock = sources / ".source-files.lock"
    lock.write_bytes(b"")
    database = state / "tasks.sqlite3"
    sqlite3.connect(database).close()
    outside = tmp_path / "outside"
    outside.touch()
    symlink = state / "outside-link"
    symlink.symlink_to(outside)
    ownership: list[tuple[Path, int, int]] = []

    def record_fchown(fd: int, uid: int, gid: int) -> None:
        ownership.append((Path(os.readlink(f"/proc/self/fd/{fd}")), uid, gid))

    monkeypatch.setattr(os, "fchown", record_fchown)

    initialize_state(state, uid=10001, gid=10001, max_entries=100)

    assert (lock, 10001, 10001) in ownership
    assert (database, 10001, 10001) in ownership
    assert not any(path == symlink for path, _, _ in ownership)
    assert all((uid, gid) == (10001, 10001) for _, uid, gid in ownership)


def test_state_migration_is_bounded(tmp_path: Path, monkeypatch) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "one").touch()
    (state / "two").touch()
    monkeypatch.setattr(os, "fchown", lambda *_args, **_kwargs: None)

    with pytest.raises(StateMigrationLimit):
        initialize_state(state, uid=10001, gid=10001, max_entries=2)


def test_state_migration_rejects_excess_depth_before_metadata_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    state = tmp_path / "state"
    (state / "one" / "two" / "three").mkdir(parents=True)
    mutations: list[str] = []
    monkeypatch.setattr(os, "fchown", lambda *_args: mutations.append("chown"))
    monkeypatch.setattr(os, "fchmod", lambda *_args: mutations.append("chmod"))

    with pytest.raises(StateMigrationLimit, match="depth limit exceeded"):
        initialize_state(state, uid=10001, gid=10001, max_entries=100, max_depth=2)

    assert mutations == []


def test_state_migration_within_depth_bound_is_complete_and_closes_descriptors(
    tmp_path: Path, monkeypatch
) -> None:
    state = tmp_path / "state"
    leaf = state / "one" / "two" / "payload"
    leaf.parent.mkdir(parents=True)
    leaf.write_bytes(b"legacy")
    real_scandir = state_module.os.scandir
    active_iterators = 0
    peak_iterators = 0
    migrated: list[Path] = []
    migrated_fds: set[int] = set()

    class TrackingScandir:
        def __init__(self, fd: int) -> None:
            nonlocal active_iterators, peak_iterators
            self._iterator = real_scandir(fd)
            self._closed = False
            active_iterators += 1
            peak_iterators = max(peak_iterators, active_iterators)

        def __next__(self) -> os.DirEntry[str]:
            return next(self._iterator)

        def close(self) -> None:
            nonlocal active_iterators
            if not self._closed:
                self._iterator.close()
                self._closed = True
                active_iterators -= 1

    def tracking_scandir(fd: int) -> TrackingScandir:
        return TrackingScandir(fd)

    def record_fchown(fd: int, _uid: int, _gid: int) -> None:
        migrated.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
        migrated_fds.add(fd)

    monkeypatch.setattr(state_module.os, "scandir", tracking_scandir)
    monkeypatch.setattr(state_module.os, "fchown", record_fchown)
    monkeypatch.setattr(state_module.os, "fchmod", lambda *_args: None)

    initialize_state(state, uid=10001, gid=10001, max_entries=100, max_depth=2)

    assert {state, state / ".state-init.lock", state / "one", leaf.parent, leaf} == set(migrated)
    assert peak_iterators == 3
    assert active_iterators == 0
    for fd in migrated_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_state_migration_depth_limit_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="depth limit must be positive"):
        initialize_state(tmp_path / "state", max_depth=0)


def test_state_migration_does_not_scan_a_different_device_directory(
    tmp_path: Path, monkeypatch
) -> None:
    state = tmp_path / "state"
    mounted = state / "mounted"
    mounted.mkdir(parents=True)
    (mounted / "foreign").touch()
    real_stat = state_module.os.stat
    real_scandir = state_module.os.scandir
    scanned: list[Path] = []

    def simulated_stat(
        path: str | bytes,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        result = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if path == "mounted" and dir_fd is not None and not follow_symlinks:
            values = list(result)
            values[2] = result.st_dev + 1
            return os.stat_result(values)
        return result

    def recording_scandir(path: str | bytes | int):
        resolved = (
            Path(os.readlink(f"/proc/self/fd/{path}"))
            if isinstance(path, int)
            else Path(os.fsdecode(path))
        )
        scanned.append(resolved)
        return real_scandir(path)

    monkeypatch.setattr(state_module.os, "stat", simulated_stat)
    monkeypatch.setattr(state_module.os, "scandir", recording_scandir)
    monkeypatch.setattr(state_module.os, "fchown", lambda *_args, **_kwargs: None)

    initialize_state(state, uid=10001, gid=10001, max_entries=100)

    assert mounted not in scanned


def test_state_migration_does_not_follow_directory_swapped_to_symlink(
    tmp_path: Path, monkeypatch
) -> None:
    state = tmp_path / "state"
    raced = state / "raced"
    raced.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret"
    secret.touch()
    moved = state / "raced-before-swap"
    real_open = state_module.os.open
    real_fchown = state_module.os.fchown
    touched: list[Path] = []
    swapped = False

    def swap_before_nofollow_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "raced" and dir_fd is not None and not swapped:
            raced.rename(moved)
            raced.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def recording_fchown(fd: int, uid: int, gid: int) -> None:
        touched.append(Path(os.readlink(f"/proc/self/fd/{fd}")).resolve(strict=False))
        real_fchown(fd, uid, gid)

    monkeypatch.setattr(state_module.os, "open", swap_before_nofollow_open)
    monkeypatch.setattr(state_module.os, "fchown", recording_fchown)

    initialize_state(state, uid=os.getuid(), gid=os.getgid(), max_entries=100)

    assert swapped is True
    assert secret not in touched


def test_state_readiness_proves_source_lock_file_and_sqlite_writes(tmp_path: Path) -> None:
    state = tmp_path / "state"

    assert state_write_ready(state, state / "tasks.sqlite3", state / "sources") is True
    assert (state / "sources" / ".source-files.lock").is_file()
    assert (state / "tasks.sqlite3").is_file()


def test_state_readiness_preserves_legacy_marker_and_conflicting_table(tmp_path: Path) -> None:
    state = tmp_path / "state"
    sources = state / "sources"
    sources.mkdir(parents=True)
    marker = sources / ".write-readiness"
    marker.write_bytes(b"must remain unchanged")
    database = state / "tasks.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 77")
        connection.execute(
            "CREATE TABLE state_readiness (singleton INTEGER PRIMARY KEY, value BLOB)"
        )
        connection.execute(
            "INSERT INTO state_readiness(singleton, value) VALUES (?, ?)", (1, b"application-data")
        )

    assert state_write_ready(state, database, sources) is True

    assert marker.read_bytes() == b"must remain unchanged"
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (77,)
        assert connection.execute("SELECT singleton, value FROM state_readiness").fetchall() == [
            (1, b"application-data")
        ]


def test_state_readiness_fails_closed_when_source_lock_is_contended(tmp_path: Path) -> None:
    state = tmp_path / "state"
    sources = state / "sources"
    sources.mkdir(parents=True)
    lock_path = sources / ".source-files.lock"
    with lock_path.open("a+b") as lock:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert state_write_ready(state, state / "tasks.sqlite3", sources) is False


def test_compose_uses_bounded_root_init_and_waits_before_every_shared_writer() -> None:
    compose = (ROOT / "compose.yml").read_text()

    assert "state-init:" in compose
    assert 'user: "0:0"' in compose
    assert "python -m image_api.state init" in compose
    assert "IMAGE_API_STATE_INIT_MAX_ENTRIES" in compose
    assert "IMAGE_API_STATE_INIT_MAX_DEPTH" in compose
    for service in SERVICES:
        match = re.search(rf"(?ms)^  {re.escape(service)}:\n(?P<body>.*?)(?=^  \S|\Z)", compose)
        assert match is not None
        service_block = match.group("body")
        assert 'user: "10001:10001"' in service_block
        assert "state-init:" in service_block
        assert "condition: service_completed_successfully" in service_block


def test_runtime_images_define_the_same_uid_and_gid() -> None:
    for name in (
        "Dockerfile.gateway",
        "Dockerfile.test",
        "Dockerfile.upscale",
        "Dockerfile.background",
        "Dockerfile.generation",
    ):
        text = (ROOT / name).read_text()
        assert "--gid 10001" in text
        assert "--uid 10001 --gid 10001" in text
        assert "USER 10001:10001" in text


def test_model_mounts_remain_read_only() -> None:
    compose = (ROOT / "compose.yml").read_text()
    model_mounts = [line for line in compose.splitlines() if ":/models/" in line]
    assert model_mounts
    assert all(line.rstrip().endswith(":ro") for line in model_mounts)
