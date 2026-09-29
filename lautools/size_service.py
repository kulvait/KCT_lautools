"""Background directory-size service.

Components request sizes by path. Worker threads count them, update every
matching database row (projects, workspaces and beamtime storage areas) and
notify subscribed listeners.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import itertools
import logging
import os
from pathlib import Path
import queue
import sqlite3
import threading
import time
from typing import Callable, Iterable

log = logging.getLogger(__name__)

BEAMTIME_AREAS = ("raw", "processed", "scratch_cc")
MAX_WORKERS = 4


class SizeEventKind(str, Enum):
    QUEUED = "queued"
    STARTED = "started"
    PROGRESS = "progress"
    FINISHED = "finished"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SizeEvent:
    kind: SizeEventKind
    path: Path
    size_bytes: int | None = None
    # Every directory size measured by this scan: the root, its immediate
    # children and every registered project/workspace/beamtime area below it.
    sizes: dict[Path, int] = field(default_factory=dict)
    files_scanned: int = 0
    errors: int = 0
    updated_projects: tuple[int, ...] = ()
    updated_workspaces: tuple[int, ...] = ()
    updated_beamtimes: tuple[int, ...] = ()
    message: str | None = None

    def concerns(self, path: Path) -> bool:
        """True if this event carries information about `path`."""
        path = Path(path)
        return path == self.path or path in self.sizes


SizeListener = Callable[[SizeEvent], None]


@dataclass
class _Job:
    path: Path
    force: bool
    seq: int


@dataclass
class _ScanResult:
    total: int
    sizes: dict[Path, int]
    files: int
    errors: int


class _Cancelled(Exception):
    pass


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def scan_directory(
    root: Path,
    tracked: Iterable[Path] = (),
    cancel: threading.Event | None = None,
    progress: Callable[[int, int], None] | None = None,
    progress_interval_s: float = 5.0,
) -> _ScanResult:
    """Count apparent file sizes below `root` in a single pass.

    Records the total for `root`, for each immediate child directory and for
    every directory listed in `tracked`. Symlinks are not followed; hard-linked
    files are counted once; unreadable entries increase `errors`.
    """
    tracked_str = {str(p) for p in tracked}
    sizes: dict[str, int] = {}
    seen_inodes: set[tuple[int, int]] = set()
    files = 0
    errors = 0
    running_total = 0
    last_progress = time.monotonic()

    def walk(path_str: str, depth: int) -> int:
        nonlocal files, errors, running_total, last_progress

        if cancel is not None and cancel.is_set():
            raise _Cancelled()

        total = 0
        try:
            iterator = os.scandir(path_str)
        except OSError:
            errors += 1
            return 0

        with iterator:
            for entry in iterator:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        total += walk(entry.path, depth + 1)
                        continue
                    if entry.is_symlink():
                        continue
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    errors += 1
                    continue

                if st.st_nlink > 1:
                    key = (st.st_dev, st.st_ino)
                    if key in seen_inodes:
                        continue
                    seen_inodes.add(key)

                total += st.st_size
                running_total += st.st_size
                files += 1

                if progress is not None:
                    now = time.monotonic()
                    if now - last_progress >= progress_interval_s:
                        last_progress = now
                        progress(running_total, files)

        if depth <= 1 or path_str in tracked_str:
            sizes[path_str] = total
        return total

    total = walk(str(root), 0)
    return _ScanResult(
        total=total,
        sizes={Path(p): s for p, s in sizes.items()},
        files=files,
        errors=errors,
    )


class SizeService:
    """Queue of directory-size requests processed by a few worker threads."""

    def __init__(
        self,
        db_path: Path,
        workers: int = 2,
        cooldown_s: float = 60.0,
        progress_interval_s: float = 5.0,
    ):
        self.db_path = Path(db_path)
        self.workers = max(1, min(int(workers), MAX_WORKERS))
        self.cooldown_s = cooldown_s
        self.progress_interval_s = progress_interval_s

        self._queue: queue.Queue[_Job | None] = queue.Queue()
        self._lock = threading.RLock()
        self._pending: dict[Path, _Job] = {}
        self._running: set[Path] = set()
        self._recent: dict[Path, tuple[float, int]] = {}
        self._listeners: list[SizeListener] = []
        self._threads: list[threading.Thread] = []
        self._cancel = threading.Event()
        self._seq = itertools.count()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._threads:
            return
        self._cancel.clear()
        for index in range(self.workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"lautools-size-{index}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout: float = 5.0) -> None:
        """Cancel running scans and stop workers."""
        self._cancel.set()
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout)
        self._threads.clear()
        with self._lock:
            self._pending.clear()

    # ------------------------------------------------------------------
    # Listeners
    # ------------------------------------------------------------------

    def subscribe(self, listener: SizeListener) -> Callable[[], None]:
        """Register a listener; returns a function that unsubscribes it.

        Listeners are called from worker threads (or from the requesting
        thread for QUEUED/SKIPPED). GUI code must use the Qt bridge.
        """
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def _emit(self, event: SizeEvent) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                log.exception("Size listener failed")

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    def request(self, path: Path, force: bool = False) -> bool:
        """Queue a size count of `path`.

        Returns True if queued (or already queued/running), False if skipped
        because a result younger than the cooldown exists.
        """
        path = _resolve(Path(path))
        skipped_size: int | None = None

        with self._lock:
            pending = self._pending.get(path)
            if pending is not None:
                pending.force = pending.force or force
                return True

            if path in self._running and not force:
                return True

            if not force:
                recent = self._recent.get(path)
                if recent is not None:
                    age = time.monotonic() - recent[0]
                    if age < self.cooldown_s:
                        skipped_size = recent[1]

            if skipped_size is None:
                job = _Job(path=path, force=force, seq=next(self._seq))
                self._pending[path] = job

        if skipped_size is not None:
            self._emit(SizeEvent(
                kind=SizeEventKind.SKIPPED,
                path=path,
                size_bytes=skipped_size,
                sizes={path: skipped_size},
                message=f"Counted less than {self.cooldown_s:.0f} s ago",
            ))
            return False

        self._queue.put(job)
        self._emit(SizeEvent(kind=SizeEventKind.QUEUED, path=path))
        return True

    def request_many(self, paths: Iterable[Path], force: bool = False) -> None:
        for path in paths:
            self.request(path, force=force)

    def request_beamtime(
        self,
        core_path: Path,
        areas: Iterable[str] = BEAMTIME_AREAS,
        force: bool = False,
    ) -> None:
        """Queue beamtime areas. A scratch_cc scan also updates every
        project and workspace below it."""
        for area in areas:
            if area not in BEAMTIME_AREAS:
                raise ValueError(f"Unknown beamtime area: {area}")
            self.request(Path(core_path) / area, force=force)

    def cached_size(self, path: Path) -> int | None:
        """Last size measured during this session, regardless of age."""
        with self._lock:
            recent = self._recent.get(_resolve(Path(path)))
        return recent[1] if recent else None

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _worker_loop(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            while True:
                job = self._queue.get()
                if job is None or self._cancel.is_set():
                    break

                with self._lock:
                    if self._pending.get(job.path) is job:
                        del self._pending[job.path]
                    self._running.add(job.path)

                try:
                    if connection is None:
                        connection = self._connect()
                    self._run_job(connection, job)
                except _Cancelled:
                    self._emit(SizeEvent(
                        kind=SizeEventKind.CANCELLED, path=job.path,
                    ))
                except Exception as exc:
                    log.exception("Size count failed for %s", job.path)
                    self._emit(SizeEvent(
                        kind=SizeEventKind.FAILED,
                        path=job.path,
                        message=str(exc),
                    ))
                finally:
                    with self._lock:
                        self._running.discard(job.path)
        finally:
            if connection is not None:
                connection.close()

    def _run_job(self, connection: sqlite3.Connection, job: _Job) -> None:
        root = job.path
        if not root.is_dir():
            self._emit(SizeEvent(
                kind=SizeEventKind.FAILED,
                path=root,
                message="Not an accessible directory",
            ))
            return

        self._emit(SizeEvent(kind=SizeEventKind.STARTED, path=root))

        tracked = self._tracked_paths(connection, root)

        def on_progress(size_bytes: int, files: int) -> None:
            self._emit(SizeEvent(
                kind=SizeEventKind.PROGRESS,
                path=root,
                size_bytes=size_bytes,
                files_scanned=files,
            ))

        result = scan_directory(
            root,
            tracked=tracked,
            cancel=self._cancel,
            progress=on_progress,
            progress_interval_s=self.progress_interval_s,
        )

        projects, workspaces, beamtimes = self._store(connection, result)

        finished = time.monotonic()
        with self._lock:
            for path, size in result.sizes.items():
                self._recent[path] = (finished, size)

        self._emit(SizeEvent(
            kind=SizeEventKind.FINISHED,
            path=root,
            size_bytes=result.total,
            sizes=result.sizes,
            files_scanned=result.files,
            errors=result.errors,
            updated_projects=projects,
            updated_workspaces=workspaces,
            updated_beamtimes=beamtimes,
            message=(
                f"{result.errors} entries could not be read"
                if result.errors else None
            ),
        ))

    # ------------------------------------------------------------------
    # Database mapping (by path)
    # ------------------------------------------------------------------

    @staticmethod
    def _beamtime_areas(
        connection: sqlite3.Connection,
    ) -> list[tuple[int, str, Path]]:
        """(beamtime row id, area name, resolved area path)."""
        areas = []
        rows = connection.execute(
            "SELECT id, core_path FROM beamtime WHERE core_path IS NOT NULL"
        ).fetchall()
        for beamtime_id, core_path in rows:
            core = _resolve(Path(core_path))
            for area in BEAMTIME_AREAS:
                areas.append((beamtime_id, area, core / area))
        return areas

    def _tracked_paths(
        self,
        connection: sqlite3.Connection,
        root: Path,
    ) -> set[Path]:
        candidates: list[Path] = []
        for (path,) in connection.execute("SELECT path FROM laupy_project"):
            candidates.append(Path(path))
        for (path,) in connection.execute(
            "SELECT path FROM laupy_project_workspace"
        ):
            candidates.append(Path(path))
        candidates.extend(p for _, _, p in self._beamtime_areas(connection))

        return {p for p in candidates if _is_within(p, root)}

    def _store(
        self,
        connection: sqlite3.Connection,
        result: _ScanResult,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        timestamp = datetime.now().isoformat(timespec="seconds")
        sizes = {str(p): s for p, s in result.sizes.items()}

        projects: list[int] = []
        for project_id, path in connection.execute(
            "SELECT id, path FROM laupy_project"
        ).fetchall():
            size = sizes.get(str(Path(path)))
            if size is None:
                continue
            connection.execute(
                """
                UPDATE laupy_project
                SET project_size_bytes = ?,
                    project_size_bytes_timestamp = ?,
                    last_inspected = ?
                WHERE id = ?
                """,
                (size, timestamp, timestamp, project_id),
            )
            projects.append(project_id)

        workspaces: list[int] = []
        for workspace_id, path in connection.execute(
            "SELECT id, path FROM laupy_project_workspace"
        ).fetchall():
            size = sizes.get(str(Path(path)))
            if size is None:
                continue
            connection.execute(
                """
                UPDATE laupy_project_workspace
                SET workspace_size_bytes = ?,
                    workspace_size_bytes_timestamp = ?,
                    last_inspected = ?
                WHERE id = ?
                """,
                (size, timestamp, timestamp, workspace_id),
            )
            workspaces.append(workspace_id)

        per_beamtime: dict[int, dict[str, int]] = {}
        for beamtime_id, area, area_path in self._beamtime_areas(connection):
            size = sizes.get(str(area_path))
            if size is not None:
                per_beamtime.setdefault(beamtime_id, {})[area] = size

        for beamtime_id, area_sizes in per_beamtime.items():
            connection.execute(
                """
                INSERT INTO beamtime_storage (beamtime_id)
                VALUES (?)
                ON CONFLICT(beamtime_id) DO NOTHING
                """,
                (beamtime_id,),
            )
            # Column names come from the fixed BEAMTIME_AREAS tuple.
            assignments = ", ".join(
                f"{area}_size_bytes = ?, {area}_size_bytes_timestamp = ?"
                for area in area_sizes
            )
            params: list[object] = []
            for size in area_sizes.values():
                params.extend([size, timestamp])
            connection.execute(
                f"UPDATE beamtime_storage SET {assignments}, "
                f"last_inspected = ? WHERE beamtime_id = ?",
                (*params, timestamp, beamtime_id),
            )

        connection.commit()
        return tuple(projects), tuple(workspaces), tuple(per_beamtime)

    def active_paths(self) -> dict[Path, str]:
        """Paths currently queued or being counted."""
        with self._lock:
            state = {path: "queued" for path in self._pending}
            state.update({path: "running" for path in self._running})
        return state
