"""Scan the GPFS tree for beamtime directories the user can access.

Mirrors the logic of scripts/find-beamtimes.sh: beamtimes live at depth 4
below the base (``<base>/<beamline>/<year>/data/<beamtime_id>``); directories
that cannot be listed are skipped, and scratch_cc writability is verified by
actually creating a file rather than trusting the permission bits.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import os
from pathlib import Path
import threading
from typing import Callable, Iterator

log = logging.getLogger(__name__)

DEFAULT_BASE = Path("/asap3/petra3/gpfs")
BEAMTIME_DEPTH = 4


class ScanEventKind(str, Enum):
    STARTED = "started"
    PROGRESS = "progress"
    FOUND = "found"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class BeamtimeCandidate:
    """One accessible beamtime directory found on disk."""

    path: Path
    beamtime_id: str
    beamline: str | None
    year: str | None
    readable: bool
    raw_exists: bool
    processed_exists: bool
    scratch_cc_exists: bool
    scratch_cc_writable: bool
    has_metadata: bool


@dataclass(frozen=True)
class ScanEvent:
    kind: ScanEventKind
    candidate: BeamtimeCandidate | None = None
    scanned: int = 0
    found: int = 0
    current: Path | None = None
    message: str | None = None


ScanListener = Callable[[ScanEvent], None]


def _is_listable(path: Path) -> bool:
    """Equivalent of `ls -1A "$d" >/dev/null 2>&1` in the shell version."""
    try:
        with os.scandir(path) as entries:
            next(iter(entries), None)
    except OSError:
        return False
    return True


def _probe_writable(scratch: Path) -> bool:
    """Create and remove a file; permission bits alone are not conclusive."""
    probe = scratch / f".access_test_{os.getuid()}_{os.getpid()}"
    try:
        fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError:
        return False

    try:
        os.close(fd)
    finally:
        try:
            os.unlink(probe)
        except OSError:
            log.warning("Could not remove access probe: %s", probe)
    return True


def _iter_depth(base: Path, depth: int, cancel: threading.Event | None) -> Iterator[Path]:
    """Yield directories exactly `depth` levels below `base`.

    Pruning at the target depth keeps the walk cheap: beamtime contents are
    never descended into.
    """
    level = [base]
    for _ in range(depth):
        children: list[Path] = []
        for directory in level:
            if cancel is not None and cancel.is_set():
                return
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                children.append(Path(entry.path))
                        except OSError:
                            continue
            except OSError:
                # Unreadable branch; the shell version silently skips these.
                continue
        level = children
    yield from level


def inspect_candidate(path: Path) -> BeamtimeCandidate | None:
    """Return a candidate for `path`, or None when it cannot be listed."""
    if not _is_listable(path):
        return None

    raw = path / "raw"
    processed = path / "processed"
    scratch = path / "scratch_cc"

    scratch_exists = scratch.is_dir()
    parts = path.parts

    return BeamtimeCandidate(
        path=path,
        beamtime_id=path.name,
        beamline=parts[-4] if len(parts) >= 4 else None,
        year=parts[-3] if len(parts) >= 3 else None,
        readable=True,
        raw_exists=raw.is_dir(),
        processed_exists=processed.is_dir(),
        scratch_cc_exists=scratch_exists,
        scratch_cc_writable=_probe_writable(scratch) if scratch_exists else False,
        has_metadata=(path / f"beamtime-metadata-{path.name}.json").is_file(),
    )


def scan_beamtimes(
    base: Path = DEFAULT_BASE,
    writable_only: bool = False,
    numeric_ids_only: bool = True,
    listener: ScanListener | None = None,
    cancel: threading.Event | None = None,
) -> list[BeamtimeCandidate]:
    """Walk `base` and return accessible beamtime directories.

    Runs synchronously; use BeamtimeScanner for a background scan.
    """
    base = Path(base)
    emit = listener or (lambda event: None)

    if not base.is_dir():
        emit(ScanEvent(
            kind=ScanEventKind.FAILED,
            message=f"Base directory is not accessible: {base}",
        ))
        return []

    emit(ScanEvent(kind=ScanEventKind.STARTED, current=base))

    candidates: list[BeamtimeCandidate] = []
    scanned = 0

    for path in _iter_depth(base, BEAMTIME_DEPTH, cancel):
        if cancel is not None and cancel.is_set():
            emit(ScanEvent(
                kind=ScanEventKind.CANCELLED,
                scanned=scanned,
                found=len(candidates),
            ))
            return candidates

        scanned += 1
        emit(ScanEvent(
            kind=ScanEventKind.PROGRESS,
            scanned=scanned,
            found=len(candidates),
            current=path,
        ))

        if numeric_ids_only and not path.name.isdigit():
            continue

        candidate = inspect_candidate(path)
        if candidate is None:
            continue
        if writable_only and not candidate.scratch_cc_writable:
            continue

        candidates.append(candidate)
        emit(ScanEvent(
            kind=ScanEventKind.FOUND,
            candidate=candidate,
            scanned=scanned,
            found=len(candidates),
        ))

    emit(ScanEvent(
        kind=ScanEventKind.FINISHED,
        scanned=scanned,
        found=len(candidates),
    ))
    return candidates


class BeamtimeScanner:
    """Runs scan_beamtimes on a worker thread and reports events."""

    def __init__(self, listener: ScanListener):
        self._listener = listener
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        base: Path = DEFAULT_BASE,
        writable_only: bool = False,
        numeric_ids_only: bool = True,
    ) -> None:
        if self.running:
            return

        self._cancel.clear()

        def run() -> None:
            try:
                scan_beamtimes(
                    base=base,
                    writable_only=writable_only,
                    numeric_ids_only=numeric_ids_only,
                    listener=self._listener,
                    cancel=self._cancel,
                )
            except Exception as exc:
                log.exception("Beamtime scan failed")
                self._listener(ScanEvent(
                    kind=ScanEventKind.FAILED, message=str(exc),
                ))

        self._thread = threading.Thread(
            target=run, name="lautools-beamtime-scan", daemon=True,
        )
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
