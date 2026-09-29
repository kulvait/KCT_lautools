"""Discover beamtime directories without treating archive stubs as empty data."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
from pathlib import Path
import threading
from typing import Callable, Iterable, Iterator

log = logging.getLogger(__name__)

DEFAULT_BASE = Path("/asap3/petra3/gpfs")
BEAMTIME_DEPTH = 4


class ScanEventKind(str, Enum):
    STARTED = "started"
    PROGRESS = "progress"
    FOUND = "found"
    OFFLOADED = "offloaded"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class BeamtimeCandidate:
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
    readme_text: str | None = None

    @property
    def archived(self) -> bool:
        return self.readme_text is not None and "archived" in self.readme_text.lower()

    @property
    def accepted(self) -> bool:
        # A storage area is sufficient. A README-only archive stub must
        # additionally have valid metadata so an arbitrary directory is not
        # mistaken for a beamtime.
        return (
            self.raw_exists
            or self.processed_exists
            or self.scratch_cc_exists
            or (self.readme_text is not None and self.has_metadata)
        )


@dataclass(frozen=True)
class ScanEvent:
    kind: ScanEventKind
    candidate: BeamtimeCandidate | None = None
    scanned: int = 0
    found: int = 0
    current: Path | None = None
    message: str | None = None


ScanListener = Callable[[ScanEvent], None]


def path_key(path: Path) -> str:
    """Compare GPFS paths without resolving potentially slow automounts."""
    return os.path.normpath(os.path.abspath(os.fspath(path)))


def _metadata_readable(path: Path) -> bool:
    for name in (f"beamtime-metadata-{path.name}.json", "metadata.json"):
        try:
            with (path / name).open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            metadata_id = data.get("beamtimeId")
            if metadata_id is None or str(metadata_id) == path.name:
                return True
    return False


def _read_readme(path: Path) -> str | None:
    try:
        return (path / "README.txt").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _probe_writable(scratch: Path) -> bool:
    """Test actual writability, cleaning up the probe afterwards."""
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


def _iter_depth(
    base: Path, depth: int, cancel: threading.Event | None,
    errors: list[Path],
) -> Iterator[Path]:
    """Yield depth-four directories; record unreadable branches."""
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
                            errors.append(Path(entry.path))
            except OSError:
                errors.append(directory)
        level = children
    yield from level


def inspect_candidate(path: Path) -> BeamtimeCandidate | None:
    """Inspect one directory. None means it could not be listed."""
    path = Path(path)
    try:
        with os.scandir(path) as entries:
            next(iter(entries), None)
    except OSError:
        return None

    raw = (path / "raw").is_dir()
    processed = (path / "processed").is_dir()
    scratch = path / "scratch_cc"
    scratch_exists = scratch.is_dir()
    parts = path.parts

    # README is relevant when scratch_cc is absent. It may also describe
    # a beamtime that still has raw or processed data.
    readme = _read_readme(path) if not scratch_exists else None
    return BeamtimeCandidate(
        path=path,
        beamtime_id=path.name,
        beamline=parts[-4] if len(parts) >= 4 else None,
        year=parts[-3] if len(parts) >= 3 else None,
        readable=True,
        raw_exists=raw,
        processed_exists=processed,
        scratch_cc_exists=scratch_exists,
        scratch_cc_writable=_probe_writable(scratch) if scratch_exists else False,
        has_metadata=_metadata_readable(path),
        readme_text=readme,
    )


def scan_beamtimes(
    base: Path = DEFAULT_BASE,
    writable_only: bool = False,
    numeric_ids_only: bool = True,
    listener: ScanListener | None = None,
    cancel: threading.Event | None = None,
    known_on_gpfs: Iterable[Path] = (),
) -> list[BeamtimeCandidate]:
    """Scan disk; emit OFFLOADED only for demonstrably missing known roots.

    An incomplete or unreadable tree cannot establish that an unseen
    beamtime disappeared. OFFLOADED is not emitted on cancellation.
    """
    base = Path(base)
    emit = listener or (lambda event: None)
    if not base.is_dir():
        emit(ScanEvent(
            ScanEventKind.FAILED,
            message=f"Base directory is not accessible: {base}",
        ))
        return []

    base_key = path_key(base)
    known = {
        path_key(p): Path(p)
        for p in known_on_gpfs
        if path_key(p).startswith(base_key + os.sep)
    }
    seen: set[str] = set()
    errors: list[Path] = []
    candidates: list[BeamtimeCandidate] = []
    scanned = 0
    emit(ScanEvent(ScanEventKind.STARTED, current=base))

    for path in _iter_depth(base, BEAMTIME_DEPTH, cancel, errors):
        if cancel is not None and cancel.is_set():
            emit(ScanEvent(
                ScanEventKind.CANCELLED, scanned=scanned, found=len(candidates)
            ))
            return candidates

        scanned += 1
        seen.add(path_key(path))
        emit(ScanEvent(
            ScanEventKind.PROGRESS,
            scanned=scanned,
            found=len(candidates),
            current=path,
        ))
        if numeric_ids_only and not path.name.isdigit():
            continue

        candidate = inspect_candidate(path)
        if candidate is None or not candidate.accepted:
            # Do not infer absence from an inaccessible or ambiguous root.
            continue
        if writable_only and not candidate.scratch_cc_writable:
            continue

        candidates.append(candidate)
        emit(ScanEvent(
            ScanEventKind.FOUND,
            candidate=candidate,
            scanned=scanned,
            found=len(candidates),
        ))

    if cancel is not None and cancel.is_set():
        emit(ScanEvent(
            ScanEventKind.CANCELLED, scanned=scanned, found=len(candidates)
        ))
        return candidates

    # If any branch was unreadable, no inference about missing entries
    # under that branch is safe. In particular, do not mark an existing
    # but unrecognized root as offloaded.
    for key, path in known.items():
        if key in seen:
            continue
        if any(
            key == path_key(error)
            or key.startswith(path_key(error) + os.sep)
            for error in errors
        ):
            continue
        try:
            exists = path.exists()
        except OSError:
            continue
        if not exists:
            emit(ScanEvent(
                ScanEventKind.OFFLOADED,
                scanned=scanned,
                found=len(candidates),
                current=path,
            ))

    emit(ScanEvent(
        ScanEventKind.FINISHED, scanned=scanned, found=len(candidates)
    ))
    return candidates


class BeamtimeScanner:
    """Run the filesystem walk outside the GUI thread."""

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
        known_on_gpfs: Iterable[Path] = (),
    ) -> None:
        if self.running:
            return
        self._cancel.clear()
        known = tuple(known_on_gpfs)

        def run() -> None:
            try:
                scan_beamtimes(
                    base=base,
                    writable_only=writable_only,
                    numeric_ids_only=numeric_ids_only,
                    known_on_gpfs=known,
                    listener=self._listener,
                    cancel=self._cancel,
                )
            except Exception as exc:
                log.exception("Beamtime scan failed")
                self._listener(ScanEvent(ScanEventKind.FAILED, message=str(exc)))

        self._thread = threading.Thread(
            target=run, name="lautools-beamtime-scan", daemon=True
        )
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
