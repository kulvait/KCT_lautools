"""Discover beamtime directories, including README-only archive stubs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
    readme_mtime: datetime | None = None

    @property
    def archived(self) -> bool:
        return (
            self.readme_text is not None
            and "archived" in self.readme_text.lower()
        )

    @property
    def has_data(self) -> bool:
        """At least one data area is really present on GPFS."""
        return self.raw_exists or self.processed_exists or self.scratch_cc_exists

    @property
    def is_stub(self) -> bool:
        """Only a reference (metadata + README) remains; no data on GPFS."""
        return (
            not self.has_data
            and self.readme_text is not None
            and self.has_metadata
        )

    @property
    def accepted(self) -> bool:
        return self.has_data or self.is_stub


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

def parse_metadata_text(text: str) -> tuple[dict, str]:
    """Parse beamtime metadata, tolerating text around the JSON object.

    Older DOOR dumps wrap the object in explanatory lines, e.g.
    "The following metadata are a dump from DOOR ..." before it and
    "file created at: ..." after it. The object starting at the first
    "{" is decoded and everything outside it is ignored.

    Returns (data, json_text), where json_text is only the object itself.
    Raises ValueError if no JSON object can be decoded.
    """
    try:
        data = json.loads(text)
    except ValueError:
        start = text.find("{")
        if start < 0:
            raise ValueError("No JSON object found in metadata") from None
        # raw_decode stops at the end of the first complete value, so
        # trailing text is ignored and braces inside strings are handled.
        data, end = json.JSONDecoder().raw_decode(text, start)
        json_text = text[start:end]
    else:
        json_text = text

    if not isinstance(data, dict):
        raise ValueError("Metadata is not a JSON object")
    return data, json_text

def _metadata_readable(path: Path) -> bool:
    for name in (f"beamtime-metadata-{path.name}.json", "metadata.json", f"beamtime-metadata-{path.name}.txt"):
        try:
            text = (path / name).read_text(encoding="utf-8")
            data, _ = parse_metadata_text(text)
        except (OSError, ValueError):
            continue
        metadata_id = data.get("beamtime_id")
        if metadata_id is None or str(metadata_id) == path.name:
            return True
    return False


def _read_readme(path: Path) -> tuple[str | None, datetime | None]:
    readme = path / "README.txt"
    try:
        text = readme.read_text(encoding="utf-8")
        mtime = datetime.fromtimestamp(readme.stat().st_mtime).replace(
            microsecond=0
        )
    except (OSError, UnicodeError):
        return None, None
    return text, mtime


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

    readme, readme_mtime = (
        _read_readme(path) if not scratch_exists else (None, None)
    )
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
        readme_mtime=readme_mtime,
    )


def scan_beamtimes(
    base: Path = DEFAULT_BASE,
    numeric_ids_only: bool = True,
    listener: ScanListener | None = None,
    cancel: threading.Event | None = None,
    known_on_gpfs: Iterable[Path] = (),
) -> list[BeamtimeCandidate]:
    """Scan everything below `base`.

    OFFLOADED is emitted only for known on-GPFS roots that demonstrably no
    longer exist, never after cancellation or below unreadable branches.
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

    for key, path in known.items():
        if key in seen:
            continue
        if any(
            key == path_key(error) or key.startswith(path_key(error) + os.sep)
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

def refresh_beamtimes(
    paths: Iterable[Path],
    known_on_gpfs: Iterable[Path] = (),
    listener: ScanListener | None = None,
    cancel: threading.Event | None = None,
) -> list[BeamtimeCandidate]:
    """Re-inspect known beamtime roots without crawling the GPFS tree.

    Emits the same events as scan_beamtimes. OFFLOADED is emitted only when
    a root recorded as on GPFS demonstrably no longer exists; unreadable or
    unrecognizable directories are left untouched.
    """
    emit = listener or (lambda event: None)
    known = {path_key(p) for p in known_on_gpfs}
    candidates: list[BeamtimeCandidate] = []
    scanned = 0
    emit(ScanEvent(ScanEventKind.STARTED))

    for path in paths:
        if cancel is not None and cancel.is_set():
            emit(ScanEvent(
                ScanEventKind.CANCELLED, scanned=scanned, found=len(candidates)
            ))
            return candidates

        path = Path(path)
        scanned += 1
        emit(ScanEvent(
            ScanEventKind.PROGRESS,
            scanned=scanned,
            found=len(candidates),
            current=path,
        ))

        candidate = inspect_candidate(path)
        if candidate is not None and candidate.accepted:
            candidates.append(candidate)
            emit(ScanEvent(
                ScanEventKind.FOUND,
                candidate=candidate,
                scanned=scanned,
                found=len(candidates),
            ))
            continue

        if path_key(path) in known:
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

    def start_refresh(
        self,
        paths: Iterable[Path],
        known_on_gpfs: Iterable[Path] = (),
    ) -> None:
        """Re-inspect known beamtime roots on the worker thread."""
        if self.running:
            return
        self._cancel.clear()
        paths = tuple(paths)
        known = tuple(known_on_gpfs)

        def run() -> None:
            try:
                refresh_beamtimes(
                    paths,
                    known_on_gpfs=known,
                    listener=self._listener,
                    cancel=self._cancel,
                )
            except Exception as exc:
                log.exception("Beamtime refresh failed")
                self._listener(ScanEvent(ScanEventKind.FAILED, message=str(exc)))

        self._thread = threading.Thread(
            target=run, name="lautools-beamtime-refresh", daemon=True
        )
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
